from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Iterator

import gradio as gr
import numpy as np
import soundfile as sf
import torch


ROOT = Path(__file__).resolve().parent
LOCAL_ENGINE = ROOT / "BetterDeclipper"
if LOCAL_ENGINE.is_dir() and str(LOCAL_ENGINE) not in sys.path:
    sys.path.insert(0, str(LOCAL_ENGINE))

from betterdeclipper import __version__ as betterdeclipper_version  # noqa: E402
from betterdeclipper import declip  # noqa: E402
from betterdeclipper.auto import load_analysis, save_analysis  # noqa: E402
from betterdeclipper.cli import analysis_lines, restoration_label  # noqa: E402


OUTPUT_ROOT = Path(os.getenv("BETTERDECLIPPER_OUTPUT_DIR", "/content/betterdeclipper_resultados" if Path("/content").exists() else ROOT / "resultados_colab"))
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
PROCESS_LOCK = threading.Lock()
SUPPORTED_SUFFIXES = {".wav", ".flac", ".mp3"}


CSS = """
:root { --bd-green:#30d89b; --bd-panel:#151b24; --bd-border:#293443; }
.gradio-container { max-width: 1180px !important; margin: auto !important; background:#0b0f14 !important; }
.bd-hero { padding:24px; border:1px solid var(--bd-border); border-radius:18px;
  background:radial-gradient(circle at top right,rgba(48,216,155,.16),transparent 35%),linear-gradient(135deg,#151b24,#0d1219); }
.bd-hero h1 { margin:0 0 6px; color:#f7fafc; font-size:clamp(1.8rem,4vw,2.8rem); }
.bd-hero p { margin:0; color:#aab7c6; }
.bd-card { border:1px solid var(--bd-border) !important; border-radius:16px !important; background:var(--bd-panel) !important; }
.bd-note { color:#9eacba; font-size:.92rem; }
footer { display:none !important; }
"""


def _safe_name(value: str) -> str:
    value = re.sub(r"[^\w.-]+", "_", value, flags=re.UNICODE).strip("._")
    return value[:90] or "audio"


def _blank_state() -> dict[str, Any]:
    return {"source": None, "analysis": None, "analysis_path": None, "diagnosis": "", "job_dir": None}


def gpu_status() -> str:
    if not torch.cuda.is_available():
        return (
            f"⚠️ **CPU ativo** · PyTorch {torch.__version__} não confirmou CUDA. "
            "O processamento funciona, mas pode ser muito mais lento."
        )
    index = torch.cuda.current_device()
    name = torch.cuda.get_device_name(index)
    props = torch.cuda.get_device_properties(index)
    try:
        free, total = torch.cuda.mem_get_info(index)
        memory = f"{free / 2**30:.1f} GB livres de {total / 2**30:.1f} GB"
    except Exception:
        memory = f"{props.total_memory / 2**30:.1f} GB totais"
    return f"✅ **CUDA confirmada pelo PyTorch** · {name} · {memory} · PyTorch {torch.__version__}"


def _source_path(value: Any) -> Path:
    if isinstance(value, str):
        path = Path(value)
    elif isinstance(value, dict) and value.get("path"):
        path = Path(value["path"])
    elif hasattr(value, "name"):
        path = Path(value.name)
    else:
        raise gr.Error("Envie um arquivo WAV, FLAC ou MP3.")
    if not path.is_file():
        raise gr.Error("O arquivo de áudio não está mais disponível nesta sessão.")
    if path.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise gr.Error("Formato não aceito. Use WAV, FLAC ou MP3.")
    return path


def _file_path(value: Any) -> Path | None:
    if value is None:
        return None
    if isinstance(value, str):
        return Path(value)
    if isinstance(value, dict) and value.get("path"):
        return Path(value["path"])
    if hasattr(value, "name"):
        return Path(value.name)
    return None


def _read_audio(source: Path, temp_dir: Path) -> tuple[np.ndarray, int]:
    try:
        data, sr = sf.read(source, always_2d=True, dtype="float64")
        return data, int(sr)
    except Exception as first_error:
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise RuntimeError(f"Não foi possível abrir o áudio e o FFmpeg não está instalado: {first_error}") from first_error
        converted = temp_dir / f"{_safe_name(source.stem)}.wav"
        run = subprocess.run(
            [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(source), "-map_metadata", "-1", "-c:a", "pcm_f32le", str(converted)],
            capture_output=True,
            text=True,
        )
        if run.returncode:
            raise RuntimeError(f"Falha ao converter o áudio com FFmpeg: {run.stderr.strip()}")
        return sf.read(converted, always_2d=True, dtype="float64")


def _diagnosis(info: dict[str, Any], channels: int) -> tuple[str, str]:
    fraction = float(info.get("clipped_frac", 0.0)) * 100
    official = "\n".join(analysis_lines(info, channels)) if info.get("analysis") else "Análise oficial indisponível."
    if fraction <= 0:
        headline = "Nenhum clipping, teto de limiter ou saturação recuperável foi identificado; o sinal será mantido."
    else:
        kinds = (info.get("analysis") or {}).get("kinds") or []
        labels = {"clip": "clipping duro", "smeared": "clipping borrado", "limiter": "limitador brickwall"}
        found = ", ".join(labels.get(kind, kind) for kind in kinds) or str(info.get("mode", "detectado"))
        headline = f"Detectado: {found}. {fraction:.3f}% das amostras foram marcadas para reconstrução."
    return headline, f"{headline}\n\n{official}"


def _new_job_dir(source: Path) -> Path:
    job = OUTPUT_ROOT / f"{_safe_name(source.stem)}-{uuid.uuid4().hex[:8]}"
    job.mkdir(parents=True, exist_ok=False)
    return job


def _device_from_info(info: dict[str, Any]) -> str:
    used = str(info.get("device", "cpu"))
    if used.startswith("cuda") and torch.cuda.is_available():
        return f"Processamento confirmado em **{used} — {torch.cuda.get_device_name(torch.cuda.current_device())}**."
    return f"Processamento confirmado em **{used}**. CUDA disponível pelo PyTorch: **{torch.cuda.is_available()}**."


def _run_analysis(source: Path, progress: gr.Progress) -> tuple[dict[str, Any], str, str, Path, str]:
    job_dir = _new_job_dir(source)
    with tempfile.TemporaryDirectory(prefix="betterdeclipper_colab_") as tmp:
        y, sr = _read_audio(source, Path(tmp))
        if not len(y):
            raise RuntimeError("O arquivo não contém amostras de áudio.")

        progress(0.02, desc="Executando diagnóstico oficial...")

        def on_progress(step: int, total: int, elapsed: float) -> None:
            progress((step, max(total, 1)), desc=f"Análise real · etapa {step}/{total} · {elapsed:.0f}s")

        _discard, info = declip(
            y,
            sr,
            preset="fast",
            mode="auto",
            device="auto",
            verbose=False,
            progress=on_progress,
            full_analysis=True,
        )
        headline, full_log = _diagnosis(info, y.shape[1])
        data = info["analysis"]["data"]
        analysis_path = job_dir / f"{_safe_name(source.stem)}.analysis.json"
        save_analysis(analysis_path, data, file=source.name, sr=sr, samples=len(y))
        details = f"{len(y) / sr:.1f} s · {sr} Hz · {y.shape[1]} canal(is)\n{_device_from_info(info)}"
        return data, headline, full_log, analysis_path, details


def analyze_upload(audio: Any, state: dict[str, Any] | None, progress: gr.Progress = gr.Progress()) -> tuple[str, str, str | None, dict[str, Any], str]:
    source = _source_path(audio)
    if not PROCESS_LOCK.acquire(blocking=False):
        raise gr.Error("Já existe um processamento em execução. Aguarde a conclusão.")
    try:
        data, headline, log, analysis_path, details = _run_analysis(source, progress)
        new_state = {
            "source": str(source),
            "analysis": data,
            "analysis_path": str(analysis_path),
            "diagnosis": headline,
            "job_dir": str(analysis_path.parent),
        }
        return f"### Diagnóstico\n{headline}\n\n{details}", log, str(analysis_path), new_state, gpu_status()
    except Exception as exc:
        raise gr.Error(f"Falha no diagnóstico: {exc}") from exc
    finally:
        PROCESS_LOCK.release()


def load_saved_analysis(audio: Any, analysis_file: Any, state: dict[str, Any] | None) -> tuple[str, str, dict[str, Any]]:
    source = _source_path(audio)
    path = _file_path(analysis_file)
    if not path or not path.is_file():
        raise gr.Error("Selecione um arquivo .analysis.json.")
    data = load_analysis(path)
    current = dict(state or _blank_state())
    current.update({"source": str(source), "analysis": data, "analysis_path": str(path)})
    message = "Análise oficial carregada. Confirme que ela pertence ao mesmo áudio ou à mesma cadeia de masterização."
    return f"### Diagnóstico\n{message}", f"Análise reutilizada: {path.name}", current


def _slice_audio(y: np.ndarray, sr: int, start_seconds: float, duration: str) -> tuple[np.ndarray, str]:
    if duration == "Música completa":
        return y, "completa"
    seconds = int(duration.split()[0])
    start = max(0, int(float(start_seconds or 0) * sr))
    end = min(len(y), start + seconds * sr)
    if start >= len(y) or end <= start:
        raise ValueError("O início escolhido está fora da duração do áudio.")
    return y[start:end], f"teste-{seconds}s-inicio-{start / sr:g}s"


def _export_pcm24(x: np.ndarray, sr: int, output: Path) -> tuple[float, bool]:
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    attenuated = peak > 10 ** (-0.1 / 20)
    if attenuated:
        x = x * (10 ** (-0.1 / 20) / peak)
    sf.write(output, x.astype(np.float64), sr, subtype="PCM_24")
    return peak, attenuated


def _process_one(
    source: Path,
    state: dict[str, Any],
    preset: str,
    mode: str,
    start_seconds: float,
    duration: str,
    progress: gr.Progress,
) -> tuple[str, str, dict[str, Any], str]:
    with tempfile.TemporaryDirectory(prefix="betterdeclipper_colab_") as tmp:
        y, sr = _read_audio(source, Path(tmp))
        y, selection_label = _slice_audio(y, sr, start_seconds, duration)
        analysis = state.get("analysis")

        def on_progress(step: int, total: int, elapsed: float) -> None:
            progress((step, max(total, 1)), desc=f"{mode} / {preset} · etapa {step}/{total} · {elapsed:.0f}s")

        started = time.time()
        x, info = declip(
            y,
            sr,
            preset=preset.lower(),
            mode=mode.lower(),
            device="auto",
            verbose=False,
            progress=on_progress,
            analysis=analysis,
            full_analysis=analysis is None,
        )
        headline, official_log = _diagnosis(info, y.shape[1])
        if state.get("analysis") is None and info.get("analysis"):
            state["analysis"] = info["analysis"]["data"]

        job_dir = Path(state.get("job_dir") or _new_job_dir(source))
        mode_dir = job_dir / f"{preset.lower()}-{mode.lower()}"
        mode_dir.mkdir(parents=True, exist_ok=True)
        label = _safe_name(restoration_label(mode.lower(), info))
        output = mode_dir / f"{_safe_name(source.stem)}_{label}_{preset.lower()}_{selection_label}_24bit.wav"
        raw_peak, attenuated = _export_pcm24(x, sr, output)
        export_note = (
            f"Exportação PCM 24 bits recebeu apenas atenuação técnica para -0,1 dBFS porque o pico reconstruído era "
            f"{20 * np.log10(max(raw_peak, 1e-12)):+.2f} dBFS. Isso não é apresentado como restauração."
            if attenuated
            else "Exportação WAV PCM 24 bits sem normalização/equalização adicional."
        )
        log = (
            f"Concluído em {time.time() - started:.1f}s.\n{_device_from_info(info)}\n{export_note}\n"
            f"Saída: {output}\n\n{official_log}"
        )
        return str(output), log, state, headline


def process_audio(
    audio: Any,
    preset: str,
    mode: str,
    duration: str,
    start_seconds: float,
    state: dict[str, Any] | None,
    progress: gr.Progress = gr.Progress(),
) -> Iterator[tuple[Any, Any, str, str, dict[str, Any], str]]:
    source = _source_path(audio)
    current = dict(state or _blank_state())
    current["source"] = str(source)
    if not PROCESS_LOCK.acquire(blocking=False):
        raise gr.Error("Já existe um processamento em execução. Aguarde a conclusão.")
    try:
        yield None, None, "Processamento iniciado...", "### Em andamento\nAguarde a reconstrução real da forma de onda.", current, gpu_status()
        output, log, current, headline = _process_one(source, current, preset, mode, start_seconds, duration, progress)
        yield output, output, log, f"### Resultado\n{headline}", current, gpu_status()
    except Exception as exc:
        raise gr.Error(f"Falha no processamento: {exc}") from exc
    finally:
        PROCESS_LOCK.release()


def compare_modes(
    audio: Any,
    preset: str,
    duration: str,
    start_seconds: float,
    state: dict[str, Any] | None,
    progress: gr.Progress = gr.Progress(),
) -> Iterator[tuple[Any, Any, Any, str, dict[str, Any], str]]:
    source = _source_path(audio)
    current = dict(state or _blank_state())
    current["source"] = str(source)
    if not PROCESS_LOCK.acquire(blocking=False):
        raise gr.Error("Já existe um processamento em execução. Aguarde a conclusão.")
    try:
        outputs: list[str | None] = [None, None, None]
        logs = ["Comparação iniciada. A mesma análise e o mesmo trecho serão usados nos três modos."]
        yield *outputs, "\n".join(logs), current, gpu_status()
        for index, mode in enumerate(("auto", "hard", "soft")):
            output, log, current, _headline = _process_one(source, current, preset, mode, start_seconds, duration, progress)
            outputs[index] = output
            logs.append(f"\n===== {mode.upper()} =====\n{log}")
            yield *outputs, "\n".join(logs), current, gpu_status()
    except Exception as exc:
        raise gr.Error(f"Falha na comparação: {exc}") from exc
    finally:
        PROCESS_LOCK.release()


def clear_session() -> tuple[None, str, str, None, None, dict[str, Any], str]:
    return None, "### Diagnóstico\nEnvie um áudio para iniciar a análise oficial.", "Pronto.", None, None, _blank_state(), gpu_status()


def create_app() -> gr.Blocks:
    with gr.Blocks(css=CSS, theme=gr.themes.Base(primary_hue="emerald", neutral_hue="slate"), title="BetterDeclipper Colab") as demo:
        state = gr.State(_blank_state())
        gr.HTML(
            "<section class='bd-hero'><h1>BetterDeclipper Colab</h1>"
            "<p>Recuperação real de clipping com a API oficial, PyTorch e CUDA quando confirmada.</p></section>"
        )
        device = gr.Markdown(gpu_status(), elem_classes="bd-card")

        with gr.Row(equal_height=False):
            with gr.Column(scale=6, elem_classes="bd-card"):
                audio = gr.Audio(label="Áudio original — WAV, FLAC ou MP3", type="filepath", sources=["upload"])
                diagnosis = gr.Markdown("### Diagnóstico\nEnvie um áudio para iniciar a análise oficial.")
                with gr.Row():
                    analysis_download = gr.File(label="Baixar análise do clipping", interactive=False)
                    analysis_upload = gr.File(label="Reutilizar análise .json", file_types=[".json"], type="filepath")
                use_analysis = gr.Button("Carregar análise selecionada")
            with gr.Column(scale=4, elem_classes="bd-card"):
                preset = gr.Radio(["Fast", "Normal", "High", "Best"], value="Normal", label="Preset")
                mode = gr.Radio(["Auto", "Hard", "Soft", "Limiter"], value="Auto", label="Modo")
                duration = gr.Radio(["15 segundos", "30 segundos", "60 segundos", "Música completa"], value="15 segundos", label="Trecho")
                start_seconds = gr.Number(value=0, minimum=0, label="Início do trecho (segundos)")
                gr.Markdown("<span class='bd-note'>Limiter é experimental. WAV 24 bits recebe somente atenuação de segurança se os picos ultrapassarem 0 dBFS.</span>")
                run_button = gr.Button("Recuperar áudio", variant="primary")
                compare_button = gr.Button("Comparar Auto × Hard × Soft")
                clear_button = gr.Button("Limpar sessão")

        with gr.Row():
            recovered = gr.Audio(label="Áudio recuperado", type="filepath")
            download = gr.File(label="Download WAV 24 bits", interactive=False)
        result_summary = gr.Markdown("### Resultado\nAguardando processamento.")
        logs = gr.Textbox(label="Logs e diagnóstico detalhado", value="Pronto.", lines=14, max_lines=28, interactive=False)

        with gr.Accordion("Comparação dos modos (mesmo áudio, análise e trecho)", open=False):
            with gr.Row():
                compare_auto = gr.Audio(label="Auto", type="filepath")
                compare_hard = gr.Audio(label="Hard", type="filepath")
                compare_soft = gr.Audio(label="Soft", type="filepath")

        audio.upload(
            analyze_upload,
            inputs=[audio, state],
            outputs=[diagnosis, logs, analysis_download, state, device],
            concurrency_limit=1,
        )
        use_analysis.click(
            load_saved_analysis,
            inputs=[audio, analysis_upload, state],
            outputs=[diagnosis, logs, state],
            concurrency_limit=1,
        )
        run_button.click(
            process_audio,
            inputs=[audio, preset, mode, duration, start_seconds, state],
            outputs=[recovered, download, logs, result_summary, state, device],
            concurrency_limit=1,
        )
        compare_button.click(
            compare_modes,
            inputs=[audio, preset, duration, start_seconds, state],
            outputs=[compare_auto, compare_hard, compare_soft, logs, state, device],
            concurrency_limit=1,
        )
        clear_button.click(
            clear_session,
            outputs=[audio, diagnosis, logs, recovered, download, state, device],
            concurrency_limit=1,
        )
    return demo


if __name__ == "__main__":
    print(gpu_status())
    create_app().queue(default_concurrency_limit=1, max_size=8).launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", "7860")),
        share=os.getenv("GRADIO_SHARE", "false").lower() == "true",
    )
