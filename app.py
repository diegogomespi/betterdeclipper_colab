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
import matplotlib
import numpy as np
import soundfile as sf
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pyloudnorm as pyln  # noqa: E402

from audio_diagnostics import candidate_metrics, diagnose_signal, format_diagnosis_pt, representative_window  # noqa: E402


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
PRESET_KEYS = {"Rápido (Fast)": "fast", "Normal": "normal", "Alta qualidade (High)": "high", "Melhor qualidade (Best)": "best"}
MODE_KEYS = {"Automático (Auto)": "auto", "Clipping duro (Hard)": "hard", "Saturação suave (Soft)": "soft", "Limiter experimental": "limiter"}
PRESET_FACTORS = {"fast": 1.0, "normal": 2.4, "high": 3.0, "best": 4.5}


CSS = """
:root { --bd-green:#30d89b; --bd-panel:#151b24; --bd-border:#293443; }
.gradio-container { max-width: 1180px !important; margin: auto !important; background:#0b0f14 !important; }
.bd-hero { padding:24px; border:1px solid var(--bd-border); border-radius:18px;
  background:radial-gradient(circle at top right,rgba(48,216,155,.16),transparent 35%),linear-gradient(135deg,#151b24,#0d1219); }
.bd-hero h1 { margin:0 0 6px; color:#f7fafc; font-size:clamp(1.8rem,4vw,2.8rem); }
.bd-hero p { margin:0; color:#aab7c6; }
.bd-card { border:1px solid var(--bd-border) !important; border-radius:16px !important; background:var(--bd-panel) !important; }
.bd-note { color:#9eacba; font-size:.92rem; }
.compact-actions { gap:10px !important; }
footer { display:none !important; }
"""


def _safe_name(value: str) -> str:
    value = re.sub(r"[^\w.-]+", "_", value, flags=re.UNICODE).strip("._")
    return value[:90] or "audio"


def _blank_state() -> dict[str, Any]:
    return {
        "source": None,
        "analysis": None,
        "analysis_path": None,
        "analysis_scope": None,
        "diagnosis": "",
        "job_dir": None,
        "duration_s": None,
        "sample_rate": None,
        "channels": None,
        "benchmark_rtf": None,
    }


def _format_seconds(seconds: float | None) -> str:
    if seconds is None or not np.isfinite(seconds):
        return "calculando..."
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}min {secs:02d}s"
    if minutes:
        return f"{minutes}min {secs:02d}s"
    return f"{secs}s"


def _preset_key(label: str) -> str:
    return PRESET_KEYS.get(label, str(label).lower())


def _mode_key(label: str) -> str:
    return MODE_KEYS.get(label, str(label).lower())


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


def _dbfs(value: float | None) -> str:
    if value is None or abs(value) < 1e-12:
        return "indisponível"
    return f"{20 * np.log10(abs(value)):+.2f} dBFS"


def _analysis_lines_pt(info: dict[str, Any], channels: int) -> list[str]:
    analysis = info.get("analysis") or {}
    refs = analysis.get("refs") or []
    knees = analysis.get("knees") or []
    flagged = info.get("flagged")
    if flagged is None:
        flagged = []
    channel_names = ["Esquerdo", "Direito"] if channels == 2 else [f"Canal {index + 1}" for index in range(channels)]
    kind_labels = {
        "clip": "clipping digital com platô",
        "smeared": "teto de clipping borrado/irregular (possível codec, reamostragem ou cadeia analógica)",
        "limiter": "teto de limitador",
        "none": "nenhum teto confiável",
    }
    lines = ["Relatório técnico do diagnóstico oficial:"]
    for channel in range(min(channels, len(refs))):
        for polarity_index, polarity in enumerate(("positivo", "negativo")):
            ref = refs[channel][polarity_index]
            knee = knees[channel][polarity_index] if channel < len(knees) else None
            fraction = float(flagged[channel][polarity_index]) * 100 if channel < len(flagged) else 0.0
            kind = kind_labels.get(ref.get("kind"), str(ref.get("kind", "indefinido")))
            ceiling = ref.get("theta")
            peak = ref.get("peak")
            blur = ""
            if ref.get("kind") == "smeared" and ceiling:
                blur = f"; dispersão {100 * float(ref.get('sigma', 0)) / abs(ceiling):.2f}%"
            action = "sem região recuperável"
            if knee is not None:
                action = f"reconstrução acima de {_dbfs(knee)}"
            lines.append(
                f"- {channel_names[channel]}, polo {polarity}: {kind}; teto {_dbfs(ceiling)}; "
                f"pico {_dbfs(peak)}{blur}; {action}; {fraction:.3f}% marcado."
            )
    kinds = analysis.get("kinds") or []
    if kinds:
        lines.append("Tipos reconhecidos: " + ", ".join(kind_labels.get(kind, kind) for kind in kinds) + ".")
    if analysis.get("knee_found"):
        lines.append("O motor encontrou indício real de ombro suave/saturação não linear abaixo do teto.")
    else:
        lines.append("O motor não confirmou ombro suave adicional no trecho analisado.")
    return lines


def _diagnosis(info: dict[str, Any], channels: int) -> tuple[str, str]:
    fraction = float(info.get("clipped_frac", 0.0)) * 100
    official = "\n".join(_analysis_lines_pt(info, channels)) if info.get("analysis") else "Análise oficial indisponível."
    if fraction <= 0:
        headline = "Nenhum clipping, teto de limiter ou saturação recuperável foi identificado; o sinal será mantido."
    else:
        analysis = info.get("analysis") or {}
        kinds = [kind for kind in (analysis.get("kinds") or []) if kind != "none"]
        labels = {"clip": "clipping duro", "smeared": "clipping borrado", "limiter": "limitador brickwall"}
        found_parts = [labels.get(kind, kind) for kind in kinds]
        if analysis.get("knee_found"):
            found_parts.append("saturação não linear/ombro suave")
        found = ", ".join(found_parts) or "região não linear marcada pelo motor"
        headline = f"Detectado: {found}. {fraction:.3f}% das amostras foram marcadas para reconstrução."
    return headline, f"{headline}\n\n{official}"


def _new_job_dir(source: Path) -> Path:
    job = OUTPUT_ROOT / f"{_safe_name(source.stem)}-{uuid.uuid4().hex[:8]}"
    job.mkdir(parents=True, exist_ok=False)
    return job


def _device_from_info(info: dict[str, Any]) -> str:
    used = str(info.get("device", "cpu"))
    if used.startswith("cuda") and torch.cuda.is_available():
        index = torch.cuda.current_device()
        free, total = torch.cuda.mem_get_info(index)
        return (
            f"Processamento confirmado em **{used} — {torch.cuda.get_device_name(index)}** · "
            f"memória CUDA após a tarefa: **{free / 2**30:.1f} GB livres de {total / 2**30:.1f} GB**."
        )
    return f"Processamento confirmado em **{used}**. CUDA disponível pelo PyTorch: **{torch.cuda.is_available()}**."


def _plot_waveforms(y: np.ndarray, x: np.ndarray, sr: int, output: Path, title: str) -> Path:
    mono_y = np.mean(y, axis=1)
    mono_x = np.mean(x, axis=1)
    points = min(len(mono_y), 5000)
    indices = np.linspace(0, len(mono_y) - 1, points, dtype=int)
    seconds = indices / sr
    original_view = mono_y[indices]
    restored_view = mono_x[indices]
    difference = restored_view - original_view
    changed = np.abs(difference) > max(1e-7, float(np.max(np.abs(difference))) * 0.015)
    spectrum_stride = max(1, sr // 24_000)
    spectrum_sr = sr / spectrum_stride
    spectrum_y = mono_y[::spectrum_stride]
    spectrum_x = mono_x[::spectrum_stride]

    plt.style.use("dark_background")
    figure, axes = plt.subplots(2, 2, figsize=(13, 7.2))
    figure.patch.set_facecolor("#0b0f14")
    for axis in axes.flat:
        axis.set_facecolor("#111822")
        axis.grid(color="#2a3543", alpha=0.55, linewidth=0.6)
        axis.tick_params(colors="#aab7c6")
    axes[0, 0].plot(seconds, original_view, color="#8fa4bd", linewidth=0.7, label="Original")
    axes[0, 0].plot(seconds, restored_view, color="#30d89b", linewidth=0.75, alpha=0.9, label="Recuperado")
    axes[0, 0].scatter(seconds[changed], original_view[changed], color="#ff6b6b", s=3, alpha=0.55, label="Região alterada")
    axes[0, 0].set_ylabel("Amplitude")
    axes[0, 0].set_xlabel("Tempo (segundos)")
    axes[0, 0].legend(loc="upper right", frameon=False, ncol=2, fontsize=8)
    axes[0, 0].set_title(title, loc="left", fontsize=12, weight="bold")
    axes[0, 1].fill_between(seconds, difference, 0, color="#f6b94a", alpha=0.85)
    axes[0, 1].set_ylabel("Alteração real")
    axes[0, 1].set_xlabel("Tempo (segundos)")
    axes[0, 1].set_title("Amostras modificadas pelo motor", loc="left", fontsize=10)
    axes[1, 0].specgram(spectrum_y, NFFT=1024, Fs=spectrum_sr, noverlap=768, cmap="magma")
    axes[1, 0].set_title("Espectrograma original", loc="left", fontsize=10)
    axes[1, 0].set_ylabel("Frequência (Hz)")
    axes[1, 0].set_xlabel("Tempo (segundos)")
    axes[1, 1].specgram(spectrum_x, NFFT=1024, Fs=spectrum_sr, noverlap=768, cmap="magma")
    axes[1, 1].set_title("Espectrograma recuperado", loc="left", fontsize=10)
    axes[1, 1].set_ylabel("Frequência (Hz)")
    axes[1, 1].set_xlabel("Tempo (segundos)")
    figure.tight_layout()
    figure.savefig(output, dpi=140, bbox_inches="tight", facecolor=figure.get_facecolor())
    plt.close(figure)
    return output


def _run_analysis(source: Path, progress: gr.Progress) -> tuple[dict[str, Any], str, str, Path, Path, dict[str, Any]]:
    job_dir = _new_job_dir(source)
    with tempfile.TemporaryDirectory(prefix="betterdeclipper_colab_") as tmp:
        y, sr = _read_audio(source, Path(tmp))
        if not len(y):
            raise RuntimeError("O arquivo não contém amostras de áudio.")
        duration_s = len(y) / sr
        analysis_samples = min(len(y), int(60 * sr))
        y_analysis = y[:analysis_samples]
        analysis_duration_s = len(y_analysis) / sr
        progress(0.02, desc=f"Analisando os primeiros {_format_seconds(analysis_duration_s)}...")

        def on_progress(step: int, total: int, elapsed: float) -> None:
            eta = (elapsed / step) * (total - step) if step > 0 and total > step else 0
            progress((step, max(total, 1)), desc=f"Diagnóstico real · etapa {step}/{total} · restante ~{_format_seconds(eta)}")

        started = time.time()
        restored_preview, info = declip(
            y_analysis,
            sr,
            preset="fast",
            mode="auto",
            device="auto",
            verbose=False,
            progress=on_progress,
            full_analysis=True,
        )
        elapsed = time.time() - started
        headline, full_log = _diagnosis(info, y_analysis.shape[1])
        data = info["analysis"]["data"]
        analysis_path = job_dir / f"{_safe_name(source.stem)}.preview.analysis.json"
        save_analysis(analysis_path, data, file=source.name, sr=sr, samples=len(y_analysis))
        chart_path = _plot_waveforms(
            y_analysis,
            restored_preview,
            sr,
            job_dir / "diagnostico_previa.png",
            f"Diagnóstico oficial — primeiros {_format_seconds(analysis_duration_s)}",
        )
        metadata = {
            "duration_s": duration_s,
            "sample_rate": sr,
            "channels": y.shape[1],
            "analysis_duration_s": analysis_duration_s,
            "analysis_elapsed_s": elapsed,
            "benchmark_rtf": elapsed / max(analysis_duration_s, 0.001),
            "device_line": _device_from_info(info),
        }
        return data, headline, full_log, analysis_path, chart_path, metadata


def estimate_text(preset: str, duration: str, start_seconds: float, state: dict[str, Any] | None) -> str:
    current = state or {}
    total = current.get("duration_s")
    rtf = current.get("benchmark_rtf")
    if not total or not rtf:
        return "### Tempo estimado\nA estimativa aparecerá depois do diagnóstico inicial."
    factor = PRESET_FACTORS.get(_preset_key(preset), 2.4)
    available = max(0.0, float(total) - max(0.0, float(start_seconds or 0)))
    selected = available if duration == "Música completa" else min(available, float(duration.split()[0]))
    selected_eta = selected * float(rtf) * factor
    full_eta = float(total) * float(rtf) * factor
    return (
        "### Tempo estimado\n"
        f"**Trecho escolhido:** {_format_seconds(selected_eta)} · **Música completa:** {_format_seconds(full_eta)}  \n"
        "Estimativa baseada no teste real desta sessão; pode variar conforme o áudio, preset, GPU e memória disponível."
    )


def analyze_upload(audio: Any, preset: str, state: dict[str, Any] | None, progress: gr.Progress = gr.Progress()) -> tuple[Any, ...]:
    source = _source_path(audio)
    if not PROCESS_LOCK.acquire(blocking=False):
        raise gr.Error("Já existe um processamento em execução. Aguarde a conclusão.")
    try:
        data, headline, log, analysis_path, chart_path, metadata = _run_analysis(source, progress)
        new_state = {
            "source": str(source),
            "analysis": data,
            "analysis_path": str(analysis_path),
            "analysis_scope": "prévia",
            "diagnosis": headline,
            "job_dir": str(analysis_path.parent),
            **metadata,
        }
        details = (
            f"**Duração total:** {_format_seconds(metadata['duration_s'])} · **Formato interno:** "
            f"{metadata['sample_rate']} Hz, {metadata['channels']} canal(is)  \n"
            f"**Diagnóstico:** primeiros {_format_seconds(metadata['analysis_duration_s'])}, concluído em "
            f"{_format_seconds(metadata['analysis_elapsed_s'])}  \n{metadata['device_line']}"
        )
        estimate = estimate_text(preset, "15 segundos", 0, new_state)
        return (
            f"### Diagnóstico\n{headline}\n\n{details}",
            log,
            gr.update(value=str(analysis_path), visible=True),
            new_state,
            gpu_status(),
            estimate,
            gr.update(value=str(chart_path), visible=True),
        )
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
    current.update({"source": str(source), "analysis": data, "analysis_path": str(path), "analysis_scope": "arquivo reutilizado"})
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


def _export_mp3(wav_path: Path, mp3_path: Path) -> Path | None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return None
    run = subprocess.run(
        [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(wav_path), "-codec:a", "libmp3lame", "-b:a", "320k", str(mp3_path)],
        capture_output=True,
        text=True,
    )
    return mp3_path if run.returncode == 0 and mp3_path.is_file() else None


def _export_flac(wav_path: Path, flac_path: Path) -> Path:
    audio, sr = sf.read(wav_path, always_2d=True, dtype="float64")
    sf.write(flac_path, audio, sr, format="FLAC", subtype="PCM_24")
    return flac_path


def _loudness_matched_players(source: Path, start_seconds: float, outputs: list[str], destination: Path) -> list[str]:
    with tempfile.TemporaryDirectory(prefix="betterdeclipper_ab_") as tmp:
        original, sr = _read_audio(source, Path(tmp))
    original, _label = _slice_audio(original, sr, start_seconds, "30 segundos")
    signals = [original]
    for path in outputs:
        audio, output_sr = sf.read(path, always_2d=True, dtype="float64")
        if output_sr != sr:
            raise RuntimeError("A taxa de amostragem mudou durante a comparação A/B.")
        signals.append(audio)
    meter = pyln.Meter(sr)
    loudness = [float(meter.integrated_loudness(signal)) for signal in signals]
    finite = [value for value in loudness if np.isfinite(value)]
    if not finite:
        raise RuntimeError("Não foi possível medir o volume perceptivo do trecho.")
    maximum_targets = []
    for signal, value in zip(signals, loudness):
        peak = float(np.max(np.abs(signal))) if signal.size else 0.0
        if not np.isfinite(value) or peak <= 0:
            maximum_targets.append(-24.0)
        else:
            maximum_targets.append(value + 20 * np.log10(0.98 / peak))
    target_lufs = min(-14.0, *maximum_targets)
    names = ["original", "automatico", "clipping_duro", "saturacao_suave"]
    matched_paths: list[str] = []
    for name, signal, value in zip(names, signals, loudness):
        matched = signal if not np.isfinite(value) else pyln.normalize.loudness(signal, value, target_lufs)
        path = destination / f"comparacao_30s_{name}_{target_lufs:.1f}LUFS.wav"
        sf.write(path, matched, sr, subtype="PCM_24")
        matched_paths.append(str(path))
    return matched_paths


def _change_measurement(reference: np.ndarray, candidate: np.ndarray) -> tuple[str, dict[str, float | bool]]:
    """Mede alteração numérica sem confundi-la com melhora de qualidade."""
    if reference.shape != candidate.shape:
        raise RuntimeError("A saída não manteve o mesmo tamanho e número de canais do trecho de entrada.")
    ref = np.asarray(reference, dtype=np.float64).reshape(-1)
    out = np.asarray(candidate, dtype=np.float64).reshape(-1)
    difference = out - ref
    rms_ref = float(np.sqrt(np.mean(ref * ref))) if ref.size else 0.0
    rms_difference = float(np.sqrt(np.mean(difference * difference))) if difference.size else 0.0
    peak_difference = float(np.max(np.abs(difference))) if difference.size else 0.0
    changed_fraction = float(np.mean(np.abs(difference) > 1e-4)) if difference.size else 0.0
    identical = bool(np.array_equal(ref, out))
    relative_db = (
        float(20 * np.log10(rms_difference / rms_ref))
        if rms_ref > 1e-15 and rms_difference > 1e-15
        else float("-inf")
    )
    correlation = (
        float(np.corrcoef(ref, out)[0, 1])
        if ref.size > 1 and np.std(ref) > 1e-15 and np.std(out) > 1e-15
        else 1.0 if identical else 0.0
    )
    if identical:
        text = "Verificação da execução: a saída ficou numericamente idêntica ao trecho de entrada."
    else:
        relative_text = f"{relative_db:+.2f} dB" if np.isfinite(relative_db) else "abaixo do limite numérico"
        text = (
            "Verificação da execução: a saída não é uma cópia do original. "
            f"Diferença RMS relativa {relative_text}; correlação {correlation:.6f}; "
            f"{changed_fraction * 100:.3f}% das amostras mudaram mais de -80 dBFS; "
            f"maior diferença {peak_difference:.6f}."
        )
    text += " Esses valores medem alteração, não comprovam melhora sonora."
    return text, {
        "identical": identical,
        "relative_db": relative_db,
        "correlation": correlation,
        "changed_fraction": changed_fraction,
        "peak_difference": peak_difference,
    }


def _process_one(
    source: Path,
    state: dict[str, Any],
    preset: str,
    mode: str,
    start_seconds: float,
    duration: str,
    progress: gr.Progress,
) -> tuple[str, str, str | None, str, dict[str, Any], str, str]:
    with tempfile.TemporaryDirectory(prefix="betterdeclipper_colab_") as tmp:
        y, sr = _read_audio(source, Path(tmp))
        y, selection_label = _slice_audio(y, sr, start_seconds, duration)
        preset_key = _preset_key(preset)
        mode_key = _mode_key(mode)
        needs_full_analysis = duration == "Música completa" and state.get("analysis_scope") != "completa"
        analysis = None if needs_full_analysis else state.get("analysis")

        def on_progress(step: int, total: int, elapsed: float) -> None:
            eta = (elapsed / step) * (total - step) if step > 0 and total > step else 0
            progress((step, max(total, 1)), desc=f"{mode} / {preset} · etapa {step}/{total} · restante ~{_format_seconds(eta)}")

        started = time.time()
        x, info = declip(
            y,
            sr,
            preset=preset_key,
            mode=mode_key,
            device="auto",
            verbose=False,
            progress=on_progress,
            analysis=analysis,
            full_analysis=analysis is None,
        )
        change_note, change_metrics = _change_measurement(y, x)
        info["change_metrics"] = change_metrics
        headline, official_log = _diagnosis(info, y.shape[1])
        if info.get("analysis") and (analysis is None or state.get("analysis") is None):
            state["analysis"] = info["analysis"]["data"]

        job_dir = Path(state.get("job_dir") or _new_job_dir(source))
        mode_dir = job_dir / f"{preset_key}-{mode_key}"
        mode_dir.mkdir(parents=True, exist_ok=True)
        if info.get("analysis") and analysis is None:
            analysis_path = job_dir / f"{_safe_name(source.stem)}.{selection_label}.analysis.json"
            save_analysis(analysis_path, info["analysis"]["data"], file=source.name, sr=sr, samples=len(y))
            state["analysis_path"] = str(analysis_path)
            state["analysis_scope"] = "completa" if duration == "Música completa" else "prévia"
        label = _safe_name(restoration_label(mode_key, info))
        output = mode_dir / f"{_safe_name(source.stem)}_{label}_{preset_key}_{selection_label}_24bit.wav"
        raw_peak, attenuated = _export_pcm24(x, sr, output)
        flac_path = _export_flac(output, output.with_suffix(".flac"))
        mp3_output = mode_dir / f"{_safe_name(source.stem)}_{label}_{preset_key}_{selection_label}_320kbps.mp3"
        mp3_path = _export_mp3(output, mp3_output)
        chart_path = _plot_waveforms(y, x, sr, mode_dir / f"grafico_{selection_label}.png", f"Antes e depois — {mode} / {preset}")
        export_note = (
            f"Exportação PCM 24 bits recebeu apenas atenuação técnica para -0,1 dBFS porque o pico reconstruído era "
            f"{20 * np.log10(max(raw_peak, 1e-12)):+.2f} dBFS. Isso não é apresentado como restauração."
            if attenuated
            else "Exportação WAV PCM 24 bits sem normalização/equalização adicional."
        )
        mp3_note = "MP3 320 kbps criado como cópia comprimida." if mp3_path else "MP3 não foi criado porque o FFmpeg não está disponível."
        elapsed_total = time.time() - started
        log = (
            f"Concluído em {_format_seconds(elapsed_total)}.\n{_device_from_info(info)}\n{change_note}\n{export_note}\n{mp3_note}\n"
            f"WAV: {output}\nFLAC: {flac_path}\n" + (f"MP3: {mp3_path}\n" if mp3_path else "") + f"\n{official_log}"
        )
        measured_headline = f"{headline}\n\n{change_note}"
        return str(output), str(flac_path), str(mp3_path) if mp3_path else None, log, state, measured_headline, str(chart_path)


def process_audio(
    audio: Any,
    preset: str,
    mode: str,
    duration: str,
    start_seconds: float,
    state: dict[str, Any] | None,
    progress: gr.Progress = gr.Progress(),
) -> Iterator[tuple[Any, ...]]:
    source = _source_path(audio)
    current = dict(state or _blank_state())
    current["source"] = str(source)
    if not PROCESS_LOCK.acquire(blocking=False):
        raise gr.Error("Já existe um processamento em execução. Aguarde a conclusão.")
    try:
        yield (
            gr.update(value=None, visible=False),
            gr.update(value=None, visible=False),
            gr.update(value=None, visible=False),
            gr.update(value=None, visible=False),
            gr.update(value=None, visible=False),
            "Processamento iniciado. O painel Progresso mostra as etapas e o tempo restante estimado.",
            "### Em andamento\nAguarde a reconstrução real da forma de onda.",
            gr.update(),
            current,
            gpu_status(),
        )
        output, flac_output, mp3_output, log, current, headline, chart_path = _process_one(source, current, preset, mode, start_seconds, duration, progress)
        yield (
            gr.update(value=output, visible=True),
            gr.update(value=output, visible=True),
            gr.update(value=flac_output, visible=True),
            gr.update(value=mp3_output, visible=bool(mp3_output)),
            gr.update(value=chart_path, visible=True),
            log,
            f"### Resultado\n{headline}",
            gr.update(value=current.get("analysis_path"), visible=bool(current.get("analysis_path"))),
            current,
            gpu_status(),
        )
    except Exception as exc:
        raise gr.Error(f"Falha no processamento: {exc}") from exc
    finally:
        PROCESS_LOCK.release()


def compare_modes(
    audio: Any,
    preset: str,
    start_seconds: float,
    state: dict[str, Any] | None,
    progress: gr.Progress = gr.Progress(),
) -> Iterator[tuple[Any, ...]]:
    source = _source_path(audio)
    current = dict(state or _blank_state())
    current["source"] = str(source)
    if not PROCESS_LOCK.acquire(blocking=False):
        raise gr.Error("Já existe um processamento em execução. Aguarde a conclusão.")
    try:
        outputs: list[str] = []
        logs = ["Comparação inteligente iniciada: 30 segundos, mesmos níveis detectados, modos Automático, Clipping duro e Saturação suave."]
        yield None, None, None, None, "\n".join(logs), current, gpu_status()
        modes = (("Automático (Auto)", "AUTOMÁTICO"), ("Clipping duro (Hard)", "CLIPPING DURO"), ("Saturação suave (Soft)", "SATURAÇÃO SUAVE"))
        for mode, heading in modes:
            output, _flac, _mp3, log, current, _headline, _chart = _process_one(
                source, current, preset, mode, start_seconds, "30 segundos", progress
            )
            outputs.append(output)
            logs.append(f"\n===== {heading} =====\n{log}")
        destination = Path(current.get("job_dir") or _new_job_dir(source)) / "comparacao-inteligente"
        destination.mkdir(parents=True, exist_ok=True)
        original_ab, auto_ab, hard_ab, soft_ab = _loudness_matched_players(source, start_seconds, outputs, destination)
        auto_signal, _ = sf.read(outputs[0], always_2d=True, dtype="float64")
        hard_signal, _ = sf.read(outputs[1], always_2d=True, dtype="float64")
        if np.array_equal(auto_signal, hard_signal):
            logs.append(
                "\nAutomático e Clipping duro ficaram numericamente idênticos neste trecho. "
                "Isso é esperado quando o diagnóstico automático escolhe o tratamento de clipping duro."
            )
        else:
            auto_hard_note, _metrics = _change_measurement(auto_signal, hard_signal)
            logs.append("\nDiferença Automático × Clipping duro: " + auto_hard_note)
        logs.append("\nPlayers A/B ajustados para o mesmo volume perceptivo (LUFS), sem alterar os arquivos de download.")
        yield original_ab, auto_ab, hard_ab, soft_ab, "\n".join(logs), current, gpu_status()
    except Exception as exc:
        raise gr.Error(f"Falha na comparação: {exc}") from exc
    finally:
        PROCESS_LOCK.release()


def _automatic_preset(duration_s: float) -> str:
    if torch.cuda.is_available() and duration_s <= 180:
        return "best"
    if torch.cuda.is_available() and duration_s <= 900:
        return "high"
    return "normal"


def automatic_restore(
    audio: Any,
    state: dict[str, Any] | None,
    progress: gr.Progress = gr.Progress(),
) -> Iterator[tuple[Any, ...]]:
    """Diagnostica, testa modos oficiais e restaura sem exigir decisão técnica do usuário."""
    source = _source_path(audio)
    current = dict(state or _blank_state())
    current["source"] = str(source)
    if not PROCESS_LOCK.acquire(blocking=False):
        raise gr.Error("Já existe um processamento em execução. Aguarde a conclusão.")
    try:
        yield (
            "### Diagnóstico automático\nPreparando o áudio...",
            gr.update(value=None, visible=False),
            gr.update(value=None, visible=False),
            gr.update(value=None, visible=False),
            "Iniciando diagnóstico automático.",
            "### Resultado\nProcessamento em andamento.",
            gr.update(value=None, visible=False),
            current,
            gpu_status(),
        )

        job_dir = _new_job_dir(source)
        with tempfile.TemporaryDirectory(prefix="betterdeclipper_auto_") as tmp:
            y, sr = _read_audio(source, Path(tmp))
        if not len(y):
            raise RuntimeError("O arquivo não contém amostras de áudio.")
        duration_s = len(y) / sr
        representative, representative_start = representative_window(y, sr, seconds=30.0)
        progress(0.03, desc="Analisando o trecho mais representativo...")

        def preview_progress(step: int, total: int, _elapsed: float) -> None:
            fraction = step / max(total, 1)
            progress(0.04 + 0.18 * fraction, desc=f"Diagnóstico oficial · etapa {step}/{total}")

        preview_started = time.time()
        auto_preview, preview_info = declip(
            representative,
            sr,
            preset="fast",
            mode="auto",
            device="auto",
            verbose=False,
            progress=preview_progress,
            full_analysis=True,
        )
        analysis_data = preview_info["analysis"]["data"]
        dsp_metrics = diagnose_signal(representative, sr, source.suffix, preview_info)
        dsp_report = format_diagnosis_pt(dsp_metrics)
        clipped_fraction = float(preview_info.get("clipped_frac", 0.0))

        candidates: list[tuple[str, np.ndarray, dict[str, Any], dict[str, Any]]] = []
        auto_metrics = candidate_metrics(representative, auto_preview, sr, clipped_fraction)
        candidates.append(("auto", auto_preview, preview_info, auto_metrics))

        if clipped_fraction > 0:
            for index, mode_key in enumerate(("hard", "soft"), start=1):
                progress(0.22 + index * 0.08, desc=f"Testando candidato {mode_key} no mesmo trecho...")
                candidate, candidate_info = declip(
                    representative,
                    sr,
                    preset="fast",
                    mode=mode_key,
                    device="auto",
                    verbose=False,
                    analysis=analysis_data,
                    full_analysis=False,
                )
                candidates.append(
                    (mode_key, candidate, candidate_info, candidate_metrics(representative, candidate, sr, clipped_fraction))
                )

        priority = {"auto": 4.0, "hard": 2.0, "soft": 0.0}
        accepted = [item for item in candidates if bool(item[3].get("accepted"))]
        selected = max(accepted, key=lambda item: float(item[3].get("safety_score", -999)) + priority[item[0]]) if accepted else None
        selected_mode = selected[0] if selected and clipped_fraction > 0 else "original"
        selected_preview = selected[1] if selected and clipped_fraction > 0 else representative

        candidate_lines = ["Comparação automática dos candidatos (métricas sem referência limpa):"]
        for mode_key, _candidate, _candidate_info, metrics in candidates:
            status = "aceito" if metrics.get("accepted") else "descartado"
            candidate_lines.append(
                f"- {mode_key}: {status}; correlação {float(metrics.get('correlation', 0)):.6f}; "
                f"diferença RMS {float(metrics.get('relative_delta_db', 0)):+.2f} dB; "
                f"desvio espectral {float(metrics.get('spectral_distance_db', 0)):.2f} dB; "
                f"veredito: {metrics.get('reason')}."
            )

        analysis_path = job_dir / f"{_safe_name(source.stem)}.automatico.analysis.json"
        save_analysis(analysis_path, analysis_data, file=source.name, sr=sr, samples=len(representative))
        current.update(
            {
                "analysis": analysis_data,
                "analysis_path": str(analysis_path),
                "analysis_scope": "trecho representativo automático",
                "job_dir": str(job_dir),
                "duration_s": duration_s,
                "sample_rate": sr,
                "channels": y.shape[1],
                "automatic_diagnostics": dsp_metrics,
                "automatic_mode": selected_mode,
            }
        )

        if selected_mode == "original":
            progress(0.80, desc="Nenhuma reconstrução segura foi confirmada; preservando o original...")
            restored = y.copy()
            full_info = preview_info
            preset_key = "nenhum"
            limitation = (
                "O sistema não encontrou uma restauração BetterDeclipper segura para aplicar. "
                "O WAV entregue preserva o sinal; ruído, reverberação ou distorção analógica ampla não foram removidos."
            )
        else:
            preset_key = _automatic_preset(duration_s)

            def full_progress(step: int, total: int, elapsed: float) -> None:
                fraction = step / max(total, 1)
                eta = (elapsed / step) * (total - step) if step > 0 and total > step else 0
                progress(
                    0.42 + 0.53 * fraction,
                    desc=f"Restaurando música completa · etapa {step}/{total} · restante ~{_format_seconds(eta)}",
                )

            progress(0.42, desc=f"Candidato aprovado: {selected_mode}. Processando a música completa em {preset_key}...")
            restored, full_info = declip(
                y,
                sr,
                preset=preset_key,
                mode=selected_mode,
                device="auto",
                verbose=False,
                progress=full_progress,
                full_analysis=True,
            )
            limitation = (
                "O arquivo foi reconstruído pelo BetterDeclipper. A seleção automática reduz risco de artefatos, "
                "mas sem uma gravação limpa de referência não pode comprovar recuperação de toda a distorção analógica."
            )

        output_dir = job_dir / "restauracao-automatica"
        output_dir.mkdir(parents=True, exist_ok=True)
        output = output_dir / f"{_safe_name(source.stem)}_restaurado_automaticamente_24bit.wav"
        raw_peak, attenuated = _export_pcm24(restored, sr, output)
        chart_path = _plot_waveforms(
            representative,
            selected_preview,
            sr,
            output_dir / "comparacao_trecho_representativo.png",
            f"Seleção automática — trecho a partir de {_format_seconds(representative_start)}",
        )
        change_note, _change = _change_measurement(y, restored)
        elapsed = time.time() - preview_started
        full_headline, full_official = _diagnosis(full_info, y.shape[1])
        attenuation_note = (
            f"A exportação recebeu somente atenuação técnica porque o pico reconstruído era {_dbfs(raw_peak)}."
            if attenuated
            else "WAV PCM 24 bits exportado sem normalização, equalização ou masterização adicional."
        )
        sonic_note = (
            "SonicMaster não foi executado: a versão oficial ainda não foi validada em T4 para músicas longas e "
            "o pipeline atual pode alterar a duração. Demucs também não foi instalado porque não há benefício demonstrado neste áudio."
        )
        log = "\n".join(
            [
                f"Concluído em {_format_seconds(elapsed)}. Trecho representativo iniciado em {_format_seconds(representative_start)}.",
                _device_from_info(full_info),
                dsp_report,
                *candidate_lines,
                f"Decisão automática: modo {selected_mode}; preset final {preset_key}.",
                change_note,
                attenuation_note,
                sonic_note,
                limitation,
                "",
                full_official,
            ]
        )
        diagnosis_markdown = (
            f"### Diagnóstico automático\n{full_headline}\n\n"
            f"**Trecho representativo:** {_format_seconds(representative_start)} até "
            f"{_format_seconds(representative_start + len(representative) / sr)}  \n"
            f"**Decisão:** {selected_mode} / {preset_key}  \n"
            f"**Confiança do clipping:** {dsp_metrics['clipping_confidence']}  \n\n"
            "A análise de ruído, dinâmica, espectro e codec está disponível nos detalhes avançados."
        )
        summary = f"### Resultado automático\n{limitation}\n\n{change_note}"
        progress(1.0, desc="Restauração automática concluída.")
        yield (
            diagnosis_markdown,
            gr.update(value=str(output), visible=True),
            gr.update(value=str(output), visible=True),
            gr.update(value=str(chart_path), visible=True),
            log,
            summary,
            gr.update(value=str(analysis_path), visible=True),
            current,
            gpu_status(),
        )
    except Exception as exc:
        raise gr.Error(f"Falha na restauração automática: {type(exc).__name__}: {exc}") from exc
    finally:
        PROCESS_LOCK.release()


def add_segment(
    start_seconds: float,
    duration: str,
    preset: str,
    mode: str,
    rows: Any,
    state: dict[str, Any] | None,
) -> tuple[list[list[Any]], str]:
    total = float((state or {}).get("duration_s") or 0)
    if total <= 0:
        raise gr.Error("Envie e analise um áudio antes de adicionar trechos.")
    start = max(0.0, float(start_seconds or 0))
    end = total if duration == "Música completa" else min(total, start + float(duration.split()[0]))
    if start >= total or end <= start:
        raise gr.Error("O trecho escolhido está fora da duração do áudio.")
    current_rows = rows.tolist() if hasattr(rows, "tolist") else list(rows or [])
    current_rows.append([round(start, 3), round(end, 3), preset, mode])
    return current_rows, f"Trecho adicionado: {_format_seconds(start)} até {_format_seconds(end)} · {preset} · {mode}."


def process_segment_plan(
    audio: Any,
    rows: Any,
    state: dict[str, Any] | None,
    progress: gr.Progress = gr.Progress(),
) -> Iterator[tuple[Any, ...]]:
    source = _source_path(audio)
    plan = rows.tolist() if hasattr(rows, "tolist") else list(rows or [])
    plan = [row for row in plan if row and len(row) >= 4 and row[0] is not None and row[1] is not None]
    if not plan:
        raise gr.Error("Adicione pelo menos um trecho ao plano.")
    parsed = sorted([(float(row[0]), float(row[1]), str(row[2]), str(row[3])) for row in plan], key=lambda item: item[0])
    for index, (start, end, _preset, _mode) in enumerate(parsed):
        if end <= start:
            raise gr.Error(f"Trecho {index + 1}: o fim precisa ser maior que o início.")
        if index and start < parsed[index - 1][1]:
            raise gr.Error("Há trechos sobrepostos. Ajuste o plano para que cada intervalo seja independente.")
    if not PROCESS_LOCK.acquire(blocking=False):
        raise gr.Error("Já existe um processamento em execução. Aguarde a conclusão.")
    current = dict(state or _blank_state())
    try:
        yield (
            gr.update(value=None, visible=False), gr.update(value=None, visible=False),
            gr.update(value=None, visible=False), gr.update(value=None, visible=False),
            gr.update(value=None, visible=False),
            "Plano de trechos iniciado. Cada intervalo será analisado e recuperado separadamente.",
            "### Em andamento\nProcessando o plano de trechos com transições suaves.",
            gr.update(), current, gpu_status(),
        )
        with tempfile.TemporaryDirectory(prefix="betterdeclipper_trechos_") as tmp:
            original, sr = _read_audio(source, Path(tmp))
        mixed = original.copy()
        context = int(1.5 * sr)
        fade = max(2, int(0.05 * sr))
        logs = []
        last_info: dict[str, Any] = {}
        started_all = time.time()
        for segment_index, (start_s, end_s, preset_label, mode_label) in enumerate(parsed):
            start = max(0, int(start_s * sr))
            end = min(len(original), int(end_s * sr))
            context_start = max(0, start - context)
            context_end = min(len(original), end + context)
            segment = original[context_start:context_end]

            def on_progress(step: int, total: int, elapsed: float, number: int = segment_index) -> None:
                part = step / max(total, 1)
                progress(((number + part), len(parsed)), desc=f"Trecho {number + 1}/{len(parsed)} · etapa {step}/{total} · {_format_seconds(elapsed)}")

            restored, info = declip(
                segment,
                sr,
                preset=_preset_key(preset_label),
                mode=_mode_key(mode_label),
                device="auto",
                verbose=False,
                progress=on_progress,
                full_analysis=True,
            )
            last_info = info
            inner_start = start - context_start
            inner_end = inner_start + (end - start)
            restored_inner = restored[inner_start:inner_end]
            weights = np.ones(len(restored_inner), dtype=np.float64)
            fade_len = min(fade, len(weights) // 2)
            if fade_len and start > 0:
                weights[:fade_len] = np.linspace(0, 1, fade_len, endpoint=False)
            if fade_len and end < len(original):
                weights[-fade_len:] = np.minimum(weights[-fade_len:], np.linspace(1, 0, fade_len, endpoint=False))
            mixed[start:end] = mixed[start:end] * (1 - weights[:, None]) + restored_inner * weights[:, None]
            headline, _detail = _diagnosis(info, original.shape[1])
            logs.append(f"Trecho {segment_index + 1} ({start_s:.2f}s–{end_s:.2f}s, {preset_label}, {mode_label}): {headline}")

        job_dir = Path(current.get("job_dir") or _new_job_dir(source)) / "plano-de-trechos"
        job_dir.mkdir(parents=True, exist_ok=True)
        wav_path = job_dir / f"{_safe_name(source.stem)}_trechos_24bit.wav"
        _peak, attenuated = _export_pcm24(mixed, sr, wav_path)
        flac_path = _export_flac(wav_path, wav_path.with_suffix(".flac"))
        mp3_path = _export_mp3(wav_path, wav_path.with_name(f"{wav_path.stem}_320kbps.mp3"))
        chart_path = _plot_waveforms(original, mixed, sr, job_dir / "grafico_plano_de_trechos.png", "Plano de trechos — antes e depois")
        safety = "Foi aplicada atenuação final de segurança para evitar clipping na exportação." if attenuated else "A exportação não precisou de atenuação final."
        log = (
            "\n".join(logs)
            + f"\n\nPlano concluído em {_format_seconds(time.time() - started_all)}. {safety}\n"
            + (_device_from_info(last_info) if last_info else "")
        )
        yield (
            gr.update(value=str(wav_path), visible=True), gr.update(value=str(wav_path), visible=True),
            gr.update(value=str(flac_path), visible=True), gr.update(value=str(mp3_path), visible=bool(mp3_path)),
            gr.update(value=str(chart_path), visible=True), log,
            f"### Resultado\n{len(parsed)} trecho(s) recuperado(s) e unido(s) com transições de 50 ms.",
            gr.update(), current, gpu_status(),
        )
    except Exception as exc:
        raise gr.Error(f"Falha no plano de trechos: {type(exc).__name__}: {exc}") from exc
    finally:
        PROCESS_LOCK.release()


def clear_session() -> tuple[Any, ...]:
    return (
        None,
        "### Diagnóstico\nEnvie um áudio para iniciar a análise oficial.",
        "Pronto.",
        gr.update(value=None, visible=False),
        gr.update(value=None, visible=False),
        gr.update(value=None, visible=False),
        gr.update(value=None, visible=False),
        gr.update(value=None, visible=False),
        gr.update(value=None, visible=False),
        "### Tempo estimado\nA estimativa aparecerá depois do diagnóstico inicial.",
        [],
        _blank_state(),
        gpu_status(),
    )


def create_app() -> gr.Blocks:
    with gr.Blocks(css=CSS, theme=gr.themes.Base(primary_hue="emerald", neutral_hue="slate"), title="BetterDeclipper Colab") as demo:
        state = gr.State(_blank_state())
        gr.HTML(
            "<section class='bd-hero'><h1>BetterDeclipper AI Studio</h1>"
            "<p>Restauração musical automática: envie o áudio, clique uma vez e acompanhe a análise.</p></section>"
        )
        device = gr.Markdown(gpu_status(), elem_classes="bd-card")

        with gr.Row(equal_height=False):
            with gr.Column(scale=6, elem_classes="bd-card"):
                audio = gr.Audio(label="Áudio original — WAV, FLAC ou MP3", type="filepath", sources=["upload"])
            with gr.Column(scale=4, elem_classes="bd-card"):
                automatic_button = gr.Button("Analisar e restaurar automaticamente", variant="primary", size="lg")
                gr.Markdown(
                    "<span class='bd-note'>O sistema escolhe um trecho representativo, testa os modos compatíveis, "
                    "descarta candidatos com artefatos evidentes e processa a música completa.</span>"
                )
                progress_target = gr.Markdown("### Progresso\nAguardando uma tarefa.", elem_classes="bd-card")

        diagnosis = gr.Markdown("### Diagnóstico automático\nEnvie um áudio e clique no botão de restauração automática.")
        with gr.Row():
            recovered = gr.Audio(label="Áudio recuperado", type="filepath", visible=False)
            with gr.Column():
                download = gr.DownloadButton("Baixar WAV PCM 24 bits", visible=False, variant="primary")
        result_summary = gr.Markdown("### Resultado\nAguardando processamento.")

        with gr.Accordion("Recursos avançados e relatório técnico", open=False):
            visualization = gr.Image(label="Formas de onda, espectrogramas e regiões alteradas", visible=False, interactive=False)
            logs = gr.Textbox(label="Logs, diagnóstico DSP e decisões automáticas", value="Pronto.", lines=12, max_lines=30, interactive=False)
            with gr.Row(elem_classes="compact-actions"):
                analysis_download = gr.DownloadButton("Baixar análise do clipping", visible=False)
                analysis_upload = gr.UploadButton("Reutilizar análise salva (.json)", file_types=[".json"], file_count="single", type="filepath")
            gr.Markdown(
                "<span class='bd-note'>O JSON guarda os níveis medidos pelo BetterDeclipper. Use apenas com o mesmo áudio "
                "ou com uma gravação da mesma cadeia.</span>"
            )

            with gr.Accordion("Controles manuais do BetterDeclipper", open=False):
                preset = gr.Radio(list(PRESET_KEYS), value="Normal", label="Qualidade (preset oficial)")
                mode = gr.Radio(list(MODE_KEYS), value="Automático (Auto)", label="Tipo de recuperação")
                duration = gr.Radio(["15 segundos", "30 segundos", "60 segundos", "Música completa"], value="15 segundos", label="Duração a processar")
                start_seconds = gr.Number(value=0, minimum=0, label="Início do trecho (segundos)")
                estimate = gr.Markdown("### Tempo estimado\nFaça o diagnóstico técnico para obter uma estimativa.")
                with gr.Row():
                    advanced_analyze_button = gr.Button("Executar diagnóstico técnico")
                    run_button = gr.Button("Recuperar com configuração manual", variant="primary")
                    compare_button = gr.Button("Comparar Auto × Hard × Soft")
                with gr.Row():
                    download_flac = gr.DownloadButton("Baixar FLAC 24 bits", visible=False)
                    download_mp3 = gr.DownloadButton("Baixar MP3 320 kbps", visible=False)

            with gr.Accordion("Plano por trechos", open=False):
                gr.Markdown("Adicione intervalos sem sobreposição. As junções usam transições de 50 ms.")
                segment_plan = gr.Dataframe(
                    headers=["Início (s)", "Fim (s)", "Qualidade", "Recuperação"],
                    datatype=["number", "number", "str", "str"],
                    row_count=0,
                    col_count=(4, "fixed"),
                    type="array",
                    interactive=True,
                    label="Trechos configurados",
                )
                with gr.Row():
                    add_segment_button = gr.Button("Adicionar configuração atual")
                    process_plan_button = gr.Button("Processar plano", variant="primary")
                segment_status = gr.Markdown("Nenhum trecho adicionado.")

            gr.Markdown(
                "### Modelos experimentais\n"
                "SonicMaster ainda não é executado automaticamente: o pipeline oficial não preserva a duração do último bloco "
                "e ainda não foi validado em uma T4 para músicas longas. Demucs não é instalado sem benefício demonstrado."
            )

        with gr.Accordion("Players A/B avançados — volumes perceptivos equivalentes", open=False):
            with gr.Row():
                compare_original = gr.Audio(label="Original (A)", type="filepath")
                compare_auto = gr.Audio(label="Automático", type="filepath")
            with gr.Row():
                compare_hard = gr.Audio(label="Clipping duro", type="filepath")
                compare_soft = gr.Audio(label="Saturação suave", type="filepath")

        clear_button = gr.Button("Limpar sessão")

        automatic_button.click(
            automatic_restore,
            inputs=[audio, state],
            outputs=[diagnosis, recovered, download, visualization, logs, result_summary, analysis_download, state, device],
            concurrency_limit=1,
            show_progress_on=progress_target,
        )
        advanced_analyze_button.click(
            analyze_upload,
            inputs=[audio, preset, state],
            outputs=[diagnosis, logs, analysis_download, state, device, estimate, visualization],
            concurrency_limit=1,
            show_progress_on=progress_target,
        )
        analysis_upload.upload(
            load_saved_analysis,
            inputs=[audio, analysis_upload, state],
            outputs=[diagnosis, logs, state],
            concurrency_limit=1,
            show_progress_on=progress_target,
        )
        run_button.click(
            process_audio,
            inputs=[audio, preset, mode, duration, start_seconds, state],
            outputs=[recovered, download, download_flac, download_mp3, visualization, logs, result_summary, analysis_download, state, device],
            concurrency_limit=1,
            show_progress_on=progress_target,
        )
        compare_button.click(
            compare_modes,
            inputs=[audio, preset, start_seconds, state],
            outputs=[compare_original, compare_auto, compare_hard, compare_soft, logs, state, device],
            concurrency_limit=1,
            show_progress_on=progress_target,
        )
        add_segment_button.click(
            add_segment,
            inputs=[start_seconds, duration, preset, mode, segment_plan, state],
            outputs=[segment_plan, segment_status],
            show_progress="hidden",
        )
        process_plan_button.click(
            process_segment_plan,
            inputs=[audio, segment_plan, state],
            outputs=[recovered, download, download_flac, download_mp3, visualization, logs, result_summary, analysis_download, state, device],
            concurrency_limit=1,
            show_progress_on=progress_target,
        )
        for control in (preset, duration, start_seconds):
            control.change(
                estimate_text,
                inputs=[preset, duration, start_seconds, state],
                outputs=estimate,
                show_progress="hidden",
            )
        clear_button.click(
            clear_session,
            outputs=[audio, diagnosis, logs, recovered, download, download_flac, download_mp3, visualization, analysis_download, estimate, segment_plan, state, device],
            concurrency_limit=1,
            show_progress="hidden",
        )
    return demo


if __name__ == "__main__":
    console_encoding = sys.stdout.encoding or "utf-8"
    print(gpu_status().encode(console_encoding, errors="replace").decode(console_encoding))
    create_app().queue(default_concurrency_limit=1, max_size=8).launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", "7860")),
        share=os.getenv("GRADIO_SHARE", "false").lower() == "true",
    )
