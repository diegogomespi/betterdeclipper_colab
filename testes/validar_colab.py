from __future__ import annotations

import tempfile
from pathlib import Path
import sys

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import app
from audio_diagnostics import candidate_metrics, diagnose_signal, representative_window
from betterdeclipper import declip
from voice_restoration import restore_vocal


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main() -> int:
    sr = 16_000
    t = np.arange(int(sr * 0.75), dtype=np.float64) / sr
    clean = 0.8 * np.sin(2 * np.pi * 220 * t) + 0.35 * np.sin(2 * np.pi * 440 * t)
    clipped = np.clip(clean, -0.35, 0.35)[:, None]

    with tempfile.TemporaryDirectory(prefix="betterdeclipper_colab_test_") as tmp:
        tmp_path = Path(tmp)
        source = tmp_path / "exemplo_clipado.wav"
        sf.write(source, clipped, sr, subtype="FLOAT")
        state = {"source": str(source), "analysis": None, "analysis_path": None, "job_dir": str(tmp_path / "resultado")}

        class Progress:
            def __call__(self, *_args, **_kwargs):
                return None

        restored_reference, engine_info = declip(clipped, sr, preset="fast", mode="hard", device="auto", verbose=False)
        mse_clipped = float(np.mean((clean[:, None] - clipped) ** 2))
        mse_restored = float(np.mean((clean[:, None] - restored_reference) ** 2))
        if not mse_restored < mse_clipped:
            raise RuntimeError(f"A restauração não melhorou o erro conhecido: {mse_clipped} -> {mse_restored}")
        if engine_info.get("clipped_frac", 0) <= 0:
            raise RuntimeError("O motor não marcou o clipping sintético conhecido.")

        output, flac, mp3, log, _state, headline, chart = app._process_one(
            source, state, "Rápido (Fast)", "Clipping duro (Hard)", 0, "Música completa", Progress()
        )
        info = sf.info(output)
        if info.subtype != "PCM_24":
            raise RuntimeError(f"Formato inesperado: {info.subtype}")
        if info.frames != len(clipped):
            raise RuntimeError("A saída não preservou a quantidade de amostras.")
        if "Processamento confirmado em" not in log:
            raise RuntimeError("O dispositivo realmente usado não foi registrado.")
        if "Detectado" not in headline:
            raise RuntimeError("O clipping sintético não foi detectado.")
        if sf.info(flac).subtype != "PCM_24":
            raise RuntimeError("A exportação FLAC não preservou PCM 24 bits.")
        if not Path(mp3).is_file():
            raise RuntimeError("A exportação MP3 320 kbps não foi criada.")
        if not Path(chart).is_file():
            raise RuntimeError("O gráfico de diagnóstico não foi criado.")

        selected, selected_start = representative_window(clipped, sr, seconds=30)
        if selected_start != 0 or len(selected) != len(clipped):
            raise RuntimeError("A seleção representativa alterou indevidamente um arquivo curto.")
        diagnostics = diagnose_signal(clipped, sr, ".wav", engine_info)
        if diagnostics["clipped_fraction"] <= 0 or diagnostics["clipping_confidence"] != "alta":
            raise RuntimeError("O diagnóstico integrado não reconheceu o clipping sintético severo.")
        safety = candidate_metrics(clipped, restored_reference, sr, diagnostics["clipped_fraction"])
        if not safety["accepted"]:
            raise RuntimeError(f"Uma restauração conhecida como melhor foi rejeitada: {safety}")

        app.OUTPUT_ROOT = tmp_path / "automatico"
        automatic_steps = list(app.automatic_restore(str(source), {}, Progress()))
        automatic_final = automatic_steps[-1]
        automatic_update = automatic_final[1]
        automatic_path = Path(str(automatic_update.get("value") if isinstance(automatic_update, dict) else automatic_update))
        if not automatic_path.is_file():
            raise RuntimeError("A restauração automática não criou o WAV final.")
        automatic_info = sf.info(automatic_path)
        if automatic_info.subtype != "PCM_24" or automatic_info.frames != len(clipped):
            raise RuntimeError("A restauração automática não preservou formato ou duração.")
        if "Comparação automática dos candidatos" not in str(automatic_final[4]):
            raise RuntimeError("O controlador não registrou a comparação dos candidatos.")
        if "SonicMaster não foi executado" not in str(automatic_final[4]):
            raise RuntimeError("O relatório não registrou corretamente a situação do SonicMaster.")

        vocal_clean = 0.5 * np.sin(2 * np.pi * 190 * np.arange(sr * 2) / sr)
        vocal_damaged = np.clip(vocal_clean * 2.5, -0.32, 0.32)
        vocal_source = tmp_path / "vocal_separado_clipado.wav"
        vocal_output = tmp_path / "vocal_restaurado.wav"
        sf.write(vocal_source, vocal_damaged, sr, subtype="FLOAT")
        vocal_path, vocal_info = restore_vocal(vocal_source, vocal_output, preset="fast")
        if not vocal_path.is_file() or sf.info(vocal_path).frames != len(vocal_clean):
            raise RuntimeError("A restauração vocal não preservou o arquivo e sua duração.")
        if vocal_info.get("identity_replaced") is not False:
            raise RuntimeError("A restauração vocal declarou substituição indevida da identidade.")
        if not Path(str(vocal_info.get("neural_candidate_path"))).is_file():
            raise RuntimeError("O candidato DeepFilterNet3 não foi realmente gerado.")

    demo = app.create_app()
    if demo is None:
        raise RuntimeError("A interface Gradio não foi criada.")
    print("VALIDAÇÃO COLAB CONCLUÍDA")
    print(app.gpu_status())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
