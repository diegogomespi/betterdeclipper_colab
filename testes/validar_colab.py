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

        output, log, _state, headline = app._process_one(source, state, "Fast", "Hard", 0, "Música completa", Progress())
        info = sf.info(output)
        if info.subtype != "PCM_24":
            raise RuntimeError(f"Formato inesperado: {info.subtype}")
        if info.frames != len(clipped):
            raise RuntimeError("A saída não preservou a quantidade de amostras.")
        if "Processamento confirmado em" not in log:
            raise RuntimeError("O dispositivo realmente usado não foi registrado.")
        if "Detectado" not in headline:
            raise RuntimeError("O clipping sintético não foi detectado.")

    demo = app.create_app()
    if demo is None:
        raise RuntimeError("A interface Gradio não foi criada.")
    print("VALIDAÇÃO COLAB CONCLUÍDA")
    print(ascii(app.gpu_status()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
