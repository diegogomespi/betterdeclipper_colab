from __future__ import annotations

import sys
import types
import importlib.util
from math import gcd
from pathlib import Path
from typing import Any, Callable

import numpy as np
import soundfile as sf
import torch
from scipy import signal

from betterdeclipper import declip
from audio_diagnostics import candidate_metrics


ProgressCallback = Callable[[float, str], None]


def deepfilternet_available() -> tuple[bool, str]:
    if importlib.util.find_spec("df") is None:
        return False, "DeepFilterNet não instalado."
    return True, "DeepFilterNet disponível."


def _install_torchaudio_import_shim() -> None:
    """Permite usar o núcleo do DeepFilterNet quando torchaudio não está instalado.

    O áudio e a reamostragem continuam sendo feitos por SoundFile/SciPy. O shim só
    satisfaz imports do utilitário de I/O do pacote; não simula o modelo.
    """
    try:
        import torchaudio  # noqa: F401
        return
    except ImportError:
        pass

    torchaudio = types.ModuleType("torchaudio")
    functional = types.ModuleType("torchaudio.functional")
    backend = types.ModuleType("torchaudio.backend")
    common = types.ModuleType("torchaudio.backend.common")

    class AudioMetaData:  # pragma: no cover - apenas anotação usada pelo pacote
        pass

    common.AudioMetaData = AudioMetaData
    functional.resample = lambda value, *_args, **_kwargs: value
    torchaudio.functional = functional
    torchaudio.backend = backend
    torchaudio.info = lambda *_args, **_kwargs: None
    torchaudio.load = lambda *_args, **_kwargs: None
    torchaudio.save = lambda *_args, **_kwargs: None
    sys.modules.update(
        {
            "torchaudio": torchaudio,
            "torchaudio.functional": functional,
            "torchaudio.backend": backend,
            "torchaudio.backend.common": common,
        }
    )


def _resample(audio: np.ndarray, source_sr: int, target_sr: int) -> np.ndarray:
    if source_sr == target_sr:
        return np.asarray(audio, dtype=np.float64)
    divisor = gcd(source_sr, target_sr)
    return signal.resample_poly(audio, target_sr // divisor, source_sr // divisor, axis=0)


def restore_vocal(
    source: str | Path,
    destination: str | Path,
    progress: ProgressCallback | None = None,
    preset: str = "normal",
    noise_attenuation_limit_db: float = 12.0,
) -> tuple[Path, dict[str, Any]]:
    """Restaura um vocal existente sem síntese ou substituição de identidade."""
    source_path = Path(source)
    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    audio, sr = sf.read(source_path, always_2d=True, dtype="float64")
    if not len(audio):
        raise ValueError("O arquivo de vocal está vazio.")
    original_frames = len(audio)

    def notify(value: float, text: str) -> None:
        if progress:
            progress(value, text)

    notify(0.05, "Reconstruindo clipping e saturação do vocal...")
    declipped, clip_info = declip(
        audio,
        sr,
        preset=preset,
        mode="auto",
        device="auto",
        verbose=False,
        full_analysis=True,
    )

    available, reason = deepfilternet_available()
    neural_candidate_path: Path | None = None
    neural_accepted = False
    neural_reason = reason
    model_name = "indisponível neste Python"
    device_noise = "não executado"
    selected = declipped
    if available:
        notify(0.45, "Carregando o modelo neural DeepFilterNet3...")
        _install_torchaudio_import_shim()
        from df.enhance import enhance, init_df

        model, df_state, model_name = init_df(
            default_model="DeepFilterNet3",
            post_filter=False,
            log_file=None,
            log_level="WARNING",
        )
        model_sr = 48_000
        model_audio = _resample(declipped, sr, model_sr)
        tensor = torch.from_numpy(model_audio.T.astype(np.float32, copy=False))
        notify(0.62, "Reduzindo ruído do vocal com o modelo neural...")
        enhanced = enhance(
            model,
            df_state,
            tensor,
            pad=True,
            atten_lim_db=float(noise_attenuation_limit_db),
        )
        restored_48k = enhanced.detach().cpu().numpy().T.astype(np.float64, copy=False)
        restored = _resample(restored_48k, model_sr, sr)
        if len(restored) < original_frames:
            restored = np.pad(restored, ((0, original_frames - len(restored)), (0, 0)))
        restored = restored[:original_frames]
        if not np.all(np.isfinite(restored)):
            raise RuntimeError("O modelo neural produziu valores inválidos.")

        neural_candidate_path = destination_path.with_name(destination_path.stem + "_deepfilternet_teste.wav")
        neural_peak = float(np.max(np.abs(restored)))
        neural_export = restored.copy()
        if neural_peak > 10 ** (-0.1 / 20):
            neural_export *= 10 ** (-0.1 / 20) / neural_peak
        sf.write(neural_candidate_path, neural_export, sr, subtype="PCM_24")
        neural_safety = candidate_metrics(audio, restored, sr, float(clip_info.get("clipped_frac", 0.0)))
        neural_accepted = bool(neural_safety.get("accepted"))
        neural_reason = str(neural_safety.get("reason"))
        selected = restored if neural_accepted else declipped
        device_noise = str(next(model.parameters()).device)

    selected_engine = "BetterDeclipper + DeepFilterNet3" if neural_accepted else "BetterDeclipper"
    peak = float(np.max(np.abs(selected)))
    safety_gain = 1.0
    if peak > 10 ** (-0.1 / 20):
        safety_gain = 10 ** (-0.1 / 20) / peak
        selected = selected * safety_gain
    sf.write(destination_path, selected, sr, subtype="PCM_24")
    notify(1.0, "Vocal restaurado.")
    return destination_path, {
        "engine_clipping": "BetterDeclipper",
        "engine_selected": selected_engine,
        "engine_noise_tested": model_name,
        "device_clipping": clip_info.get("device", "desconhecido"),
        "device_noise": device_noise,
        "clipped_fraction": float(clip_info.get("clipped_frac", 0.0)),
        "sample_rate": sr,
        "channels": audio.shape[1],
        "frames_input": original_frames,
        "frames_output": len(selected),
        "noise_attenuation_limit_db": float(noise_attenuation_limit_db),
        "safety_gain_db": float(20 * np.log10(safety_gain)) if safety_gain < 1 else 0.0,
        "identity_replaced": False,
        "neural_candidate_accepted": neural_accepted,
        "neural_candidate_reason": neural_reason,
        "neural_candidate_path": str(neural_candidate_path) if neural_candidate_path else None,
    }
