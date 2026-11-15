from __future__ import annotations

from typing import Any

import numpy as np
from scipy import signal


EPS = 1e-12


def _mono(audio: np.ndarray) -> np.ndarray:
    value = np.asarray(audio, dtype=np.float64)
    if value.ndim == 1:
        return value
    return np.mean(value, axis=1)


def _db(value: float) -> float:
    return float(20.0 * np.log10(max(float(value), EPS)))


def representative_window(audio: np.ndarray, sr: int, seconds: float = 30.0) -> tuple[np.ndarray, float]:
    """Escolhe o trecho sustentado de maior energia, sem alterar o áudio."""
    total = len(audio)
    wanted = min(total, max(1, int(seconds * sr)))
    if wanted >= total:
        return np.asarray(audio), 0.0

    mono = _mono(audio)
    block = max(1, int(sr * 0.5))
    usable = (len(mono) // block) * block
    powers = np.mean(mono[:usable].reshape(-1, block) ** 2, axis=1)
    blocks = max(1, int(np.ceil(wanted / block)))
    kernel = np.ones(blocks, dtype=np.float64)
    sustained = np.convolve(powers, kernel, mode="valid")
    start_block = int(np.argmax(sustained))
    start = min(start_block * block, total - wanted)
    return np.asarray(audio[start : start + wanted]), start / sr


def diagnose_signal(
    audio: np.ndarray,
    sr: int,
    source_suffix: str,
    betterdeclipper_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Mede descritores objetivos; não tenta adivinhar uma referência limpa inexistente."""
    mono = _mono(audio)
    if mono.size == 0:
        raise ValueError("O trecho está vazio.")

    frame = min(len(mono), max(256, int(sr * 0.050)))
    hop = max(1, frame // 2)
    if len(mono) < frame:
        framed = mono[None, :]
    else:
        count = 1 + (len(mono) - frame) // hop
        framed = np.lib.stride_tricks.sliding_window_view(mono, frame)[::hop][:count]
    frame_rms = np.sqrt(np.mean(framed * framed, axis=1) + EPS)
    frame_db = 20 * np.log10(frame_rms + EPS)

    peak = float(np.max(np.abs(mono)))
    rms = float(np.sqrt(np.mean(mono * mono) + EPS))
    crest_db = _db(peak / max(rms, EPS))
    quiet_db = float(np.percentile(frame_db, 10))
    median_db = float(np.percentile(frame_db, 50))
    loud_db = float(np.percentile(frame_db, 95))
    dynamic_range_db = loud_db - quiet_db
    noise_distance_db = median_db - quiet_db

    nperseg = min(4096, len(mono))
    if nperseg >= 64:
        frequencies, _times, spectrum = signal.stft(
            mono,
            fs=sr,
            nperseg=nperseg,
            noverlap=nperseg // 2,
            boundary=None,
            padded=False,
        )
        power = np.mean(np.abs(spectrum) ** 2, axis=1) + EPS
        total_power = float(np.sum(power))
        centroid = float(np.sum(frequencies * power) / total_power)
        low_ratio = float(np.sum(power[frequencies < 100]) / total_power)
        high_limit = min(12000.0, sr * 0.40)
        high_ratio = float(np.sum(power[frequencies >= high_limit]) / total_power)
        geometric = float(np.exp(np.mean(np.log(power))))
        spectral_flatness = geometric / float(np.mean(power))
        mid_mask = (frequencies >= 4000) & (frequencies < min(12000, sr / 2))
        top_mask = frequencies >= min(16000, sr * 0.42)
        mid_energy = float(np.mean(power[mid_mask])) if np.any(mid_mask) else EPS
        top_energy = float(np.mean(power[top_mask])) if np.any(top_mask) else EPS
        top_drop_db = _db(np.sqrt(top_energy / max(mid_energy, EPS)))
    else:
        centroid = low_ratio = high_ratio = spectral_flatness = 0.0
        top_drop_db = 0.0

    info = betterdeclipper_info or {}
    analysis = info.get("analysis") or {}
    clipped_fraction = float(info.get("clipped_frac", 0.0))
    kinds = [str(kind) for kind in analysis.get("kinds", []) if kind != "none"]
    knee_found = bool(analysis.get("knee_found"))

    if clipped_fraction >= 0.01:
        clipping_confidence = "alta"
    elif clipped_fraction >= 0.0005:
        clipping_confidence = "moderada"
    else:
        clipping_confidence = "baixa"

    suffix = source_suffix.lower()
    lossy_container = suffix in {".mp3", ".aac", ".m4a", ".ogg", ".opus"}
    possible_cutoff = sr >= 40_000 and top_drop_db < -24.0
    codec_confidence = "alta" if lossy_container else "baixa" if possible_cutoff else "indeterminada"

    return {
        "sample_peak_dbfs": _db(peak),
        "rms_dbfs": _db(rms),
        "crest_factor_db": crest_db,
        "quiet_floor_dbfs": quiet_db,
        "median_level_dbfs": median_db,
        "loud_level_dbfs": loud_db,
        "dynamic_range_db": dynamic_range_db,
        "noise_distance_db": noise_distance_db,
        "spectral_centroid_hz": centroid,
        "low_energy_percent": low_ratio * 100,
        "high_energy_percent": high_ratio * 100,
        "spectral_flatness": spectral_flatness,
        "high_frequency_drop_db": top_drop_db,
        "clipped_fraction": clipped_fraction,
        "clipping_kinds": kinds,
        "soft_knee_found": knee_found,
        "clipping_confidence": clipping_confidence,
        "codec_confidence": codec_confidence,
        "lossy_container": lossy_container,
        "possible_frequency_cutoff": possible_cutoff,
    }


def format_diagnosis_pt(metrics: dict[str, Any]) -> str:
    clipping = metrics["clipped_fraction"] * 100
    kinds = ", ".join(metrics["clipping_kinds"]) or "nenhum tipo confirmado"
    saturation = (
        "ombro de saturação não linear confirmado pelo BetterDeclipper"
        if metrics["soft_knee_found"]
        else "causa de saturação ampla não confirmada com segurança"
    )
    codec = (
        "o arquivo enviado já usa contêiner com perdas"
        if metrics["lossy_container"]
        else "possível corte de altas frequências; isso sozinho não comprova compressão com perdas"
        if metrics["possible_frequency_cutoff"]
        else "não há evidência suficiente para atribuir artefatos a um codec"
    )
    return "\n".join(
        [
            "Diagnóstico automático do trecho representativo:",
            f"- Clipping: {clipping:.3f}% marcado; confiança {metrics['clipping_confidence']}; tipos: {kinds}.",
            f"- Saturação/distorção não linear: {saturation}.",
            f"- Dinâmica: crest factor {metrics['crest_factor_db']:.2f} dB; faixa entre quadros quietos e fortes {metrics['dynamic_range_db']:.2f} dB.",
            f"- Ruído: piso quieto estimado em {metrics['quiet_floor_dbfs']:.2f} dBFS; distância até o nível mediano {metrics['noise_distance_db']:.2f} dB. É uma estimativa, não uma medição isolada do ruído.",
            f"- Espectro: centroide {metrics['spectral_centroid_hz']:.0f} Hz; energia grave {metrics['low_energy_percent']:.2f}%; energia alta {metrics['high_energy_percent']:.3f}%.",
            f"- Compressão/codec: {codec}; confiança {metrics['codec_confidence']}.",
            "- Resposta de frequência: descrita pelo espectro acima; sem referência limpa não é possível afirmar qual curva seria a correta.",
        ]
    )


def candidate_metrics(
    reference: np.ndarray,
    candidate: np.ndarray,
    sr: int,
    expected_change_fraction: float = 0.0,
) -> dict[str, Any]:
    if reference.shape != candidate.shape or reference.size == 0:
        return {"accepted": False, "reason": "duração ou canais diferentes do original"}
    if not np.all(np.isfinite(candidate)):
        return {"accepted": False, "reason": "a saída contém valores inválidos"}

    ref = np.asarray(reference, dtype=np.float64).reshape(-1)
    out = np.asarray(candidate, dtype=np.float64).reshape(-1)
    delta = out - ref
    rms_ref = float(np.sqrt(np.mean(ref * ref)) + EPS)
    rms_delta = float(np.sqrt(np.mean(delta * delta)) + EPS)
    relative_delta_db = _db(rms_delta / rms_ref)
    correlation = float(np.corrcoef(ref, out)[0, 1]) if np.std(out) > EPS else 0.0

    mono_ref = _mono(reference)
    mono_out = _mono(candidate)
    nperseg = min(2048, len(mono_ref))
    if nperseg >= 64:
        _f, _t, z_ref = signal.stft(mono_ref, fs=sr, nperseg=nperseg, noverlap=nperseg // 2, boundary=None)
        _f, _t, z_out = signal.stft(mono_out, fs=sr, nperseg=nperseg, noverlap=nperseg // 2, boundary=None)
        ref_profile = np.mean(np.abs(z_ref), axis=1) + EPS
        out_profile = np.mean(np.abs(z_out), axis=1) + EPS
        meaningful = ref_profile >= float(np.max(ref_profile)) * 1e-3
        spectral_delta = np.abs(20 * np.log10(out_profile[meaningful] / ref_profile[meaningful]))
        spectral_weights = ref_profile[meaningful]
        spectral_distance_db = float(np.average(spectral_delta, weights=spectral_weights)) if spectral_delta.size else 0.0
    else:
        spectral_distance_db = 0.0

    derivative_ref = np.diff(mono_ref, prepend=mono_ref[0])
    derivative_out = np.diff(mono_out, prepend=mono_out[0])
    ref_impulse = float(np.percentile(np.abs(derivative_ref), 99.95)) + EPS
    out_impulse = float(np.percentile(np.abs(derivative_out), 99.95))
    impulse_ratio = out_impulse / ref_impulse

    peak = float(np.max(np.abs(candidate)))
    allowed_spectral_distance = 4.5 + min(12.0, max(0.0, expected_change_fraction) * 20.0)
    accepted = (
        correlation >= 0.97
        and relative_delta_db <= -8.0
        and spectral_distance_db <= allowed_spectral_distance
        and impulse_ratio <= 3.0
    )
    reasons = []
    if correlation < 0.97:
        reasons.append("correlação muito baixa")
    if relative_delta_db > -8.0:
        reasons.append("alteração excessiva")
    if spectral_distance_db > allowed_spectral_distance:
        reasons.append("desvio espectral excessivo")
    if impulse_ratio > 3.0:
        reasons.append("possíveis impulsos/estalos")

    safety_score = (
        100.0
        - max(0.0, (1.0 - correlation) * 500.0)
        - max(0.0, relative_delta_db + 24.0) * 1.2
        - spectral_distance_db * 2.0
        - max(0.0, impulse_ratio - 1.0) * 5.0
    )
    return {
        "accepted": accepted,
        "reason": ", ".join(reasons) if reasons else "sem artefato evidente pelos testes automáticos",
        "correlation": correlation,
        "relative_delta_db": relative_delta_db,
        "spectral_distance_db": spectral_distance_db,
        "allowed_spectral_distance_db": allowed_spectral_distance,
        "impulse_ratio": impulse_ratio,
        "peak": peak,
        "safety_score": safety_score,
    }
