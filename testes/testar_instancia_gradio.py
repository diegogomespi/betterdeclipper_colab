from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from gradio_client import Client, handle_file


URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:7860"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def path_from_output(value) -> Path:
    if isinstance(value, dict):
        value = value.get("value") or value.get("path")
    path = Path(str(value))
    if not path.is_file():
        raise RuntimeError(f"Arquivo de saída ausente: {path}")
    return path


def main() -> int:
    report: dict[str, object] = {"url": URL}
    with tempfile.TemporaryDirectory(prefix="betterdeclipper_remote_test_") as temp:
        root = Path(temp)
        sr = 16_000
        seconds = 2
        t = np.arange(sr * seconds, dtype=np.float64) / sr
        clean = 0.78 * np.sin(2 * np.pi * 220 * t) + 0.34 * np.sin(2 * np.pi * 440 * t)
        clipped = np.clip(clean, -0.35, 0.35)
        clipped_path = root / "clipping_conhecido.wav"
        sf.write(clipped_path, clipped, sr, subtype="FLOAT")

        client = Client(URL, verbose=False)
        diagnosis = client.predict(handle_file(clipped_path), "Rápido (Fast)", api_name="/analyze_upload")
        if "Detectado:" not in diagnosis[0]:
            raise RuntimeError(f"Clipping conhecido não detectado: {diagnosis[0]}")
        analysis_path = path_from_output(diagnosis[2])
        chart_path = path_from_output(diagnosis[5])
        device_text = str(diagnosis[3])
        report["diagnostico"] = diagnosis[0]
        report["dispositivo"] = device_text
        report["estimativa"] = diagnosis[4]
        report["analise_json"] = analysis_path.suffix == ".json"
        report["grafico"] = chart_path.suffix.lower() == ".png"

        processed = client.predict(
            handle_file(clipped_path),
            "Rápido (Fast)",
            "Clipping duro (Hard)",
            "Música completa",
            0,
            api_name="/process_audio",
        )
        wav = path_from_output(processed[1])
        flac = path_from_output(processed[2])
        mp3 = path_from_output(processed[3])
        process_chart = path_from_output(processed[4])
        if "medem alteração, não comprovam melhora sonora" not in str(processed[5]):
            raise RuntimeError("O relatório não exibiu a verificação numérica da alteração.")
        if sf.info(wav).subtype != "PCM_24" or sf.info(flac).subtype != "PCM_24":
            raise RuntimeError("WAV/FLAC não foram exportados em PCM 24 bits.")
        if sf.info(wav).samplerate != sr or sf.info(flac).samplerate != sr:
            raise RuntimeError("A taxa de amostragem original não foi preservada.")
        report["exportacoes"] = {"wav": wav.name, "flac": flac.name, "mp3": mp3.name, "grafico": process_chart.name}

        automatic = client.predict(handle_file(clipped_path), api_name="/automatic_restore")
        automatic_wav = path_from_output(automatic[2])
        if sf.info(automatic_wav).subtype != "PCM_24" or sf.info(automatic_wav).frames != len(clipped):
            raise RuntimeError("O modo automático não preservou WAV PCM 24 bits e duração.")
        if "Decisão:" not in str(automatic[0]) or "Comparação automática dos candidatos" not in str(automatic[4]):
            raise RuntimeError("O modo automático não apresentou diagnóstico e seleção dos candidatos.")
        report["restauracao_automatica"] = automatic_wav.name

        vocal_result = client.predict(handle_file(clipped_path), api_name="/process_vocal")
        vocal_wav = path_from_output(vocal_result[1])
        vocal_neural_test = path_from_output(vocal_result[2])
        if sf.info(vocal_wav).frames != len(clipped) or sf.info(vocal_neural_test).frames != len(clipped):
            raise RuntimeError("A restauração vocal alterou a duração do vocal importado.")
        if "DeepFilterNet3" not in str(vocal_result[3]):
            raise RuntimeError("O relatório vocal não comprovou a execução do modelo neural.")
        report["restauracao_vocal"] = {
            "selecionado": vocal_wav.name,
            "candidato_neural": vocal_neural_test.name,
        }

        loaded = client.predict(handle_file(clipped_path), handle_file(analysis_path), api_name="/load_saved_analysis")
        if "carregada" not in str(loaded[0]).lower():
            raise RuntimeError(f"A análise JSON não foi reutilizada: {loaded}")
        report["reutilizacao_json"] = True

        comparison = client.predict(handle_file(clipped_path), "Rápido (Fast)", 0, api_name="/compare_modes")
        players = [path_from_output(item) for item in comparison[:4]]
        if len({sf.info(path).samplerate for path in players}) != 1:
            raise RuntimeError("Os players A/B não preservaram a mesma taxa de amostragem.")
        report["comparacao_inteligente"] = [path.name for path in players]

        rows = {
            "headers": ["Início (s)", "Fim (s)", "Qualidade", "Recuperação"],
            "data": [[0, 1.5, "Rápido (Fast)", "Clipping duro (Hard)"]],
            "metadata": None,
        }
        planned = client.predict(handle_file(clipped_path), rows, api_name="/process_segment_plan")
        planned_wav = path_from_output(planned[1])
        if sf.info(planned_wav).frames != len(clipped):
            raise RuntimeError("O plano de trechos alterou a duração do arquivo.")
        report["plano_de_trechos"] = planned_wav.name

        clean_path = root / "musica_limpa.wav"
        clean_t = np.arange(sr * 10, dtype=np.float64) / sr
        envelope = 0.35 + 0.25 * np.sin(2 * np.pi * 0.37 * clean_t)
        clean_music = envelope * (
            0.55 * np.sin(2 * np.pi * 223 * clean_t)
            + 0.23 * np.sin(2 * np.pi * 449.3 * clean_t)
            + 0.12 * np.sin(2 * np.pi * 901.7 * clean_t)
        )
        sf.write(clean_path, clean_music, sr, subtype="FLOAT")
        clean_client = Client(URL, verbose=False)
        clean_result = clean_client.predict(handle_file(clean_path), "Rápido (Fast)", api_name="/analyze_upload")
        if "Nenhum clipping" not in str(clean_result[0]):
            raise RuntimeError(f"Sinal limpo recebeu diagnóstico inesperado: {clean_result[0]}")
        report["sinal_limpo_preservado"] = True

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("TESTE REMOTO CONCLUÍDO COM SUCESSO")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
