from __future__ import annotations

import html
import importlib
import os
import subprocess
import sys
import time
from pathlib import Path

try:
    from IPython import display
except ImportError:  # Permite validar o launcher fora do notebook.
    display = None


PROJECT_MODULES = ("app", "audio_diagnostics", "voice_restoration")


def _run(command: list[str], cwd: str | Path | None = None) -> None:
    print(">", " ".join(command), flush=True)
    subprocess.check_call(command, cwd=str(cwd) if cwd else None)


def _sync_repo(repo_url: str, branch: str, project_dir: str) -> Path:
    destination = Path(project_dir).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not (destination / ".git").exists():
        _run(["git", "clone", "--branch", branch, "--depth", "1", repo_url, str(destination)])
    else:
        _run(["git", "fetch", "origin", branch, "--depth", "1"], destination)
        _run(["git", "reset", "--hard", f"origin/{branch}"], destination)
    return destination


def _torch_report() -> dict[str, object]:
    import torch

    available = bool(torch.cuda.is_available())
    report: dict[str, object] = {
        "torch": torch.__version__,
        "cuda_available": available,
        "torch_cuda_build": torch.version.cuda,
        "device": "cpu",
        "memory_gb": None,
    }
    if available:
        index = torch.cuda.current_device()
        report["device"] = torch.cuda.get_device_name(index)
        report["memory_gb"] = round(torch.cuda.get_device_properties(index).total_memory / 2**30, 2)
    return report


def _clear_cached_project_modules() -> None:
    """Evita misturar código antigo e novo ao executar novamente a célula do Colab."""
    importlib.invalidate_caches()
    for module_name in PROJECT_MODULES:
        sys.modules.pop(module_name, None)


def _status_card(report: dict[str, object], share_url: str | None = None) -> None:
    cuda = bool(report["cuda_available"])
    color = "#30d89b" if cuda else "#f6b94a"
    title = "CUDA confirmada" if cuda else "CPU ativa"
    device = html.escape(str(report["device"]))
    link = (
        f"<a href='{html.escape(share_url)}' target='_blank' style='display:inline-block;margin-top:18px;padding:13px 22px;"
        "border-radius:10px;background:#30d89b;color:#06130e;text-decoration:none;font-weight:700'>ABRIR EM NOVA ABA</a>"
        if share_url
        else ""
    )
    if display is None:
        print(f"{title}: {device} · PyTorch {report['torch']} · build CUDA {report['torch_cuda_build']}", flush=True)
        return
    display.display(
        display.HTML(
            f"""
            <section style='max-width:820px;margin:12px auto;padding:26px;border-radius:18px;background:#0e141c;
              border:1px solid #293443;color:#ecf3f8;font-family:Inter,Arial,sans-serif'>
              <h2 style='margin:0 0 8px;color:{color}'>{title}</h2>
              <p style='margin:0;color:#aab7c6'>Dispositivo: <strong>{device}</strong> · PyTorch {html.escape(str(report['torch']))}
              · build CUDA {html.escape(str(report['torch_cuda_build']))}</p>
              {link}
              <p style='margin:18px 0 0;color:#8291a2;font-size:13px'>Mantenha esta célula em execução durante o uso.</p>
            </section>
            """
        )
    )


def run_from_colab(
    repo_url: str,
    branch: str = "main",
    project_dir: str = "/content/betterdeclipper_colab",
    share: bool = True,
) -> None:
    print("BetterDeclipper Colab — preparando sessão", flush=True)
    repo_dir = _sync_repo(repo_url, branch, project_dir)

    before = _torch_report()
    print("Diagnóstico PyTorch antes da instalação:", before, flush=True)
    if not before["cuda_available"]:
        print(
            "AVISO: CUDA não foi confirmada. No Colab, selecione Ambiente de execução > Alterar tipo de ambiente > T4 GPU. "
            "O sistema continuará em CPU.",
            flush=True,
        )

    _run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], repo_dir)
    _run([sys.executable, "-m", "pip", "install", "-r", str(repo_dir / "requirements.txt")], repo_dir)

    after = _torch_report()
    print("Diagnóstico PyTorch após a instalação:", after, flush=True)
    _status_card(after)

    os.environ["BETTERDECLIPPER_OUTPUT_DIR"] = "/content/betterdeclipper_resultados"
    if str(repo_dir) not in sys.path:
        sys.path.insert(0, str(repo_dir))
    _clear_cached_project_modules()
    app = importlib.import_module("app")
    print(f"BetterDeclipper oficial carregado: {app.betterdeclipper_version}", flush=True)

    demo = app.create_app().queue(default_concurrency_limit=1, max_size=8)
    _app, local_url, share_url = demo.launch(
        share=share,
        inline=True,
        quiet=False,
        prevent_thread_lock=True,
        show_error=True,
    )
    print("URL local:", local_url, flush=True)
    print("URL temporária:", share_url, flush=True)
    _status_card(after, share_url or local_url)
    while True:
        time.sleep(1)


if __name__ == "__main__":
    raise SystemExit("Execute run_from_colab() pelo notebook betterdeclipper_colab.ipynb.")
