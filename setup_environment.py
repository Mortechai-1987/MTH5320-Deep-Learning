"""Create the course environment with CUDA, MPS, or CPU PyTorch."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parent
ENVIRONMENT = REPOSITORY / ".venv"
TORCH_VERSION = "2.11.0"
TORCHVISION_VERSION = "0.26.0"


def run(*command: str) -> None:
    """Run one setup command and stop immediately if it fails."""
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=REPOSITORY, check=True)


def environment_python() -> Path:
    """Return the virtual environment's Python executable on any OS."""
    if os.name == "nt":
        return ENVIRONMENT / "Scripts" / "python.exe"
    return ENVIRONMENT / "bin" / "python"


def has_nvidia_gpu() -> bool:
    """Use the NVIDIA driver utility as the pre-installation CUDA check."""
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        return False

    result = subprocess.run(
        [nvidia_smi],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def torch_install_command(python: Path) -> tuple[list[str], str]:
    """Choose the best available PyTorch wheel before PyTorch is installed."""
    base = [
        str(python),
        "-m",
        "pip",
        "install",
        f"torch=={TORCH_VERSION}",
        f"torchvision=={TORCHVISION_VERSION}",
    ]

    if has_nvidia_gpu():
        return (
            base + ["--index-url", "https://download.pytorch.org/whl/cu128"],
            "CUDA",
        )

    if platform.system() == "Darwin":
        # Official macOS wheels include MPS support on compatible Macs.
        return base, "MPS when available, otherwise CPU"

    return (
        base + ["--index-url", "https://download.pytorch.org/whl/cpu"],
        "CPU",
    )


def main() -> None:
    if sys.version_info < (3, 10):
        raise SystemExit("Python 3.10 or newer is required.")

    if not ENVIRONMENT.exists():
        run(sys.executable, "-m", "venv", str(ENVIRONMENT))

    python = environment_python()
    run(str(python), "-m", "pip", "install", "--upgrade", "pip")
    run(str(python), "-m", "pip", "install", "-r", "requirements.txt")

    torch_command, expected_backend = torch_install_command(python)
    print(f"Installing PyTorch for: {expected_backend}")
    run(*torch_command)

    run(
        str(python),
        "-m",
        "ipykernel",
        "install",
        "--user",
        "--name",
        "fall-2026-deep-learning",
        "--display-name",
        "Python (Fall 2026 Deep Learning)",
    )

    verification = """
import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

x = torch.ones(1, device=device)
print(f"PyTorch {torch.__version__}; selected device: {device}; test: {x.item()}")
"""
    run(str(python), "-c", verification)

    print("\nEnvironment ready. Select the Jupyter kernel:")
    print("Python (Fall 2026 Deep Learning)")


if __name__ == "__main__":
    main()
