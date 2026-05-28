#!/usr/bin/env python3
"""
setup.py — Bootstrap the project environment (venv + dependencies).

What it does:
  - Creates venv/ if it does not exist
  - Upgrades pip and installs packages from requirements.txt
  - Verifies that torch, cv2, numpy, optuna, ultralytics, etc. import correctly

How to run (from repo root):
  python3 setup.py

Then activate the venv in your shell:
  source venv/bin/activate    # macOS / Linux
  venv\\Scripts\\activate     # Windows
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV_DIR = ROOT / "venv"
REQUIREMENTS = ROOT / "requirements.txt"

VERIFY_SNIPPET = """
import torch
import cv2
import numpy
import tqdm
import matplotlib
import sklearn
import scipy
import optuna
import ultralytics
from PIL import Image
from device_utils import resolve_device, best_torch_device
print("OK")
print("Best device:", best_torch_device())
print("Resolved auto:", resolve_device("auto"))
"""


def venv_python() -> Path:
    if sys.platform == "win32":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def run(cmd: list[str], *, step: str) -> None:
    print(f"\n>> {step}")
    print(f"   {' '.join(cmd)}")
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    if not REQUIREMENTS.exists():
        print(f"Missing {REQUIREMENTS}")
        sys.exit(1)

    if not VENV_DIR.exists():
        run([sys.executable, "-m", "venv", str(VENV_DIR)], step="Creating virtual environment")
    else:
        print(f"\n>> Virtual environment already exists: {VENV_DIR}")

    py = venv_python()
    if not py.exists():
        print(f"Expected venv Python not found: {py}")
        sys.exit(1)

    run([str(py), "-m", "pip", "install", "--upgrade", "pip"], step="Upgrading pip")
    run([str(py), "-m", "pip", "install", "-r", str(REQUIREMENTS)], step="Installing requirements")

    print("\n>> Verifying imports")
    result = subprocess.run(
        [str(py), "-c", VERIFY_SNIPPET.strip()],
        cwd=ROOT,
        check=False,
    )
    if result.returncode != 0:
        print("\nImport verification failed.")
        sys.exit(result.returncode)

    print("\nSetup complete.")
    if sys.platform == "win32":
        print(r"Activate with:  venv\Scripts\activate")
    else:
        print("Activate with:  source venv/bin/activate")


if __name__ == "__main__":
    main()
