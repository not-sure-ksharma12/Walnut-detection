#!/usr/bin/env python3
"""
device_utils.py — Shared PyTorch / Ultralytics device selection (CUDA-first).

What it does:
  - Resolves --device auto to cuda (if available), else mps, else cpu
  - Provides helpers for Ultralytics YOLO and torch.cuda/mps cache clearing

Used by all training, evaluation, and inference scripts in this repo.
"""

from __future__ import annotations

import argparse
from typing import Optional, Union

import torch


def best_torch_device() -> str:
    """Pick the best available device: cuda, then mps, then cpu."""
    if torch.cuda.is_available():
        return "cuda"
    if _mps_usable():
        return "mps"
    return "cpu"


def _mps_usable() -> bool:
    if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
        return False
    try:
        torch.zeros(1, device="mps")
        return True
    except Exception:
        return False


def resolve_device(device: Optional[str] = "auto", *, warn: bool = True) -> str:
    """
    Resolve a device string to 'cuda', 'mps', or 'cpu'.

    - auto / None / '' → best_torch_device() (CUDA preferred)
    - cuda → cuda if available, else cpu (with optional warning)
    - mps → mps if usable, else cuda, else cpu
    - cpu → cpu
    """
    if device is None or device == "" or str(device).lower() == "auto":
        return best_torch_device()

    device = str(device).lower()

    if device == "cuda":
        if torch.cuda.is_available():
            return "cuda"
        if warn:
            print("Warning: CUDA requested but not available; using CPU.")
        return "cpu"

    if device == "mps":
        if _mps_usable():
            return "mps"
        if torch.cuda.is_available():
            if warn:
                print("Warning: MPS requested but not available; using CUDA.")
            return "cuda"
        if warn:
            print("Warning: MPS requested but not available; using CPU.")
        return "cpu"

    if device == "cpu":
        return "cpu"

    if warn:
        print(f"Warning: Unknown device '{device}'; using best available.")
    return best_torch_device()


def empty_torch_cache(device: str) -> None:
    """Release GPU memory after a trial or heavy inference."""
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif device == "mps" and _mps_usable():
        torch.mps.empty_cache()


def ultralytics_device(device: str) -> Union[str, int]:
    """Device string/int for ultralytics YOLO train/predict."""
    if device == "cuda" and torch.cuda.is_available():
        return 0
    return device


def add_device_argument(
    parser: argparse.ArgumentParser,
    *,
    default: str = "auto",
    help_text: Optional[str] = None,
) -> None:
    """Register --device with cuda / mps / cpu / auto choices."""
    parser.add_argument(
        "--device",
        type=str,
        default=default,
        choices=["auto", "cpu", "cuda", "mps"],
        help=help_text
        or "Compute device: auto (CUDA if available, else MPS, else CPU), or cpu/cuda/mps",
    )
