#!/usr/bin/env python3
"""
extract_walnuts.py — Cut out walnut RGBA patches from training positives.

What it does:
  - Reads 32×32 patches from train/positive/ (one walnut per patch)
  - Builds an alpha mask and saves RGBA cutouts to walnut_cutouts/
  - Writes metadata JSON for downstream synthetic compositing

Prerequisites:
  - python3 setup.py
  - train/positive/ with PNG patches

How to run:
  cd Walnut-detection
  source venv/bin/activate
  python extract_walnuts.py --train_dir train --output_dir .
"""

import argparse
import json
import math
import os
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def create_alpha_mask(patch: np.ndarray, center_x: int = 16, center_y: int = 16) -> np.ndarray:
    """Create alpha mask for walnut in 32×32 patch (walnut centered)."""
    h, w = patch.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    radius = min(12, min(h, w) // 3)
    cv2.circle(mask, (center_x, center_y), radius, cv2.GC_FGD, -1)
    cv2.circle(mask, (center_x, center_y), min(radius + 4, min(h, w) // 2), cv2.GC_PR_FGD, -1)
    mask[mask == 0] = cv2.GC_BGD

    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(patch, mask, None, bgd, fgd, 3, cv2.GC_INIT_WITH_MASK)
        alpha = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    except Exception:
        alpha = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(alpha, (center_x, center_y), radius, 255, -1)

    alpha = cv2.GaussianBlur(alpha, (3, 3), 0)
    return alpha


def mask_diameter(alpha: np.ndarray) -> float:
    """Equivalent diameter from alpha mask."""
    contours, _ = cv2.findContours(alpha, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    area = max(cv2.contourArea(c) for c in contours)
    return 2 * math.sqrt(max(area, 0) / math.pi)


def main():
    parser = argparse.ArgumentParser(description="Extract walnut cutouts from train/positive patches")
    parser.add_argument("--train_dir", default="train", help="Path to train (contains positive/)")
    parser.add_argument("--output_dir", default=".", help="Output root (walnut_cutouts/ created here)")
    args = parser.parse_args()

    positive_dir = Path(args.train_dir) / "positive"
    if not positive_dir.exists():
        raise SystemExit(f"Not found: {positive_dir}")

    out_root = Path(args.output_dir)
    cutouts_dir = out_root / "walnut_cutouts"
    cutouts_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(positive_dir.glob("*.png"))
    if not paths:
        raise SystemExit(f"No PNGs in {positive_dir}")

    metadata = []
    patch_size = 32
    center = patch_size // 2

    for path in tqdm(paths, desc="Extracting walnuts"):
        img = cv2.imread(str(path))
        if img is None or img.shape[:2] != (patch_size, patch_size):
            continue
        alpha = create_alpha_mask(img, center, center)
        d = mask_diameter(alpha)
        if d < 4:
            continue

        out_name = path.stem + "_cutout.png"
        out_path = cutouts_dir / out_name
        rgba = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
        rgba[:, :, 3] = alpha
        cv2.imwrite(str(out_path), cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))

        metadata.append({
            "path": str(out_path),
            "stem": path.stem,
            "diameter": float(d),
        })

    meta_path = out_root / "walnut_cutouts_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved {len(metadata)} cutouts to {cutouts_dir}")
    print(f"Metadata: {meta_path}")


if __name__ == "__main__":
    main()
