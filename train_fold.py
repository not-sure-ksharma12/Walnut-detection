#!/usr/bin/env python3
"""
train_fold.py — Train one fold of k-fold CV via binary_classifier.py.

Each fold: Phase 1 (epochs) → hard-negative mining → Phase 2 (second_phase_epochs).
Models saved to models/fold_{k}/.

Prerequisites:
  - all_patches/positive|negative/
  - fold_{k}.json from build_fold_jsons.py

How to run:
  python train_fold.py 0 --device auto
  python train_fold.py 3 --epochs 20 --second_phase_epochs 10
"""

import argparse
import subprocess
import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(description="Train binary classifier for one CV fold")
    parser.add_argument("fold", type=int, help="Fold index (e.g. 0..9)")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--second_phase_epochs", type=int, default=10)
    from device_utils import add_device_argument

    add_device_argument(parser, default="auto")
    args = parser.parse_args()

    fold_file = WORKSPACE / f"fold_{args.fold}.json"
    output_dir = WORKSPACE / "models" / f"fold_{args.fold}"

    if not fold_file.exists():
        print(f"Missing {fold_file}. Run: python build_fold_jsons.py")
        return 1

    cmd = [
        sys.executable,
        str(WORKSPACE / "binary_classifier.py"),
        "--dataset_dir", str(WORKSPACE),
        "--fold_file", str(fold_file),
        "--output_dir", str(output_dir),
        "--device", args.device,
        "--epochs", str(args.epochs),
        "--loss_type", "focal",
        "--mine_hard_negatives",
        "--hard_negative_threshold", "0.5",
        "--hard_negative_duplicate", "3",
        "--second_phase_epochs", str(args.second_phase_epochs),
    ]
    print(
        f"Fold {args.fold}: device={args.device}, epochs={args.epochs}, "
        f"focal + hard negatives, phase2={args.second_phase_epochs}"
    )
    print(" ".join(cmd))
    return subprocess.run(cmd, cwd=str(WORKSPACE)).returncode


if __name__ == "__main__":
    sys.exit(main())
