#!/usr/bin/env python3
"""
build_fold_jsons.py — Create 10-fold CV split JSONs from pooled patches.

What it does:
  - Scans all_patches/positive and all_patches/negative for unique image stems
  - Writes fold_0.json ... fold_9.json (train_stems, val_stems, test_stems per fold)
  - Does not copy patches; binary_classifier.py filters by stem when --fold_file is set

Prerequisites:
  - all_patches/positive/*.png and all_patches/negative/*.png
  - Patch names like image_q01_p0001.png / image_q01_n0001.png (stem = image_q01)
  - At least 10 unique image stems (one per fold as test)

How to run:
  cd Walnut-detection
  source venv/bin/activate
  python build_fold_jsons.py
  python build_fold_jsons.py --patches_dir all_patches --n_folds 10 --seed 42
  python binary_classifier.py --dataset_dir . --fold_file fold_0.json --output_dir models/fold0
"""

import argparse
import json
import random
import re
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parent


def get_all_stems(patches_dir: Path) -> list[str]:
    pos = patches_dir / "positive"
    neg = patches_dir / "negative"
    stems = set()
    for p in list(pos.glob("*.png")) + list(neg.glob("*.png")):
        s = p.stem
        m = re.match(r"^(.+)_[pn]\d+$", s)
        stems.add(m.group(1) if m else s)
    return sorted(stems)


def main():
    parser = argparse.ArgumentParser(
        description="Build fold JSON files for image-level k-fold CV on all_patches/"
    )
    parser.add_argument(
        "--patches_dir",
        type=str,
        default=str(WORKSPACE / "all_patches"),
        help="Pooled patches root with positive/ and negative/ subdirs",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(WORKSPACE),
        help="Where to write fold_0.json ... fold_{n-1}.json",
    )
    parser.add_argument(
        "--n_folds",
        type=int,
        default=10,
        help="Number of folds (default: 10)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed")
    args = parser.parse_args()

    patches_dir = Path(args.patches_dir)
    output_dir = Path(args.output_dir)
    n_folds = args.n_folds

    if not (patches_dir / "positive").is_dir():
        raise SystemExit(
            f"Missing {patches_dir / 'positive'}. "
            "Populate all_patches/positive and all_patches/negative first "
            "(patches from all images, not split into train/val/test folders)."
        )

    stems = get_all_stems(patches_dir)
    n = len(stems)
    if n < n_folds:
        raise SystemExit(f"Too few stems ({n}) for {n_folds}-fold CV (need at least {n_folds}).")

    random.seed(args.seed)
    order = list(range(n))
    random.shuffle(order)
    shuffled = [stems[i] for i in order]

    fold_size = n // n_folds
    remainder = n % n_folds
    fold_info = []

    for fold_idx in range(n_folds):
        # Test: one fold
        start = fold_idx * fold_size + min(fold_idx, remainder)
        end = start + fold_size + (1 if fold_idx < remainder else 0)
        test_stems = shuffled[start:end]
        train_val_stems = shuffled[:start] + shuffled[end:]
        # Val: 1/9 of train_val
        n_train_val = len(train_val_stems)
        n_val = max(1, n_train_val // 9)
        n_train = n_train_val - n_val
        train_stems = train_val_stems[:n_train]
        val_stems = train_val_stems[n_train:]

        data = {
            "train_stems": train_stems,
            "val_stems": val_stems,
            "test_stems": test_stems,
        }
        out_path = output_dir / f"fold_{fold_idx}.json"
        with out_path.open("w") as f:
            json.dump(data, f, indent=2)
        fold_info.append({
            "fold": fold_idx,
            "train": len(train_stems),
            "val": len(val_stems),
            "test": len(test_stems),
        })

    # Summary table
    print("Per-fold image (stem) counts:")
    print("Fold   Train   Val   Test")
    print("-" * 28)
    for row in fold_info:
        print(f"  {row['fold']}     {row['train']:4d}   {row['val']:3d}   {row['test']:4d}")
    print("-" * 28)
    avg_train = sum(r["train"] for r in fold_info) / n_folds
    avg_val = sum(r["val"] for r in fold_info) / n_folds
    avg_test = sum(r["test"] for r in fold_info) / n_folds
    print(f"Avg    {avg_train:.1f}   {avg_val:.1f}   {avg_test:.1f}")
    print(f"\nTotal unique stems: {n}")
    print(f"Wrote fold_0.json ... fold_{n_folds - 1}.json to {output_dir}")
    print("Train with: python binary_classifier.py --dataset_dir . --fold_file fold_0.json ...")


if __name__ == "__main__":
    main()
