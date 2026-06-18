#!/usr/bin/env python3
"""
Build a binary classifier dataset from Annotations11-10 with train / val / test splits.

Input:
  - cropped_images/ = full images
  - annotated_images/*.txt = walnut center (x, y) per line

Output directory structure:
  <output_dir>/
    train/
      positive/   (32x32 patches centered on annotations)
      negative/   (background patches)
    val/
      positive/
      negative/
    test/
      positive/
      negative/

Splits are at IMAGE level (no leakage between train/val/test).

Fixed split (e.g. 4 images → 2 train, 1 val, 1 test):
  python build_annotations11_10_dataset.py --image_dir output --annotation_dir output/annotations \\
      --output_dir output/dataset --n_train 2 --n_val 1 --n_test 1 --clean

Train:
  python binary_classifier.py --dataset_dir <output_dir> --output_dir models --epochs 50

Fine-tune:
  python binary_classifier.py --dataset_dir <output_dir> --output_dir models --epochs 30 --pretrained models/walnut_classifier_best_precision.pth
"""

import argparse
import json
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def load_centers(txt_path: Path) -> list[tuple[int, int]]:
    centers = []
    if not txt_path.exists():
        return centers
    with txt_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    x, y = int(float(parts[0])), int(float(parts[1]))
                    centers.append((x, y))
                except ValueError:
                    continue
    return centers


def build_dataset(
    image_dir: str,
    annotation_dir: str,
    output_dir: str,
    patch_size: int = 32,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    neg_per_positive: float = 2.0,
    neg_min_distance: float = 24.0,
    seed: int = 42,
    n_train: int | None = None,
    n_val: int | None = None,
    n_test: int | None = None,
    split_json: str | None = None,
    clean: bool = False,
) -> None:
    random.seed(seed)
    np.random.seed(seed)

    image_dir = Path(image_dir)
    annotation_dir = Path(annotation_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if clean:
        for sub in ("train", "val", "test"):
            p = output_dir / sub
            if p.exists():
                shutil.rmtree(p)
        print(f"🧹 Removed existing train/val/test under {output_dir}")

    exts = ["*.png", "*.PNG", "*.jpg", "*.JPG"]
    image_files = []
    for ext in exts:
        image_files.extend(image_dir.glob(ext))
    image_files = sorted(set(image_files))

    if not image_files:
        raise SystemExit(f"No images in {image_dir}")

    stem_to_path = {p.stem: p for p in image_files}
    n = len(image_files)

    if split_json:
        split_json_path = Path(split_json)
        if not split_json_path.exists():
            raise SystemExit(f"split_json not found: {split_json_path}")
        with split_json_path.open() as f:
            predefined = json.load(f)

        train_files, val_files, test_files = set(), set(), set()
        for split_name, target in (
            ("train", train_files),
            ("val", val_files),
            ("test", test_files),
        ):
            for stem in predefined.get(split_name, []):
                if stem not in stem_to_path:
                    raise SystemExit(
                        f"Stem '{stem}' in split_json {split_name} not found under {image_dir}"
                    )
                target.add(stem_to_path[stem])

        assigned = train_files | val_files | test_files
        missing = set(stem_to_path) - {p.stem for p in assigned}
        if missing:
            raise SystemExit(
                f"split_json does not assign {len(missing)} image(s), e.g. {sorted(missing)[:3]}"
            )
        print(
            f"📊 Predefined split: {len(train_files)} train, "
            f"{len(val_files)} val, {len(test_files)} test (of {n} images)"
        )
        split_record = {
            "train": [p.stem for p in sorted(train_files)],
            "val": [p.stem for p in sorted(val_files)],
            "test": [p.stem for p in sorted(test_files)],
        }
    else:
        # Image-level split: train / val / test
        random.shuffle(image_files)

        use_fixed_counts = n_train is not None and n_val is not None and n_test is not None
        if use_fixed_counts:
            if n_train + n_val + n_test != n:
                raise SystemExit(
                    f"--n_train + --n_val + --n_test must equal number of images ({n}), "
                    f"got {n_train} + {n_val} + {n_test} = {n_train + n_val + n_test}"
                )
            n_train_n, n_val_n, n_test_n = n_train, n_val, n_test
            print(f"📊 Fixed split: {n_train_n} train, {n_val_n} val, {n_test_n} test (of {n} images)")
        else:
            n_train_n = max(1, int(n * train_ratio))
            n_val_n = max(0, int(n * val_ratio))
            n_test_n = n - n_train_n - n_val_n
            if n_test_n < 0:
                n_test_n = 0
                n_val_n = n - n_train_n
            print(f"📊 Ratio split: {n_train_n} train, {n_val_n} val, {n_test_n} test (of {n} images)")

        train_files = set(image_files[:n_train_n])
        val_files = set(image_files[n_train_n : n_train_n + n_val_n])
        test_files = set(image_files[n_train_n + n_val_n :])
        split_record = {
            "train": [p.stem for p in sorted(train_files)],
            "val": [p.stem for p in sorted(val_files)],
            "test": [p.stem for p in sorted(test_files)],
        }

    with (output_dir / "split.json").open("w") as f:
        json.dump(split_record, f, indent=2)
    print(f"📄 Saved {output_dir / 'split.json'} (train/val/test stems)")

    splits = {
        "train": train_files,
        "val": val_files,
        "test": test_files,
    }

    for split_name in splits:
        (output_dir / split_name / "positive").mkdir(parents=True, exist_ok=True)
        (output_dir / split_name / "negative").mkdir(parents=True, exist_ok=True)

    def valid_center(x: int, y: int, w: int, h: int) -> bool:
        half = patch_size // 2
        return half <= x < w - half and half <= y < h - half

    counts = {"train": {"pos": 0, "neg": 0}, "val": {"pos": 0, "neg": 0}, "test": {"pos": 0, "neg": 0}}

    for img_path in tqdm(image_files, desc="Building dataset"):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        stem = img_path.stem
        ann_path = annotation_dir / f"{stem}.txt"
        centers = load_centers(ann_path)

        # Which split this image belongs to
        if img_path in train_files:
            split_name = "train"
        elif img_path in val_files:
            split_name = "val"
        else:
            split_name = "test"

        pos_dir = output_dir / split_name / "positive"
        neg_dir = output_dir / split_name / "negative"

        # Positive patches
        for i, (cx, cy) in enumerate(centers):
            if not valid_center(cx, cy, w, h):
                continue
            half = patch_size // 2
            patch = img[cy - half : cy + half, cx - half : cx + half]
            if patch.size == 0 or patch.shape[0] != patch_size or patch.shape[1] != patch_size:
                continue
            out_name = f"{stem}_p{i:04d}.png"
            cv2.imwrite(str(pos_dir / out_name), patch)
            counts[split_name]["pos"] += 1

        # Negative patches
        centers_arr = np.array(centers, dtype=np.float32) if centers else np.zeros((0, 2))
        n_neg_target = max(10, int(len(centers) * neg_per_positive)) if centers else 50
        n_neg = 0
        attempts = 0
        max_attempts = n_neg_target * 50

        half_p = patch_size // 2
        cx_lo, cx_hi = half_p, w - 1 - half_p
        cy_lo, cy_hi = half_p, h - 1 - half_p
        if cx_hi < cx_lo or cy_hi < cy_lo:
            continue

        while n_neg < n_neg_target and attempts < max_attempts:
            attempts += 1
            cx = random.randint(cx_lo, cx_hi)
            cy = random.randint(cy_lo, cy_hi)
            if centers_arr.size > 0:
                dists = np.sqrt(np.sum((centers_arr - [cx, cy]) ** 2, axis=1))
                if np.any(dists < neg_min_distance):
                    continue
            half = patch_size // 2
            patch = img[cy - half : cy + half, cx - half : cx + half]
            if patch.size == 0 or patch.shape[0] != patch_size or patch.shape[1] != patch_size:
                continue
            out_name = f"{stem}_n{n_neg:04d}.png"
            cv2.imwrite(str(neg_dir / out_name), patch)
            n_neg += 1
            counts[split_name]["neg"] += 1

    print(f"\n📁 Output: {output_dir}")
    print(f"   train: {len(train_files)} images -> {counts['train']['pos']} pos, {counts['train']['neg']} neg")
    print(f"   val:   {len(val_files)} images -> {counts['val']['pos']} pos, {counts['val']['neg']} neg")
    print(f"   test:  {len(test_files)} images -> {counts['test']['pos']} pos, {counts['test']['neg']} neg")
    print("\nTrain:  python binary_classifier.py --dataset_dir", str(output_dir), "--output_dir models --epochs 50")
    print("Test:   python evaluate_walnut_annotations.py --image_dir <cropped_images> --annotation_dir <annotated_images> --split_file", str(output_dir / "split.json"), "--split test ...")


def main():
    parser = argparse.ArgumentParser(description="Build train/val/test dataset from Annotations11-10")
    parser.add_argument("--image_dir", default="images_all/Annotations11-10/cropped_images", help="Cropped images directory")
    parser.add_argument("--annotation_dir", default="images_all/Annotations11-10/annotated_images", help="Annotation .txt directory")
    parser.add_argument("--output_dir", default="data_annotations11_10", help="Output root (will create train/val/test with positive/negative)")
    parser.add_argument("--patch_size", type=int, default=32, help="Patch size")
    parser.add_argument("--train_ratio", type=float, default=0.7, help="Fraction of images for train (ignored if --n_train/--n_val/--n_test set)")
    parser.add_argument("--val_ratio", type=float, default=0.15, help="Fraction of images for val (ignored if fixed counts set)")
    parser.add_argument("--test_ratio", type=float, default=0.15, help="Fraction of images for test (ignored if fixed counts set)")
    parser.add_argument("--n_train", type=int, default=None, help="Exact number of images for train (requires --n_val and --n_test)")
    parser.add_argument("--n_val", type=int, default=None, help="Exact number of images for val")
    parser.add_argument("--n_test", type=int, default=None, help="Exact number of images for test")
    parser.add_argument(
        "--split_json",
        default=None,
        help="Use existing split.json (train/val/test stems) instead of random shuffling",
    )
    parser.add_argument("--clean", action="store_true", help="Delete existing train/val/test folders before rebuilding")
    parser.add_argument("--neg_per_positive", type=float, default=2.0, help="Target negatives per image = neg_per_positive * num_positives")
    parser.add_argument("--neg_min_distance", type=float, default=24.0, help="Min distance from negative patch center to any GT center (px)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling images into splits")
    args = parser.parse_args()

    if (args.n_train, args.n_val, args.n_test) != (None, None, None) and not all(
        x is not None for x in (args.n_train, args.n_val, args.n_test)
    ):
        parser.error("If using fixed splits, set all of --n_train, --n_val, and --n_test")

    build_dataset(
        image_dir=args.image_dir,
        annotation_dir=args.annotation_dir,
        output_dir=args.output_dir,
        patch_size=args.patch_size,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        neg_per_positive=args.neg_per_positive,
        neg_min_distance=args.neg_min_distance,
        seed=args.seed,
        n_train=args.n_train,
        n_val=args.n_val,
        n_test=args.n_test,
        split_json=args.split_json,
        clean=args.clean,
    )


if __name__ == "__main__":
    main()
