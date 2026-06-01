#!/usr/bin/env python3
"""
generate_yolo_synthetic_dataset.py — Build a YOLO-format dataset from the best synthetic config.

What it does:
  - Loads optuna_results/best_config.json and composites synthetic walnut images
  - patch mode: 32×32 images (one walnut or background per file); writes imgsz.txt
  - full mode: 640×640 images with 1–4 walnuts and YOLO bbox labels
  - Outputs images/train|val and labels/train|val under yolo_walnut_synthetic/ (default)

Prerequisites:
  - python3 setup.py
  - optuna_results/best_config.json (run optimize_synthetic_config.py first)
  - walnut_cutouts/ and train/negative/

How to run:
  cd Walnut-detection
  source venv/bin/activate
  python generate_yolo_synthetic_dataset.py \\
    --config optuna_results/best_config.json \\
    --cutouts_dir output/dataset/walnut_cutouts \\
    --neg_dir output/dataset/train/negative \\
    --mode patch --num_images 5000
  python train_yolov8_synthetic.py --data_dir yolo_walnut_synthetic --imgsz 32
"""

import copy
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np

WORKSPACE = Path(__file__).resolve().parent
sys.path.insert(0, str(WORKSPACE))

import optimize_synthetic_config as m

IMG_SIZE = 640
PATCH = 32
MIN_WALNUTS = 1
MAX_WALNUTS = 4


def _resolve_path(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else WORKSPACE / p


def make_background_640(neg_patches, green_bgs):
    """Build 640x640 image by tiling 32x32 patches."""
    n_tiles = IMG_SIZE // PATCH  # 20
    bg = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    for i in range(n_tiles):
        for j in range(n_tiles):
            p = random.choice(green_bgs) if green_bgs else random.choice(neg_patches)
            if p.shape[0] != PATCH or p.shape[1] != PATCH:
                p = cv2.resize(p, (PATCH, PATCH))
            y1, y2 = i * PATCH, (i + 1) * PATCH
            x1, x2 = j * PATCH, (j + 1) * PATCH
            bg[y1:y2, x1:x2] = p
    return bg


def apply_augmentations(img, config):
    """Apply brightness, contrast, HSV, blur, noise (same as optimize_synthetic_config)."""
    brightness = config["brightness_delta"]
    contrast = config["contrast_factor"]
    blur_sigma = config["blur_sigma"]
    noise_std = config["noise_std"]
    hue_shift = config["hue_shift"]
    sat_factor = config["saturation_factor"]

    out = img.astype(np.float32)

    if abs(brightness) > 0.01:
        out = out + brightness * 255
        out = np.clip(out, 0, 255)

    if abs(contrast - 1.0) > 0.01:
        mean = out.mean()
        out = (out - mean) * contrast + mean
        out = np.clip(out, 0, 255)

    if abs(hue_shift) > 0.01 or abs(sat_factor - 1.0) > 0.01:
        hsv = cv2.cvtColor(out.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift * 180) % 180
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat_factor, 0, 255)
        out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)

    if blur_sigma > 0.3:
        ksize = int(blur_sigma * 4) | 1
        out = cv2.GaussianBlur(out.astype(np.uint8), (ksize, ksize), blur_sigma).astype(np.float32)

    if noise_std > 0.001:
        noise = np.random.randn(*out.shape).astype(np.float32) * noise_std * 255
        out = np.clip(out + noise, 0, 255)

    return out.astype(np.uint8)


def generate_one_image(config, cutouts, neg_patches, green_bgs):
    """Generate one 640x640 image and return (BGR image, list of YOLO bbox lines)."""
    bg = make_background_640(neg_patches, green_bgs)
    scale_min = config["cutout_scale_min"]
    scale_max = config["cutout_scale_max"]
    pos_jitter = config["position_jitter"]

    num_walnuts = random.randint(MIN_WALNUTS, MAX_WALNUTS)
    boxes = []  # (x_center, y_center, w, h) in pixels

    for _ in range(num_walnuts):
        cutout_bgr, cutout_alpha = random.choice(cutouts)
        # Scale walnut to a reasonable size on 640x640 (e.g. 20–120 px)
        max_side = max(cutout_bgr.shape[0], cutout_bgr.shape[1])
        base_scale = (80 / max_side) * random.uniform(0.6, 1.4)
        rel = random.uniform(scale_min, scale_max)
        scale = rel * base_scale
        new_w = max(16, min(IMG_SIZE - 20, int(cutout_bgr.shape[1] * scale)))
        new_h = max(16, min(IMG_SIZE - 20, int(cutout_bgr.shape[0] * scale)))

        bgr_s = cv2.resize(cutout_bgr, (new_w, new_h))
        alpha_s = cv2.resize(cutout_alpha, (new_w, new_h))

        max_x = IMG_SIZE - new_w
        max_y = IMG_SIZE - new_h
        if max_x <= 0 or max_y <= 0:
            continue
        jitter = int(pos_jitter * IMG_SIZE * 0.15)
        x = random.randint(0, max(0, max_x - 1))
        y = random.randint(0, max(0, max_y - 1))
        if jitter > 0:
            x = max(0, min(max_x, x + random.randint(-jitter, jitter)))
            y = max(0, min(max_y, y + random.randint(-jitter, jitter)))

        m.composite_one(bg, bgr_s, alpha_s, x, y)
        x_center = x + new_w / 2.0
        y_center = y + new_h / 2.0
        boxes.append((x_center, y_center, new_w, new_h))

    img = apply_augmentations(bg, config)

    # YOLO format: class x_center y_center width height (normalized 0-1)
    lines = []
    for (xc, yc, w, h) in boxes:
        xc_n = xc / IMG_SIZE
        yc_n = yc / IMG_SIZE
        w_n = w / IMG_SIZE
        h_n = h / IMG_SIZE
        lines.append(f"0 {xc_n:.6f} {yc_n:.6f} {w_n:.6f} {h_n:.6f}")
    return img, lines


def generate_patch_dataset(config, cutouts, neg_patches, num_pos, num_neg, images_train, labels_train, images_val, labels_val, val_ratio, seed):
    """Generate 32x32 patch dataset: positives from synthetic (best config), negatives from neg_patches."""
    random.seed(seed)
    np.random.seed(seed)
    green_bgs = [p for p in neg_patches if m.is_green(p, config["green_ratio_thresh"])]
    if len(green_bgs) < 10:
        green_bgs = neg_patches[:200]

    pos_patches = m.generate_synthetic_batch(
        config, cutouts, neg_patches, num_samples=num_pos, patch_size=PATCH
    )
    # Ensure 32x32
    pos_patches = [cv2.resize(p, (PATCH, PATCH)) if p.shape[0] != PATCH or p.shape[1] != PATCH else p for p in pos_patches]

    n_total = num_pos + num_neg
    n_val = int(n_total * val_ratio)
    n_train = n_total - n_val
    indices = list(range(n_total))
    random.shuffle(indices)
    train_indices = set(indices[:n_train])

    # YOLO label for positive: one box covering full 32x32 → normalized (0.5, 0.5, 1.0, 1.0)
    pos_label_line = "0 0.5 0.5 1.0 1.0"
    neg_list = list(neg_patches)

    for idx in range(n_total):
        is_train = idx in train_indices
        img_dir = images_train if is_train else images_val
        lbl_dir = labels_train if is_train else labels_val
        stem = f"patch_{idx:06d}"
        if idx < num_pos:
            patch = pos_patches[idx]
            if patch.shape[0] != PATCH or patch.shape[1] != PATCH:
                patch = cv2.resize(patch, (PATCH, PATCH))
            cv2.imwrite(str(img_dir / f"{stem}.jpg"), patch)
            (lbl_dir / f"{stem}.txt").write_text(pos_label_line + "\n")
        else:
            p = neg_list[(idx - num_pos) % len(neg_list)]
            if p.shape[0] != PATCH or p.shape[1] != PATCH:
                p = cv2.resize(p, (PATCH, PATCH))
            cv2.imwrite(str(img_dir / f"{stem}.jpg"), p)
            (lbl_dir / f"{stem}.txt").write_text("")
        if (idx + 1) % 1000 == 0:
            print(f"  {idx + 1}/{n_total}")

    # Signal to train script to use imgsz=32
    (images_train.parent.parent / "imgsz.txt").write_text("32")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate YOLO synthetic walnut dataset (best config)")
    parser.add_argument("--mode", choices=["patch", "full"], default="patch",
                        help="patch: 32x32 one-walnut/neg per image. full: 640x640 multi-walnut.")
    parser.add_argument("--num_images", type=int, default=5000,
                        help="Total images (patch: num_pos when --pos_only else half pos half neg; full: composites)")
    parser.add_argument("--pos_only", action="store_true",
                        help="Patch mode only: no negative images (synthetic positives only; neg patches still used as bg)")
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output dataset root (default: yolo_walnut_synthetic/)",
    )
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Fraction of images for val split (0.1 = 10%%)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for compositing and train/val split")
    parser.add_argument(
        "--config",
        type=str,
        default=str(WORKSPACE / "optuna_results" / "best_config.json"),
        help="Path to best_config.json from optimize_synthetic_config.py",
    )
    parser.add_argument(
        "--cutouts_dir",
        type=str,
        default="output/dataset/walnut_cutouts",
        help="Directory of RGBA walnut cutout PNGs",
    )
    parser.add_argument(
        "--neg_dir",
        type=str,
        default="output/dataset/train/negative",
        help="Directory of negative background patches (32×32 PNGs)",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else WORKSPACE / "yolo_walnut_synthetic"
    val_ratio = getattr(args, "val_ratio", 0.1)
    images_train = out_dir / "images" / "train"
    labels_train = out_dir / "labels" / "train"
    images_val = out_dir / "images" / "val"
    labels_val = out_dir / "labels" / "val"
    for d in (images_train, labels_train, images_val, labels_val):
        d.mkdir(parents=True, exist_ok=True)

    config_path = _resolve_path(args.config)
    if not config_path.is_file():
        print(f"Missing {config_path}. Run optimize_synthetic_config.py first.")
        sys.exit(1)
    with open(config_path, "r") as f:
        best_cfg = json.load(f)
    if "best_params" not in best_cfg:
        print(f"Invalid config file (expected 'best_params'): {config_path}")
        sys.exit(1)
    config = copy.deepcopy(best_cfg["best_params"])
    if config.get("cutout_scale_min", 0) >= config.get("cutout_scale_max", 1):
        config["cutout_scale_max"] = config["cutout_scale_min"] + 0.1

    cutouts_dir = _resolve_path(args.cutouts_dir)
    neg_dir = _resolve_path(args.neg_dir)
    if not cutouts_dir.is_dir():
        print(f"Cutouts directory not found: {cutouts_dir}")
        sys.exit(1)
    if not neg_dir.is_dir():
        print(f"Negative patches directory not found: {neg_dir}")
        sys.exit(1)

    random.seed(args.seed)
    np.random.seed(args.seed)

    cutouts = m.load_cutouts(cutouts_dir)
    neg_patches = m.load_negative_patches(neg_dir, m.PATCH_SIZE)
    print(f"Config:    {config_path}")
    print(f"Cutouts:   {cutouts_dir} ({len(cutouts)} PNGs)")
    print(f"Negatives: {neg_dir} ({len(neg_patches)} PNGs)")
    if not cutouts:
        print("No cutout PNGs found. Run extract_walnuts.py first.")
        sys.exit(1)
    if not neg_patches:
        print("No negative PNGs found. Build dataset first.")
        sys.exit(1)

    if args.mode == "patch":
        if getattr(args, "pos_only", False):
            num_pos = args.num_images
            num_neg = 0
            print(f"Patch mode (pos only): {num_pos} synthetic positives -> {out_dir}")
        else:
            num_pos = args.num_images // 2
            num_neg = args.num_images - num_pos
            print(f"Patch mode: {num_pos} pos + {num_neg} neg -> {out_dir}")
        generate_patch_dataset(
            config, cutouts, neg_patches,
            num_pos, num_neg,
            images_train, labels_train, images_val, labels_val,
            val_ratio, args.seed,
        )
        print(f"Done. Train: {images_train}, Val: {images_val}. Run train with --imgsz 32")
        return

    # --- full mode: 640x640 ---
    n_val = max(0, int(args.num_images * val_ratio))
    n_train = args.num_images - n_val
    green_bgs = [p for p in neg_patches if m.is_green(p, config["green_ratio_thresh"])]
    if len(green_bgs) < 10:
        green_bgs = neg_patches[:200]
    print(f"Full mode: {args.num_images} images ({n_train} train, {n_val} val) -> {out_dir}")

    for i in range(args.num_images):
        img, lines = generate_one_image(config, cutouts, neg_patches, green_bgs)
        stem = f"walnut_synth_{i:06d}"
        if i < n_train:
            img_dir, lbl_dir = images_train, labels_train
        else:
            img_dir, lbl_dir = images_val, labels_val
        img_path = img_dir / f"{stem}.jpg"
        label_path = lbl_dir / f"{stem}.txt"
        cv2.imwrite(str(img_path), img)
        with open(label_path, "w") as f:
            f.write("\n".join(lines))
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{args.num_images}")

    print(f"Done. Train: {images_train}, Val: {images_val}")
    print("Next: run train_yolov8_synthetic.py --data_dir", out_dir)


if __name__ == "__main__":
    main()
