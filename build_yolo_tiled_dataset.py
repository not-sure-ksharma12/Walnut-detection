#!/usr/bin/env python3
"""
build_yolo_tiled_dataset.py — Tile full images into overlapping 640×640 YOLO crops.

What it does:
  - Slices cropped_images/ into 640×640 tiles (stride 480) with YOLO labels
  - Keeps annotations whose centre falls in a tile; samples empty tiles as negatives
  - Writes yolo_walnut_tiled/ (images/labels for train and val from split.json)

Prerequisites:
  - python3 setup.py
  - cropped_images/, annotated_images/, split.json

How to run:
  cd Walnut-detection
  source venv/bin/activate
  python build_yolo_tiled_dataset.py
  python build_yolo_tiled_dataset.py --image_dir cropped_images --split_file split.json
"""

import json
import random
import sys
from pathlib import Path
from typing import List, Optional, Tuple

WORKSPACE = Path(__file__).resolve().parent
CROPPED_IMAGES = WORKSPACE / "cropped_images"
ANNOTATED_IMAGES = WORKSPACE / "annotated_images"
SPLIT_FILE = WORKSPACE / "split.json"
OUT_DIR = WORKSPACE / "yolo_walnut_tiled"

TILE_SIZE = 640
STRIDE = 480
BOX_SIZE = 48
BG_FRACTION = 0.25
CLASS_ID = 0


def _find_image(image_dir: Path, stem: str) -> Optional[Path]:
    for ext in (".JPG", ".jpg", ".png", ".PNG", ".jpeg"):
        p = image_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def parse_annotations(annotation_path: Path) -> List[Tuple[int, int]]:
    points = []
    with open(annotation_path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    points.append((int(parts[0]), int(parts[1])))
                except ValueError:
                    continue
    return points


def tile_positions(img_dim: int, tile_size: int, stride: int) -> List[int]:
    """Return top-left positions for tiles along one dimension."""
    positions = []
    pos = 0
    while pos + tile_size <= img_dim:
        positions.append(pos)
        pos += stride
    if not positions or positions[-1] + tile_size < img_dim:
        positions.append(max(0, img_dim - tile_size))
    return positions


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Build tiled YOLO dataset")
    parser.add_argument(
        "--image_dir",
        type=str,
        default=str(CROPPED_IMAGES),
        help="Directory of full-resolution cropped images",
    )
    parser.add_argument(
        "--annotation_dir",
        type=str,
        default=str(ANNOTATED_IMAGES),
        help="Directory of annotation .txt files (one per image stem)",
    )
    parser.add_argument(
        "--split_file",
        type=str,
        default=str(SPLIT_FILE),
        help="JSON file with train/val/test image stem lists",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=str(OUT_DIR),
        help="Output YOLO dataset root (images/ and labels/ subdirs created)",
    )
    parser.add_argument(
        "--tile_size",
        type=int,
        default=TILE_SIZE,
        help="Tile width and height in pixels",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=STRIDE,
        help="Stride between tile origins in pixels",
    )
    parser.add_argument(
        "--box_size",
        type=int,
        default=BOX_SIZE,
        help="YOLO box side length in pixels for each walnut in a tile",
    )
    parser.add_argument(
        "--bg_fraction",
        type=float,
        default=BG_FRACTION,
        help="Fraction of empty (background) tiles to keep (0–1)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for tile sampling")
    args = parser.parse_args()

    random.seed(args.seed)

    image_dir = Path(args.image_dir)
    annotation_dir = Path(args.annotation_dir)
    split_file = Path(args.split_file)
    out_dir = Path(args.out_dir)
    tile_size = args.tile_size
    stride = args.stride
    box_size = args.box_size
    half = box_size // 2

    if not split_file.exists():
        print(f"Split file not found: {split_file}")
        sys.exit(1)
    with open(split_file) as f:
        splits = json.load(f)

    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    try:
        import cv2
    except ImportError:
        print("cv2 required: pip install opencv-python")
        sys.exit(1)

    stats = {"train": {"pos": 0, "bg": 0}, "val": {"pos": 0, "bg": 0}}

    for split_name in ("train", "val"):
        stems = splits.get(split_name, [])
        img_sub = out_dir / "images" / split_name
        lbl_sub = out_dir / "labels" / split_name

        for stem in stems:
            ann_path = annotation_dir / f"{stem}.txt"
            if not ann_path.exists():
                continue
            img_path = _find_image(image_dir, stem)
            if img_path is None:
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                continue
            img_h, img_w = img.shape[:2]
            points = parse_annotations(ann_path)

            x_positions = tile_positions(img_w, tile_size, stride)
            y_positions = tile_positions(img_h, tile_size, stride)

            for ty in y_positions:
                for tx in x_positions:
                    tile_x2 = tx + tile_size
                    tile_y2 = ty + tile_size

                    # Find annotations whose center is inside this tile
                    tile_labels = []
                    for px, py in points:
                        if tx <= px < tile_x2 and ty <= py < tile_y2:
                            # Box in tile-relative coordinates, clipped
                            bx1 = max(0, px - half - tx)
                            by1 = max(0, py - half - ty)
                            bx2 = min(tile_size, px + half - tx)
                            by2 = min(tile_size, py + half - ty)
                            if bx2 <= bx1 or by2 <= by1:
                                continue
                            cx = (bx1 + bx2) / 2.0
                            cy = (by1 + by2) / 2.0
                            bw = bx2 - bx1
                            bh = by2 - by1
                            tile_labels.append(
                                f"{CLASS_ID} {cx/tile_size:.6f} {cy/tile_size:.6f} "
                                f"{bw/tile_size:.6f} {bh/tile_size:.6f}"
                            )

                    is_bg = len(tile_labels) == 0
                    if is_bg and random.random() > args.bg_fraction:
                        continue

                    tile_name = f"{stem}_t{tx}_{ty}"
                    tile_img = img[ty:tile_y2, tx:tile_x2]
                    cv2.imwrite(str(img_sub / f"{tile_name}.jpg"), tile_img,
                                [cv2.IMWRITE_JPEG_QUALITY, 95])
                    label_text = "\n".join(tile_labels) + "\n" if tile_labels else ""
                    (lbl_sub / f"{tile_name}.txt").write_text(label_text, encoding="utf-8")

                    if is_bg:
                        stats[split_name]["bg"] += 1
                    else:
                        stats[split_name]["pos"] += 1

    for s in ("train", "val"):
        total = stats[s]["pos"] + stats[s]["bg"]
        print(f"{s}: {total} tiles ({stats[s]['pos']} with walnuts, {stats[s]['bg']} background)")

    data_yaml = out_dir / "data.yaml"
    data_yaml.write_text(
        f"""# Walnut tiled dataset (640x640 crops, stride {stride}, box {box_size}px)
path: {out_dir.resolve()}
train: images/train
val: images/val
nc: 1
names:
  0: walnut
""",
        encoding="utf-8",
    )
    print(f"Wrote {data_yaml}")


if __name__ == "__main__":
    main()
