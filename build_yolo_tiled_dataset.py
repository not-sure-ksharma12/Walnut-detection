#!/usr/bin/env python3
"""
build_yolo_tiled_dataset.py — Tile full images into overlapping 640×640 YOLO crops.

What it does:
  - Slices full images into tiles (default 640×640, stride 480) with YOLO labels
  - Keeps annotations whose centre falls in a tile; samples empty tiles as negatives
  - Optional: --num_tiles N random 640×640 crops with real annotation labels (--composite_cutouts for extra cutouts)
  - Writes yolo_walnut_tiled/ (images/labels for train and val from split.json)

Prerequisites:
  - python3 setup.py
  - Full images, annotation .txt files, split.json (from build_annotations11_10_dataset.py)

How to run:
  cd Walnut-detection
  source venv/bin/activate
  python build_yolo_tiled_dataset.py \\
    --config optuna_results/best_config.json \\
    --num_tiles 500 \\
    --image_dir output \\
    --annotation_dir output/annotations \\
    --split_file output/dataset/split.json \\
    --out_dir yolo_walnut_tiled

  Random count in a range:
  python build_yolo_tiled_dataset.py --num_tiles_min 200 --num_tiles_max 800
"""

from __future__ import annotations

import copy
import json
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

WORKSPACE = Path(__file__).resolve().parent
sys.path.insert(0, str(WORKSPACE))

TILE_SIZE = 640
STRIDE = 480
BOX_SIZE = 48
BG_FRACTION = 0.25
CLASS_ID = 0
DEFAULT_SYNTH_PATCHES_MIN = 1
DEFAULT_SYNTH_PATCHES_MAX = 4


def _resolve_path(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else WORKSPACE / p


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


def load_best_config(config_path: Path) -> Dict[str, Any]:
    if not config_path.is_file():
        raise SystemExit(f"Config not found: {config_path}")
    with open(config_path, "r") as f:
        cfg = json.load(f)
    if "best_params" not in cfg and "detection_params" not in cfg:
        raise SystemExit(f"Invalid config (expected best_params / detection_params): {config_path}")
    return cfg


def apply_detection_config(cfg: Dict[str, Any], box_size: int) -> int:
    det = cfg.get("detection_params") or {}
    if "match_radius" in det:
        return max(8, int(round(float(det["match_radius"]))))
    return box_size


def tile_positions(img_dim: int, tile_size: int, stride: int) -> List[int]:
    positions = []
    pos = 0
    while pos + tile_size <= img_dim:
        positions.append(pos)
        pos += stride
    if not positions or positions[-1] + tile_size < img_dim:
        positions.append(max(0, img_dim - tile_size))
    return positions


def labels_for_tile(
    points: List[Tuple[int, int]],
    tx: int,
    ty: int,
    tile_size: int,
    half: int,
) -> List[str]:
    tile_x2 = tx + tile_size
    tile_y2 = ty + tile_size
    lines: List[str] = []
    for px, py in points:
        if tx <= px < tile_x2 and ty <= py < tile_y2:
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
            lines.append(
                f"{CLASS_ID} {cx/tile_size:.6f} {cy/tile_size:.6f} "
                f"{bw/tile_size:.6f} {bh/tile_size:.6f}"
            )
    return lines


@dataclass
class TileCandidate:
    split_name: str
    stem: str
    tx: int
    ty: int
    is_bg: bool
    label_lines: List[str]


def collect_candidates(
    splits: Dict[str, List[str]],
    image_dir: Path,
    annotation_dir: Path,
    tile_size: int,
    stride: int,
    half: int,
    bg_fraction: float,
) -> Dict[str, List[TileCandidate]]:
    import cv2

    pools: Dict[str, List[TileCandidate]] = {"train": [], "val": []}
    for split_name in ("train", "val"):
        for stem in splits.get(split_name, []):
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
            for ty in tile_positions(img_h, tile_size, stride):
                for tx in tile_positions(img_w, tile_size, stride):
                    label_lines = labels_for_tile(points, tx, ty, tile_size, half)
                    is_bg = len(label_lines) == 0
                    if is_bg and random.random() > bg_fraction:
                        continue
                    pools[split_name].append(
                        TileCandidate(split_name, stem, tx, ty, is_bg, label_lines)
                    )
    return pools


def resolve_tile_count(
    num_tiles: Optional[int],
    num_tiles_min: Optional[int],
    num_tiles_max: Optional[int],
) -> Optional[int]:
    if num_tiles_min is not None or num_tiles_max is not None:
        lo = num_tiles_min if num_tiles_min is not None else num_tiles_max
        hi = num_tiles_max if num_tiles_max is not None else num_tiles_min
        if lo is None or hi is None:
            raise SystemExit("Provide both --num_tiles_min and --num_tiles_max for a random count.")
        if lo > hi:
            raise SystemExit("--num_tiles_min must be <= --num_tiles_max")
        return random.randint(lo, hi)
    return num_tiles


def image_fits_tile(img_w: int, img_h: int, tile_size: int) -> bool:
    return img_w >= tile_size and img_h >= tile_size


def flip_labels_horizontal(label_lines: List[str]) -> List[str]:
    flipped: List[str] = []
    for line in label_lines:
        parts = line.split()
        if len(parts) < 5:
            continue
        cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        flipped.append(f"{parts[0]} {1.0 - cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return flipped


def pick_positive_crop_origin(
    points: List[Tuple[int, int]],
    img_w: int,
    img_h: int,
    tile_size: int,
    half: int,
    max_tries: int = 40,
) -> Tuple[int, int]:
    """Pick a crop that contains at least one annotated walnut."""
    if not points:
        return pick_random_crop_origin(img_w, img_h, tile_size)
    for _ in range(max_tries):
        tx, ty = pick_random_crop_origin(img_w, img_h, tile_size)
        if labels_for_tile(points, tx, ty, tile_size, half):
            return tx, ty
    px, py = random.choice(points)
    tx = max(0, min(int(px - tile_size / 2), max(0, img_w - tile_size)))
    ty = max(0, min(int(py - tile_size / 2), max(0, img_h - tile_size)))
    return tx, ty


def map_points_to_zoomed(
    points: List[Tuple[int, int]],
    img_w: int,
    img_h: int,
    zoom_w: int,
    zoom_h: int,
) -> Tuple[List[Tuple[int, int]], float, float]:
    sx = zoom_w / img_w
    sy = zoom_h / img_h
    mapped = [(int(round(px * sx)), int(round(py * sy))) for px, py in points]
    return mapped, sx, sy


def crop_tile_with_labels(
    img,
    points: List[Tuple[int, int]],
    img_w: int,
    img_h: int,
    tile_size: int,
    half: int,
    prefer_empty: bool,
    prefer_positive: bool,
) -> Tuple[Any, int, int, List[str]]:
    """Return a tile_size×tile_size crop and YOLO labels from preexisting annotations."""
    import cv2

    if image_fits_tile(img_w, img_h, tile_size):
        if prefer_empty:
            tx, ty = pick_background_crop_origin(points, img_w, img_h, tile_size, half)
        elif prefer_positive:
            tx, ty = pick_positive_crop_origin(points, img_w, img_h, tile_size, half)
        else:
            tx, ty = pick_random_crop_origin(img_w, img_h, tile_size)
        tile = img[ty : ty + tile_size, tx : tx + tile_size].copy()
        labels = labels_for_tile(points, tx, ty, tile_size, half)
        return tile, tx, ty, labels

    # Source smaller than tile_size (e.g. 519×482): upscale, then crop.
    zoom_w = max(tile_size + 1, int(round(tile_size * random.uniform(1.08, 1.35))))
    zoom_h = max(tile_size + 1, int(round(tile_size * random.uniform(1.08, 1.35))))
    zoomed = cv2.resize(img, (zoom_w, zoom_h), interpolation=cv2.INTER_LINEAR)
    mapped, sx, sy = map_points_to_zoomed(points, img_w, img_h, zoom_w, zoom_h)
    scaled_half = max(4, int(round(half * (sx + sy) / 2)))
    if prefer_empty:
        tx, ty = pick_background_crop_origin(mapped, zoom_w, zoom_h, tile_size, scaled_half)
    elif prefer_positive:
        tx, ty = pick_positive_crop_origin(mapped, zoom_w, zoom_h, tile_size, scaled_half)
    else:
        tx, ty = pick_random_crop_origin(zoom_w, zoom_h, tile_size)
    tile = zoomed[ty : ty + tile_size, tx : tx + tile_size].copy()
    labels = labels_for_tile(mapped, tx, ty, tile_size, scaled_half)
    return tile, tx, ty, labels


def scale_points_to_tile(
    points: List[Tuple[int, int]],
    img_w: int,
    img_h: int,
    tile_size: int,
) -> List[Tuple[int, int]]:
    if img_w <= 0 or img_h <= 0:
        return points
    sx = tile_size / img_w
    sy = tile_size / img_h
    return [(int(round(px * sx)), int(round(py * sy))) for px, py in points]


def pick_random_crop_origin(img_w: int, img_h: int, tile_size: int) -> Tuple[int, int]:
    max_tx = max(0, img_w - tile_size)
    max_ty = max(0, img_h - tile_size)
    tx = random.randint(0, max_tx) if max_tx > 0 else 0
    ty = random.randint(0, max_ty) if max_ty > 0 else 0
    return tx, ty


def pick_background_crop_origin(
    points: List[Tuple[int, int]],
    img_w: int,
    img_h: int,
    tile_size: int,
    half: int,
    max_tries: int = 40,
) -> Tuple[int, int]:
    for _ in range(max_tries):
        tx, ty = pick_random_crop_origin(img_w, img_h, tile_size)
        if not labels_for_tile(points, tx, ty, tile_size, half):
            return tx, ty
    return pick_random_crop_origin(img_w, img_h, tile_size)


def composite_random_walnuts(
    tile_bgr,
    cutouts: list,
    config: Dict[str, Any],
    tile_size: int,
    box_size: int,
    apply_augmentations: Optional[Callable],
    synth_patches_min: int,
    synth_patches_max: int,
) -> Tuple[Any, List[str]]:
    """Paste synthetic cutouts at random positions; return (BGR image, YOLO label lines)."""
    import cv2
    import numpy as np

    from optimize_synthetic_config import composite_one

    if not cutouts:
        return tile_bgr, []

    img = tile_bgr.copy()
    th, tw = img.shape[:2]
    if th != tile_size or tw != tile_size:
        img = cv2.resize(img, (tile_size, tile_size), interpolation=cv2.INTER_LINEAR)
        th = tw = tile_size

    scale_min = config["cutout_scale_min"]
    scale_max = config["cutout_scale_max"]
    pos_jitter = config.get("position_jitter", 0.2)
    # Match generate_yolo_synthetic_dataset (~20–120 px on 640×640)
    target_px = max(box_size, 80)

    if synth_patches_max <= 0:
        return img, []

    n_patches = random.randint(synth_patches_min, synth_patches_max)
    lines: List[str] = []
    for _ in range(n_patches):
        cutout_bgr, cutout_alpha = random.choice(cutouts)
        max_side = max(cutout_bgr.shape[0], cutout_bgr.shape[1], 1)
        base_scale = (target_px / max_side) * random.uniform(0.6, 1.4)
        rel = random.uniform(scale_min, scale_max)
        scale = rel * base_scale
        new_w = max(20, min(tile_size - 20, int(cutout_bgr.shape[1] * scale)))
        new_h = max(20, min(tile_size - 20, int(cutout_bgr.shape[0] * scale)))

        bgr_s = cv2.resize(cutout_bgr, (new_w, new_h))
        alpha_s = cv2.resize(cutout_alpha, (new_w, new_h))
        if apply_augmentations is not None:
            bgr_s = apply_augmentations(bgr_s, config)

        max_x = tile_size - new_w
        max_y = tile_size - new_h
        if max_x < 0 or max_y < 0:
            continue
        x = random.randint(0, max_x)
        y = random.randint(0, max_y)
        jitter_px = int(pos_jitter * tile_size * 0.15)
        if jitter_px > 0:
            x = max(0, min(max_x, x + random.randint(-jitter_px, jitter_px)))
            y = max(0, min(max_y, y + random.randint(-jitter_px, jitter_px)))

        composite_one(img, bgr_s, alpha_s, x, y)
        cx = (x + new_w / 2.0) / tile_size
        cy = (y + new_h / 2.0) / tile_size
        bw = new_w / tile_size
        bh = new_h / tile_size
        lines.append(f"{CLASS_ID} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

    return img, lines


@dataclass
class ImageRecord:
    split_name: str
    stem: str
    img_path: Path
    points: List[Tuple[int, int]]
    img_w: int
    img_h: int


def load_image_records(
    splits: Dict[str, List[str]],
    image_dir: Path,
    annotation_dir: Path,
) -> Dict[str, List[ImageRecord]]:
    import cv2

    records: Dict[str, List[ImageRecord]] = {"train": [], "val": []}
    for split_name in ("train", "val"):
        for stem in splits.get(split_name, []):
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
            records[split_name].append(
                ImageRecord(split_name, stem, img_path, parse_annotations(ann_path), img_w, img_h)
            )
    return records


def write_random_crop_tiles(
    records: Dict[str, List[ImageRecord]],
    total: int,
    val_ratio: float,
    out_dir: Path,
    tile_size: int,
    half: int,
    bg_fraction: float,
    cutouts: list,
    aug_config: Dict[str, Any],
    box_size: int,
    apply_augmentations: Optional[Callable],
    composite_cutouts: bool,
    synth_patches_min: int,
    synth_patches_max: int,
) -> Dict[str, Dict[str, int]]:
    import cv2

    train_recs = records["train"]
    val_recs = records["val"]
    if not train_recs and not val_recs:
        raise SystemExit("No images found for random tile generation.")

    if not val_recs:
        n_train, n_val = total, 0
    elif not train_recs:
        n_train, n_val = 0, total
    else:
        n_val = int(round(total * val_ratio))
        if total >= 2:
            n_val = max(1, min(n_val, total - 1))
        else:
            n_val = 0
        n_train = total - n_val

    stats = {"train": {"pos": 0, "bg": 0}, "val": {"pos": 0, "bg": 0}}
    counters = {"train": 0, "val": 0}

    def emit(split_name: str, rec: ImageRecord) -> None:
        img = cv2.imread(str(rec.img_path))
        if img is None:
            return

        is_bg = random.random() < bg_fraction
        tile_img, tx, ty, label_lines = crop_tile_with_labels(
            img,
            rec.points,
            rec.img_w,
            rec.img_h,
            tile_size,
            half,
            prefer_empty=is_bg,
            prefer_positive=not is_bg,
        )
        if random.random() < 0.5:
            tile_img = cv2.flip(tile_img, 1)
            label_lines = flip_labels_horizontal(label_lines)

        if composite_cutouts and cutouts:
            tile_img, synth_lines = composite_random_walnuts(
                tile_img,
                cutouts,
                aug_config,
                tile_size,
                box_size,
                apply_augmentations,
                synth_patches_min,
                synth_patches_max,
            )
            label_lines = label_lines + synth_lines

        if label_lines:
            stats[split_name]["pos"] += 1
        else:
            stats[split_name]["bg"] += 1

        idx = counters[split_name]
        counters[split_name] += 1
        img_sub = out_dir / "images" / split_name
        lbl_sub = out_dir / "labels" / split_name
        tile_name = f"{rec.stem}_r{tx}_{ty}_s{idx:05d}"
        cv2.imwrite(
            str(img_sub / f"{tile_name}.jpg"),
            tile_img,
            [cv2.IMWRITE_JPEG_QUALITY, 95],
        )
        label_text = "\n".join(label_lines) + "\n" if label_lines else ""
        (lbl_sub / f"{tile_name}.txt").write_text(label_text, encoding="utf-8")

    for _ in range(n_train):
        emit("train", random.choice(train_recs))
    for _ in range(n_val):
        emit("val", random.choice(val_recs))
    return stats


def sample_tiles(
    pools: Dict[str, List[TileCandidate]],
    total: int,
    val_ratio: float,
) -> List[Tuple[TileCandidate, int]]:
    """Return (candidate, sample_index) pairs; uses replacement if total > pool size."""
    train_pool = pools["train"]
    val_pool = pools["val"]
    if not train_pool and not val_pool:
        raise SystemExit("No tile candidates found. Check image_dir, annotations, and split.")

    if not val_pool:
        n_train, n_val = total, 0
    elif not train_pool:
        n_train, n_val = 0, total
    else:
        n_val = int(round(total * val_ratio))
        if total >= 2:
            n_val = max(1, min(n_val, total - 1))
        else:
            n_val = 0
        n_train = total - n_val

    out: List[Tuple[TileCandidate, int]] = []

    def draw(pool: List[TileCandidate], n: int) -> None:
        if n <= 0:
            return
        if not pool:
            return
        picks = random.choices(pool, k=n)
        for i, cand in enumerate(picks):
            out.append((cand, i))

    draw(train_pool, n_train)
    draw(val_pool, n_val)
    random.shuffle(out)
    return out


def write_tile(
    cand: TileCandidate,
    sample_idx: int,
    image_dir: Path,
    annotation_dir: Path,
    out_dir: Path,
    tile_size: int,
    half: int,
    apply_aug: bool,
    apply_augmentations: Optional[Callable],
    aug_config: Dict[str, Any],
) -> None:
    import cv2

    img_path = _find_image(image_dir, cand.stem)
    if img_path is None:
        return
    img = cv2.imread(str(img_path))
    if img is None:
        return

    img_h, img_w = img.shape[:2]
    ty, tx = cand.ty, cand.tx
    ann_path = annotation_dir / f"{cand.stem}.txt"
    points = parse_annotations(ann_path) if ann_path.exists() else []
    if image_fits_tile(img_w, img_h, tile_size):
        tile_img = img[ty : ty + tile_size, tx : tx + tile_size].copy()
        label_lines = list(cand.label_lines)
    else:
        tile_img, tx, ty, label_lines = crop_tile_with_labels(
            img,
            points,
            img_w,
            img_h,
            tile_size,
            half,
            prefer_empty=cand.is_bg,
            prefer_positive=not cand.is_bg,
        )

    img_sub = out_dir / "images" / cand.split_name
    lbl_sub = out_dir / "labels" / cand.split_name
    suffix = f"_s{sample_idx:05d}" if sample_idx > 0 else ""
    tile_name = f"{cand.stem}_t{cand.tx}_{cand.ty}{suffix}"
    cv2.imwrite(
        str(img_sub / f"{tile_name}.jpg"),
        tile_img,
        [cv2.IMWRITE_JPEG_QUALITY, 95],
    )
    label_text = "\n".join(label_lines) + "\n" if label_lines else ""
    (lbl_sub / f"{tile_name}.txt").write_text(label_text, encoding="utf-8")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Build tiled YOLO dataset")
    parser.add_argument("--image_dir", type=str, default="output")
    parser.add_argument("--annotation_dir", type=str, default="output/annotations")
    parser.add_argument("--split_file", type=str, default="output/dataset/split.json")
    parser.add_argument("--out_dir", type=str, default="yolo_walnut_tiled")
    parser.add_argument("--tile_size", type=int, default=TILE_SIZE)
    parser.add_argument("--stride", type=int, default=STRIDE)
    parser.add_argument("--box_size", type=int, default=BOX_SIZE)
    parser.add_argument("--bg_fraction", type=float, default=BG_FRACTION)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--config",
        type=str,
        default=str(WORKSPACE / "optuna_results" / "best_config.json"),
    )
    parser.add_argument("--no_augment", action="store_true")
    parser.add_argument(
        "--num_tiles",
        type=int,
        default=None,
        help="Randomly sample this many tiles total (train/val split by --val_ratio)",
    )
    parser.add_argument(
        "--num_tiles_min",
        type=int,
        default=None,
        help="With --num_tiles_max: sample a random count in [min, max] each run",
    )
    parser.add_argument(
        "--num_tiles_max",
        type=int,
        default=None,
        help="With --num_tiles_min: sample a random count in [min, max] each run",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="Fraction of --num_tiles assigned to val (default 0.1)",
    )
    parser.add_argument(
        "--cutouts_dir",
        type=str,
        default="output/dataset/walnut_cutouts",
        help="Walnut RGBA cutouts for random placement (--num_tiles mode)",
    )
    parser.add_argument(
        "--composite_cutouts",
        action="store_true",
        help="Also paste augmented walnut cutouts on tiles (labels = annotations + cutouts)",
    )
    parser.add_argument(
        "--random_walnuts",
        action="store_true",
        default=None,
        help="Alias for --composite_cutouts",
    )
    parser.add_argument(
        "--grid_tiles",
        action="store_true",
        help="With --num_tiles: resample fixed grid crops instead of random 640×640 cuts",
    )
    parser.add_argument(
        "--num_synthetic_patches",
        type=int,
        default=None,
        help="Fixed number of synthetic cutouts per tile (with --composite_cutouts)",
    )
    parser.add_argument(
        "--synthetic_patches_min",
        type=int,
        default=DEFAULT_SYNTH_PATCHES_MIN,
        help="Min synthetic cutouts per tile when --composite_cutouts (default: 1)",
    )
    parser.add_argument(
        "--synthetic_patches_max",
        type=int,
        default=DEFAULT_SYNTH_PATCHES_MAX,
        help="Max synthetic cutouts per tile when --composite_cutouts (default: 4)",
    )
    args = parser.parse_args()

    if args.num_tiles is not None and (
        args.num_tiles_min is not None or args.num_tiles_max is not None
    ):
        raise SystemExit("Use either --num_tiles or --num_tiles_min/--num_tiles_max, not both.")

    random.seed(args.seed)

    image_dir = _resolve_path(args.image_dir)
    annotation_dir = _resolve_path(args.annotation_dir)
    split_file = _resolve_path(args.split_file)
    out_dir = _resolve_path(args.out_dir)
    config_path = _resolve_path(args.config)
    best_cfg = load_best_config(config_path)
    aug_config = copy.deepcopy(best_cfg.get("best_params") or {})
    if aug_config.get("cutout_scale_min", 0) >= aug_config.get("cutout_scale_max", 1):
        aug_config["cutout_scale_max"] = aug_config.get("cutout_scale_min", 0.5) + 0.1

    tile_size = args.tile_size
    stride = args.stride
    box_size = apply_detection_config(best_cfg, args.box_size)
    half = box_size // 2
    apply_aug = not args.no_augment and bool(aug_config)

    if apply_aug:
        from generate_yolo_synthetic_dataset import apply_augmentations
    else:
        apply_augmentations = None

    target_tiles = resolve_tile_count(args.num_tiles, args.num_tiles_min, args.num_tiles_max)
    composite_cutouts = args.composite_cutouts or bool(args.random_walnuts)

    if args.num_synthetic_patches is not None:
        if args.num_synthetic_patches < 0:
            raise SystemExit("--num_synthetic_patches must be >= 0")
        synth_patches_min = synth_patches_max = args.num_synthetic_patches
    else:
        synth_patches_min = args.synthetic_patches_min
        synth_patches_max = args.synthetic_patches_max
    if synth_patches_min < 0 or synth_patches_max < 0:
        raise SystemExit("--synthetic_patches_min/max must be >= 0")
    if synth_patches_min > synth_patches_max:
        raise SystemExit("--synthetic_patches_min must be <= --synthetic_patches_max")

    cutouts_dir = _resolve_path(args.cutouts_dir)
    cutouts: list = []
    if composite_cutouts:
        from optimize_synthetic_config import load_cutouts

        cutouts = load_cutouts(cutouts_dir)
        if not cutouts:
            raise SystemExit(
                f"--composite_cutouts needs cutout PNGs in {cutouts_dir}. "
                "Run extract_walnuts.py first."
            )

    print(f"Config:      {config_path}")
    print(f"Box size:    {box_size}px")
    print(f"Augment:     {apply_aug} (cutouts only, foliage unchanged)")
    if target_tiles is not None:
        mode = "grid resample" if args.grid_tiles else "random 640×640 crop + annotation labels"
        print(f"Tile target: {target_tiles} ({mode})")
        if composite_cutouts:
            if synth_patches_min == synth_patches_max:
                synth_desc = f"{synth_patches_min} per tile"
            else:
                synth_desc = f"{synth_patches_min}–{synth_patches_max} per tile"
            print(f"Cutouts:     {cutouts_dir} ({len(cutouts)} PNGs, {synth_desc})")
    else:
        print("Tile target: all grid positions (bg_fraction filter applies)")

    if not split_file.exists():
        raise SystemExit(f"Split file not found: {split_file}")
    with open(split_file) as f:
        splits = json.load(f)

    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        p = out_dir / sub
        if target_tiles is not None and p.exists():
            shutil.rmtree(p)
        p.mkdir(parents=True, exist_ok=True)

    pools = collect_candidates(
        splits, image_dir, annotation_dir, tile_size, stride, half, args.bg_fraction
    )
    print(
        f"Candidates: train={len(pools['train'])}, val={len(pools['val'])} "
        f"(unique grid cells after bg filter)"
    )

    stats = {"train": {"pos": 0, "bg": 0}, "val": {"pos": 0, "bg": 0}}

    if target_tiles is not None:
        if target_tiles < 1:
            raise SystemExit("--num_tiles must be >= 1")
        if not args.grid_tiles:
            records = load_image_records(splits, image_dir, annotation_dir)
            stats = write_random_crop_tiles(
                records,
                target_tiles,
                args.val_ratio,
                out_dir,
                tile_size,
                half,
                args.bg_fraction,
                cutouts,
                aug_config,
                box_size,
                apply_augmentations if apply_aug else None,
                composite_cutouts,
                synth_patches_min,
                synth_patches_max,
            )
        else:
            sampled = sample_tiles(pools, target_tiles, args.val_ratio)
            for idx, (cand, _) in enumerate(sampled):
                write_tile(
                    cand, idx, image_dir, annotation_dir, out_dir, tile_size, half,
                    apply_aug, apply_augmentations, aug_config,
                )
                if cand.is_bg:
                    stats[cand.split_name]["bg"] += 1
                else:
                    stats[cand.split_name]["pos"] += 1
            if target_tiles > len(pools["train"]) + len(pools["val"]):
                print(
                    "Note: requested more tiles than unique grid cells; "
                    "duplicates reuse the same crop positions (_s##### suffix). "
                    "Use default --num_tiles (no --grid_tiles) for random annotated crops."
                )
    else:
        for split_name in ("train", "val"):
            for cand in pools[split_name]:
                write_tile(
                    cand, 0, image_dir, annotation_dir, out_dir, tile_size, half,
                    apply_aug, apply_augmentations, aug_config,
                )
                if cand.is_bg:
                    stats[split_name]["bg"] += 1
                else:
                    stats[split_name]["pos"] += 1

    for s in ("train", "val"):
        total = stats[s]["pos"] + stats[s]["bg"]
        print(f"{s}: {total} tiles ({stats[s]['pos']} with walnuts, {stats[s]['bg']} background)")

    config_snapshot = out_dir / "best_config_used.json"
    with open(config_snapshot, "w") as f:
        json.dump(
            {
                "source_config": str(config_path.resolve()),
                "detection_params": best_cfg.get("detection_params"),
                "best_params": best_cfg.get("best_params"),
                "applied": {
                    "box_size": box_size,
                    "tile_size": tile_size,
                    "tile_stride": stride,
                    "augment_cutouts": apply_aug,
                    "num_tiles": target_tiles,
                    "val_ratio": args.val_ratio,
                    "composite_cutouts": composite_cutouts,
                    "synthetic_patches_min": synth_patches_min if composite_cutouts else None,
                    "synthetic_patches_max": synth_patches_max if composite_cutouts else None,
                    "cutouts_dir": str(cutouts_dir) if composite_cutouts else None,
                },
            },
            f,
            indent=2,
        )
    print(f"Wrote {config_snapshot}")

    data_yaml = out_dir / "data.yaml"
    data_yaml.write_text(
        f"""# Walnut tiled dataset (tile {tile_size}px, stride {stride}, box {box_size}px)
# Built with config: {config_path.name}
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
