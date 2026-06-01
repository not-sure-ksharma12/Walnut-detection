#!/usr/bin/env python3
"""
build_yolo_tiled_dataset.py — Tile full images into overlapping 640×640 YOLO crops.

What it does:
  - Slices full images into tiles (default 640×640, stride 480) with YOLO labels
  - Keeps annotations whose centre falls in a tile; samples empty tiles as negatives
  - Optional: --num_tiles N or --num_tiles_min/--num_tiles_max for random subsampling
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
    out_dir: Path,
    tile_size: int,
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

    ty, tx = cand.ty, cand.tx
    tile_img = img[ty : ty + tile_size, tx : tx + tile_size]
    if apply_aug and apply_augmentations is not None:
        tile_img = apply_augmentations(tile_img, aug_config)

    img_sub = out_dir / "images" / cand.split_name
    lbl_sub = out_dir / "labels" / cand.split_name
    suffix = f"_s{sample_idx:05d}" if sample_idx > 0 else ""
    tile_name = f"{cand.stem}_t{cand.tx}_{cand.ty}{suffix}"
    cv2.imwrite(
        str(img_sub / f"{tile_name}.jpg"),
        tile_img,
        [cv2.IMWRITE_JPEG_QUALITY, 95],
    )
    label_text = "\n".join(cand.label_lines) + "\n" if cand.label_lines else ""
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

    print(f"Config:      {config_path}")
    print(f"Box size:    {box_size}px")
    print(f"Augment:     {apply_aug}")
    if target_tiles is not None:
        print(f"Tile target: {target_tiles} (random sample, replacement OK)")
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
        sampled = sample_tiles(pools, target_tiles, args.val_ratio)
        for idx, (cand, _) in enumerate(sampled):
            write_tile(
                cand, idx, image_dir, out_dir, tile_size,
                apply_aug, apply_augmentations, aug_config,
            )
            if cand.is_bg:
                stats[cand.split_name]["bg"] += 1
            else:
                stats[cand.split_name]["pos"] += 1
        if target_tiles > len(pools["train"]) + len(pools["val"]):
            print(
                "Note: requested more tiles than unique candidates; "
                "duplicates use new augmentations (_s##### suffix)."
            )
    else:
        for split_name in ("train", "val"):
            for cand in pools[split_name]:
                write_tile(
                    cand, 0, image_dir, out_dir, tile_size,
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
                    "augment_tiles": apply_aug,
                    "num_tiles": target_tiles,
                    "val_ratio": args.val_ratio,
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
