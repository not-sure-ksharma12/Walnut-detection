#!/usr/bin/env python3
"""
train_yolov8_synthetic.py — Train YOLOv8n on the synthetic walnut dataset.

What it does:
  - Optionally runs generate_yolo_synthetic_dataset.py (--generate or missing dataset)
  - Writes data.yaml and trains ultralytics YOLOv8n (patch imgsz=32 or full imgsz=640)
  - Saves runs under yolo_runs/ (default name: walnut_synthetic)

Prerequisites:
  - python3 setup.py (includes ultralytics)
  - YOLO dataset at yolo_walnut_synthetic/ or use --generate

How to run:
  cd Walnut-detection
  source venv/bin/activate
  python generate_yolo_synthetic_dataset.py --mode patch --num_images 5000
  python train_yolov8_synthetic.py
  python train_yolov8_synthetic.py --generate --num_images 5000 --epochs 100
"""

import random
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

WORKSPACE = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = WORKSPACE / "yolo_walnut_synthetic"
DEFAULT_EPOCHS = 100
DEFAULT_IMGSZ = 640
DEFAULT_BATCH = 16
IMAGE_GLOBS = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")


def _list_images(folder: Path) -> List[Path]:
    if not folder.is_dir():
        return []
    files: List[Path] = []
    for pattern in IMAGE_GLOBS:
        files.extend(folder.glob(pattern))
    return sorted({p.resolve() for p in files})


def ensure_val_split(
    data_dir: Path,
    min_val: int = 1,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> Tuple[str, str]:
    """Ensure images/val is non-empty. Returns (train_rel, val_rel) paths for data.yaml."""
    train_img = data_dir / "images" / "train"
    val_img = data_dir / "images" / "val"
    train_lbl = data_dir / "labels" / "train"
    val_lbl = data_dir / "labels" / "val"
    val_img.mkdir(parents=True, exist_ok=True)
    val_lbl.mkdir(parents=True, exist_ok=True)

    train_files = _list_images(train_img)
    val_files = _list_images(val_img)

    if len(val_files) >= min_val:
        return "images/train", "images/val"

    if not train_files:
        raise SystemExit(f"No training images in {train_img}")

    if len(train_files) == 1:
        print(
            "Warning: only one training image; YOLO val will reuse images/train "
            "(not ideal, but required by ultralytics)."
        )
        return "images/train", "images/train"

    rng = random.Random(seed)
    n_move = max(min_val, int(round(len(train_files) * val_fraction)))
    n_move = min(n_move, len(train_files) - 1)
    to_move = rng.sample(train_files, n_move)
    for src in to_move:
        shutil.move(str(src), str(val_img / src.name))
        lbl_src = train_lbl / f"{src.stem}.txt"
        lbl_dst = val_lbl / f"{src.stem}.txt"
        if lbl_src.exists():
            shutil.move(str(lbl_src), str(lbl_dst))
        else:
            lbl_dst.write_text("", encoding="utf-8")
    print(f"Moved {n_move} image(s) from train → val (val folder was empty)")
    return "images/train", "images/val"


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train YOLOv8n on synthetic walnut dataset")
    parser.add_argument("--data_dir", type=str, default=str(DEFAULT_DATA_DIR),
                        help="Path to YOLO dataset (images/train, labels/train, images/val, labels/val)")
    parser.add_argument("--generate", action="store_true",
                        help="Run dataset generator first (uses --num_images)")
    parser.add_argument("--num_images", type=int, default=5000,
                        help="Number of synthetic images to generate (if --generate)")
    parser.add_argument("--mode", choices=["patch", "full"], default="patch",
                        help="Dataset mode when using --generate")
    parser.add_argument("--pos_only", action="store_true",
                        help="When --generate in patch mode: only synthetic positives, no negative images")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Training epochs for YOLOv8n")
    parser.add_argument(
        "--imgsz",
        type=int,
        default=DEFAULT_IMGSZ,
        help="Training image size (auto-reads imgsz.txt=32 for patch datasets)",
    )
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH, help="Training batch size")
    from device_utils import add_device_argument, resolve_device, ultralytics_device

    add_device_argument(parser, default="auto")
    parser.add_argument(
        "--project",
        type=str,
        default=str(WORKSPACE / "yolo_runs"),
        help="Ultralytics project directory for run outputs",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="walnut_synthetic",
        help="Run name under project/ (weights saved to project/name/weights/)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if args.generate or not (data_dir / "images" / "train").exists():
        print("Generating synthetic YOLO dataset...")
        cmd = [
            sys.executable,
            str(Path(__file__).parent / "generate_yolo_synthetic_dataset.py"),
            "--mode", args.mode,
            "--num_images", str(args.num_images),
            "--out_dir", str(data_dir),
        ]
        if getattr(args, "pos_only", False):
            cmd.append("--pos_only")
        subprocess.run(cmd, check=True, cwd=str(WORKSPACE))
        if not (data_dir / "images" / "train").exists():
            print("Dataset generation failed.")
            sys.exit(1)

    train_rel, val_rel = ensure_val_split(data_dir)
    n_train = len(_list_images(data_dir / train_rel))
    n_val = len(_list_images(data_dir / val_rel))
    print(f"Dataset: {n_train} train, {n_val} val images")

    if n_train < 1 or n_val < 1:
        raise SystemExit("Need at least one train and one val image for YOLO training.")

    batch = min(args.batch, max(1, n_train))
    if batch < args.batch:
        print(f"Reduced batch size {args.batch} → {batch} (small train set)")

    data_yaml = data_dir / "data.yaml"
    data_yaml.write_text(
        f"""# Walnut YOLO dataset
path: {data_dir.resolve()}
train: {train_rel}
val: {val_rel}
nc: 1
names:
  0: walnut
""",
        encoding="utf-8",
    )
    print(f"Wrote {data_yaml}")

    imgsz = args.imgsz
    imgsz_file = data_dir / "imgsz.txt"
    if imgsz_file.exists():
        imgsz = int(imgsz_file.read_text().strip())
        print(f"Using imgsz={imgsz} from {imgsz_file} (patch dataset)")
    else:
        print(f"Using imgsz={imgsz}")

    try:
        from ultralytics import YOLO
    except ImportError:
        print("Install ultralytics: pip install ultralytics")
        sys.exit(1)

    model = YOLO("yolov8n.pt")
    train_kw = dict(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=imgsz,
        batch=batch,
        project=args.project,
        name=args.name,
    )
    device = resolve_device(args.device)
    print(f"Using device: {device}")
    train_kw["device"] = ultralytics_device(device)
    model.train(**train_kw)
    print(f"Training done. Best weights: {Path(args.project) / args.name / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
