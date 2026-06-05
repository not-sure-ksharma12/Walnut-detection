#!/usr/bin/env python3
"""
Two-stage walnut detection: YOLOv8 tiled proposals + patch classifier filter.

Stage 1 (YOLO):
  - Slide a window (tile_size, stride) over each full image.
  - Run YOLOv8 on each tile.
  - Map detections back to full-image coordinates.
  - Run global NMS over all proposals for that image.

Stage 2 (patch classifier):
  - For each YOLO proposal box, crop a small patch around its center.
  - Run the binary walnut classifier (same as WalnutDetector uses).
  - Keep only proposals where classifier probability >= clf_conf.

Evaluation:
  - Convert kept boxes to centres.
  - Match centres to ground-truth walnut centres using match_detections
    (Hungarian + radius).
  - Report per-image GT/Det/TP/FP/FN and global precision/recall/F1/MAE.

Usage example:
  cd Walnut-detection
  source venv/bin/activate
  python "evaluate_yolo_two_stage copy.py" \\
    --yolo_model_path yolo_runs/walnut_synthetic/weights/best.pt \\
    --clf_model_path models/walnut_classifier_phase1.pth \\
    --image_dir output \\
    --annotation_dir output/annotations \\
    --split_file output/dataset/split.json \\
    --split test \\
    --device auto
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

from device_utils import add_device_argument, resolve_device, ultralytics_device
from evaluate_walnut_annotations import (
    parse_annotations,
    match_detections,
    _find_image,
)
from walnut_detector import WalnutDetector


WORKSPACE = Path(__file__).resolve().parent
DEFAULT_TILE_SIZE = 640
DEFAULT_STRIDE = 480
DEFAULT_MATCH_RADIUS = 48.0
DEFAULT_YOLO_MODEL = WORKSPACE / "yolo_runs" / "walnut_synthetic" / "weights" / "best.pt"


def resolve_model_path(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else WORKSPACE / p


def find_latest_yolo_best() -> Path | None:
    """Most recently modified best.pt under yolo_runs/*/weights/."""
    candidates = sorted(
        WORKSPACE.glob("yolo_runs/*/weights/best.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def resolve_yolo_model_path(path: str | None) -> Path:
    if path:
        resolved = resolve_model_path(path)
        if resolved.is_file():
            return resolved
        fallback = find_latest_yolo_best()
        if fallback is not None:
            print(f"⚠️  YOLO model not found: {resolved}")
            print(f"   Using latest: {fallback}")
            return fallback
        raise SystemExit(
            f"YOLO model not found: {resolved}\n"
            f"Train first: python train_yolov8_synthetic.py --data_dir yolo_walnut_tiled\n"
            f"Or pass --yolo_model_path yolo_runs/walnut_synthetic/weights/best.pt"
        )
    fallback = find_latest_yolo_best()
    if fallback is not None:
        print(f"Using YOLO model: {fallback}")
        return fallback
    raise SystemExit(
        "No YOLO weights found under yolo_runs/*/weights/best.pt. "
        "Run train_yolov8_synthetic.py first."
    )


def load_binary_sweep_params(results_path: Path, metric: str = "mae") -> Dict[str, float]:
    """Load best patch_size / stride / threshold from parameter_sweep_binary.py output."""
    if not results_path.is_file():
        raise SystemExit(f"Sweep results not found: {results_path}")
    with open(results_path) as f:
        data = json.load(f)
    key = "best_by_mae" if metric == "mae" else "best_by_f1"
    best = data.get(key)
    if not best:
        raise SystemExit(f"Missing '{key}' in {results_path}")
    return best


def apply_sweep_to_args(args, sweep: Dict[str, float]) -> None:
    """Map binary sweep fields onto two-stage classifier / eval settings."""
    args.clf_conf = float(sweep["threshold"])
    args.clf_patch_size = int(sweep["patch_size"])
    args.clf_window_size = int(sweep["patch_size"])
    args.match_radius = DEFAULT_MATCH_RADIUS
    print(
        f"Binary sweep ({args.sweep_metric}): "
        f"patch_size={args.clf_patch_size} stride={sweep['stride']} "
        f"threshold={args.clf_conf} match_radius={args.match_radius}"
    )
    print(
        f"  (sweep stride={sweep['stride']} is for sliding-window detector; "
        f"YOLO tiling still uses --tile_size/--stride)"
    )


def tile_positions(img_dim: int, tile_size: int, stride: int) -> List[int]:
    """Return top-left positions for tiles along one dimension."""
    positions: List[int] = []
    pos = 0
    while pos + tile_size <= img_dim:
        positions.append(pos)
        pos += stride
    if not positions or positions[-1] + tile_size < img_dim:
        positions.append(max(0, img_dim - tile_size))
    return positions


def run_tiled_yolo(
    model,
    img_bgr: np.ndarray,
    tile_size: int,
    stride: int,
    conf: float,
    iou: float,
    device: str,
    imgsz: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run YOLO on tiled image, return full-image boxes (xyxy) and scores."""
    from PIL import Image

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_h, img_w = img_rgb.shape[:2]

    xs = tile_positions(img_w, tile_size, stride)
    ys = tile_positions(img_h, tile_size, stride)

    all_boxes: List[List[float]] = []
    all_scores: List[float] = []

    for ty in ys:
        for tx in xs:
            tile = img_rgb[ty : ty + tile_size, tx : tx + tile_size]
            pil_tile = Image.fromarray(tile)
            results = model.predict(
                source=pil_tile,
                imgsz=imgsz,
                conf=conf,
                iou=iou,
                device=device,
                verbose=False,
            )
            if not results:
                continue
            r = results[0]
            if r.boxes is None or len(r.boxes) == 0:
                continue
            xyxy = r.boxes.xyxy.cpu().numpy()
            scores = r.boxes.conf.cpu().numpy()
            for (x1, y1, x2, y2), s in zip(xyxy, scores):
                # Map to full-image coords
                all_boxes.append([x1 + tx, y1 + ty, x2 + tx, y2 + ty])
                all_scores.append(float(s))

    if not all_boxes:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    boxes_arr = np.array(all_boxes, dtype=np.float32)
    scores_arr = np.array(all_scores, dtype=np.float32)
    return boxes_arr, scores_arr


def nms_xyxy(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float,
) -> np.ndarray:
    """Simple NMS over xyxy boxes. Returns indices of boxes to keep."""
    if boxes.shape[0] == 0:
        return np.zeros((0,), dtype=np.int64)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep: List[int] = []

    while order.size > 0:
        i = int(order[0])
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)

        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return np.array(keep, dtype=np.int64)


class WalnutPatchClassifier:
    """Thin wrapper around WalnutDetector's classifier for single-patch scoring."""

    def __init__(self, model_path: str, patch_size: int = 32, device: str = "auto"):
        # Reuse WalnutDetector's loading and transform
        self.detector = WalnutDetector(
            model_path=model_path,
            patch_size=patch_size,
            stride=patch_size,
            confidence_threshold=0.5,
            device=device,
            cluster_eps=24.0,
            local_max_size=5,
            nms_radius=0.0,
        )
        self.model = self.detector.model
        self.model.eval()
        self.transform = self.detector.transform
        self.device = self.detector.device
        self.patch_size = patch_size

    def classify(
        self,
        img_bgr: np.ndarray,
        cx: int,
        cy: int,
        window_size: int = 32,
    ) -> float:
        """Return walnut probability for a patch centered at (cx, cy)."""
        import torch

        h, w = img_bgr.shape[:2]
        half = window_size // 2
        x1 = max(0, cx - half)
        y1 = max(0, cy - half)
        x2 = min(w, cx + half)
        y2 = min(h, cy + half)
        if x2 <= x1 or y2 <= y1:
            return 0.0

        patch_bgr = img_bgr[y1:y2, x1:x2]
        patch_rgb = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2RGB)

        tensor = self.transform(patch_rgb).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(tensor)
            probs = torch.softmax(outputs, dim=1)
            p_walnut = probs[0, 1].item()
        return float(p_walnut)


def evaluate_two_stage(
    yolo_model_path: str,
    clf_model_path: str,
    image_dir: str,
    annotation_dir: str,
    split_file: str,
    split: str = "test",
    tile_size: int = DEFAULT_TILE_SIZE,
    stride: int = DEFAULT_STRIDE,
    imgsz: int = 640,
    yolo_conf: float = 0.20,
    yolo_iou: float = 0.40,
    clf_conf: float = 0.60,
    clf_patch_size: int = 32,
    clf_window_size: int = 32,
    nms_iou: float = 0.40,
    match_radius: float = DEFAULT_MATCH_RADIUS,
    device: str = "auto",
    verbose: bool = True,
) -> Dict:
    """Evaluate two-stage YOLO + classifier on a split of full images."""
    from ultralytics import YOLO

    device = resolve_device(device)
    yolo_device = ultralytics_device(device)
    image_dir_p = Path(image_dir)
    annotation_dir_p = Path(annotation_dir)
    split_file_p = Path(split_file)

    if not split_file_p.exists():
        raise FileNotFoundError(f"split_file not found: {split_file_p}")

    with open(split_file_p) as f:
        splits = json.load(f)
    stems = splits.get(split)
    if stems is None:
        raise KeyError(f"Split '{split}' not found in {split_file}")

    # Filter to stems that actually have both an image and annotation
    valid_stems: List[str] = []
    for stem in stems:
        img_path = _find_image(image_dir_p, stem)
        ann_path = annotation_dir_p / f"{stem}.txt"
        if img_path is not None and ann_path.exists():
            valid_stems.append(stem)

    if not valid_stems:
        raise RuntimeError(f"No valid images found for split '{split}'")

    if verbose:
        print(f"Evaluating {len(valid_stems)} images in split '{split}'")

    yolo_model = YOLO(yolo_model_path)
    clf = WalnutPatchClassifier(
        model_path=clf_model_path,
        patch_size=clf_patch_size,
        device=device,
    )

    total_tp, total_fp, total_fn = 0, 0, 0
    count_errors: List[int] = []
    per_image: List[Dict] = []

    for stem in valid_stems:
        img_path = _find_image(image_dir_p, stem)
        assert img_path is not None
        ann_path = annotation_dir_p / f"{stem}.txt"

        gt = parse_annotations(str(ann_path))

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            raise ValueError(f"Could not load image: {img_path}")

        # Stage 1: YOLO tiled proposals
        boxes, scores = run_tiled_yolo(
            model=yolo_model,
            img_bgr=img_bgr,
            tile_size=tile_size,
            stride=stride,
            conf=yolo_conf,
            iou=yolo_iou,
            device=yolo_device,
            imgsz=imgsz,
        )

        if boxes.shape[0] == 0:
            centers: List[Tuple[int, int]] = []
        else:
            keep = nms_xyxy(boxes, scores, nms_iou)
            boxes_nms = boxes[keep]

            # Stage 2: classifier filter
            centers: List[Tuple[int, int]] = []
            for (x1, y1, x2, y2) in boxes_nms:
                cx = int(round((x1 + x2) / 2.0))
                cy = int(round((y1 + y2) / 2.0))
                p_walnut = clf.classify(img_bgr, cx, cy, window_size=clf_window_size)
                if p_walnut >= clf_conf:
                    centers.append((cx, cy))

        tp, fp, fn = match_detections(centers, gt, match_radius)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        count_errors.append(abs(len(centers) - len(gt)))

        per_image.append(
            {
                "stem": stem,
                "gt_count": len(gt),
                "det_count": len(centers),
                "tp": tp,
                "fp": fp,
                "fn": fn,
            }
        )

        if verbose:
            print(
                f"  {stem}: GT={len(gt)} Det={len(centers)} "
                f"TP={tp} FP={fp} FN={fn}"
            )

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    mae = float(np.mean(count_errors)) if count_errors else 0.0

    metrics = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mae": mae,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "total_gt": total_tp + total_fn,
        "total_det": total_tp + total_fp,
        "per_image": per_image,
    }

    if verbose:
        print(
            f"\nPrecision: {precision:.3f}  Recall: {recall:.3f}  "
            f"F1: {f1:.3f}  MAE: {mae:.2f}"
        )

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Two-stage YOLOv8 + patch classifier walnut evaluation"
    )
    parser.add_argument(
        "--yolo_model_path",
        type=str,
        default=None,
        help="Path to YOLO .pt (default: latest yolo_runs/*/weights/best.pt)",
    )
    parser.add_argument(
        "--clf_model_path",
        required=True,
        help="Path to patch classifier .pth model (used by WalnutDetector)",
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default=str(WORKSPACE / "output"),
        help="Directory with full images",
    )
    parser.add_argument(
        "--annotation_dir",
        type=str,
        default=str(WORKSPACE / "output" / "annotations"),
        help="Directory with annotation .txt files",
    )
    parser.add_argument(
        "--split_file",
        type=str,
        default=str(WORKSPACE / "output" / "dataset" / "split.json"),
        help="Path to split.json",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
    )
    parser.add_argument(
        "--tile_size",
        type=int,
        default=DEFAULT_TILE_SIZE,
        help="Tile size in pixels (should match training, typically 640)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=DEFAULT_STRIDE,
        help="Tile stride in pixels (should match training, typically 480)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="YOLO inference image size",
    )
    parser.add_argument(
        "--yolo_conf",
        type=float,
        default=0.20,
        help="YOLO confidence threshold",
    )
    parser.add_argument(
        "--yolo_iou",
        type=float,
        default=0.40,
        help="YOLO NMS IoU threshold (per-tile)",
    )
    parser.add_argument(
        "--clf_conf",
        type=float,
        default=0.60,
        help="Classifier walnut probability threshold",
    )
    parser.add_argument(
        "--clf_window_size",
        type=int,
        default=32,
        help="Patch extraction window size in pixels around YOLO center (e.g. 32 or 48). "
             "Window is resized to 32x32 for the classifier.",
    )
    parser.add_argument(
        "--nms_iou",
        type=float,
        default=0.40,
        help="Global NMS IoU threshold over YOLO proposals",
    )
    parser.add_argument(
        "--match_radius",
        type=float,
        default=DEFAULT_MATCH_RADIUS,
        help="Matching radius in pixels for detection <-> GT centres",
    )
    parser.add_argument(
        "--sweep_results",
        type=str,
        default=None,
        help="binary_parameter_sweep_results.json — sets clf_conf, patch/window size, match_radius",
    )
    parser.add_argument(
        "--sweep_metric",
        type=str,
        default="mae",
        choices=["mae", "f1"],
        help="Use best_by_mae or best_by_f1 from sweep results (default: mae)",
    )
    parser.add_argument(
        "--clf_patch_size",
        type=int,
        default=32,
        help="Classifier input patch size (overridden by --sweep_results)",
    )
    add_device_argument(parser, default="auto")
    args = parser.parse_args()
    args.device = resolve_device(args.device)

    if args.sweep_results:
        sweep = load_binary_sweep_params(
            resolve_model_path(args.sweep_results), args.sweep_metric
        )
        apply_sweep_to_args(args, sweep)

    yolo_model_path = resolve_yolo_model_path(args.yolo_model_path)

    metrics = evaluate_two_stage(
        yolo_model_path=str(yolo_model_path),
        clf_model_path=args.clf_model_path,
        image_dir=args.image_dir,
        annotation_dir=args.annotation_dir,
        split_file=args.split_file,
        split=args.split,
        tile_size=args.tile_size,
        stride=args.stride,
        imgsz=args.imgsz,
        yolo_conf=args.yolo_conf,
        yolo_iou=args.yolo_iou,
        clf_conf=args.clf_conf,
        clf_patch_size=args.clf_patch_size,
        clf_window_size=args.clf_window_size,
        nms_iou=args.nms_iou,
        match_radius=args.match_radius,
        device=args.device,
        verbose=True,
    )

    print(
        f"\nSummary ({args.split} split): "
        f"GT={metrics['total_gt']} Det={metrics['total_det']} "
        f"TP={metrics['tp']} FP={metrics['fp']} FN={metrics['fn']}"
    )


if __name__ == "__main__":
    main()

