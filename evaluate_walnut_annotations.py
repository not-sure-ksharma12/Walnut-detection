#!/usr/bin/env python3
"""
evaluate_walnut_annotations.py — Score patch-classifier detections against GT annotations.

What it does:
  - Runs WalnutDetector on images from a split (train/val/test)
  - Matches predicted centres to GT with Hungarian assignment within a radius
  - Reports precision, recall, F1, and MAE (count error)
  - Exports evaluate_dataset(), parse_annotations(), match_detections() for other scripts

Prerequisites:
  - python3 setup.py
  - Classifier .pth, cropped_images/, annotated_images/, split.json

How to run:
  cd Walnut-detection
  source venv/bin/activate
  python evaluate_walnut_annotations.py \\
    --model_path models/glenn/walnut_classifier_phase1.pth \\
    --image_dir InputData/GlennDormancyRootstock/cropped_images \\
    --annotation_dir InputData/GlennDormancyRootstock/annotated_images \\
    --split_file InputData/GlennDormancyRootstock/dataset/split.json \\
    --split test \\
    --sweep_results InputData/GlennDormancyRootstock/binary_parameter_sweep_results_Glenn.json \\
    --dataset_name Glenn \\
    --device auto

  Opens red-box overlays from detection_visualizations_Glenn/overlays/
  Coordinates in detection_visualizations_Glenn/coordinates/*.txt
"""

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

WORKSPACE = Path(__file__).resolve().parent

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from walnut_detector import WalnutDetector


# ---------------------------------------------------------------------------
# Annotation helpers
# ---------------------------------------------------------------------------

def parse_annotations(annotation_path: str) -> List[Tuple[int, int]]:
    """Parse annotation file and return list of (x, y) coordinates."""
    walnuts = []
    with open(annotation_path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    walnuts.append((int(parts[0]), int(parts[1])))
                except ValueError:
                    continue
    return walnuts


# ---------------------------------------------------------------------------
# Matching detections <-> ground truth
# ---------------------------------------------------------------------------

def match_detections(
    detections: List[Tuple[int, int]],
    ground_truth: List[Tuple[int, int]],
    match_radius: float = 48.0,
) -> Tuple[int, int, int]:
    """Match detections to ground truth using Hungarian algorithm.

    Returns (true_positives, false_positives, false_negatives).
    """
    if len(detections) == 0 and len(ground_truth) == 0:
        return 0, 0, 0
    if len(detections) == 0:
        return 0, 0, len(ground_truth)
    if len(ground_truth) == 0:
        return 0, len(detections), 0

    det_arr = np.array(detections, dtype=np.float64)
    gt_arr = np.array(ground_truth, dtype=np.float64)

    # Cost matrix: Euclidean distance between every (det, gt) pair
    diff = det_arr[:, None, :] - gt_arr[None, :, :]  # (D, G, 2)
    cost = np.sqrt((diff ** 2).sum(axis=2))            # (D, G)

    row_idx, col_idx = linear_sum_assignment(cost)

    tp = int(np.sum(cost[row_idx, col_idx] <= match_radius))
    fp = len(detections) - tp
    fn = len(ground_truth) - tp
    return tp, fp, fn


# ---------------------------------------------------------------------------
# Detection visualization (red boxes + coordinate files)
# ---------------------------------------------------------------------------

def center_to_box(x: int, y: int, box_size: int) -> Tuple[int, int, int, int]:
    """Convert center (x, y) to axis-aligned box x1,y1,x2,y2 (inclusive)."""
    half = box_size // 2
    return x - half, y - half, x + half, y + half


def save_detection_visualization(
    image_path: str,
    centers: List[Tuple[int, int]],
    confidences: List[float],
    output_dir: Path,
    stem: str,
    box_size: int = 48,
) -> Dict[str, str]:
    """Save red-box overlay JPG and a coordinates .txt for one image."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")

    h, w = image.shape[:2]
    overlay = image.copy()

    coords_dir = output_dir / "coordinates"
    overlays_dir = output_dir / "overlays"
    coords_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir.mkdir(parents=True, exist_ok=True)

    txt_path = coords_dir / f"{stem}.txt"
    lines = [
        "# Model walnut detections (binary classifier sliding window)",
        f"# Image: {stem}",
        f"# Box size: {box_size} px (square centered on each detection)",
        "# Format: x_center y_center confidence x1 y1 x2 y2",
    ]

    red = (0, 0, 255)  # BGR
    for (x, y), conf in zip(centers, confidences):
        x1, y1, x2, y2 = center_to_box(x, y, box_size)
        x1c = max(0, min(x1, w - 1))
        y1c = max(0, min(y1, h - 1))
        x2c = max(0, min(x2, w - 1))
        y2c = max(0, min(y2, h - 1))
        cv2.rectangle(overlay, (x1c, y1c), (x2c, y2c), red, 2)
        label = f"{conf:.2f}"
        cv2.putText(
            overlay, label, (x1c, max(12, y1c - 4)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, red, 1, cv2.LINE_AA,
        )
        lines.append(f"{x} {y} {conf:.4f} {x1c} {y1c} {x2c} {y2c}")

    cv2.putText(
        overlay, f"Detections: {len(centers)}", (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX, 0.9, red, 2, cv2.LINE_AA,
    )

    overlay_path = overlays_dir / f"{stem}_detections.jpg"
    cv2.imwrite(str(overlay_path), overlay)
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {
        "overlay": str(overlay_path),
        "coordinates": str(txt_path),
    }


def load_sweep_detector_params(results_path: str, metric: str = "mae") -> Dict[str, Any]:
    """Load best patch_size / stride / threshold from parameter_sweep_binary.py output."""
    path = Path(results_path)
    if not path.is_file():
        raise FileNotFoundError(f"Sweep results not found: {path}")
    with path.open() as f:
        data = json.load(f)
    key = "best_by_mae" if metric == "mae" else "best_by_f1"
    best = data.get(key)
    if not best:
        raise KeyError(f"Missing '{key}' in {path}")
    return best


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def evaluate_dataset(
    model_path: str,
    image_dir: str,
    annotation_dir: str,
    split_file: str,
    split: str = "test",
    patch_size: int = 32,
    stride: int = 16,
    threshold: float = 0.5,
    cluster: bool = True,
    cluster_eps: float = 24.0,
    local_max_size: int = 5,
    nms_radius: float = 12.0,
    match_radius: float = 48.0,
    device: str = "auto",
    verbose: bool = True,
    detector: "WalnutDetector | None" = None,
    window_size: int = None,
    visualize_dir: str | None = None,
    box_size: int = 48,
) -> Optional[Dict]:
    """Run detector on a split and return aggregated metrics dict.

    If *detector* is provided it is reused (reconfigured with stride /
    threshold / cluster_eps / local_max_size / nms_radius).  Otherwise a
    new WalnutDetector is created from *model_path*.

    Returns dict with keys: precision, recall, f1, mae, tp, fp, fn,
    total_gt, total_det, per_image.  Returns None on failure.
    """
    # Load split
    with open(split_file) as f:
        splits = json.load(f)
    stems = splits.get(split)
    if stems is None:
        print(f"Split '{split}' not found in {split_file}")
        return None

    # Filter to stems that actually have both an image and annotation
    image_dir_p = Path(image_dir)
    annotation_dir_p = Path(annotation_dir)
    valid_stems = []
    for stem in stems:
        img_path = _find_image(image_dir_p, stem)
        ann_path = annotation_dir_p / f"{stem}.txt"
        if img_path is not None and ann_path.exists():
            valid_stems.append(stem)

    if not valid_stems:
        if verbose:
            print(f"No valid images found for split '{split}'")
        return None

    # Build or reconfigure detector
    if detector is not None:
        detector.reconfigure(
            patch_size=patch_size,
            stride=stride,
            confidence_threshold=threshold,
            cluster_eps=cluster_eps,
            local_max_size=local_max_size,
            nms_radius=nms_radius,
        )
    else:
        detector = WalnutDetector(
            model_path=model_path,
            patch_size=patch_size,
            stride=stride,
            confidence_threshold=threshold,
            device=device,
            cluster_eps=cluster_eps,
            local_max_size=local_max_size,
            nms_radius=nms_radius,
        )

    total_tp, total_fp, total_fn = 0, 0, 0
    count_errors = []
    per_image = []
    vis_root = Path(visualize_dir) if visualize_dir else None
    if vis_root:
        vis_root.mkdir(parents=True, exist_ok=True)

    for stem in valid_stems:
        img_path = str(_find_image(image_dir_p, stem))
        ann_path = str(annotation_dir_p / f"{stem}.txt")

        gt = parse_annotations(ann_path)
        results = detector.process_image(img_path, output_dir=None, cluster=cluster)
        detections = results["centers"]
        confidences = results["confidences"]

        tp, fp, fn = match_detections(detections, gt, match_radius)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        count_errors.append(abs(len(detections) - len(gt)))

        row: Dict[str, Any] = {
            "stem": stem,
            "gt_count": len(gt),
            "det_count": len(detections),
            "tp": tp, "fp": fp, "fn": fn,
        }

        if vis_root:
            paths = save_detection_visualization(
                img_path, detections, confidences, vis_root, stem, box_size=box_size,
            )
            row["overlay_path"] = paths["overlay"]
            row["coordinates_path"] = paths["coordinates"]
            if verbose:
                print(f"  💾 {stem}: overlay + coordinates saved")

        per_image.append(row)

        if verbose:
            print(
                f"  {stem}: GT={len(gt)} Det={len(detections)} "
                f"TP={tp} FP={fp} FN={fn}"
            )

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
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
        print(f"\n  Precision: {precision:.3f}  Recall: {recall:.3f}  "
              f"F1: {f1:.3f}  MAE: {mae:.2f}")

    return metrics


def _find_image(image_dir: Path, stem: str) -> Optional[Path]:
    """Find image file for a given stem, trying common extensions."""
    for ext in (".JPG", ".jpg", ".png", ".PNG", ".jpeg"):
        p = image_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def _infer_dataset_name(image_dir: str) -> str:
    """Derive a dataset label from the image directory path."""
    p = Path(image_dir)
    if p.name.lower() in ("cropped_images", "cropped", "images", "output"):
        return p.parent.name
    return p.name


def _default_output_path(dataset_name: str, split: str) -> Path:
    """evaluation_results_{dataset}_{split}.json in repo root."""
    safe_name = dataset_name.replace(" ", "_")
    if split == "test":
        return WORKSPACE / f"evaluation_results_{safe_name}.json"
    return WORKSPACE / f"evaluation_results_{safe_name}_{split}.json"


def _format_summary_text(report: Dict[str, Any]) -> str:
    """Human-readable summary matching the CLI printout."""
    lines = [
        "Walnut Detection Evaluation",
        "=" * 50,
        f"Dataset:    {report['dataset_name']}",
        f"Split:      {report['split']}",
        f"Model:      {report['model_path']}",
        f"Device:     {report['device']}",
        f"Timestamp:  {report['timestamp']}",
        "",
        "Detector settings:",
        f"  patch_size={report['detector']['patch_size']}  "
        f"stride={report['detector']['stride']}  "
        f"threshold={report['detector']['threshold']}",
        f"  cluster={report['detector']['cluster']}  "
        f"cluster_eps={report['detector']['cluster_eps']}  "
        f"nms_radius={report['detector']['nms_radius']}",
        f"  match_radius={report['detector']['match_radius']}",
        "",
    ]
    for row in report["metrics"]["per_image"]:
        lines.append(
            f"  {row['stem']}: GT={row['gt_count']} Det={row['det_count']} "
            f"TP={row['tp']} FP={row['fp']} FN={row['fn']}"
        )
    m = report["metrics"]
    lines.extend(
        [
            "",
            f"Summary ({report['split']} split):",
            f"   Total GT: {m['total_gt']}  Total Det: {m['total_det']}",
            f"   TP: {m['tp']}  FP: {m['fp']}  FN: {m['fn']}",
            f"   Precision: {m['precision']:.3f}",
            f"   Recall:    {m['recall']:.3f}",
            f"   F1:        {m['f1']:.3f}",
            f"   MAE:       {m['mae']:.2f}",
        ]
    )
    return "\n".join(lines) + "\n"


def save_evaluation_report(
    report: Dict[str, Any],
    output_file: str | Path,
) -> Path:
    """Write JSON report and a companion .txt summary."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    txt_path = output_path.with_suffix(".txt")
    txt_path.write_text(_format_summary_text(report), encoding="utf-8")
    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate walnut detector against ground-truth annotations"
    )
    parser.add_argument("--model_path", required=True, help="Path to .pth model")
    parser.add_argument("--image_dir", required=True, help="Cropped images directory")
    parser.add_argument("--annotation_dir", required=True, help="Annotation .txt directory")
    parser.add_argument("--split_file", required=True, help="Path to split.json")
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "val", "test"],
        help="Which split from split.json to evaluate",
    )
    parser.add_argument("--patch_size", type=int, default=32, help="Sliding-window patch size in pixels")
    parser.add_argument("--stride", type=int, default=16, help="Sliding-window stride in pixels")
    parser.add_argument("--threshold", type=float, default=0.5, help="Classifier confidence threshold")
    parser.add_argument(
        "--cluster",
        action="store_true",
        default=True,
        help="Merge nearby detections with DBSCAN (default: on)",
    )
    parser.add_argument(
        "--no_cluster",
        dest="cluster",
        action="store_false",
        help="Disable DBSCAN clustering of detections",
    )
    parser.add_argument(
        "--cluster_eps",
        type=float,
        default=24.0,
        help="DBSCAN eps in pixels for merging nearby detections",
    )
    parser.add_argument(
        "--local_max_size",
        type=int,
        default=5,
        help="Local-maxima window size (odd integer) for peak picking",
    )
    parser.add_argument(
        "--nms_radius",
        type=float,
        default=12.0,
        help="Non-maximum suppression radius in pixels (0 to disable)",
    )
    parser.add_argument(
        "--match_radius",
        type=float,
        default=48.0,
        help="Max distance in pixels to match a detection to ground truth",
    )
    from device_utils import add_device_argument, resolve_device

    add_device_argument(parser, default="auto")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="Dataset label for output files (default: inferred from --image_dir)",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Path to save results JSON (default: evaluation_results_{dataset}.json)",
    )
    parser.add_argument(
        "--no_save",
        action="store_true",
        help="Do not write results to disk",
    )
    parser.add_argument(
        "--sweep_results",
        type=str,
        default=None,
        help="binary_parameter_sweep_*.json — auto-set patch_size, stride, threshold",
    )
    parser.add_argument(
        "--sweep_metric",
        type=str,
        default="mae",
        choices=["mae", "f1"],
        help="Use best_by_mae or best_by_f1 from sweep results (default: mae)",
    )
    parser.add_argument(
        "--visualize_dir",
        type=str,
        default=None,
        help="Save red-box overlays (overlays/) and coordinate files (coordinates/)",
    )
    parser.add_argument(
        "--box_size",
        type=int,
        default=48,
        help="Side length in pixels for red detection boxes (default: 48)",
    )
    args = parser.parse_args()
    args.device = resolve_device(args.device)
    dataset_name = args.dataset_name or _infer_dataset_name(args.image_dir)

    if args.sweep_results:
        sweep = load_sweep_detector_params(args.sweep_results, args.sweep_metric)
        args.patch_size = int(sweep["patch_size"])
        args.stride = int(sweep["stride"])
        args.threshold = float(sweep["threshold"])
        print(
            f"Sweep ({args.sweep_metric}): patch_size={args.patch_size} "
            f"stride={args.stride} threshold={args.threshold}"
        )

    visualize_dir = args.visualize_dir
    if visualize_dir is None and not args.no_save:
        visualize_dir = str(
            WORKSPACE / f"detection_visualizations_{dataset_name.replace(' ', '_')}"
        )

    print("🥜 Walnut Detection Evaluation")
    print(f"Dataset: {dataset_name}")
    print(f"Device: {args.device}")
    if visualize_dir:
        print(f"Visualizations: {visualize_dir}")
    print("=" * 50)

    metrics = evaluate_dataset(
        model_path=args.model_path,
        image_dir=args.image_dir,
        annotation_dir=args.annotation_dir,
        split_file=args.split_file,
        split=args.split,
        patch_size=args.patch_size,
        stride=args.stride,
        threshold=args.threshold,
        cluster=args.cluster,
        cluster_eps=args.cluster_eps,
        local_max_size=args.local_max_size,
        nms_radius=args.nms_radius,
        match_radius=args.match_radius,
        device=args.device,
        verbose=True,
        visualize_dir=visualize_dir,
        box_size=args.box_size,
    )

    if metrics:
        print(f"\n📊 Summary ({args.split} split):")
        print(f"   Total GT: {metrics['total_gt']}  Total Det: {metrics['total_det']}")
        print(f"   TP: {metrics['tp']}  FP: {metrics['fp']}  FN: {metrics['fn']}")
        print(f"   Precision: {metrics['precision']:.3f}")
        print(f"   Recall:    {metrics['recall']:.3f}")
        print(f"   F1:        {metrics['f1']:.3f}")
        print(f"   MAE:       {metrics['mae']:.2f}")

        if not args.no_save:
            output_file = args.output_file or str(_default_output_path(dataset_name, args.split))
            report = {
                "dataset_name": dataset_name,
                "split": args.split,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "model_path": args.model_path,
                "image_dir": args.image_dir,
                "annotation_dir": args.annotation_dir,
                "split_file": args.split_file,
                "device": args.device,
                "detector": {
                    "patch_size": args.patch_size,
                    "stride": args.stride,
                    "threshold": args.threshold,
                    "cluster": args.cluster,
                    "cluster_eps": args.cluster_eps,
                    "local_max_size": args.local_max_size,
                    "nms_radius": args.nms_radius,
                    "match_radius": args.match_radius,
                },
                "metrics": metrics,
                "visualize_dir": visualize_dir,
                "box_size": args.box_size,
            }
            saved_path = save_evaluation_report(report, output_file)
            print(f"\n💾 Saved {saved_path}")
            print(f"💾 Saved {saved_path.with_suffix('.txt')}")
            if visualize_dir:
                print(f"💾 Red-box overlays: {visualize_dir}/overlays/")
                print(f"💾 Coordinates:      {visualize_dir}/coordinates/")


if __name__ == "__main__":
    main()
