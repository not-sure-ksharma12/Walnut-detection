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
    --model_path optuna_results/w0_option_b.pth \\
    --image_dir cropped_images \\
    --annotation_dir annotated_images \\
    --split_file split.json \\
    --split test --device auto
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
            window_size=window_size,
        )

    total_tp, total_fp, total_fn = 0, 0, 0
    count_errors = []
    per_image = []

    for stem in valid_stems:
        img_path = str(_find_image(image_dir_p, stem))
        ann_path = str(annotation_dir_p / f"{stem}.txt")

        gt = parse_annotations(ann_path)
        results = detector.process_image(img_path, output_dir=None, cluster=cluster)
        detections = results["centers"]

        tp, fp, fn = match_detections(detections, gt, match_radius)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        count_errors.append(abs(len(detections) - len(gt)))

        per_image.append({
            "stem": stem,
            "gt_count": len(gt),
            "det_count": len(detections),
            "tp": tp, "fp": fp, "fn": fn,
        })

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
    args = parser.parse_args()
    args.device = resolve_device(args.device)

    print("🥜 Walnut Detection Evaluation")
    print(f"Device: {args.device}")
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
    )

    if metrics:
        print(f"\n📊 Summary ({args.split} split):")
        print(f"   Total GT: {metrics['total_gt']}  Total Det: {metrics['total_det']}")
        print(f"   TP: {metrics['tp']}  FP: {metrics['fp']}  FN: {metrics['fn']}")
        print(f"   Precision: {metrics['precision']:.3f}")
        print(f"   Recall:    {metrics['recall']:.3f}")
        print(f"   F1:        {metrics['f1']:.3f}")
        print(f"   MAE:       {metrics['mae']:.2f}")


if __name__ == "__main__":
    main()
