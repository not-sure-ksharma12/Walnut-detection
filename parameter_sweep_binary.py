#!/usr/bin/env python3
"""
parameter_sweep_binary.py — Grid search detector patch_size, stride, and threshold.

What it does:
  - Runs sliding-window evaluation (via evaluate_walnut_annotations.evaluate_dataset)
  - Tries many patch_size / stride / threshold combinations on a split
  - Ranks configs by count MAE and F1; saves binary_parameter_sweep_results.json

Prerequisites:
  - Trained classifier .pth (train binary_classifier.py first, or use a pretrained checkpoint)
  - Full images, annotation .txt files, and split.json (from build_annotations11_10_dataset.py)

How to run:
  cd Walnut-detection
  source venv/bin/activate
  python parameter_sweep_binary.py \\
    --model_path models/walnut_classifier_best_precision.pth \\
    --image_dir output \\
    --annotation_dir output/annotations \\
    --split_file output/dataset/split.json \\
    --split test \\
    --device auto
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from device_utils import add_device_argument, resolve_device
from evaluate_walnut_annotations import evaluate_dataset
from walnut_detector import WalnutDetector

WORKSPACE = Path(__file__).resolve().parent


def run_sweep(
    model_path: str,
    image_dir: str,
    annotation_dir: str,
    split_file: str,
    split: str,
    device: str,
    patch_sizes: List[int],
    strides: List[int],
    thresholds: List[float],
    cluster: bool,
    cluster_eps: float,
    local_max_size: int,
    nms_radius: float,
    match_radius: float,
    quick: bool,
) -> None:
    print("🧪 Binary classifier parameter sweep")
    print("=" * 50)
    print(f"Model:      {model_path}")
    print(f"Images:     {image_dir}")
    print(f"Annots:     {annotation_dir}")
    print(f"Split file: {split_file}  ({split})")
    print(f"Device:     {device}")
    print()

    model_path_p = Path(model_path)
    if not model_path_p.exists():
        raise SystemExit(f"Model not found: {model_path_p}\nTrain first: python binary_classifier.py --dataset_dir output/dataset ...")

    combos: List[tuple[int, int, float]] = []
    for patch_size in patch_sizes:
        for stride in strides:
            if stride >= patch_size:
                continue
            for threshold in thresholds:
                combos.append((patch_size, stride, threshold))

    if quick:
        combos = combos[: min(12, len(combos))]

    print(f"Configurations to try: {len(combos)}")
    print()

    detector: Optional[WalnutDetector] = None
    loaded_patch_size: Optional[int] = None
    results: List[Dict[str, Any]] = []

    for i, (patch_size, stride, threshold) in enumerate(combos, start=1):
        print(f"[{i}/{len(combos)}] patch_size={patch_size} stride={stride} threshold={threshold}")
        if detector is None or loaded_patch_size != patch_size:
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
            loaded_patch_size = patch_size
        metrics = evaluate_dataset(
            model_path=model_path,
            image_dir=image_dir,
            annotation_dir=annotation_dir,
            split_file=split_file,
            split=split,
            patch_size=patch_size,
            stride=stride,
            threshold=threshold,
            cluster=cluster,
            cluster_eps=cluster_eps,
            local_max_size=local_max_size,
            nms_radius=nms_radius,
            match_radius=match_radius,
            device=device,
            verbose=False,
            detector=detector,
        )

        if metrics is None:
            results.append(
                {
                    "patch_size": patch_size,
                    "stride": stride,
                    "threshold": threshold,
                    "success": False,
                }
            )
            print("  ❌ failed\n")
            continue

        entry = {
            "patch_size": patch_size,
            "stride": stride,
            "threshold": threshold,
            "success": True,
            "mae": metrics["mae"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "tp": metrics["tp"],
            "fp": metrics["fp"],
            "fn": metrics["fn"],
            "total_gt": metrics["total_gt"],
            "total_det": metrics["total_det"],
        }
        results.append(entry)
        print(
            f"  ✅ MAE={metrics['mae']:.3f}  P={metrics['precision']:.3f}  "
            f"R={metrics['recall']:.3f}  F1={metrics['f1']:.3f}\n"
        )

    successful = [r for r in results if r.get("success")]
    if not successful:
        raise SystemExit("No successful configurations.")

    by_mae = sorted(successful, key=lambda x: (x["mae"], -x["f1"]))
    by_f1 = sorted(successful, key=lambda x: (-x["f1"], x["mae"]))

    print("=" * 50)
    print("TOP 10 BY COUNT MAE (lower is better)")
    print("=" * 50)
    for i, r in enumerate(by_mae[:10], start=1):
        print(
            f"{i:2d}. patch={r['patch_size']:2d} stride={r['stride']:2d} "
            f"thr={r['threshold']:.2f}  MAE={r['mae']:.3f}  F1={r['f1']:.3f}"
        )

    best_mae = by_mae[0]
    best_f1 = by_f1[0]
    print()
    print("🏆 Best by MAE:")
    print(f"   patch_size={best_mae['patch_size']} stride={best_mae['stride']} threshold={best_mae['threshold']}")
    print(f"   MAE={best_mae['mae']:.3f}  F1={best_mae['f1']:.3f}")
    print()
    print("🏆 Best by F1:")
    print(f"   patch_size={best_f1['patch_size']} stride={best_f1['stride']} threshold={best_f1['threshold']}")
    print(f"   MAE={best_f1['mae']:.3f}  F1={best_f1['f1']:.3f}")

    out_path = WORKSPACE / "binary_parameter_sweep_results.json"
    with out_path.open("w") as f:
        json.dump(
            {
                "all_results": results,
                "successful_results": successful,
                "best_by_mae": best_mae,
                "best_by_f1": best_f1,
            },
            f,
            indent=2,
        )
    print(f"\n💾 Saved {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep binary classifier detection hyperparameters")
    parser.add_argument(
        "--model_path",
        type=str,
        default=str(WORKSPACE / "models" / "walnut_classifier_best_precision.pth"),
        help="Path to trained .pth checkpoint",
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default=str(WORKSPACE / "output"),
        help="Directory of full images (e.g. output/)",
    )
    parser.add_argument(
        "--annotation_dir",
        type=str,
        default=str(WORKSPACE / "output" / "annotations"),
        help="Directory of annotation .txt files",
    )
    parser.add_argument(
        "--split_file",
        type=str,
        default=str(WORKSPACE / "output" / "dataset" / "split.json"),
        help="split.json from build_annotations11_10_dataset.py",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Which split to evaluate (default: test)",
    )
    parser.add_argument(
        "--patch_sizes",
        type=int,
        nargs="+",
        default=[16, 24, 32],
        help="Patch sizes to try",
    )
    parser.add_argument(
        "--strides",
        type=int,
        nargs="+",
        default=[8, 12, 16],
        help="Strides to try (skipped when stride >= patch_size)",
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.45, 0.5, 0.55, 0.6, 0.65],
        help="Classifier confidence thresholds to try",
    )
    parser.add_argument(
        "--cluster",
        action="store_true",
        default=True,
        help="Use DBSCAN clustering on detections (default: on)",
    )
    parser.add_argument(
        "--no_cluster",
        dest="cluster",
        action="store_false",
        help="Disable clustering",
    )
    parser.add_argument("--cluster_eps", type=float, default=28.0, help="DBSCAN eps (pixels)")
    parser.add_argument("--local_max_size", type=int, default=7, help="Local max window size (odd)")
    parser.add_argument("--nms_radius", type=float, default=14.0, help="NMS radius in pixels")
    parser.add_argument(
        "--match_radius",
        type=float,
        default=48.0,
        help="Match radius for precision/recall (pixels)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run only the first 12 combinations (faster)",
    )
    add_device_argument(parser, default="auto")
    args = parser.parse_args()
    args.device = resolve_device(args.device)

    run_sweep(
        model_path=args.model_path,
        image_dir=args.image_dir,
        annotation_dir=args.annotation_dir,
        split_file=args.split_file,
        split=args.split,
        device=args.device,
        patch_sizes=args.patch_sizes,
        strides=args.strides,
        thresholds=args.thresholds,
        cluster=args.cluster,
        cluster_eps=args.cluster_eps,
        local_max_size=args.local_max_size,
        nms_radius=args.nms_radius,
        match_radius=args.match_radius,
        quick=args.quick,
    )


if __name__ == "__main__":
    main()
