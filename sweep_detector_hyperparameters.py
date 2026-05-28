#!/usr/bin/env python3
"""
sweep_detector_hyperparameters.py — Grid search over detector stride/threshold/NMS settings.

What it does:
  - Runs evaluate_dataset() over a grid of patch_size, stride, threshold, cluster_eps, etc.
  - Prints the best precision, recall, and F1 on the chosen split
  - Suggests a command line for evaluate_walnut_annotations.py with winning settings

Prerequisites:
  - python3 setup.py
  - Same inputs as evaluate_walnut_annotations.py (model, images, annotations, split.json)

How to run:
  cd Walnut-detection
  source venv/bin/activate
  python sweep_detector_hyperparameters.py \\
    --model_path models_finetuned/walnut_classifier_best_precision.pth \\
    --image_dir cropped_images \\
    --annotation_dir annotated_images \\
    --split_file split.json \\
    --split test --device auto
  python sweep_detector_hyperparameters.py ... --quick
"""

import argparse
from itertools import product

from evaluate_walnut_annotations import evaluate_dataset


def main():
    parser = argparse.ArgumentParser(description="Sweep detector hyperparameters on test set")
    parser.add_argument("--model_path", required=True, help="Path to .pth model")
    parser.add_argument("--image_dir", required=True, help="Cropped images dir")
    parser.add_argument("--annotation_dir", required=True, help="Annotation .txt dir")
    parser.add_argument("--split_file", required=True, help="Path to split.json")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"], help="Split to evaluate")
    from device_utils import add_device_argument, resolve_device

    add_device_argument(parser, default="auto")
    parser.add_argument("--quick", action="store_true", help="Smaller grid (fewer runs)")
    args = parser.parse_args()
    args.device = resolve_device(args.device)
    print(f"Using device: {args.device}")

    base = {
        "model_path": args.model_path,
        "image_dir": args.image_dir,
        "annotation_dir": args.annotation_dir,
        "patch_size": 32,
        "cluster": True,
        "match_radius": 48.0,
        "split_file": args.split_file,
        "split": args.split,
        "device": args.device,
        "verbose": False,
    }

    if args.quick:
        strides = [16]
        thresholds = [0.48, 0.5, 0.52]
        cluster_eps_list = [28]
        nms_radii = [14]
    else:
        strides = [8, 16]
        thresholds = [0.48, 0.5, 0.52]
        cluster_eps_list = [26, 30]
        nms_radii = [12, 16]

    configs = []
    for stride, thresh, cluster_eps, nms_radius in product(
        strides, thresholds, cluster_eps_list, nms_radii
    ):
        configs.append({
            **base,
            "stride": stride,
            "threshold": thresh,
            "cluster_eps": cluster_eps,
            "nms_radius": nms_radius,
            "local_max_size": 7,
        })

    print(f"🥜 Sweeping {len(configs)} hyperparameter configs on split '{args.split}'")
    print("=" * 80)

    results = []
    for i, cfg in enumerate(configs):
        run = (
            f"stride={cfg['stride']} thresh={cfg['threshold']:.2f} "
            f"cluster_eps={cfg['cluster_eps']} nms={cfg['nms_radius']}"
        )
        print(f"  [{i+1}/{len(configs)}] {run} ...", end=" ", flush=True)
        metrics = evaluate_dataset(**cfg)
        if metrics is None:
            print("FAILED")
            continue
        results.append((run, cfg, metrics))
        print(f"P={metrics['precision']:.3f} R={metrics['recall']:.3f} F1={metrics['f1']:.3f}")

    if not results:
        print("No successful runs.")
        return

    # Sort by F1 descending
    results.sort(key=lambda x: x[2]["f1"], reverse=True)

    print()
    print("=" * 80)
    print("📊 Results (sorted by F1)")
    print("=" * 80)
    print(f"{'Config':<55} {'Prec':>6} {'Rec':>6} {'F1':>6} {'MAE':>6}")
    print("-" * 80)
    for run, cfg, m in results:
        print(f"{run:<55} {m['precision']:>6.3f} {m['recall']:>6.3f} {m['f1']:>6.3f} {m['mae']:>6.2f}")

    best_run, best_cfg, best_m = results[0]
    print()
    print("🏆 Best config (by F1):")
    print(f"   {best_run}")
    print(f"   Precision: {best_m['precision']:.3f}, Recall: {best_m['recall']:.3f}, F1: {best_m['f1']:.3f}, MAE: {best_m['mae']:.2f}")
    print()
    print("Re-run with:")
    print(
        f"  python evaluate_walnut_annotations.py --model_path {args.model_path} "
        f"--image_dir {args.image_dir} --annotation_dir {args.annotation_dir} "
        f"--split_file {args.split_file} --split {args.split} --device {args.device} "
        f"--stride {best_cfg['stride']} --threshold {best_cfg['threshold']} "
        f"--cluster_eps {best_cfg['cluster_eps']} --nms_radius {best_cfg['nms_radius']} "
        f"--local_max_size {best_cfg['local_max_size']} --match_radius {best_cfg['match_radius']} --cluster"
    )


if __name__ == "__main__":
    main()
