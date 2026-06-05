#!/usr/bin/env python3
"""
Run detector hyperparameter sweep for all k-fold models.

Each fold's classifier (models/fold_k/) is evaluated on that fold's test_stems
from fold_k.json using evaluate_dataset (sliding-window detector metrics).

Eval image/annotation dirs (first match wins):
  1. --image_dir / --annotation_dir
  2. combined_eval/images + combined_eval/annotations (if populated)
  3. output/ + output/annotations (this repo's default layout)
  4. Legacy images_all/Annotations11-10/... (Set 1 ± Set 2 merge)

  python run_sweeps_all_folds.py --device auto
  python run_sweeps_all_folds.py --quick --device auto
  python run_sweeps_all_folds.py --image_dir output --annotation_dir output/annotations

Output: sweep_results/fold_0.json ... fold_9.json + aggregated_summary.json
"""
import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parent
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

import argparse
import json
import re
from itertools import product

from evaluate_walnut_annotations import evaluate_dataset


def num_images(metrics: dict) -> int:
    return len(metrics.get("per_image", []))


def resolve_fold_model(fold_dir: Path) -> Path | None:
    for name in ("walnut_classifier_best_precision.pth", "walnut_classifier_phase1.pth"):
        path = fold_dir / name
        if path.exists():
            return path
    return None


def load_centers_and_size(txt_path: Path) -> tuple[list[tuple[int, int]], int, int]:
    """Load centers and parse # Image size: WxH. Return (centers, full_w, full_h)."""
    centers = []
    full_w, full_h = 0, 0
    if not txt_path.exists():
        return centers, full_w, full_h
    with txt_path.open("r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("# Image size:"):
                m = re.search(r"(\d+)\s*[xX]\s*(\d+)", line)
                if m:
                    full_w, full_h = int(m.group(1)), int(m.group(2))
                continue
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    x, y = int(float(parts[0])), int(float(parts[1]))
                    centers.append((x, y))
                except ValueError:
                    continue
    return centers, full_w, full_h


def build_set2_quadrant_annotations(
    set2_cropped_dir: Path,
    output_ann_dir: Path,
) -> set[str]:
    """
    From Second Set (full-image .txt per base_stem), write one .txt per quadrant image
    with local coordinates. Returns set of stems that have annotation files.
    """
    output_ann_dir.mkdir(parents=True, exist_ok=True)
    quad_suffixes = ["_q00", "_q01", "_q10", "_q11"]
    quad_bounds = [
        (0, 0, 0, 0),   # placeholder; we use half dims
        (0, 0, 0, 0),
        (0, 0, 0, 0),
        (0, 0, 0, 0),
    ]
    stems_written = set()
    txt_files = sorted(set2_cropped_dir.glob("*.txt"))
    for txt_path in txt_files:
        base_stem = txt_path.stem
        centers, full_w, full_h = load_centers_and_size(txt_path)
        if full_w <= 0 or full_h <= 0:
            continue
        w2, h2 = full_w // 2, full_h // 2
        bounds = [
            (0, w2, 0, h2, 0, 0),
            (w2, full_w, 0, h2, w2, 0),
            (0, w2, h2, full_h, 0, h2),
            (w2, full_w, h2, full_h, w2, h2),
        ]
        for qs, (x_lo, x_hi, y_lo, y_hi, ox, oy) in zip(quad_suffixes, bounds):
            stem = base_stem + qs
            img_path = set2_cropped_dir / f"{stem}.JPG"
            if not img_path.exists():
                img_path = set2_cropped_dir / f"{stem}.jpg"
            if not img_path.exists():
                continue
            local_centers = [
                (x - ox, y - oy) for x, y in centers
                if x_lo <= x < x_hi and y_lo <= y < y_hi
            ]
            out_path = output_ann_dir / f"{stem}.txt"
            with out_path.open("w") as f:
                for x, y in local_centers:
                    f.write(f"{x} {y}\n")
            stems_written.add(stem)
    return stems_written


def run_sweep_one_fold(
    fold_idx: int,
    model_path: Path,
    test_stems: list[str],
    split_file_path: Path,
    image_dir: Path,
    annotation_dir: Path,
    device: str,
    quick: bool,
    verbose: bool = False,
) -> list[dict]:
    """Run full sweep for one fold; return list of {config_key, run, metrics}."""
    with split_file_path.open("w") as f:
        json.dump({"test": test_stems}, f, indent=0)

    base = {
        "model_path": str(model_path),
        "image_dir": str(image_dir),
        "annotation_dir": str(annotation_dir),
        "patch_size": 32,
        "cluster": True,
        "match_radius": 48.0,
        "split_file": str(split_file_path),
        "split": "test",
        "device": device,
        "verbose": verbose,
    }
    if quick:
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

    results = []
    for cfg in configs:
        run = (
            f"stride={cfg['stride']} thresh={cfg['threshold']:.2f} "
            f"cluster_eps={cfg['cluster_eps']} nms={cfg['nms_radius']}"
        )
        config_key = (cfg["stride"], cfg["threshold"], cfg["cluster_eps"], cfg["nms_radius"])
        metrics = evaluate_dataset(**cfg)
        if metrics is None:
            continue
        results.append({
            "config_key": list(config_key),
            "run": run,
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "mae": metrics["mae"],
            "tp": metrics["tp"],
            "fp": metrics["fp"],
            "fn": metrics["fn"],
            "num_images": num_images(metrics),
        })
    return results


def merge_metrics(list_of_metrics: list[dict]) -> dict:
    """Merge TP/FP/FN across runs (e.g. Set1 + Set2), recompute P/R/F1."""
    total_tp = sum(m["tp"] for m in list_of_metrics)
    total_fp = sum(m["fp"] for m in list_of_metrics)
    total_fn = sum(m["fn"] for m in list_of_metrics)
    total_images = sum(m["num_images"] for m in list_of_metrics)
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    # MAE: average of MAEs weighted by num_images
    if total_images > 0:
        mae = sum(m["mae"] * m["num_images"] for m in list_of_metrics) / total_images
    else:
        mae = 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mae": mae,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "num_images": total_images,
    }


class Tee:
    """Write to both stdout and a log file."""
    def __init__(self, log_path: Path):
        self._file = log_path.open("w", encoding="utf-8")
        self._stdout = sys.stdout
    def write(self, data: str):
        self._stdout.write(data)
        self._stdout.flush()
        self._file.write(data)
        self._file.flush()
    def flush(self):
        self._stdout.flush()
        self._file.flush()
    def close(self):
        self._file.close()
        sys.stdout = self._stdout


def main():
    parser = argparse.ArgumentParser(description="Run detector sweep for all k-fold models")
    from device_utils import add_device_argument

    add_device_argument(parser, default="auto")
    parser.add_argument("--quick", action="store_true", help="Smaller grid")
    parser.add_argument("--n_folds", type=int, default=10, help="Number of folds to sweep")
    parser.add_argument("--image_dir", type=str, default=None, help="Images for detector eval (default: output/ or combined_eval/)")
    parser.add_argument("--annotation_dir", type=str, default=None, help="Annotations for detector eval (default: output/annotations/)")
    parser.add_argument("--set1_only", action="store_true", help="Legacy: only use Set 1 under images_all/")
    parser.add_argument("--log", type=str, default=None, help="Save console output (default: sweep_log.txt)")
    args = parser.parse_args()

    log_path = Path(args.log).resolve() if args.log else WORKSPACE / "sweep_log.txt"
    tee = Tee(log_path)
    sys.stdout = tee
    try:
        _run_sweeps(args, WORKSPACE)
    finally:
        tee.close()
    print(f"Log saved to {log_path}")


def _run_sweeps(args, workspace: Path):
    results_dir = workspace / "sweep_results"
    results_dir.mkdir(parents=True, exist_ok=True)

    combined_img = workspace / "combined_eval" / "images"
    combined_ann = workspace / "combined_eval" / "annotations"
    default_img = workspace / "output"
    default_ann = workspace / "output" / "annotations"

    use_single_dir = False
    image_dir = annotation_dir = None
    set1_img = set1_ann = set2_cropped = set2_ann_dir = None
    set2_stems: set[str] = set()

    if args.image_dir:
        image_dir = Path(args.image_dir)
        annotation_dir = Path(args.annotation_dir) if args.annotation_dir else image_dir / "annotations"
        use_single_dir = True
        print(f"📁 Using --image_dir: {image_dir} + {annotation_dir}")
    elif combined_img.exists() and combined_ann.exists() and next(combined_img.iterdir(), None) is not None:
        image_dir = combined_img
        annotation_dir = combined_ann
        use_single_dir = True
        print(f"📁 Using combined_eval/ (single image + annotation dir)")
    elif default_img.exists() and default_ann.exists():
        image_dir = default_img
        annotation_dir = default_ann
        use_single_dir = True
        print(f"📁 Using output/ + output/annotations/")
    else:
        set1_img = workspace / "images_all" / "Annotations11-10" / "cropped_images"
        set1_ann = workspace / "images_all" / "Annotations11-10" / "annotated_images"
        set2_cropped = workspace / "images_all" / "Annotations11-10" / "Second Set of Annotations" / "cropped_images"
        use_set2 = not args.set1_only and set2_cropped.exists()
        if use_set2:
            set2_ann_dir = workspace / "temp_second_set_annotations"
            set2_stems = build_set2_quadrant_annotations(set2_cropped, set2_ann_dir)
            print(f"📁 Set 2: wrote quadrant annotations for {len(set2_stems)} stems to {set2_ann_dir}")
            print("   Evaluating each fold on Set 1 + Set 2 (merged TP/FP/FN per config).")
        else:
            if args.set1_only:
                print("   Using Set 1 only (--set1_only).")
            else:
                print("   Set 2 dir not found; using Set 1 only.")

    # Config grid (same key for aggregation)
    if args.quick:
        strides, thresholds = [16], [0.48, 0.5, 0.52]
        cluster_eps_list, nms_radii = [28], [14]
    else:
        strides, thresholds = [8, 16], [0.48, 0.5, 0.52]
        cluster_eps_list, nms_radii = [26, 30], [12, 16]

    all_config_keys = []
    for stride, thresh, cluster_eps, nms_radius in product(
        strides, thresholds, cluster_eps_list, nms_radii
    ):
        all_config_keys.append((stride, thresh, cluster_eps, nms_radius))

    print(f"\n📐 Sweep hyperparameters (fixed: patch_size=32, cluster=True, match_radius=48, local_max_size=7)")
    print(f"   stride:       {strides}")
    print(f"   threshold:    {thresholds}")
    print(f"   cluster_eps:  {cluster_eps_list}")
    print(f"   nms_radius:   {nms_radii}")
    print(f"   Total configs: {len(all_config_keys)} (× {args.n_folds} folds = {len(all_config_keys) * args.n_folds} runs)")
    print(f"   Results: {results_dir}/fold_*.json + {results_dir}/aggregated_summary.json\n")

    split_file_path = workspace / "temp_split_test.json"
    per_fold_results = []

    for fold_idx in range(args.n_folds):
        fold_file = workspace / f"fold_{fold_idx}.json"
        model_path = resolve_fold_model(workspace / "models" / f"fold_{fold_idx}")
        if not fold_file.exists():
            print(f"⚠️  Missing {fold_file}, skipping fold {fold_idx}")
            continue
        if model_path is None:
            print(f"⚠️  Missing model in models/fold_{fold_idx}/, skipping fold {fold_idx}")
            continue

        with fold_file.open() as f:
            fold_data = json.load(f)
        test_stems = fold_data["test_stems"]

        if use_single_dir:
            test_stems_set1 = test_stems
            test_stems_set2 = []
        else:
            set1_stems = set()
            for ext in ["*.png", "*.PNG", "*.jpg", "*.JPG"]:
                for p in set1_img.glob(ext):
                    set1_stems.add(p.stem)
            test_stems_set1 = [s for s in test_stems if s in set1_stems]
            test_stems_set2 = [s for s in test_stems if s in set2_stems] if set2_stems else []

        print(f"\n{'='*60}")
        print(f"Fold {fold_idx}: test_stems={len(test_stems)}" + (f" (Set1={len(test_stems_set1)}, Set2={len(test_stems_set2)})" if not use_single_dir else ""))
        print("=" * 60)

        fold_run_results = []
        for config_key in all_config_keys:
            stride, thresh, cluster_eps, nms_radius = config_key
            run = (
                f"stride={stride} thresh={thresh:.2f} "
                f"cluster_eps={cluster_eps} nms={nms_radius}"
            )

            if use_single_dir:
                with split_file_path.open("w") as f:
                    json.dump({"test": test_stems}, f, indent=0)
                cfg = {
                    "model_path": str(model_path),
                    "image_dir": str(image_dir),
                    "annotation_dir": str(annotation_dir),
                    "patch_size": 32,
                    "cluster": True,
                    "match_radius": 48.0,
                    "split_file": str(split_file_path),
                    "split": "test",
                    "device": args.device,
                    "verbose": False,
                    "stride": stride,
                    "threshold": thresh,
                    "cluster_eps": cluster_eps,
                    "nms_radius": nms_radius,
                    "local_max_size": 7,
                }
                metrics = evaluate_dataset(**cfg)
                if metrics is None:
                    continue
                fold_run_results.append({
                    "config_key": list(config_key),
                    "run": run,
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1": metrics["f1"],
                    "mae": metrics["mae"],
                    "tp": metrics["tp"],
                    "fp": metrics["fp"],
                    "fn": metrics["fn"],
                    "num_images": num_images(metrics),
                })
            else:
                metrics_list = []
                if test_stems_set1:
                    with split_file_path.open("w") as f:
                        json.dump({"test": test_stems_set1}, f, indent=0)
                    cfg_set1 = {
                        "model_path": str(model_path),
                        "image_dir": str(set1_img),
                        "annotation_dir": str(set1_ann),
                        "patch_size": 32,
                        "cluster": True,
                        "match_radius": 48.0,
                        "split_file": str(split_file_path),
                        "split": "test",
                        "device": args.device,
                        "verbose": False,
                        "stride": stride,
                        "threshold": thresh,
                        "cluster_eps": cluster_eps,
                        "nms_radius": nms_radius,
                        "local_max_size": 7,
                    }
                    m1 = evaluate_dataset(**cfg_set1)
                    if m1 is not None:
                        metrics_list.append(m1)
                if test_stems_set2 and set2_ann_dir is not None:
                    with split_file_path.open("w") as f:
                        json.dump({"test": test_stems_set2}, f, indent=0)
                    cfg_set2 = {
                        "model_path": str(model_path),
                        "image_dir": str(set2_cropped),
                        "annotation_dir": str(set2_ann_dir),
                        "patch_size": 32,
                        "cluster": True,
                        "match_radius": 48.0,
                        "split_file": str(split_file_path),
                        "split": "test",
                        "device": args.device,
                        "verbose": False,
                        "stride": stride,
                        "threshold": thresh,
                        "cluster_eps": cluster_eps,
                        "nms_radius": nms_radius,
                        "local_max_size": 7,
                    }
                    m2 = evaluate_dataset(**cfg_set2)
                    if m2 is not None:
                        metrics_list.append(m2)
                if not metrics_list:
                    continue
                merged = merge_metrics(metrics_list)
                fold_run_results.append({
                    "config_key": list(config_key),
                    "run": run,
                    "precision": merged["precision"],
                    "recall": merged["recall"],
                    "f1": merged["f1"],
                    "mae": merged["mae"],
                    "tp": merged["tp"],
                    "fp": merged["fp"],
                    "fn": merged["fn"],
                    "num_images": merged["num_images"],
                })

        out_path = results_dir / f"fold_{fold_idx}.json"
        with out_path.open("w") as f:
            json.dump({"fold": fold_idx, "results": fold_run_results}, f, indent=2)
        print(f"   Wrote {out_path} ({len(fold_run_results)} configs)")
        per_fold_results.append({"fold": fold_idx, "results": fold_run_results})

    if not per_fold_results:
        print("No fold results to aggregate.")
        return

    # Aggregate: for each config_key, average P/R/F1 across folds
    key_to_metrics = {}
    for fold_data in per_fold_results:
        for r in fold_data["results"]:
            key = tuple(r["config_key"])
            if key not in key_to_metrics:
                key_to_metrics[key] = {"precision": [], "recall": [], "f1": [], "mae": []}
            key_to_metrics[key]["precision"].append(r["precision"])
            key_to_metrics[key]["recall"].append(r["recall"])
            key_to_metrics[key]["f1"].append(r["f1"])
            key_to_metrics[key]["mae"].append(r["mae"])

    agg_list = []
    for key, vals in key_to_metrics.items():
        stride, thresh, cluster_eps, nms_radius = key
        run = f"stride={stride} thresh={thresh:.2f} cluster_eps={cluster_eps} nms={nms_radius}"
        n = len(vals["precision"])
        agg_list.append({
            "run": run,
            "config_key": list(key),
            "mean_precision": sum(vals["precision"]) / n,
            "mean_recall": sum(vals["recall"]) / n,
            "mean_f1": sum(vals["f1"]) / n,
            "mean_mae": sum(vals["mae"]) / n,
            "std_f1": (sum((x - sum(vals["f1"])/n)**2 for x in vals["f1"]) / n) ** 0.5 if n > 1 else 0.0,
        })

    agg_list.sort(key=lambda x: x["mean_f1"], reverse=True)

    print()
    print("=" * 80)
    print("📊 Aggregated results (mean over folds, sorted by mean F1)")
    print("=" * 80)
    print(f"{'Config':<55} {'P':>6} {'R':>6} {'F1':>6} {'F1_std':>7} {'MAE':>6}")
    print("-" * 80)
    for a in agg_list:
        print(
            f"{a['run']:<55} "
            f"{a['mean_precision']:>6.3f} {a['mean_recall']:>6.3f} {a['mean_f1']:>6.3f} "
            f"{a['std_f1']:>7.3f} {a['mean_mae']:>6.2f}"
        )

    best = agg_list[0]
    print()
    print("🏆 Best config (by mean F1):")
    print(f"   {best['run']}")
    print(f"   Mean Precision: {best['mean_precision']:.3f}, Recall: {best['mean_recall']:.3f}, F1: {best['mean_f1']:.3f} (±{best['std_f1']:.3f})")
    print(f"   Mean MAE: {best['mean_mae']:.2f}")

    summary_path = results_dir / "aggregated_summary.json"
    with summary_path.open("w") as f:
        json.dump(agg_list, f, indent=2)
    print(f"\n   Wrote {summary_path}")


if __name__ == "__main__":
    main()
