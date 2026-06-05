# Walnut Detection — Runbook

Step-by-step commands for the `output/` dataset layout (4 images, 2 train / 1 val / 1 test).

## Setup

```bash
cd Walnut-detection
python setup.py
source venv/bin/activate
```

All scripts accept `--device auto` (CUDA → MPS → CPU).

---

## 1. Build patch dataset

Extract positive and negative 32×32 patches from annotations and split into train/val/test:

```bash
python build_annotations11_10_dataset.py \
  --image_dir output \
  --annotation_dir output/annotations \
  --output_dir output/dataset \
  --n_train 2 --n_val 1 --n_test 1 --clean
```

### 10-fold cross-validation (optional, ≥10 images)

For k-fold training you need **all patches from every image** in one pool (not the `train/val/test` folders above):

```
all_patches/
  positive/   # e.g. image_q01_p0001.png, image_q10_p0002.png, ...
  negative/   # e.g. image_q01_n0001.png, ...
```

Populate `all_patches/` from your full image set (≥10 unique images). With only 4 images in `output/`, 10-fold CV will not run.

Build fold split files (`train_stems`, `val_stems`, `test_stems` per fold — no patch copies):

```bash
python build_fold_jsons.py
```

Custom paths:

```bash
python build_fold_jsons.py \
  --patches_dir all_patches \
  --output_dir . \
  --n_folds 10 \
  --seed 42
```

Output: `fold_0.json` … `fold_9.json` in the repo root (or `--output_dir`).

**Train one fold** (`train_fold.py` wraps `binary_classifier.py` with focal loss + hard-negative mining):

```bash
python train_fold.py 0 --device auto
python train_fold.py 3 --epochs 20 --second_phase_epochs 10 --device auto
```

Saves to `models/fold_0/walnut_classifier_best_precision.pth` (or `walnut_classifier_phase1.pth` if phase 2 did not beat phase 1).

**Train all folds** in sequence (logs to `training_log.txt`):

```bash
python run_all_folds.py --device auto
python run_all_folds.py --n_folds 5 --device auto   # if you used --n_folds 5 above
```

**Sweep detector hyperparameters per fold** (each fold’s model on that fold’s `test_stems`; aggregates mean F1 across folds):

```bash
python run_sweeps_all_folds.py --device auto
python run_sweeps_all_folds.py --quick --device auto
python run_sweeps_all_folds.py --image_dir output --annotation_dir output/annotations
```

Writes `sweep_results/fold_0.json` … `fold_9.json` and `sweep_results/aggregated_summary.json`. Use the best config from `aggregated_summary.json` (top row by mean F1) for `parameter_sweep_binary.py`-style detector settings on held-out data.

Pipeline order: `build_fold_jsons.py` → `run_all_folds.py` → `run_sweeps_all_folds.py`.

---

## 2. Train binary classifier

Train from scratch (needed once so `models/*.pth` exists for the sweep below):

```bash
python binary_classifier.py \
  --dataset_dir output/dataset \
  --output_dir models \
  --epochs 20 \
  --loss_type focal \
  --mine_hard_negatives \
  --hard_negative_threshold 0.5 \
  --hard_negative_duplicate 3 \
  --second_phase_epochs 10 \
  --device auto
```

Fine-tune from an existing checkpoint (optional):

```bash
python binary_classifier.py \
  --dataset_dir output/dataset \
  --output_dir models_finetuned \
  --pretrained models/walnut_classifier_best_precision.pth \
  --learning_rate 0.0003 \
  --epochs 25 \
  --loss_type focal \
  --mine_hard_negatives \
  --hard_negative_threshold 0.5 \
  --hard_negative_duplicate 3 \
  --second_phase_epochs 20 \
  --patch_size 32 \
  --device auto
```

---

## 3. Parameter sweep (classifier detector settings)

Run after training. Use `phase1.pth` if `best_precision` is missing:

```bash
python parameter_sweep_binary.py \
  --model_path models/walnut_classifier_phase1.pth \
  --image_dir output \
  --annotation_dir output/annotations \
  --split_file output/dataset/split.json \
  --split test \
  --device auto
```

Quick sweep (fewer combinations):

```bash
python parameter_sweep_binary.py --quick
```

Results: `binary_parameter_sweep_results.json` (`best_by_mae`, `best_by_f1`).

---

## 4. Extract cutouts & run sliding-window detector

### Extract walnut cutouts

Creates `walnut_cutouts/` under `--output_dir`:

```bash
python extract_walnuts.py \
  --train_dir output/dataset/train \
  --output_dir output/dataset
```

### Run `walnut_detector.py` on full images

Uses the trained classifier with hyperparameters from the parameter sweep (`best_by_mae`: patch=16, stride=8, threshold=0.5). Saves overlays and JSON under `output/detections/`:

```bash
python walnut_detector.py \
  --model_path models/walnut_classifier_phase1.pth \
  --image_dir output \
  --output_dir output/detections \
  --patch_size 16 \
  --stride 8 \
  --threshold 0.5 \
  --cluster \
  --cluster_eps 28.0 \
  --nms_radius 14.0 \
  --device auto
```

Single image:

```bash
python walnut_detector.py \
  --model_path models/walnut_classifier_phase1.pth \
  --image_path output/image_q00.png \
  --output_dir output/detections \
  --patch_size 16 --stride 8 --threshold 0.5 --cluster --device auto
```

For `best_by_f1` from the sweep instead, use `--patch_size 16 --stride 12 --threshold 0.5`.

Metrics vs annotations (precision/recall/F1):

```bash
python evaluate_walnut_annotations.py \
  --model_path models/walnut_classifier_phase1.pth \
  --image_dir output \
  --annotation_dir output/annotations \
  --split_file output/dataset/split.json \
  --split test \
  --patch_size 16 --stride 8 --threshold 0.5 \
  --device auto
```

---

## 5. Optimize synthetic config (Optuna)

```bash
python optimize_synthetic_config.py \
  --cutouts_dir output/dataset/walnut_cutouts \
  --neg_dir output/dataset/train/negative \
  --w0_path models/walnut_classifier_phase1.pth \
  --image_dir output \
  --annotation_dir output/annotations \
  --split_file output/dataset/split.json \
  --n_trials 10 \
  --device auto
```

Repo-root layout (after `extract_walnuts.py --train_dir train --output_dir .`):

```bash
python optimize_synthetic_config.py \
  --cutouts_dir walnut_cutouts \
  --neg_dir train/negative \
  --w0_path models/walnut_classifier_phase1.pth \
  --n_trials 10 \
  --device auto
```

Output: `optuna_results/best_config.json`

---

## 6. Build YOLO tiled dataset

Real **640×640 crops** from images with **preexisting annotation labels** (YOLO boxes from `.txt` files):

```bash
python build_yolo_tiled_dataset.py \
  --config optuna_results/best_config.json \
  --image_dir output \
  --annotation_dir output/annotations \
  --split_file output/dataset/split.json \
  --out_dir yolo_walnut_tiled
```

Random tile count (default: **random 640×640 crop + real annotation labels**):

```bash
python build_yolo_tiled_dataset.py \
  --config optuna_results/best_config.json \
  --num_tiles 500
```

Optional: paste extra augmented cutouts on top (`best_config` augments cutouts only):

```bash
python build_yolo_tiled_dataset.py \
  --config optuna_results/best_config.json \
  --num_tiles 500 \
  --composite_cutouts \
  --cutouts_dir output/dataset/walnut_cutouts \
  --synthetic_patches_min 1 \
  --synthetic_patches_max 4
```

Fixed count (e.g. exactly 2 synthetic cutouts per tile):

```bash
python build_yolo_tiled_dataset.py \
  --config optuna_results/best_config.json \
  --num_tiles 500 \
  --composite_cutouts \
  --num_synthetic_patches 2
```

Resample fixed grid cells instead of random crops: add `--grid_tiles`.

```bash
python build_yolo_tiled_dataset.py \
  --config optuna_results/best_config.json \
  --num_tiles_min 200 \
  --num_tiles_max 800
```

Synthetic-only YOLO data (composited cutouts, same `best_config.json`):

```bash
python generate_yolo_synthetic_dataset.py \
  --config optuna_results/best_config.json \
  --cutouts_dir output/dataset/walnut_cutouts \
  --neg_dir output/dataset/train/negative \
  --mode patch \
  --num_images 5000 \
  --out_dir yolo_walnut_synthetic
```

---

## 7. Train YOLO

```bash
python train_yolov8_synthetic.py \
  --data_dir yolo_walnut_tiled \
  --imgsz 640 \
  --epochs 100 \
  --device auto
```

Weights: `yolo_runs/walnut_synthetic/weights/best.pt` (or `walnut_synthetic-N` if the name collides).

---

## 8. Two-stage evaluation (YOLO + classifier)

Uses classifier thresholds from `binary_parameter_sweep_results.json`:

```bash
python "evaluate_yolo_two_stage copy.py" \
  --yolo_model_path yolo_runs/walnut_synthetic/weights/best.pt \
  --clf_model_path models/walnut_classifier_phase1.pth \
  --image_dir output \
  --annotation_dir output/annotations \
  --split_file output/dataset/split.json \
  --sweep_results binary_parameter_sweep_results.json \
  --sweep_metric mae \
  --split test \
  --device auto
```

Best F1 sweep config instead of best MAE:

```bash
python "evaluate_yolo_two_stage copy.py" \
  ... \
  --sweep_metric f1
```

If YOLO returns no proposals, lower `--yolo_conf` (e.g. `0.05`).

---

## Data layout (`output/`)

| Path | Contents |
|------|----------|
| `output/image_q*.png` | Full images |
| `output/annotations/*.txt` | Walnut centre annotations |
| `output/dataset/` | Patch dataset (`train/`, `val/`, `test/`, `split.json`) |
| `all_patches/` | Pooled patches for k-fold CV (`positive/`, `negative/`) |
| `fold_0.json` … `fold_9.json` | Per-fold stem lists from `build_fold_jsons.py` |
| `output/dataset/walnut_cutouts/` | RGBA cutouts |
| `models/` | Classifier checkpoints (`.pth`) |
| `optuna_results/best_config.json` | Best synthetic compositing config |
| `binary_parameter_sweep_results.json` | Best detector hyperparameters |
| `yolo_walnut_tiled/` | YOLO train/val tiles |
| `yolo_runs/` | YOLO training outputs |



flowchart TD
    A[build_fold_jsons.py] -->|fold_0.json ... fold_N.json| B[run_all_folds.py]
    B -->|calls train_fold.py 0..N| C[binary_classifier.py per fold]
    C -->|models/fold_k/*.pth| D[run_sweeps_all_folds.py]
    D -->|eval on fold test_stems| E[sweep_results/fold_k.json]
    E --> F[aggregated_summary.json]
