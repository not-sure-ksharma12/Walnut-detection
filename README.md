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

## 4. Extract walnut cutouts

Creates `walnut_cutouts/` under `--output_dir`:

```bash
python extract_walnuts.py \
  --train_dir output/dataset/train \
  --output_dir output/dataset
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

Real-image 640×640 tiles using `best_config.json`:

```bash
python build_yolo_tiled_dataset.py \
  --config optuna_results/best_config.json \
  --image_dir output \
  --annotation_dir output/annotations \
  --split_file output/dataset/split.json \
  --out_dir yolo_walnut_tiled
```

Random tile count (samples with replacement when N > unique grid cells):

```bash
python build_yolo_tiled_dataset.py \
  --config optuna_results/best_config.json \
  --num_tiles 500
```

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
  --yolo_model_path yolo_runs/walnut_synthetic-3/weights/best.pt \
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
| `output/dataset/walnut_cutouts/` | RGBA cutouts |
| `models/` | Classifier checkpoints (`.pth`) |
| `optuna_results/best_config.json` | Best synthetic compositing config |
| `binary_parameter_sweep_results.json` | Best detector hyperparameters |
| `yolo_walnut_tiled/` | YOLO train/val tiles |
| `yolo_runs/` | YOLO training outputs |
