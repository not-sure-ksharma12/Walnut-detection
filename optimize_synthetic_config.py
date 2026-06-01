#!/usr/bin/env python3
"""
optimize_synthetic_config.py — Optuna search over synthetic data generation settings (Stage 1).

What it does:
  - Samples compositing/augmentation configs with Optuna TPE
  - Generates synthetic 32×32 patches in memory, fine-tunes classifier head on W0
  - Evaluates sliding-window detection F1 on val images and saves best config
  - Writes optuna_results/best_config.json and logs to logs/

Prerequisites:
  - python3 setup.py
  - walnut_cutouts/ (from extract_walnuts.py), train/negative/ (or output/dataset/train/negative/)
  - A trained .pth for W0 (--w0_path)
  - Full images, annotations, split.json for val evaluation
  - Optional: cyclegan_models/G_AB_best.pth

How to run:
  cd Walnut-detection
  source venv/bin/activate
  python optimize_synthetic_config.py \\
    --cutouts_dir walnut_cutouts \\
    --neg_dir train/negative \\
    --w0_path models/walnut_classifier_phase1.pth \\
    --image_dir output \\
    --annotation_dir output/annotations \\
    --split_file output/dataset/split.json \\
    --n_trials 100 --device auto
"""

import argparse
import copy
import json
import logging
import os
import random
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms

# ---------------------------------------------------------------------------
# Paths (relative to repo root = directory containing this file)
# ---------------------------------------------------------------------------
WORKSPACE = Path(__file__).resolve().parent

STUDY_DIR = WORKSPACE / "optuna_results"
LOG_DIR = WORKSPACE / "logs"


@dataclass(frozen=True)
class RunPaths:
    w0_path: Path
    cutouts_dir: Path
    neg_dir: Path
    image_dir: Path
    annotation_dir: Path
    split_file: Path
    cyclegan_ckpt: Path


def _resolve_path(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else WORKSPACE / p


def _require_dir(path: Path, label: str) -> None:
    if not path.is_dir():
        raise SystemExit(f"{label} not found: {path}")


def _require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise SystemExit(f"{label} not found: {path}")

# Detection hyper-params (fixed — best config from real-model sweep)
DET_STRIDE = 16
DET_THRESH = 0.48
DET_EPS = 26.0
DET_NMS = 16.0
DET_LMS = 7
DET_MATCH_RADIUS = 48.0

# Fine-tune budget
FT_LR = 3e-5
FT_STEPS = 100
FT_BATCH = 32
NUM_SYNTHETIC = 500
PATCH_SIZE = 32

# ---------------------------------------------------------------------------
# Imports from workspace (add workspace to sys.path)
# ---------------------------------------------------------------------------
sys.path.insert(0, str(WORKSPACE))

def setup_logger(log_dir: Path) -> logging.Logger:
    """Create a logger that writes to both console and a timestamped file."""
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"search_{timestamp}.log"

    logger = logging.getLogger("config_search")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-5s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    logger.info(f"Log file: {log_file}")
    return logger


# Global logger — initialized in main()
log: logging.Logger = logging.getLogger("config_search")

from binary_classifier import WalnutClassifier, calculate_class_weights
from evaluate_walnut_annotations import (
    evaluate_dataset,
    parse_annotations,
    match_detections,
    _find_image,
)
from walnut_detector import WalnutDetector
from refine_synthetic import Generator32


# ---------------------------------------------------------------------------
# Preload all assets into memory once
# ---------------------------------------------------------------------------
def load_cutouts(cutouts_dir: Path):
    """Load all RGBA cutouts into memory as (bgr, alpha) pairs."""
    cutouts = []
    for p in sorted(cutouts_dir.glob("*.png")):
        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        if img.shape[2] == 4:
            bgr, alpha = img[:, :, :3], img[:, :, 3]
        else:
            bgr = img
            alpha = np.ones(img.shape[:2], dtype=np.uint8) * 255
        cutouts.append((bgr, alpha))
    return cutouts


def load_negative_patches(neg_dir: Path, patch_size: int = 32,
                          green_ratio_thresh: float = 0.3):
    """Load and resize negative patches, keeping green ones."""
    patches = []
    for p in sorted(neg_dir.glob("*.png")):
        img = cv2.imread(str(p))
        if img is None:
            continue
        if img.shape[:2] != (patch_size, patch_size):
            img = cv2.resize(img, (patch_size, patch_size))
        patches.append(img)
    return patches


def is_green(img, thresh):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (25, 30, 20), (95, 255, 255))
    return (mask > 0).sum() / mask.size >= thresh


def load_cyclegan(ckpt_path: Path, device: str):
    """Load CycleGAN generator G_AB."""
    G = Generator32()
    G.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    G.to(device)
    G.eval()
    return G


# ---------------------------------------------------------------------------
# Synthetic patch generation (in-memory, config-driven)
# ---------------------------------------------------------------------------
def composite_one(bg, bgr, alpha, x, y):
    h, w = bgr.shape[:2]
    bh, bw = bg.shape[:2]
    x2, y2 = min(x + w, bw), min(y + h, bh)
    w2, h2 = x2 - x, y2 - y
    if w2 <= 0 or h2 <= 0:
        return
    src_a = (alpha[:h2, :w2].astype(np.float32) / 255.0)[:, :, np.newaxis]
    bg[y:y2, x:x2] = (bgr[:h2, :w2] * src_a + bg[y:y2, x:x2] * (1 - src_a)).astype(np.uint8)


def generate_synthetic_batch(
    config: dict,
    cutouts: list,
    neg_patches: list,
    num_samples: int = 500,
    patch_size: int = 32,
) -> list:
    """Generate synthetic patches in-memory using the given config.

    Returns a list of BGR numpy arrays (patch_size x patch_size).
    """
    green_thresh = config["green_ratio_thresh"]
    scale_min = config["cutout_scale_min"]
    scale_max = config["cutout_scale_max"]
    pos_jitter = config["position_jitter"]

    # Augmentation params
    brightness = config["brightness_delta"]
    contrast = config["contrast_factor"]
    blur_sigma = config["blur_sigma"]
    noise_std = config["noise_std"]
    hue_shift = config["hue_shift"]
    sat_factor = config["saturation_factor"]

    # Filter backgrounds by green threshold
    green_bgs = [p for p in neg_patches if is_green(p, green_thresh)]
    if len(green_bgs) < 10:
        green_bgs = neg_patches[:200]

    results = []
    for _ in range(num_samples):
        bg = random.choice(green_bgs).copy()
        cutout_bgr, cutout_alpha = random.choice(cutouts)

        base_scale = min(
            (patch_size - 2) / max(1, cutout_bgr.shape[1]),
            (patch_size - 2) / max(1, cutout_bgr.shape[0]),
        )
        if base_scale <= 0:
            results.append(bg)
            continue

        rel = random.uniform(scale_min, scale_max)
        scale = rel * base_scale
        new_w = max(4, int(cutout_bgr.shape[1] * scale))
        new_h = max(4, int(cutout_bgr.shape[0] * scale))
        if new_w > patch_size - 1 or new_h > patch_size - 1:
            shrink = min((patch_size - 2) / new_w, (patch_size - 2) / new_h)
            new_w = max(4, int(new_w * shrink))
            new_h = max(4, int(new_h * shrink))

        bgr_s = cv2.resize(cutout_bgr, (new_w, new_h))
        alpha_s = cv2.resize(cutout_alpha, (new_w, new_h))

        # Position: center + jitter
        max_x = max(0, patch_size - new_w)
        max_y = max(0, patch_size - new_h)
        cx, cy = max_x // 2, max_y // 2
        jitter_px = int(pos_jitter * patch_size / 2)
        x = max(0, min(max_x, cx + random.randint(-jitter_px, jitter_px)))
        y = max(0, min(max_y, cy + random.randint(-jitter_px, jitter_px)))

        composite_one(bg, bgr_s, alpha_s, x, y)

        # --- Post-composition augmentations ---
        img = bg.astype(np.float32)

        # Brightness
        if abs(brightness) > 0.01:
            img = img + brightness * 255
            img = np.clip(img, 0, 255)

        # Contrast
        if abs(contrast - 1.0) > 0.01:
            mean = img.mean()
            img = (img - mean) * contrast + mean
            img = np.clip(img, 0, 255)

        # Hue + saturation (via HSV)
        if abs(hue_shift) > 0.01 or abs(sat_factor - 1.0) > 0.01:
            hsv = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift * 180) % 180
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat_factor, 0, 255)
            img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)

        # Gaussian blur
        if blur_sigma > 0.3:
            ksize = int(blur_sigma * 4) | 1
            img = cv2.GaussianBlur(img.astype(np.uint8), (ksize, ksize), blur_sigma).astype(np.float32)

        # Gaussian noise
        if noise_std > 0.001:
            noise = np.random.randn(*img.shape).astype(np.float32) * noise_std * 255
            img = np.clip(img + noise, 0, 255)

        results.append(img.astype(np.uint8))

    return results


# ---------------------------------------------------------------------------
# CycleGAN batch refinement (in-memory)
# ---------------------------------------------------------------------------
def refine_batch_cyclegan(patches_bgr: list, G: nn.Module, device: str) -> list:
    """Apply CycleGAN G_AB to a batch of BGR patches. Returns refined BGR patches."""
    tf = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    refined = []
    batch_size = 64
    for i in range(0, len(patches_bgr), batch_size):
        batch_bgr = patches_bgr[i:i + batch_size]
        tensors = []
        for p in batch_bgr:
            rgb = cv2.cvtColor(p, cv2.COLOR_BGR2RGB)
            tensors.append(tf(rgb))
        batch_tensor = torch.stack(tensors).to(device)

        with torch.no_grad():
            out = G(batch_tensor)
        out = (out.clamp(-1, 1) * 0.5 + 0.5).cpu()

        for j in range(out.shape[0]):
            arr = (out[j].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            refined.append(cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))

    return refined


# ---------------------------------------------------------------------------
# Build train tensors from in-memory patches
# ---------------------------------------------------------------------------
def patches_to_tensor(patches_bgr: list, patch_size: int = 32) -> torch.Tensor:
    """Convert list of BGR numpy patches to normalized tensor (N, 3, H, W)."""
    tf = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((patch_size, patch_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    tensors = []
    for p in patches_bgr:
        rgb = cv2.cvtColor(p, cv2.COLOR_BGR2RGB)
        tensors.append(tf(rgb))
    return torch.stack(tensors)


# ---------------------------------------------------------------------------
# Load W0 fresh (CRITICAL: always from disk, never reuse)
# ---------------------------------------------------------------------------
def load_w0(w0_path: Path, device: str, freeze_backbone: bool = True) -> WalnutClassifier:
    """Load W0 from disk. Fresh copy every time."""
    model = WalnutClassifier(input_size=PATCH_SIZE, num_classes=2)
    try:
        ckpt = torch.load(w0_path, map_location="cpu", weights_only=True)
    except Exception:
        ckpt = torch.load(w0_path, map_location="cpu", weights_only=False)

    state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    model.load_state_dict(state, strict=False)
    model.to(device)

    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False

    return model


# ---------------------------------------------------------------------------
# Brief fine-tune (head only)
# ---------------------------------------------------------------------------
def finetune_head(
    model: WalnutClassifier,
    pos_tensor: torch.Tensor,
    neg_patches_bgr: list,
    device: str,
    lr: float = FT_LR,
    steps: int = FT_STEPS,
    batch_size: int = FT_BATCH,
):
    """Fine-tune classifier head for a fixed number of steps.

    Samples a balanced mini-batch each step (half pos, half neg).
    """
    model.train()
    # Only optimize trainable params (head)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable, lr=lr, weight_decay=1e-4)

    n_pos = pos_tensor.shape[0]
    half_batch = batch_size // 2

    # Pre-convert a random subset of negatives to tensor
    neg_sample = random.sample(neg_patches_bgr, min(len(neg_patches_bgr), n_pos * 2))
    neg_tensor = patches_to_tensor(neg_sample, PATCH_SIZE)

    n_neg = neg_tensor.shape[0]

    weights = calculate_class_weights(n_pos, n_neg).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    losses = []
    for step in range(steps):
        p_idx = torch.randint(0, n_pos, (half_batch,))
        n_idx = torch.randint(0, n_neg, (half_batch,))

        x = torch.cat([pos_tensor[p_idx], neg_tensor[n_idx]], dim=0).to(device)
        y = torch.cat([torch.ones(half_batch, dtype=torch.long),
                        torch.zeros(half_batch, dtype=torch.long)]).to(device)

        perm = torch.randperm(x.shape[0])
        x, y = x[perm], y[perm]

        out = model(x)
        loss = criterion(out, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    model.eval()
    return losses


# ---------------------------------------------------------------------------
# Evaluate detector on val split (Option A — full sliding window)
# ---------------------------------------------------------------------------
def evaluate_on_val(model: WalnutClassifier, device: str, paths: RunPaths) -> dict:
    """Run full detector pipeline on val split and return full metrics dict.

    Saves model to a temp file, creates WalnutDetector, runs evaluate_dataset.
    Returns dict with f1, precision, recall, mae, per_image, etc.
    """
    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
        tmp_path = f.name
        torch.save({"model_state_dict": model.state_dict()}, tmp_path)

    try:
        metrics = evaluate_dataset(
            model_path=tmp_path,
            image_dir=str(paths.image_dir),
            annotation_dir=str(paths.annotation_dir),
            split_file=str(paths.split_file),
            split="val",
            patch_size=PATCH_SIZE,
            stride=DET_STRIDE,
            threshold=DET_THRESH,
            cluster=True,
            cluster_eps=DET_EPS,
            local_max_size=DET_LMS,
            nms_radius=DET_NMS,
            match_radius=DET_MATCH_RADIUS,
            device=device,
            verbose=False,
        )
    finally:
        os.unlink(tmp_path)

    if metrics is None:
        return {"f1": 0.0, "precision": 0.0, "recall": 0.0, "mae": 999.0,
                "tp": 0, "fp": 0, "fn": 0, "per_image": []}
    return metrics


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------
def create_objective(cutouts, neg_patches, cyclegan_G, device, baseline_f1, paths: RunPaths):

    def objective(trial: optuna.Trial) -> float:
        t0 = time.time()

        # Fixed seed so every trial draws the same cutouts/backgrounds.
        # Only the config (scale, jitter, augmentations) changes between trials.
        random.seed(0)
        np.random.seed(0)

        # --- Sample config ---
        config = {
            "green_ratio_thresh": trial.suggest_float("green_ratio_thresh", 0.3, 0.8),
            "cutout_scale_min": trial.suggest_float("cutout_scale_min", 0.3, 0.7),
            "cutout_scale_max": trial.suggest_float("cutout_scale_max", 0.65, 1.0),
            "position_jitter": trial.suggest_float("position_jitter", 0.0, 0.4),
            "brightness_delta": trial.suggest_float("brightness_delta", -0.3, 0.3),
            "contrast_factor": trial.suggest_float("contrast_factor", 0.5, 1.5),
            "blur_sigma": trial.suggest_float("blur_sigma", 0.0, 3.0),
            "noise_std": trial.suggest_float("noise_std", 0.0, 0.05),
            "hue_shift": trial.suggest_float("hue_shift", -0.1, 0.1),
            "saturation_factor": trial.suggest_float("saturation_factor", 0.5, 1.5),
            "use_cyclegan": trial.suggest_categorical("use_cyclegan", [True, False]),
        }

        if config["cutout_scale_min"] >= config["cutout_scale_max"]:
            config["cutout_scale_max"] = config["cutout_scale_min"] + 0.1

        log.info("=" * 60)
        log.info(f"TRIAL {trial.number} START")
        log.info("=" * 60)
        log.debug("Config:")
        for k, v in config.items():
            log.debug(f"  {k}: {v}")

        # --- Generate synthetic patches ---
        t_gen = time.time()
        patches = generate_synthetic_batch(
            config, cutouts, neg_patches,
            num_samples=NUM_SYNTHETIC, patch_size=PATCH_SIZE,
        )
        gen_time = time.time() - t_gen
        log.debug(f"Generated {len(patches)} patches in {gen_time:.1f}s")

        # --- Optionally refine with CycleGAN ---
        if config["use_cyclegan"] and cyclegan_G is not None:
            t_cyc = time.time()
            patches = refine_batch_cyclegan(patches, cyclegan_G, device)
            cyc_time = time.time() - t_cyc
            log.debug(f"CycleGAN refinement: {cyc_time:.1f}s")

        # --- Convert to tensor ---
        pos_tensor = patches_to_tensor(patches, PATCH_SIZE)
        log.debug(f"Positive tensor shape: {pos_tensor.shape}")

        # --- Load W0 FRESH (always reset — never carry across trials) ---
        model = load_w0(paths.w0_path, device, freeze_backbone=True)
        log.debug("Loaded W0 fresh (backbone frozen)")

        # --- Fine-tune head ---
        t_ft = time.time()
        losses = finetune_head(model, pos_tensor, neg_patches, device)
        ft_time = time.time() - t_ft
        log.debug(f"Fine-tune: {FT_STEPS} steps in {ft_time:.1f}s")
        log.debug(f"  Loss: start={losses[0]:.4f} end={losses[-1]:.4f} "
                   f"min={min(losses):.4f} mean={np.mean(losses):.4f}")

        # --- Evaluate on val (full detector, Option A) ---
        t_eval = time.time()
        metrics = evaluate_on_val(model, device, paths)
        eval_time = time.time() - t_eval
        f1 = metrics["f1"]
        precision = metrics["precision"]
        recall = metrics["recall"]
        mae = metrics["mae"]

        log.debug(f"Detection eval: {eval_time:.1f}s")
        log.info(f"  Results: P={precision:.4f} R={recall:.4f} F1={f1:.4f} MAE={mae:.1f}")
        log.info(f"  TP={metrics['tp']} FP={metrics['fp']} FN={metrics['fn']}")

        # Per-image breakdown (DEBUG level — only in log file)
        for img_info in metrics.get("per_image", []):
            log.debug(f"    {img_info['stem']}: GT={img_info['gt_count']} "
                       f"Det={img_info['det_count']} TP={img_info['tp']} "
                       f"FP={img_info['fp']} FN={img_info['fn']}")

        # --- Discard model ---
        del model
        from device_utils import empty_torch_cache

        empty_torch_cache(device)

        elapsed = time.time() - t0
        delta = f1 - baseline_f1
        delta_str = f"{delta:+.4f}" if delta != 0 else "+0.0000"

        log.info(f"  Timing: gen={gen_time:.1f}s ft={ft_time:.1f}s eval={eval_time:.1f}s total={elapsed:.0f}s")
        log.info(f"  vs baseline: {delta_str}  (baseline={baseline_f1:.4f})")
        log.info(f"TRIAL {trial.number} END — F1={f1:.4f}")

        # Store extra metrics as trial user attributes for CSV export
        trial.set_user_attr("precision", precision)
        trial.set_user_attr("recall", recall)
        trial.set_user_attr("mae", mae)
        trial.set_user_attr("tp", metrics["tp"])
        trial.set_user_attr("fp", metrics["fp"])
        trial.set_user_attr("fn", metrics["fn"])
        trial.set_user_attr("elapsed_s", elapsed)
        trial.set_user_attr("loss_start", losses[0])
        trial.set_user_attr("loss_end", losses[-1])
        trial.set_user_attr("delta_vs_baseline", delta)

        return f1

    return objective


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    global log

    parser = argparse.ArgumentParser(description="Stage 1: Search synthetic config space")
    from device_utils import add_device_argument, resolve_device

    parser.add_argument("--n_trials", type=int, default=100, help="Number of Optuna trials to run")
    parser.add_argument(
        "--cutouts_dir",
        type=str,
        default="walnut_cutouts",
        help="Directory of RGBA walnut cutout PNGs (from extract_walnuts.py)",
    )
    parser.add_argument(
        "--neg_dir",
        type=str,
        default="train/negative",
        help="Directory of negative background patches (32×32 PNGs)",
    )
    parser.add_argument(
        "--w0_path",
        type=str,
        default="models/walnut_classifier_phase1.pth",
        help="Base classifier checkpoint (W0) to fine-tune each trial",
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default="output",
        help="Full images for val detection evaluation",
    )
    parser.add_argument(
        "--annotation_dir",
        type=str,
        default="output/annotations",
        help="Annotation .txt files for val evaluation",
    )
    parser.add_argument(
        "--split_file",
        type=str,
        default="output/dataset/split.json",
        help="split.json with train/val/test stems",
    )
    parser.add_argument(
        "--cyclegan_ckpt",
        type=str,
        default="cyclegan_models/G_AB_best.pth",
        help="Optional CycleGAN generator checkpoint (skipped if missing)",
    )
    add_device_argument(
        parser,
        default="auto",
        help_text="Device for trials: auto (CUDA preferred), or cpu/cuda/mps",
    )
    parser.add_argument("--resume", action="store_true", help="Resume an existing Optuna study from study.db")
    parser.add_argument(
        "--study_name",
        default="walnut_synthetic_search",
        help="Optuna study name (SQLite DB stored under optuna_results/)",
    )
    args = parser.parse_args()
    args.device = resolve_device(args.device)

    paths = RunPaths(
        w0_path=_resolve_path(args.w0_path),
        cutouts_dir=_resolve_path(args.cutouts_dir),
        neg_dir=_resolve_path(args.neg_dir),
        image_dir=_resolve_path(args.image_dir),
        annotation_dir=_resolve_path(args.annotation_dir),
        split_file=_resolve_path(args.split_file),
        cyclegan_ckpt=_resolve_path(args.cyclegan_ckpt),
    )
    _require_file(paths.w0_path, "W0 checkpoint")
    _require_dir(paths.cutouts_dir, "Cutouts directory (--cutouts_dir)")
    _require_dir(paths.neg_dir, "Negative patches directory (--neg_dir)")
    _require_file(paths.split_file, "Split file")
    _require_dir(paths.image_dir, "Image directory")
    _require_dir(paths.annotation_dir, "Annotation directory")

    STUDY_DIR.mkdir(parents=True, exist_ok=True)
    db_path = STUDY_DIR / "study.db"

    # --- Initialize logger ---
    log = setup_logger(LOG_DIR)

    # Suppress Optuna's own logging to keep our log clean
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    log.info("=" * 70)
    log.info("  Stage 1: Synthetic Config Search")
    log.info("=" * 70)
    log.info(f"  W0:           {paths.w0_path}")
    log.info(f"  Cutouts dir:  {paths.cutouts_dir}")
    log.info(f"  Negatives:    {paths.neg_dir}")
    log.info(f"  CycleGAN:     {paths.cyclegan_ckpt}")
    log.info(f"  Images:       {paths.image_dir}")
    log.info(f"  Annotations:  {paths.annotation_dir}")
    log.info(f"  Split file:   {paths.split_file}")
    log.info(f"  Detection:    stride={DET_STRIDE} thresh={DET_THRESH} "
             f"eps={DET_EPS} nms={DET_NMS} lms={DET_LMS} match_r={DET_MATCH_RADIUS}")
    log.info(f"  Fine-tune:    lr={FT_LR} steps={FT_STEPS} batch={FT_BATCH}")
    log.info(f"  Synthetic:    {NUM_SYNTHETIC} patches per trial")
    log.info(f"  Trials:       {args.n_trials}")
    log.info(f"  Device:       {args.device}")
    log.info(f"  Study DB:     {db_path}")
    log.info("=" * 70)

    # --- Preload everything ---
    log.info("Loading assets...")
    cutouts = load_cutouts(paths.cutouts_dir)
    log.info(f"  Cutouts: {len(cutouts)}")
    if not cutouts:
        raise SystemExit(
            f"No cutout PNGs in {paths.cutouts_dir}. "
            "Run: python extract_walnuts.py --train_dir <dataset>/train --output_dir <parent>"
        )

    neg_patches = load_negative_patches(paths.neg_dir, PATCH_SIZE)
    log.info(f"  Negatives: {len(neg_patches)}")
    if not neg_patches:
        raise SystemExit(
            f"No negative PNGs in {paths.neg_dir}. "
            "Build dataset first: python build_annotations11_10_dataset.py ..."
        )

    cyclegan_G = None
    if paths.cyclegan_ckpt.exists():
        cyclegan_G = load_cyclegan(paths.cyclegan_ckpt, args.device)
        log.info(f"  CycleGAN: loaded")
    else:
        log.info(f"  CycleGAN: not found, will skip refinement")

    # --- Baseline: W0 without any fine-tuning ---
    log.info("Computing baseline (W0 unmodified on val)...")
    baseline_model = load_w0(paths.w0_path, args.device, freeze_backbone=False)
    baseline_metrics = evaluate_on_val(baseline_model, args.device, paths)
    del baseline_model
    baseline_f1 = baseline_metrics["f1"]
    log.info(f"  BASELINE: P={baseline_metrics['precision']:.4f} "
             f"R={baseline_metrics['recall']:.4f} F1={baseline_f1:.4f} "
             f"MAE={baseline_metrics['mae']:.1f}")
    log.info(f"  TP={baseline_metrics['tp']} FP={baseline_metrics['fp']} "
             f"FN={baseline_metrics['fn']}")
    for img_info in baseline_metrics.get("per_image", []):
        log.debug(f"    {img_info['stem']}: GT={img_info['gt_count']} "
                   f"Det={img_info['det_count']} TP={img_info['tp']} "
                   f"FP={img_info['fp']} FN={img_info['fn']}")

    # --- Create Optuna study ---
    storage = f"sqlite:///{db_path}"
    if args.resume:
        study = optuna.load_study(
            study_name=args.study_name, storage=storage,
        )
        log.info(f"Resumed study: {len(study.trials)} existing trials")
    else:
        study = optuna.create_study(
            study_name=args.study_name, storage=storage,
            direction="maximize",
            load_if_exists=True,
        )

    objective = create_objective(
        cutouts, neg_patches, cyclegan_G, args.device, baseline_f1, paths
    )

    log.info(f"\nStarting search ({args.n_trials} trials)...\n")
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=False)

    # --- Report results ---
    log.info("")
    log.info("=" * 70)
    log.info("  SEARCH COMPLETE")
    log.info("=" * 70)
    log.info(f"  Baseline F1 (W0):  {baseline_f1:.4f}")
    log.info(f"  Best trial F1:     {study.best_value:.4f}")
    log.info(f"  Improvement:       {study.best_value - baseline_f1:+.4f}")
    log.info(f"  Total trials:      {len(study.trials)}")
    log.info("")
    log.info("  Best config:")
    for k, v in study.best_params.items():
        log.info(f"    {k}: {v}")

    # Save best config
    best_config_path = STUDY_DIR / "best_config.json"
    with open(best_config_path, "w") as f:
        json.dump({
            "baseline_f1": baseline_f1,
            "best_f1": study.best_value,
            "best_params": study.best_params,
            "best_trial_number": study.best_trial.number,
            "n_trials": len(study.trials),
            "detection_params": {
                "stride": DET_STRIDE, "threshold": DET_THRESH,
                "cluster_eps": DET_EPS, "nms_radius": DET_NMS,
                "local_max_size": DET_LMS, "match_radius": DET_MATCH_RADIUS,
            },
            "finetune_params": {
                "lr": FT_LR, "steps": FT_STEPS, "batch_size": FT_BATCH,
                "num_synthetic": NUM_SYNTHETIC,
            },
        }, f, indent=2)
    log.info(f"  Saved: {best_config_path}")

    # Save all trials as CSV
    trials_path = STUDY_DIR / "trial_log.csv"
    try:
        df = study.trials_dataframe()
        df.to_csv(trials_path, index=False)
        log.info(f"  Saved: {trials_path}")
    except ImportError:
        log.warning("  pandas not installed — skipping CSV export")

    # Top 10 leaderboard
    header = (f"  {'#':>3}  {'F1':>7}  {'Prec':>6}  {'Rec':>6}  {'MAE':>5}  "
              f"{'dF1':>7}  {'CycleGAN':>8}  {'Scale':>12}  "
              f"{'Blur':>5}  {'Bright':>7}  {'Green':>6}  {'Time':>5}")
    log.info(f"\n  Top 10 configs:")
    log.info(header)
    log.info("  " + "-" * 100)
    completed_trials = [t for t in study.trials if t.value is not None]
    sorted_trials = sorted(completed_trials, key=lambda t: t.value, reverse=True)
    for t in sorted_trials[:10]:
        p = t.params
        ua = t.user_attrs
        f1 = t.value or 0.0
        delta = ua.get("delta_vs_baseline", f1 - baseline_f1)
        log.info(
            f"  {t.number:>3}  {f1:>7.4f}  "
            f"{ua.get('precision',0):>6.3f}  {ua.get('recall',0):>6.3f}  "
            f"{ua.get('mae',0):>5.1f}  {delta:>+7.4f}  "
            f"{str(p.get('use_cyclegan','')):>8}  "
            f"[{p.get('cutout_scale_min',0):.2f},{p.get('cutout_scale_max',0):.2f}]  "
            f"{p.get('blur_sigma',0):>5.1f}  "
            f"{p.get('brightness_delta',0):>7.3f}  "
            f"{p.get('green_ratio_thresh',0):>6.2f}  "
            f"{ua.get('elapsed_s',0):>5.0f}s"
        )

    # Also save a detailed JSON with all trials
    all_trials_path = STUDY_DIR / "all_trials.json"
    all_trials_data = []
    for t in sorted_trials:
        all_trials_data.append({
            "trial": t.number,
            "f1": t.value,
            "params": t.params,
            "user_attrs": t.user_attrs,
        })
    with open(all_trials_path, "w") as f:
        json.dump(all_trials_data, f, indent=2)
    log.info(f"  Saved: {all_trials_path}")


if __name__ == "__main__":
    main()
