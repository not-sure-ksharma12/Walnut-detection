#!/usr/bin/env python3
"""
train_synthetic_then_real.py — Train classifier: synthetic pretrain → real fine-tune (Option B).

What it does:
  - Phase 1: Generate synthetic patches (best config), train on synthetic data
  - Phase 2–4: Fine-tune on real train/ patches with focal loss, hard neg/pos mining
  - Tracks val F1 on val/positive|negative; saves optuna_results/w0_option_b.pth (default)

Prerequisites:
  - python3 setup.py
  - optuna_results/best_config.json, train/positive|negative/, models_finetuned W0
  - Optional: val/positive|negative/ for validation checkpoints

How to run:
  cd Walnut-detection
  source venv/bin/activate
  python train_synthetic_then_real.py --device auto
  python train_synthetic_then_real.py --device cuda --real_patches_dir train
"""

import copy
import json
import logging
import random
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Add workspace root
WORKSPACE = Path(__file__).resolve().parent
sys.path.insert(0, str(WORKSPACE))

import optimize_synthetic_config as m
from binary_classifier import (
    WalnutClassifier,
    BinaryWalnutDataset,
    AugmentationTransform,
    calculate_class_weights,
    FocalLoss,
    mine_hard_negatives,
    mine_hard_positives,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# Phase 1: full model on synthetic (moderate LR), 5000 synthetics, epochs + focal + HN mining
NUM_SYNTHETIC = 5000
PHASE1_LR = 1e-4
PHASE1_EPOCHS = 10
PHASE1_STEPS_PER_EPOCH = 200

# Phase 2: full model unfrozen on real (lower LR), focal + HN mining
REAL_EPOCHS = 25
REAL_LR = 5e-6
REAL_BATCH = 32
PATCH_SIZE = 32
FOCAL_GAMMA = 2.0
HN_THRESHOLD = 0.5
PHASE2_HN_EPOCHS = 5

# Phase 3: head-only on real (freeze backbone), optional calibration/finalization
PHASE3_EPOCHS = 5
PHASE3_LR = 1e-5

# Phase 4: hard positive mining (optional)
HP_THRESHOLD = 0.4
HP_TOP_K = 500
HP_DUPLICATE = 3
PHASE4_EPOCHS = 3
PHASE4_LR = 5e-6


def freeze_backbone(model: WalnutClassifier) -> None:
    """Freeze all backbone blocks (head only trainable)."""
    for block in [model.block1, model.block2, model.block3, model.block4]:
        for param in block.parameters():
            param.requires_grad = False


def unfreeze_head(model: WalnutClassifier) -> None:
    """Ensure classifier head parameters are trainable."""
    for param in model.classifier.parameters():
        param.requires_grad = True


def unfreeze_backbone(model: WalnutClassifier) -> None:
    """Unfreeze all backbone blocks (full model trainable)."""
    for block in [model.block1, model.block2, model.block3, model.block4]:
        for param in block.parameters():
            param.requires_grad = True


def evaluate_val_f1(
    model: WalnutClassifier,
    val_loader: DataLoader,
    device: str,
) -> tuple:
    """Run model on val patches; return (f1, precision, recall). Handles imbalance."""
    model.eval()
    tp, fp, fn = 0, 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            out = model(images.to(device))
            pred = out.argmax(dim=1).cpu()
            for i in range(labels.size(0)):
                if labels[i].item() == 1 and pred[i].item() == 1:
                    tp += 1
                elif labels[i].item() == 1 and pred[i].item() == 0:
                    fn += 1
                elif labels[i].item() == 0 and pred[i].item() == 1:
                    fp += 1
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return f1, prec, rec


def mine_hard_neg_indices(
    model: WalnutClassifier,
    neg_tensor: torch.Tensor,
    device: str,
    threshold: float = 0.5,
    batch_size: int = 64,
) -> List[int]:
    """Return indices into neg_tensor where model predicts positive with prob > threshold."""
    model.eval()
    hard_indices: List[int] = []
    for i in range(0, neg_tensor.shape[0], batch_size):
        batch = neg_tensor[i : i + batch_size].to(device)
        with torch.no_grad():
            out = model(batch)
            probs = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
        for j, p in enumerate(probs):
            if p > threshold:
                hard_indices.append(i + j)
    return hard_indices


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Train W0 on best synthetic config, then fine-tune on real patches"
    )
    from device_utils import add_device_argument, resolve_device

    add_device_argument(parser, default="auto")
    parser.add_argument(
        "--pretrained",
        type=str,
        default=None,
        help="Optional .pth checkpoint to start from (expects model_state_dict). "
             "Useful for running only Phase 3.",
    )
    parser.add_argument(
        "--real_patches_dir",
        type=str,
        default=None,
        help="Dir with positive/ and negative/ subdirs (default: train)",
    )
    parser.add_argument(
        "--val_patches_dir",
        type=str,
        default=None,
        help="Dir with positive/ and negative/ for val F1 (default: val)",
    )
    parser.add_argument("--num_synthetic", type=int, default=NUM_SYNTHETIC,
                        help="Number of synthetic patches to generate (default 5000)")
    parser.add_argument("--phase1_lr", type=float, default=PHASE1_LR,
                        help="Phase 1 (synthetic, head only) learning rate")
    parser.add_argument("--phase1_epochs", type=int, default=PHASE1_EPOCHS,
                        help="Phase 1 number of epochs")
    parser.add_argument("--phase1_steps_per_epoch", type=int, default=PHASE1_STEPS_PER_EPOCH,
                        help="Phase 1 training steps per epoch")
    parser.add_argument(
        "--phase2_epochs",
        type=int,
        default=REAL_EPOCHS,
        help="Phase 2 epochs fine-tuning full model on real patches",
    )
    parser.add_argument("--phase2_lr", type=float, default=REAL_LR,
                        help="Phase 2 (real, full model) learning rate")
    parser.add_argument("--phase2_hn_epochs", type=int, default=PHASE2_HN_EPOCHS,
                        help="Phase 2 extra epochs after hard negative mining")
    parser.add_argument("--phase3_epochs", type=int, default=PHASE3_EPOCHS,
                        help="Phase 3 head-only epochs on real train patches (0 to disable)")
    parser.add_argument("--phase3_lr", type=float, default=PHASE3_LR,
                        help="Phase 3 (real, head only) learning rate")
    parser.add_argument("--hp_threshold", type=float, default=HP_THRESHOLD,
                        help="Hard positive mining threshold: keep positives with P(walnut) < this")
    parser.add_argument("--hp_top_k", type=int, default=HP_TOP_K,
                        help="Keep top-K hardest positives (lowest confidence)")
    parser.add_argument("--hp_duplicate", type=int, default=HP_DUPLICATE,
                        help="Duplicate each hard positive this many times in training")
    parser.add_argument("--phase4_epochs", type=int, default=PHASE4_EPOCHS,
                        help="Phase 4 epochs after hard positive mining (0 to disable)")
    parser.add_argument("--phase4_lr", type=float, default=PHASE4_LR,
                        help="Phase 4 learning rate (default trains head only)")
    parser.add_argument("--focal_gamma", type=float, default=FOCAL_GAMMA,
                        help="Focal loss gamma (both phases)")
    parser.add_argument("--hn_threshold", type=float, default=HN_THRESHOLD,
                        help="Hard negative mining confidence threshold")
    parser.add_argument("--output", type=str, default=None,
                        help="Output .pth path (default: optuna_results/w0_option_b.pth)")
    args = parser.parse_args()
    args.device = resolve_device(args.device)
    print(f"Using device: {args.device}")

    if args.real_patches_dir is None:
        args.real_patches_dir = str(WORKSPACE / "train")
    real_dir = Path(args.real_patches_dir)
    train_pos = real_dir / "positive"
    train_neg = real_dir / "negative"
    if not train_pos.exists() or not train_neg.exists():
        log.error(
            "Real patches dir must contain positive/ and negative/ subdirs. "
            "Not found under %s", real_dir
        )
        sys.exit(1)

    val_dir = Path(args.val_patches_dir) if args.val_patches_dir else WORKSPACE / "val"
    val_pos = val_dir / "positive"
    val_neg = val_dir / "negative"
    use_val = val_pos.exists() and val_neg.exists()
    if use_val:
        val_dataset = BinaryWalnutDataset(
            str(val_pos), str(val_neg),
            transform=AugmentationTransform(PATCH_SIZE, training=False),
            augment=False,
        )
        val_loader = DataLoader(val_dataset, batch_size=REAL_BATCH, shuffle=False, num_workers=0)
        log.info("Val patches: %s (%d pos, %d neg)", val_dir, len(val_dataset.positive_files), len(val_dataset.negative_files))
    else:
        val_loader = None
        log.warning("Val dir %s missing positive/ or negative/; skipping val F1 and best-model save", val_dir)

    # ---------- Initialize model (from pretrained or W0) ----------
    if args.pretrained:
        log.info("Loading pretrained checkpoint: %s", args.pretrained)
        model = WalnutClassifier(input_size=PATCH_SIZE, num_classes=2)
        try:
            ckpt = torch.load(args.pretrained, map_location="cpu", weights_only=True)
        except Exception:
            ckpt = torch.load(args.pretrained, map_location="cpu", weights_only=False)
        state = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state, strict=False)
        model.to(args.device)
    else:
        model = None  # set in Phase 1

    # ---------- Phase 1: Best synthetic config ----------
    if args.phase1_epochs > 0:
        best_config_path = m.STUDY_DIR / "best_config.json"
        if not best_config_path.exists():
            log.error("Missing %s. Run Stage 1 search first.", best_config_path)
            sys.exit(1)

        with open(best_config_path, "r") as f:
            best_cfg = json.load(f)
        best_params = best_cfg["best_params"]
        config = copy.deepcopy(best_params)
        if config["cutout_scale_min"] >= config["cutout_scale_max"]:
            config["cutout_scale_max"] = config["cutout_scale_min"] + 0.1

        log.info(
            "Phase 1: FULL W0 on best synthetic config (trial %s), %d synthetics, "
            "%d epochs × %d steps/epoch, LR=%.0e, focal gamma=%.2f, HN threshold=%.2f",
            best_cfg.get("best_trial_number", "?"),
            args.num_synthetic,
            args.phase1_epochs,
            args.phase1_steps_per_epoch,
            args.phase1_lr,
            args.focal_gamma,
            args.hn_threshold,
        )

        random.seed(0)
        np.random.seed(0)

        cutouts = m.load_cutouts(m.CUTOUTS_DIR)
        neg_patches = m.load_negative_patches(m.NEG_DIR, m.PATCH_SIZE)
        log.info("  Cutouts: %d, Negatives: %d", len(cutouts), len(neg_patches))

        cyclegan_G = None
        if m.CYCLEGAN_CKPT.exists():
            cyclegan_G = m.load_cyclegan(m.CYCLEGAN_CKPT, args.device)

        patches = m.generate_synthetic_batch(
            config, cutouts, neg_patches,
            num_samples=args.num_synthetic, patch_size=m.PATCH_SIZE,
        )
        if config.get("use_cyclegan") and cyclegan_G is not None:
            patches = m.refine_batch_cyclegan(patches, cyclegan_G, args.device)
        pos_tensor = m.patches_to_tensor(patches, m.PATCH_SIZE)
        neg_tensor = m.patches_to_tensor(neg_patches, m.PATCH_SIZE)

        n_pos = pos_tensor.shape[0]
        n_neg = neg_tensor.shape[0]
        phase1_weights = calculate_class_weights(n_pos, n_neg, args.device).to(args.device)
        criterion_phase1 = FocalLoss(gamma=args.focal_gamma, alpha=phase1_weights)

        # Phase 1 trains the entire W0 (backbone + head) on synthetic
        model = m.load_w0(m.W0_PATH, args.device, freeze_backbone=False)
        optimizer_phase1 = optim.Adam(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.phase1_lr, weight_decay=1e-4,
        )
        half_batch = m.FT_BATCH // 2

        hard_neg_indices: List[int] = []
        for epoch in range(1, args.phase1_epochs + 1):
            model.train()
            epoch_losses = []
            for step in range(args.phase1_steps_per_epoch):
                p_idx = torch.randint(0, n_pos, (half_batch,))
                if hard_neg_indices and random.random() < 0.5:
                    hn = random.choices(hard_neg_indices, k=half_batch)
                    n_idx = torch.tensor(hn, dtype=torch.long)
                else:
                    n_idx = torch.randint(0, n_neg, (half_batch,))

                x = torch.cat([pos_tensor[p_idx], neg_tensor[n_idx]], dim=0).to(args.device)
                y = torch.cat([
                    torch.ones(half_batch, dtype=torch.long),
                    torch.zeros(half_batch, dtype=torch.long),
                ]).to(args.device)
                perm = torch.randperm(x.shape[0])
                x, y = x[perm], y[perm]

                optimizer_phase1.zero_grad()
                out = model(x)
                loss = criterion_phase1(out, y)
                loss.backward()
                optimizer_phase1.step()
                epoch_losses.append(loss.item())

            avg_loss = sum(epoch_losses) / len(epoch_losses)
            log.info("  Phase 1 Epoch %d/%d  loss=%.4f", epoch, args.phase1_epochs, avg_loss)

            hard_neg_indices = mine_hard_neg_indices(
                model, neg_tensor, args.device,
                threshold=args.hn_threshold, batch_size=64,
            )
            log.info("    Hard negatives mined: %d", len(hard_neg_indices))

        log.info("  Phase 1 done.")
    else:
        log.info("Phase 1 skipped (phase1_epochs=0)")

    if model is None:
        log.error("No model initialized. Provide --pretrained or enable Phase 1.")
        sys.exit(1)

    # ---------- Phase 2: Unfreeze full model on real patches (low LR), focal loss ----------
    log.info(
        "Phase 2: full model (unfrozen) on real patches from %s, low LR=%.0e, focal gamma=%.2f",
        real_dir, args.phase2_lr, args.focal_gamma,
    )
    unfreeze_backbone(model)

    train_dataset = BinaryWalnutDataset(
        str(train_pos), str(train_neg),
        transform=AugmentationTransform(PATCH_SIZE, training=True),
        augment=True,
    )
    n_pos = len(train_dataset.positive_files)
    n_neg = len(train_dataset.negative_files)
    class_weights = calculate_class_weights(n_pos, n_neg, args.device).to(args.device)
    criterion_phase2 = FocalLoss(gamma=args.focal_gamma, alpha=class_weights)

    train_loader = DataLoader(
        train_dataset,
        batch_size=REAL_BATCH,
        shuffle=True,
        num_workers=0,
    )

    optimizer_phase2 = optim.Adam(model.parameters(), lr=args.phase2_lr, weight_decay=1e-4)
    model.train()

    best_val_f1 = -1.0
    best_state = None

    for epoch in range(1, args.phase2_epochs + 1):
        total_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images = images.to(args.device)
            labels = labels.to(args.device)
            optimizer_phase2.zero_grad()
            out = model(images)
            loss = criterion_phase2(out, labels)
            loss.backward()
            optimizer_phase2.step()
            total_loss += loss.item()
            pred = out.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
        avg_loss = total_loss / len(train_loader)
        acc = 100.0 * correct / total
        log.info("  Phase 2 Epoch %d/%d  loss=%.4f  acc=%.2f%%", epoch, args.phase2_epochs, avg_loss, acc)

        if use_val and val_loader is not None:
            val_f1, val_prec, val_rec = evaluate_val_f1(model, val_loader, args.device)
            log.info("    Val F1=%.4f  P=%.4f  R=%.4f", val_f1, val_prec, val_rec)
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_state = copy.deepcopy(model.state_dict())
                val_dir.mkdir(parents=True, exist_ok=True)
                best_path = val_dir / "best_model.pth"
                torch.save({"model_state_dict": best_state}, best_path)
                log.info("    Best val F1 -> saved %s", best_path)

    # ---------- Phase 2 hard negative mining: mine then train extra epochs ----------
    if args.phase2_hn_epochs > 0:
        log.info("Phase 2 hard negative mining (threshold=%.2f)...", args.hn_threshold)
        mining_transform = AugmentationTransform(PATCH_SIZE, training=False)
        hard_negatives = mine_hard_negatives(
            model, str(train_neg), mining_transform, args.device,
            threshold=args.hn_threshold, batch_size=REAL_BATCH,
        )
        if hard_negatives:
            hard_neg_paths = [str(p) for p, _ in hard_negatives]
            neg_paths = list(Path(train_neg).glob("*.png"))
            combined_neg_files = [str(p) for p in neg_paths] + hard_neg_paths
            n_neg_combined = len(combined_neg_files)
            log.info("  Adding %d hard negatives; training %d more epochs on pos + combined negs",
                    len(hard_neg_paths), args.phase2_hn_epochs)
            pos_files = [str(p) for p in Path(train_pos).glob("*.png")]
            dataset_hn = BinaryWalnutDataset(
                str(train_pos), str(train_neg),
                transform=AugmentationTransform(PATCH_SIZE, training=True),
                augment=True,
                positive_files=pos_files,
                negative_files=combined_neg_files,
            )
            weights_hn = calculate_class_weights(n_pos, n_neg_combined, args.device).to(args.device)
            criterion_hn = FocalLoss(gamma=args.focal_gamma, alpha=weights_hn)
            loader_hn = DataLoader(dataset_hn, batch_size=REAL_BATCH, shuffle=True, num_workers=0)
            for epoch in range(1, args.phase2_hn_epochs + 1):
                total_loss = 0.0
                correct = 0
                total = 0
                for images, labels in loader_hn:
                    images = images.to(args.device)
                    labels = labels.to(args.device)
                    optimizer_phase2.zero_grad()
                    out = model(images)
                    loss = criterion_hn(out, labels)
                    loss.backward()
                    optimizer_phase2.step()
                    total_loss += loss.item()
                    pred = out.argmax(dim=1)
                    correct += (pred == labels).sum().item()
                    total += labels.size(0)
                avg_loss = total_loss / len(loader_hn)
                acc = 100.0 * correct / total
                log.info("  Phase 2 HN Epoch %d/%d  loss=%.4f  acc=%.2f%%",
                        epoch, args.phase2_hn_epochs, avg_loss, acc)
                if use_val and val_loader is not None:
                    val_f1, val_prec, val_rec = evaluate_val_f1(model, val_loader, args.device)
                    log.info("    Val F1=%.4f  P=%.4f  R=%.4f", val_f1, val_prec, val_rec)
                    if val_f1 > best_val_f1:
                        best_val_f1 = val_f1
                        best_state = copy.deepcopy(model.state_dict())
                        val_dir.mkdir(parents=True, exist_ok=True)
                        best_path = val_dir / "best_model.pth"
                        torch.save({"model_state_dict": best_state}, best_path)
                        log.info("    Best val F1 -> saved %s", best_path)
        else:
            log.info("  No hard negatives above threshold; skipping HN epochs.")

    # ---------- Phase 3: Freeze backbone, train head only on real patches ----------
    if args.phase3_epochs > 0:
        log.info(
            "Phase 3: head-only on real patches (freeze backbone), epochs=%d, LR=%.0e",
            args.phase3_epochs, args.phase3_lr,
        )
        freeze_backbone(model)
        unfreeze_head(model)

        head_params = [p for p in model.parameters() if p.requires_grad]
        optimizer_phase3 = optim.Adam(head_params, lr=args.phase3_lr, weight_decay=1e-4)

        for epoch in range(1, args.phase3_epochs + 1):
            total_loss = 0.0
            correct = 0
            total = 0
            for images, labels in train_loader:
                images = images.to(args.device)
                labels = labels.to(args.device)
                optimizer_phase3.zero_grad()
                out = model(images)
                loss = criterion_phase2(out, labels)
                loss.backward()
                optimizer_phase3.step()
                total_loss += loss.item()
                pred = out.argmax(dim=1)
                correct += (pred == labels).sum().item()
                total += labels.size(0)
            avg_loss = total_loss / len(train_loader)
            acc = 100.0 * correct / total
            log.info("  Phase 3 Epoch %d/%d  loss=%.4f  acc=%.2f%%", epoch, args.phase3_epochs, avg_loss, acc)

            if use_val and val_loader is not None:
                val_f1, val_prec, val_rec = evaluate_val_f1(model, val_loader, args.device)
                log.info("    Val F1=%.4f  P=%.4f  R=%.4f", val_f1, val_prec, val_rec)
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    best_state = copy.deepcopy(model.state_dict())
                    val_dir.mkdir(parents=True, exist_ok=True)
                    best_path = val_dir / "best_model.pth"
                    torch.save({"model_state_dict": best_state}, best_path)
                    log.info("    Best val F1 -> saved %s", best_path)

    # ---------- Phase 4: Hard positive mining + oversample ----------
    if args.phase4_epochs > 0:
        log.info(
            "Phase 4: hard positive mining (thr=%.2f, top_k=%d, dup=%d) + head-only epochs=%d, LR=%.0e",
            args.hp_threshold, args.hp_top_k, args.hp_duplicate, args.phase4_epochs, args.phase4_lr,
        )
        # Mine hard positives from real training positives
        mining_transform = AugmentationTransform(PATCH_SIZE, training=False)
        hard_pos = mine_hard_positives(
            model,
            str(train_pos),
            mining_transform,
            args.device,
            threshold=args.hp_threshold,
            batch_size=REAL_BATCH,
        )
        hard_pos = hard_pos[: max(0, args.hp_top_k)]
        if hard_pos:
            hard_pos_paths = [str(p) for p, _ in hard_pos]
            # Oversample hard positives by duplication
            oversampled_hard_pos = []
            for _ in range(max(1, args.hp_duplicate)):
                oversampled_hard_pos.extend(hard_pos_paths)

            base_pos_files = [str(p) for p in Path(train_pos).glob("*.png")]
            base_neg_files = [str(p) for p in Path(train_neg).glob("*.png")]
            combined_pos_files = base_pos_files + oversampled_hard_pos

            # Train head-only for stability (focus on decision boundary)
            freeze_backbone(model)
            unfreeze_head(model)
            trainable = [p for p in model.parameters() if p.requires_grad]
            optimizer_phase4 = optim.Adam(trainable, lr=args.phase4_lr, weight_decay=1e-4)

            ds_hp = BinaryWalnutDataset(
                str(train_pos), str(train_neg),
                transform=AugmentationTransform(PATCH_SIZE, training=True),
                augment=True,
                positive_files=combined_pos_files,
                negative_files=base_neg_files,
            )
            # Recompute weights based on oversampled counts
            cw_hp = calculate_class_weights(len(combined_pos_files), len(base_neg_files), args.device).to(args.device)
            crit_hp = FocalLoss(gamma=args.focal_gamma, alpha=cw_hp)
            loader_hp = DataLoader(ds_hp, batch_size=REAL_BATCH, shuffle=True, num_workers=0)

            for epoch in range(1, args.phase4_epochs + 1):
                total_loss = 0.0
                correct = 0
                total = 0
                for images, labels in loader_hp:
                    images = images.to(args.device)
                    labels = labels.to(args.device)
                    optimizer_phase4.zero_grad()
                    out = model(images)
                    loss = crit_hp(out, labels)
                    loss.backward()
                    optimizer_phase4.step()
                    total_loss += loss.item()
                    pred = out.argmax(dim=1)
                    correct += (pred == labels).sum().item()
                    total += labels.size(0)
                avg_loss = total_loss / len(loader_hp)
                acc = 100.0 * correct / total
                log.info("  Phase 4 Epoch %d/%d  loss=%.4f  acc=%.2f%%", epoch, args.phase4_epochs, avg_loss, acc)

                if use_val and val_loader is not None:
                    val_f1, val_prec, val_rec = evaluate_val_f1(model, val_loader, args.device)
                    log.info("    Val F1=%.4f  P=%.4f  R=%.4f", val_f1, val_prec, val_rec)
                    if val_f1 > best_val_f1:
                        best_val_f1 = val_f1
                        best_state = copy.deepcopy(model.state_dict())
                        val_dir.mkdir(parents=True, exist_ok=True)
                        best_path = val_dir / "best_model.pth"
                        torch.save({"model_state_dict": best_state}, best_path)
                        log.info("    Best val F1 -> saved %s", best_path)
        else:
            log.info("  No hard positives found below threshold; skipping Phase 4.")

    model.eval()

    # ---------- Save ----------
    if best_state is not None:
        model.load_state_dict(best_state)
        log.info("Loaded best checkpoint (val F1=%.4f) for final save", best_val_f1)
    m.STUDY_DIR.mkdir(parents=True, exist_ok=True)
    out_path = args.output
    if out_path is None:
        out_path = str(m.STUDY_DIR / "w0_option_b.pth")
    torch.save({"model_state_dict": model.state_dict()}, out_path)
    log.info("Saved: %s", out_path)
    if use_val and best_state is not None:
        log.info("Best by val F1 also saved to: %s", val_dir / "best_model.pth")
    log.info("Run evaluate_walnut_annotations.py with --model_path %s (same stride/threshold as before)", out_path)


if __name__ == "__main__":
    main()
