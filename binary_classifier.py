#!/usr/bin/env python3
"""
binary_classifier.py — Train a 32×32 CNN to classify walnut vs background patches.

What it does:
  - Trains WalnutClassifier on dataset_dir/positive and dataset_dir/negative
  - Applies augmentation, class weights, and optional hard-negative mining
  - Saves best checkpoint and training plots under --output_dir

Note: Default train/val split is at patch level (not image level); see --help for fold options.

Prerequisites:
  - python3 setup.py
  - Dataset with positive/ and negative/ subfolders (or all_patches/ + fold file)

How to run:
  cd Walnut-detection
  source venv/bin/activate
  python binary_classifier.py --dataset_dir train --output_dir models --epochs 50
  python binary_classifier.py --dataset_dir train --mine_hard_negatives
  python binary_classifier.py --help
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as Fnn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
import numpy as np
import cv2
from pathlib import Path
from typing import Tuple, List, Dict
import json
import re
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import random

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

def _stem_from_patch_path(path: Path) -> str:
    """From patch filename like DJI_xxx_q01_p0062.png or _n0001.png return stem DJI_xxx_q01."""
    s = path.stem
    m = re.match(r"^(.+)_[pn]\d+$", s)
    return m.group(1) if m else s


class BinaryWalnutDataset(Dataset):
    """Dataset for binary walnut classification"""
    
    def __init__(self, positive_dir: str, negative_dir: str, transform=None, augment=True,
                 positive_files=None, negative_files=None):
        self.positive_dir = Path(positive_dir)
        self.negative_dir = Path(negative_dir)
        self.transform = transform
        self.augment = augment
        
        # Get all image files (either from dirs or explicit lists)
        if positive_files is not None and negative_files is not None:
            self.positive_files = [Path(p) for p in positive_files]
            self.negative_files = [Path(p) for p in negative_files]
        else:
            self.positive_files = list(self.positive_dir.glob("*.png"))
            self.negative_files = list(self.negative_dir.glob("*.png"))
        
        print(f"Found {len(self.positive_files)} positive samples")
        print(f"Found {len(self.negative_files)} negative samples")
        
        # Create labels
        self.images = []
        self.labels = []
        
        # Add positive samples (label = 1)
        for img_path in self.positive_files:
            self.images.append(str(img_path))
            self.labels.append(1)
        
        # Add negative samples (label = 0)
        for img_path in self.negative_files:
            self.images.append(str(img_path))
            self.labels.append(0)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.images[idx]
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Could not load image: {img_path}")
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        if self.transform:
            img = self.transform(img)
        
        # Get label
        label = self.labels[idx]
        
        return img, label

class AugmentationTransform:
    """Custom augmentation for walnut classification"""
    
    def __init__(self, patch_size: int = 32, training: bool = True):
        self.patch_size = patch_size
        self.training = training
        
        # Base transforms
        self.base_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((patch_size, patch_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Augmentation transforms
        if training:
            self.augment_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((patch_size, patch_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.3),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.augment_transform = self.base_transform
    
    def __call__(self, img):
        if self.training and random.random() < 0.7:  # 70% chance of augmentation
            return self.augment_transform(img)
        else:
            return self.base_transform(img)

class WalnutClassifier(nn.Module):
    """CNN for binary walnut classification"""
    
    def __init__(self, input_size: int = 32, num_classes: int = 2, dropout_rate: float = 0.5):
        super(WalnutClassifier, self).__init__()
        
        self.input_size = input_size
        self.num_classes = num_classes
        
        # Feature extraction layers
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 32x32 -> 16x16
            
            # Block 2
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 16x16 -> 8x8
            
            # Block 3
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 8x8 -> 4x4
            
            # Block 4
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((2, 2))  # 2x2
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(256 * 2 * 2, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(128, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def calculate_class_weights(num_positive: int, num_negative: int, device: str = 'cpu'):
    """Calculate weights to handle class imbalance.
    We create tensors on CPU to avoid MPS backend issues on some macOS versions;
    callers move them to device with .to(device).
    """
    total = num_positive + num_negative
    if num_positive == 0 or num_negative == 0:
        return torch.tensor([1.0, 1.0], device='cpu')
    weight_positive = total / (2 * num_positive)
    weight_negative = total / (2 * num_negative)
    return torch.tensor([weight_negative, weight_positive], device='cpu')


class FocalLoss(nn.Module):
    """Focal loss for multi-class classification (here used for 2-class walnut vs background).
    
    This loss down-weights easy examples and focuses training on hard, misclassified samples.
    It is especially useful for imbalanced datasets.
    """

    def __init__(self, gamma: float = 2.0, alpha: torch.Tensor | None = None,
                 reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        # alpha is expected to be a tensor of per-class weights [w_neg, w_pos]
        self.alpha = alpha

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Logits of shape (N, C)
            targets: Ground-truth class indices of shape (N,)
        """
        # Log-softmax over classes
        logpt_all = Fnn.log_softmax(inputs, dim=1)
        pt_all = torch.exp(logpt_all)

        # Select the probability and log-probability for the true class
        logpt = logpt_all.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = pt_all.gather(1, targets.unsqueeze(1)).squeeze(1)

        # Class weighting (alpha) if provided
        if self.alpha is not None:
            alpha = self.alpha.to(inputs.device)
            at = alpha.gather(0, targets)
        else:
            at = 1.0

        loss = -at * (1.0 - pt) ** self.gamma * logpt

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss

class BinaryTrainer:
    """Trainer for binary walnut classification"""
    
    def __init__(self, model: WalnutClassifier, device: str, learning_rate: float = 0.001,
                 class_weights: torch.Tensor = None, loss_type: str = "ce",
                 focal_gamma: float = 2.0):
        self.model = model.to(device)
        self.device = device
        self.learning_rate = learning_rate
        
        # Loss and optimizer
        if loss_type == "focal":
            self.criterion = FocalLoss(gamma=focal_gamma, alpha=class_weights)
            if class_weights is not None:
                print(
                    f"📊 Using Focal Loss with class weights: "
                    f"negative={class_weights[0]:.3f}, positive={class_weights[1]:.3f}, "
                    f"gamma={focal_gamma:.2f}"
                )
            else:
                print(f"📊 Using Focal Loss without explicit class weights, gamma={focal_gamma:.2f}")
        else:
            if class_weights is not None:
                self.criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
                print(
                    f"📊 Using CrossEntropyLoss with class weights: "
                    f"negative={class_weights[0]:.3f}, positive={class_weights[1]:.3f}"
                )
            else:
                self.criterion = nn.CrossEntropyLoss()
                print("📊 Using standard CrossEntropyLoss (no class weights)")
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.best_val_acc = 0.0
        self.best_precision = 0.0
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in tqdm(train_loader, desc="Training"):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100 * correct / total
        
        return avg_loss, accuracy
    
    def validate_epoch(self, val_loader: DataLoader) -> Tuple[float, float, Dict]:
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validating"):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                # Statistics
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Store for metrics
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                probabilities = torch.softmax(outputs, dim=1)
                all_probabilities.extend(probabilities[:, 1].cpu().numpy())  # Probability of class 1 (walnut)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100 * correct / total
        
        # Calculate additional metrics
        metrics = {
            'accuracy': accuracy_score(all_labels, all_predictions),
            'precision': precision_score(all_labels, all_predictions, average='binary'),
            'recall': recall_score(all_labels, all_predictions, average='binary'),
            'f1': f1_score(all_labels, all_predictions, average='binary'),
            'auc': roc_auc_score(all_labels, all_probabilities)
        }
        
        return avg_loss, accuracy, metrics
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
              num_epochs: int, save_path: str = "walnut_classifier.pth"):
        """Train the model"""
        print(f"🚀 Starting training for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            
            # Validate
            val_loss, val_acc, metrics = self.validate_epoch(val_loader)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Print metrics
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print(f"Precision: {metrics['precision']:.3f}, Recall: {metrics['recall']:.3f}")
            print(f"F1: {metrics['f1']:.3f}, AUC: {metrics['auc']:.3f}")
            print(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Track best validation accuracy
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
            
            # Save best model based on precision
            if metrics['precision'] > self.best_precision:
                self.best_precision = metrics['precision']
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'epoch': epoch,
                    'val_acc': val_acc,
                    'precision': metrics['precision'],
                    'recall': metrics['recall'],
                    'f1': metrics['f1'],
                    'metrics': metrics
                }, save_path)
                print(f"💾 Saved best model (Precision: {metrics['precision']:.3f}, Val Acc: {val_acc:.2f}%)")
        
        print(f"\n✅ Training completed!")
        print(f"   Best validation accuracy: {self.best_val_acc:.2f}%")
        print(f"   Best precision: {self.best_precision:.3f}")
        return self.train_losses, self.val_losses, self.train_accuracies, self.val_accuracies

def mine_hard_negatives(model: WalnutClassifier, negative_dir: str, transform, 
                        device: str, threshold: float = 0.5, batch_size: int = 32,
                        negative_files: List = None):
    """Find negative patches the model incorrectly classifies as positive (hard negatives)
    
    Args:
        model: Trained model to use for mining
        negative_dir: Directory containing negative patches
        transform: Transform to apply to images
        device: Device to run inference on
        threshold: Confidence threshold above which to consider a false positive
        batch_size: Batch size for inference
        
    Returns:
        List of tuples: [(path, confidence_score), ...] sorted by confidence (highest first)
    """
    model.eval()
    hard_negatives = []
    
    if negative_files is not None:
        negative_files = list(negative_files)
    else:
        negative_path = Path(negative_dir)
        negative_files = list(negative_path.glob("*.png"))
    
    print(f"🔍 Mining hard negatives from {len(negative_files)} negative patches...")
    print(f"   Threshold: {threshold:.2f}")
    
    # Process in batches for efficiency
    for i in tqdm(range(0, len(negative_files), batch_size), desc="Mining"):
        batch_files = negative_files[i:i+batch_size]
        batch_tensors = []
        batch_paths = []
        
        for path in batch_files:
            try:
                img = cv2.imread(str(path))
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_tensor = transform(img)
                batch_tensors.append(img_tensor)
                batch_paths.append(path)
            except Exception as e:
                print(f"Warning: Could not load {path}: {e}")
                continue
        
        if len(batch_tensors) == 0:
            continue
        
        # Stack and move to device
        batch_tensor = torch.stack(batch_tensors).to(device)
        
        # Forward pass
        with torch.no_grad():
            outputs = model(batch_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            walnut_probs = probabilities[:, 1].cpu().numpy()  # P(walnut)
        
        # Find hard negatives (high confidence false positives)
        for j, prob in enumerate(walnut_probs):
            if prob > threshold:
                hard_negatives.append((batch_paths[j], float(prob)))
    
    # Sort by confidence (most confident mistakes first)
    hard_negatives.sort(key=lambda x: x[1], reverse=True)
    
    print(f"✅ Found {len(hard_negatives)} hard negatives (out of {len(negative_files)} total)")
    if len(hard_negatives) > 0:
        print(f"   Confidence range: {hard_negatives[-1][1]:.3f} - {hard_negatives[0][1]:.3f}")
    
    return hard_negatives


def mine_hard_positives(
    model: WalnutClassifier,
    positive_dir: str,
    transform,
    device: str,
    threshold: float = 0.4,
    batch_size: int = 32,
    positive_files: List = None,
):
    """Find positive patches the model is uncertain about (hard positives).

    Scores positives with P(walnut). Hard positives have probability below threshold.
    Returned list is sorted ascending (hardest first).
    """
    model.eval()
    hard_positives = []

    if positive_files is not None:
        positive_files = list(positive_files)
    else:
        positive_path = Path(positive_dir)
        positive_files = list(positive_path.glob("*.png"))

    print(f"🔍 Mining hard positives from {len(positive_files)} positive patches...")
    print(f"   Threshold: {threshold:.2f}  (keep positives with P(walnut) < threshold)")

    for i in tqdm(range(0, len(positive_files), batch_size), desc="Mining"):
        batch_files = positive_files[i:i + batch_size]
        batch_tensors = []
        batch_paths = []

        for path in batch_files:
            try:
                img = cv2.imread(str(path))
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_tensor = transform(img)
                batch_tensors.append(img_tensor)
                batch_paths.append(path)
            except Exception as e:
                print(f"Warning: Could not load {path}: {e}")
                continue

        if len(batch_tensors) == 0:
            continue

        batch_tensor = torch.stack(batch_tensors).to(device)

        with torch.no_grad():
            outputs = model(batch_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            walnut_probs = probabilities[:, 1].cpu().numpy()

        for j, prob in enumerate(walnut_probs):
            if prob < threshold:
                hard_positives.append((batch_paths[j], float(prob)))

    hard_positives.sort(key=lambda x: x[1])

    print(f"✅ Found {len(hard_positives)} hard positives (out of {len(positive_files)} total)")
    if len(hard_positives) > 0:
        print(f"   Confidence range: {hard_positives[0][1]:.3f} - {hard_positives[-1][1]:.3f}")

    return hard_positives


def create_data_loaders(dataset_dir: str, batch_size: int = 32, 
                       patch_size: int = 32, val_split: float = 0.2,
                       device: str = 'cpu', fold_file: str = None,
                       extra_negative_dir: str = None) -> Tuple[DataLoader, DataLoader, torch.Tensor]:
    """Create training and validation data loaders.

    Supports:
    0) all_patches + fold_file (CV): dataset_dir/all_patches/positive|negative, filter by fold JSON.
    1) Train/val/test layout (image-level split, no leakage):
       dataset_dir/train/positive/, train/negative/
       dataset_dir/val/positive/, val/negative/
    2) Legacy layout:
       dataset_dir/positive/, negative/
       dataset_dir/real_positive/, real_negative/ (optional val)

    Returns:
        train_loader, val_loader, class_weights
    """
    dataset_path = Path(dataset_dir)
    all_patches_pos = dataset_path / "all_patches" / "positive"
    all_patches_neg = dataset_path / "all_patches" / "negative"

    # 0) all_patches + fold_file (10-fold CV)
    if fold_file and all_patches_pos.exists() and all_patches_neg.exists():
        with open(fold_file) as f:
            fold = json.load(f)
        train_stems = set(fold["train_stems"])
        val_stems = set(fold["val_stems"])
        train_pos_paths, train_neg_paths = [], []
        val_pos_paths, val_neg_paths = [], []
        for p in all_patches_pos.glob("*.png"):
            stem = _stem_from_patch_path(p)
            if stem in train_stems:
                train_pos_paths.append(str(p))
            elif stem in val_stems:
                val_pos_paths.append(str(p))
        for p in all_patches_neg.glob("*.png"):
            stem = _stem_from_patch_path(p)
            if stem in train_stems:
                train_neg_paths.append(str(p))
            elif stem in val_stems:
                val_neg_paths.append(str(p))
        if extra_negative_dir:
            extra = list(Path(extra_negative_dir).glob("*.png"))
            train_neg_paths.extend([str(p) for p in extra])
        print("Using train/val split (fold JSON, image-level)")
        train_dataset = BinaryWalnutDataset(
            str(all_patches_pos), str(all_patches_neg),
            transform=AugmentationTransform(patch_size, training=True),
            augment=True,
            positive_files=train_pos_paths, negative_files=train_neg_paths,
        )
        val_dataset = BinaryWalnutDataset(
            str(all_patches_pos), str(all_patches_neg),
            transform=AugmentationTransform(patch_size, training=False),
            augment=False,
            positive_files=val_pos_paths, negative_files=val_neg_paths,
        )
        num_pos = len(train_pos_paths)
        num_neg = len(train_neg_paths)
        class_weights = calculate_class_weights(num_pos, num_neg, device)
        print(f"📊 Train: {num_pos} positives, {num_neg} negatives")
        if num_pos:
            print(f"📊 Class ratio: {num_neg/num_pos:.2f}:1 (neg:pos)")
        print(f"📊 Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        return train_loader, val_loader, class_weights

    # Check for train/val/test layout
    train_pos = dataset_path / "train" / "positive"
    train_neg = dataset_path / "train" / "negative"
    val_pos = dataset_path / "val" / "positive"
    val_neg = dataset_path / "val" / "negative"

    if train_pos.exists() and train_neg.exists() and val_pos.exists() and val_neg.exists() and \
       len(list(train_pos.glob("*.png"))) > 0 and len(list(train_neg.glob("*.png"))) > 0 and \
       len(list(val_pos.glob("*.png"))) > 0 and len(list(val_neg.glob("*.png"))) > 0:
        print("Using train/val split (image-level)")
        positive_dir = str(train_pos)
        negative_dir = str(train_neg)
        val_positive_dir = str(val_pos)
        val_negative_dir = str(val_neg)
        use_separate_val = True
    else:
        # Legacy: positive/, negative/, optional real_positive/, real_negative/
        positive_dir = os.path.join(dataset_dir, "positive")
        negative_dir = os.path.join(dataset_dir, "negative")
        real_pos_dir = os.path.join(dataset_dir, "real_positive")
        real_neg_dir = os.path.join(dataset_dir, "real_negative")

        if os.path.exists(real_pos_dir) and os.path.exists(real_neg_dir) and \
           len(list(Path(real_pos_dir).glob("*.png"))) > 0 and len(list(Path(real_neg_dir).glob("*.png"))) > 0:
            print("Using real validation data (real_positive/real_negative)")
            val_positive_dir = real_pos_dir
            val_negative_dir = real_neg_dir
            use_separate_val = True
        else:
            print("⚠️  Using synthetic data split for validation (PATCH level, possible leakage)")
            val_positive_dir = positive_dir
            val_negative_dir = negative_dir
            use_separate_val = False

    # Create full training dataset
    full_dataset = BinaryWalnutDataset(
        positive_dir, negative_dir,
        transform=AugmentationTransform(patch_size, training=True),
        augment=True
    )

    num_positive = len(full_dataset.positive_files)
    num_negative = len(full_dataset.negative_files)
    class_weights = calculate_class_weights(num_positive, num_negative, device)

    print(f"📊 Train: {num_positive} positives, {num_negative} negatives")
    print(f"📊 Class ratio: {num_negative/num_positive:.2f}:1 (neg:pos)" if num_positive else "📊 No positives")

    if use_separate_val:
        train_dataset = full_dataset
        val_dataset = BinaryWalnutDataset(
            val_positive_dir, val_negative_dir,
            transform=AugmentationTransform(patch_size, training=False),
            augment=False
        )
    else:
        from torch.utils.data import random_split
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
        val_dataset.dataset.transform = AugmentationTransform(patch_size, training=False)
        val_dataset.dataset.augment = False

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    print(f"📊 Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    return train_loader, val_loader, class_weights

def plot_training_history(train_losses, val_losses, train_accs, val_accs, save_path):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    ax1.plot(train_losses, label='Train Loss', color='blue')
    ax1.plot(val_losses, label='Val Loss', color='red')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy plot
    ax2.plot(train_accs, label='Train Acc', color='blue')
    ax2.plot(val_accs, label='Val Acc', color='red')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def save_hard_negatives(hard_negatives: List[Tuple[Path, float]], output_dir: str, 
                        num_to_save: int = None, duplicate_factor: int = 1):
    """Save or duplicate hard negative patches
    
    Args:
        hard_negatives: List of (path, confidence) tuples
        output_dir: Directory to save hard negatives
        num_to_save: Number of top hard negatives to save (None = all)
        duplicate_factor: Number of times to duplicate each hard negative (for oversampling)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Limit number if specified
    if num_to_save is not None:
        hard_negatives = hard_negatives[:num_to_save]
    
    print(f"💾 Saving {len(hard_negatives)} hard negatives (duplicating {duplicate_factor}x)...")
    
    saved_count = 0
    for path, confidence in tqdm(hard_negatives, desc="Saving"):
        # Read original image
        img = cv2.imread(str(path))
        if img is None:
            continue
        
        # Save multiple copies if duplicating
        for dup_idx in range(duplicate_factor):
            # Create filename with confidence score
            base_name = path.stem
            if duplicate_factor > 1:
                output_name = f"{base_name}_hardneg_{confidence:.3f}_dup{dup_idx}.png"
            else:
                output_name = f"{base_name}_hardneg_{confidence:.3f}.png"
            
            output_file = output_path / output_name
            cv2.imwrite(str(output_file), img)
            saved_count += 1
    
    print(f"✅ Saved {saved_count} hard negative patches to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Train binary walnut classifier")
    parser.add_argument("--dataset_dir", required=True, help="Path to binary dataset directory")
    parser.add_argument("--fold_file", default=None, help="Path to fold JSON (train_stems, val_stems) for CV; requires dataset_dir/all_patches/")
    parser.add_argument("--output_dir", default="./models", help="Output directory for models")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of epochs for initial training (phase 1)")
    parser.add_argument("--second_phase_epochs", type=int, default=0,
                        help="Additional epochs to train AFTER adding mined hard negatives "
                             "(phase 2). 0 = skip second phase.")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--patch_size", type=int, default=32, help="Patch size")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout rate")
    parser.add_argument("--loss_type", type=str, default="focal",
                        choices=["ce", "focal"],
                        help="Loss function: 'ce' for CrossEntropyLoss, 'focal' for Focal Loss")
    parser.add_argument("--focal_gamma", type=float, default=2.0,
                        help="Gamma parameter for focal loss (higher focuses more on hard examples)")
    parser.add_argument("--mine_hard_negatives", action="store_true", 
                        help="After training, mine hard negatives and save them")
    parser.add_argument("--hard_negative_threshold", type=float, default=0.5,
                        help="Confidence threshold for hard negative mining")
    parser.add_argument("--hard_negative_output", type=str, default=None,
                        help="Directory to save hard negatives (default: dataset_dir/hard_negatives)")
    parser.add_argument("--hard_negative_duplicate", type=int, default=1,
                        help="Number of times to duplicate each hard negative (for oversampling)")
    parser.add_argument("--pretrained", type=str, default=None,
                        help="Path to pretrained .pth to fine-tune from (loads model_state_dict). Use lower LR (e.g. 0.0003).")
    parser.add_argument("--device", type=str, default=None,
                        choices=["auto", "cpu", "cuda", "mps"],
                        help="Device to train on (default: auto = cuda if available, else mps, else cpu)")
    
    args = parser.parse_args()
    
    # Suggest lower learning rate when fine-tuning
    if args.pretrained and args.learning_rate >= 0.001:
        print("💡 Tip: When fine-tuning (--pretrained), use a lower learning rate (e.g. --learning_rate 0.0003) to avoid overwriting pretrained weights.")
    
    print("🥜 Binary Walnut Classifier Training")
    print("=" * 50)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    from device_utils import resolve_device

    device = resolve_device(args.device)
    print(f"🖥️  Using device: {device}")
    
    # Create data loaders (optionally with fold JSON for CV)
    train_loader, val_loader, class_weights = create_data_loaders(
        args.dataset_dir, args.batch_size, args.patch_size, device=device,
        fold_file=args.fold_file,
    )
    
    # Create model
    model = WalnutClassifier(
        input_size=args.patch_size, 
        num_classes=2, 
        dropout_rate=args.dropout
    )
    
    if args.pretrained and os.path.isfile(args.pretrained):
        print(f"📥 Loading pretrained weights from {args.pretrained}")
        ckpt = torch.load(args.pretrained, map_location=device, weights_only=False)
        state = ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt
        model.load_state_dict(state, strict=False)
        print("   Fine-tuning from pretrained model.")
    
    print(f"📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer with chosen loss (focal or cross-entropy) and class weights
    trainer = BinaryTrainer(
        model,
        device,
        args.learning_rate,
        class_weights=class_weights,
        loss_type=args.loss_type,
        focal_gamma=args.focal_gamma,
    )
    
    # -------------------------
    # Phase 1: Initial training
    # -------------------------
    phase1_model_path = os.path.join(args.output_dir, "walnut_classifier_phase1.pth")
    train_losses, val_losses, train_accs, val_accs = trainer.train(
        train_loader, val_loader, args.epochs, phase1_model_path
    )
    
    final_model_path = phase1_model_path
    final_best_val_acc = trainer.best_val_acc
    final_best_precision = trainer.best_precision
    
    # --------------------------------------
    # Hard negative mining + optional phase 2
    # --------------------------------------
    if args.mine_hard_negatives:
        print("\n" + "=" * 50)
        print("🔍 HARD NEGATIVE MINING")
        print("=" * 50)
        
        # Load the best model
        checkpoint = torch.load(phase1_model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Create transform for mining
        mining_transform = AugmentationTransform(args.patch_size, training=False)
        
        # When using fold_file, train negatives are in all_patches filtered by train_stems (no single dir)
        negative_files_for_mining = None
        negative_dir = None
        if args.fold_file:
            with open(args.fold_file) as f:
                fold = json.load(f)
            train_stems = set(fold["train_stems"])
            all_neg = Path(args.dataset_dir) / "all_patches" / "negative"
            negative_files_for_mining = [p for p in all_neg.glob("*.png") if _stem_from_patch_path(p) in train_stems]
            hard_neg_output = Path(args.output_dir) / "hard_negatives"
        else:
            train_neg = Path(args.dataset_dir) / "train" / "negative"
            negative_dir = str(train_neg) if train_neg.exists() and list(train_neg.glob("*.png")) else os.path.join(args.dataset_dir, "negative")
            hard_neg_output = None
        
        hard_negatives = mine_hard_negatives(
            model, negative_dir or "", mining_transform, device,
            threshold=args.hard_negative_threshold, batch_size=args.batch_size,
            negative_files=negative_files_for_mining,
        )
        
        if len(hard_negatives) > 0:
            # Determine output directory for inspection (if requested)
            if args.hard_negative_output:
                inspection_output_dir = args.hard_negative_output
                save_hard_negatives(
                    hard_negatives,
                    inspection_output_dir,
                    duplicate_factor=args.hard_negative_duplicate,
                )
                print(f"\n💾 Hard negatives for inspection saved to: {inspection_output_dir}")
            
            if args.fold_file:
                # CV mode: save to output_dir/hard_negatives for phase 2 to pick up via extra_negative_dir
                hard_neg_output.mkdir(parents=True, exist_ok=True)
                print(f"\n💾 Saving hard negatives for phase 2: {hard_neg_output}")
                save_hard_negatives(
                    hard_negatives,
                    str(hard_neg_output),
                    duplicate_factor=args.hard_negative_duplicate,
                )
            else:
                # Non-CV: add into training negative folder
                print(f"\n💾 Adding hard negatives into training negatives: {negative_dir}")
                save_hard_negatives(
                    hard_negatives,
                    negative_dir,
                    duplicate_factor=args.hard_negative_duplicate,
                )
            
            # Optional Phase 2 training with augmented negatives
            if args.second_phase_epochs > 0:
                print("\n" + "=" * 50)
                print("🚀 PHASE 2: RETRAINING WITH HARD NEGATIVES")
                print("=" * 50)
                
                extra_neg = str(hard_neg_output) if args.fold_file and hard_neg_output and hard_neg_output.exists() else None
                train_loader2, val_loader2, class_weights2 = create_data_loaders(
                    args.dataset_dir, args.batch_size, args.patch_size, device=device,
                    fold_file=args.fold_file,
                    extra_negative_dir=extra_neg,
                )
                
                # New trainer that continues from the mined model weights
                trainer2 = BinaryTrainer(
                    model,
                    device,
                    args.learning_rate,
                    class_weights=class_weights2,
                    loss_type=args.loss_type,
                    focal_gamma=args.focal_gamma,
                )
                
                final_model_path = os.path.join(
                    args.output_dir, "walnut_classifier_best_precision.pth"
                )
                train_losses2, val_losses2, train_accs2, val_accs2 = trainer2.train(
                    train_loader2, val_loader2, args.second_phase_epochs, final_model_path
                )
                
                # Append phase 2 history
                train_losses.extend(train_losses2)
                val_losses.extend(val_losses2)
                train_accs.extend(train_accs2)
                val_accs.extend(val_accs2)
                
                final_best_val_acc = trainer2.best_val_acc
                final_best_precision = trainer2.best_precision
            else:
                print("\nℹ️  Hard negatives added, but second training phase is disabled "
                      "(--second_phase_epochs = 0).")
        else:
            print("ℹ️  No hard negatives found. Model is performing well on negative samples!")
    
    # If we never ran a second phase, ensure the final model path points to a sensible name
    if not args.mine_hard_negatives or args.second_phase_epochs == 0:
        # For compatibility, also save/copy the phase1 model as the final best-precision model
        best_precision_path = os.path.join(
            args.output_dir, "walnut_classifier_best_precision.pth"
        )
        if best_precision_path != phase1_model_path:
            # Copy weights-only state dict
            torch.save(
                torch.load(phase1_model_path, map_location=device, weights_only=False),
                best_precision_path,
            )
        final_model_path = best_precision_path
    
    # Plot training history (all phases)
    plot_path = os.path.join(args.output_dir, "training_history.png")
    plot_training_history(train_losses, val_losses, train_accs, val_accs, plot_path)
    
    # Save training history (all phases)
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accs,
        'val_accuracies': val_accs,
        'best_val_acc': final_best_val_acc,
        'best_precision': final_best_precision,
        'phase1_epochs': args.epochs,
        'phase2_epochs': args.second_phase_epochs if args.mine_hard_negatives else 0,
    }
    
    history_path = os.path.join(args.output_dir, "training_history.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n✅ Training completed! Final model saved to: {final_model_path}")

if __name__ == "__main__":
    main()
