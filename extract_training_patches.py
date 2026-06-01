#!/usr/bin/env python3
"""
Extract 32x32 patches from annotated images for binary classifier training. Positives: patches centered on
annotation points. Negatives: patches from background (excluding areas near walnuts via poison-disc mask).
Writes positive/ and negative/ image directories.

How to run:
  python extract_training_patches.py --images_dir path/to/images --annotations_dir path/to/annotations --output_dir path/to/output [--patch_size 32] [--num_negatives_per_image N]

Use --help for all options.
"""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple
import argparse
from tqdm import tqdm
import random

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

def parse_annotations(annotation_path: str) -> List[Tuple[int, int]]:
    """Parse annotation file and return list of (x, y) coordinates"""
    walnuts = []
    
    with open(annotation_path, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip comments and empty lines
            if line.startswith('#') or not line:
                continue
            # Parse x y coordinates
            parts = line.split()
            if len(parts) >= 2:
                try:
                    x = int(parts[0])
                    y = int(parts[1])
                    walnuts.append((x, y))
                except ValueError:
                    continue
    
    return walnuts

def create_poison_disc_mask(image_shape: Tuple[int, int], walnuts: List[Tuple[int, int]], 
                            radius: int = 16) -> np.ndarray:
    """Create a binary mask with poison discs (circles) covering walnut locations"""
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    
    for x, y in walnuts:
        # Draw filled circle at walnut location
        cv2.circle(mask, (x, y), radius, 255, -1)
    
    return mask

def extract_positive_patches(image: np.ndarray, walnuts: List[Tuple[int, int]], 
                             patch_size: int = 32) -> List[np.ndarray]:
    """Extract positive patches centered on walnuts"""
    patches = []
    h, w = image.shape[:2]
    half_size = patch_size // 2
    
    for x, y in walnuts:
        # Calculate crop coordinates
        x1 = max(0, x - half_size)
        y1 = max(0, y - half_size)
        x2 = min(w, x + half_size)
        y2 = min(h, y + half_size)
        
        # Extract patch
        patch = image[y1:y2, x1:x2]
        
        # If patch is smaller than patch_size, pad it
        if patch.shape[0] < patch_size or patch.shape[1] < patch_size:
            # Create padded patch
            padded = np.zeros((patch_size, patch_size, 3), dtype=image.dtype)
            # Calculate padding offsets
            pad_y1 = max(0, half_size - y)
            pad_x1 = max(0, half_size - x)
            # Place patch in padded array
            patch_h, patch_w = patch.shape[:2]
            padded[pad_y1:pad_y1+patch_h, pad_x1:pad_x1+patch_w] = patch
            patch = padded
        else:
            # Resize to exact patch_size if needed (shouldn't happen, but safety check)
            if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
                patch = cv2.resize(patch, (patch_size, patch_size))
        
        patches.append(patch)
    
    return patches

def extract_negative_patches(image: np.ndarray, poison_mask: np.ndarray, 
                            num_patches: int, patch_size: int = 32,
                            min_distance: int = 16) -> List[np.ndarray]:
    """Extract negative patches from areas without walnuts"""
    patches = []
    h, w = image.shape[:2]
    half_size = patch_size // 2
    
    # Create valid sampling mask (areas where we can sample)
    # Valid if the entire patch doesn't overlap with poison discs
    valid_mask = np.ones((h - patch_size + 1, w - patch_size + 1), dtype=np.uint8)
    
    # Mark invalid areas (where patches would overlap with poison discs)
    for y in range(h - patch_size + 1):
        for x in range(w - patch_size + 1):
            # Check if patch area overlaps with poison mask
            patch_area = poison_mask[y:y+patch_size, x:x+patch_size]
            if np.any(patch_area > 0):
                valid_mask[y, x] = 0
    
    # Get all valid sampling locations
    valid_locations = np.argwhere(valid_mask > 0)
    
    if len(valid_locations) == 0:
        print(f"Warning: No valid negative patch locations found!")
        return patches
    
    # Sample random locations
    num_samples = min(num_patches, len(valid_locations))
    sampled_indices = np.random.choice(len(valid_locations), num_samples, replace=False)
    
    for idx in sampled_indices:
        y, x = valid_locations[idx]
        patch = image[y:y+patch_size, x:x+patch_size]
        patches.append(patch)
    
    return patches

def process_training_images(train_dir: str, output_dir: str, patch_size: int = 32, 
                           poison_radius: int = 16):
    """Process all training images and extract patches"""
    
    train_path = Path(train_dir)
    images_dir = train_path / "images"
    annotations_dir = train_path / "annotations"
    
    # Create output directories (matching binary_classifier.py expectations)
    positives_dir = Path(output_dir) / "positive"
    negatives_dir = Path(output_dir) / "negative"
    positives_dir.mkdir(parents=True, exist_ok=True)
    negatives_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all image files
    image_files = list(images_dir.glob("*.JPG")) + list(images_dir.glob("*.jpg"))
    
    print(f"Found {len(image_files)} training images")
    
    all_positive_patches = []
    all_negative_patches = []
    
    # Process each image
    for img_path in tqdm(image_files, desc="Processing images"):
        # Find corresponding annotation file
        # Try different possible annotation file names
        base_name = img_path.stem
        
        # Try exact match first
        annotation_path = annotations_dir / f"{base_name}.txt"
        
        # If not found, try without extension variations
        if not annotation_path.exists():
            # Try with different case
            annotation_path = annotations_dir / f"{base_name.lower()}.txt"
        
        if not annotation_path.exists():
            print(f"Warning: No annotation found for {img_path.name}, skipping...")
            continue
        
        # Load image
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"Warning: Could not load image {img_path.name}, skipping...")
            continue
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Parse annotations
        walnuts = parse_annotations(str(annotation_path))
        
        if len(walnuts) == 0:
            print(f"Warning: No walnuts found in {img_path.name}, skipping...")
            continue
        
        # Extract positive patches
        positive_patches = extract_positive_patches(image, walnuts, patch_size)
        all_positive_patches.extend(positive_patches)
        
        # Create poison disc mask
        poison_mask = create_poison_disc_mask(image.shape, walnuts, radius=poison_radius)
        
        # Extract negative patches (same number as positives from this image)
        negative_patches = extract_negative_patches(
            image, poison_mask, len(positive_patches), patch_size, min_distance=poison_radius
        )
        all_negative_patches.extend(negative_patches)
    
    print(f"\nExtracted {len(all_positive_patches)} positive patches")
    print(f"Extracted {len(all_negative_patches)} negative patches")
    
    # Balance the dataset
    min_count = min(len(all_positive_patches), len(all_negative_patches))
    
    if len(all_positive_patches) > min_count:
        # Randomly sample positives
        all_positive_patches = random.sample(all_positive_patches, min_count)
        print(f"Balanced: Using {min_count} positive patches")
    
    if len(all_negative_patches) > min_count:
        # Randomly sample negatives
        all_negative_patches = random.sample(all_negative_patches, min_count)
        print(f"Balanced: Using {min_count} negative patches")
    
    # Save positive patches
    print(f"\nSaving {len(all_positive_patches)} positive patches...")
    for i, patch in enumerate(tqdm(all_positive_patches, desc="Saving positives")):
        # Convert RGB to BGR for OpenCV
        patch_bgr = cv2.cvtColor(patch, cv2.COLOR_RGB2BGR)
        output_path = positives_dir / f"walnut_{i:06d}.png"
        cv2.imwrite(str(output_path), patch_bgr)
    
    # Save negative patches
    print(f"\nSaving {len(all_negative_patches)} negative patches...")
    for i, patch in enumerate(tqdm(all_negative_patches, desc="Saving negatives")):
        # Convert RGB to BGR for OpenCV
        patch_bgr = cv2.cvtColor(patch, cv2.COLOR_RGB2BGR)
        output_path = negatives_dir / f"background_{i:06d}.png"
        cv2.imwrite(str(output_path), patch_bgr)
    
    print(f"\n✅ Done! Saved patches to:")
    print(f"   Positives: {positives_dir}")
    print(f"   Negatives: {negatives_dir}")
    print(f"   Total: {len(all_positive_patches)} positives, {len(all_negative_patches)} negatives")

def main():
    parser = argparse.ArgumentParser(description="Extract training patches from annotated images")
    parser.add_argument("--train_dir", type=str, 
                       default="/Users/kalpit/TuesdayPresentation/Walnut_Research/train",
                       help="Path to training directory with images/ and annotations/")
    parser.add_argument("--output_dir", type=str, default=".",
                       help="Output directory for positives/ and negatives/ folders")
    parser.add_argument("--patch_size", type=int, default=32,
                       help="Patch size in pixels (default: 32)")
    parser.add_argument("--poison_radius", type=int, default=16,
                       help="Radius of poison discs to mask walnuts (default: 16)")
    
    args = parser.parse_args()
    
    process_training_images(
        args.train_dir, 
        args.output_dir, 
        patch_size=args.patch_size,
        poison_radius=args.poison_radius
    )

if __name__ == "__main__":
    main()

