#!/usr/bin/env python3
"""
walnut_detector.py — Sliding-window walnut detector using a trained binary classifier.

What it does:
  - Scans full images with overlapping 32×32 patches (WalnutDetector class or CLI)
  - Builds a confidence map, finds peaks, optionally clusters with DBSCAN
  - Saves detection overlays and JSON per image

Prerequisites:
  - python3 setup.py
  - Trained .pth from binary_classifier.py or train_synthetic_then_real.py

How to run:
  cd Walnut-detection
  source venv/bin/activate
  python walnut_detector.py \\
    --model_path models/walnut_classifier.pth \\
    --image_dir cropped_images \\
    --output_dir output/detections \\
    --threshold 0.6 --patch_size 32 --stride 16 --cluster
"""

import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import cv2
from pathlib import Path
from typing import List, Tuple, Dict
import argparse
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

class WalnutDetector:
    """Sliding window walnut detector using trained binary classifier"""
    
    def __init__(self, model_path: str, patch_size: int = 32, stride: int = 16, 
                 confidence_threshold: float = 0.5, device: str = 'auto',
                 cluster_eps: float = 24.0, local_max_size: int = 5, nms_radius: float = 0.0):
        
        self.patch_size = patch_size
        self.stride = stride
        self.confidence_threshold = confidence_threshold
        self.cluster_eps = cluster_eps
        self.local_max_size = max(3, local_max_size) if local_max_size % 2 == 1 else max(3, local_max_size + 1)
        self.nms_radius = max(0.0, nms_radius)
        
        from device_utils import resolve_device

        self.device = resolve_device(device, warn=False)
        
        print(f"🖥️  Using device: {self.device}")
        
        # Load model
        self.model = self._load_model(model_path)
        self.model.eval()
        
        # Transform for patches
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((patch_size, patch_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def _load_model(self, model_path: str) -> nn.Module:
        """Load trained model"""
        from binary_classifier import WalnutClassifier
        
        # Create model architecture
        model = WalnutClassifier(input_size=self.patch_size, num_classes=2)
        
        # Load weights
        # PyTorch 2.6 defaults weights_only=True; allow full checkpoint loading
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
        except Exception:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            # Assume the checkpoint is the state dict itself
            model.load_state_dict(checkpoint)
        
        print(f"📦 Loaded model from {model_path}")
        val_acc = checkpoint.get('val_acc', None)
        try:
            if val_acc is not None:
                print(f"📊 Model accuracy: {float(val_acc):.2f}%")
            else:
                print("📊 Model accuracy: Unknown")
        except Exception:
            print("📊 Model accuracy: Unknown")
        
        return model.to(self.device)
    
    def detect_walnuts(self, image: np.ndarray) -> Tuple[List[Tuple[int, int]], List[float], np.ndarray]:
        """
        Detect walnuts in an image using sliding window approach
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            centers: List of (x, y) coordinates of detected walnuts
            confidences: List of confidence scores for each detection
            confidence_map: 2D array of confidence scores for each pixel
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image_rgb.shape[:2]
        
        # Initialize confidence map
        confidence_map = np.zeros((h, w), dtype=np.float32)
        count_map = np.zeros((h, w), dtype=np.int32)
        
        # Extract patches using sliding window
        patches = []
        patch_coords = []
        
        for y in range(0, h - self.patch_size + 1, self.stride):
            for x in range(0, w - self.patch_size + 1, self.stride):
                # Extract patch
                patch = image_rgb[y:y+self.patch_size, x:x+self.patch_size]
                patches.append(patch)
                patch_coords.append((x, y))
        
        # Process patches in batches
        batch_size = 32
        all_confidences = []
        
        for i in tqdm(range(0, len(patches), batch_size), desc="Processing patches"):
            batch_patches = patches[i:i+batch_size]
            batch_coords = patch_coords[i:i+batch_size]
            
            # Transform patches
            batch_tensors = []
            for patch in batch_patches:
                tensor = self.transform(patch)
                batch_tensors.append(tensor)
            
            batch_tensor = torch.stack(batch_tensors).to(self.device)
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(batch_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidences = probabilities[:, 1].cpu().numpy()  # Probability of class 1 (walnut)
            
            all_confidences.extend(confidences)
            
            # Update confidence map
            for j, (x, y) in enumerate(batch_coords):
                confidence = confidences[j]
                confidence_map[y:y+self.patch_size, x:x+self.patch_size] += confidence
                count_map[y:y+self.patch_size, x:x+self.patch_size] += 1
        
        # Normalize confidence map
        confidence_map = np.divide(confidence_map, count_map, 
                                 out=np.zeros_like(confidence_map), 
                                 where=count_map != 0)
        
        # Find high-confidence regions
        high_conf_mask = confidence_map > self.confidence_threshold
        
        # Find local maxima (larger window = fewer duplicate peaks per walnut)
        centers, confidences = self._find_local_maxima(confidence_map, high_conf_mask)
        
        # Optional NMS: keep only highest-confidence peak within nms_radius
        if self.nms_radius > 0 and len(centers) > 0:
            centers, confidences = self._nms(centers, confidences, self.nms_radius)
        
        return centers, confidences, confidence_map
    
    def _find_local_maxima(self, confidence_map: np.ndarray, mask: np.ndarray) -> Tuple[List[Tuple[int, int]], List[float]]:
        """Find local maxima in confidence map (size controls merging of nearby peaks)."""
        from scipy.ndimage import maximum_filter
        
        size = self.local_max_size
        local_maxima = maximum_filter(confidence_map, size=size) == confidence_map
        peaks = local_maxima & mask
        
        y_coords, x_coords = np.where(peaks)
        confidences = confidence_map[peaks]
        sorted_indices = np.argsort(confidences)[::-1]
        
        centers = [(int(x_coords[i]), int(y_coords[i])) for i in sorted_indices]
        confidences = [float(confidences[i]) for i in sorted_indices]
        return centers, confidences
    
    def _nms(self, centers: List[Tuple[int, int]], confidences: List[float], radius: float) -> Tuple[List[Tuple[int, int]], List[float]]:
        """Non-maximum suppression: within radius keep only the highest-confidence detection."""
        if len(centers) <= 1:
            return centers, confidences
        from scipy.spatial.distance import cdist
        centers_arr = np.array(centers, dtype=np.float64)
        conf_arr = np.array(confidences, dtype=np.float64)
        keep = np.ones(len(centers), dtype=bool)
        for i in range(len(centers)):
            if not keep[i]:
                continue
            dists = np.sqrt(np.sum((centers_arr - centers_arr[i]) ** 2, axis=1))
            same_region = (dists <= radius) & (dists > 0)
            lower_conf = conf_arr < conf_arr[i]
            keep[same_region & lower_conf] = False
        kept = [i for i in range(len(centers)) if keep[i]]
        return [centers[i] for i in kept], [confidences[i] for i in kept]
    
    def cluster_detections(self, centers: List[Tuple[int, int]], confidences: List[float], 
                          eps: float = None, min_samples: int = 1) -> Tuple[List[Tuple[int, int]], List[float]]:
        """Cluster nearby detections to avoid duplicates"""
        if len(centers) == 0:
            return [], []
        
        if eps is None:
            eps = self.cluster_eps
        
        centers_array = np.array(centers)
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(centers_array)
        
        # Get cluster centers
        clustered_centers = []
        clustered_confidences = []
        
        for cluster_id in set(clustering.labels_):
            if cluster_id == -1:  # Noise points
                continue
            
            # Get points in this cluster
            cluster_mask = clustering.labels_ == cluster_id
            cluster_centers = centers_array[cluster_mask]
            cluster_confidences = np.array(confidences)[cluster_mask]
            
            # Use weighted average for cluster center
            weights = cluster_confidences
            weighted_center = np.average(cluster_centers, axis=0, weights=weights)
            max_confidence = np.max(cluster_confidences)
            
            clustered_centers.append((int(weighted_center[0]), int(weighted_center[1])))
            clustered_confidences.append(float(max_confidence))
        
        return clustered_centers, clustered_confidences
    
    def visualize_detections(self, image: np.ndarray, centers: List[Tuple[int, int]], 
                           confidences: List[float], confidence_map: np.ndarray = None,
                           save_path: str = None) -> np.ndarray:
        """Visualize detections on image"""
        vis_image = image.copy()
        
        # Draw confidence map if provided
        if confidence_map is not None:
            # Create heatmap
            heatmap = cv2.applyColorMap(
                (confidence_map * 255).astype(np.uint8), 
                cv2.COLORMAP_JET
            )
            # Blend with original image
            vis_image = cv2.addWeighted(vis_image, 0.7, heatmap, 0.3, 0)
        
        # Draw detections
        for i, (x, y) in enumerate(centers):
            confidence = confidences[i]
            
            # Color based on confidence
            if confidence > 0.8:
                color = (0, 255, 0)  # Green for high confidence
            elif confidence > 0.6:
                color = (0, 255, 255)  # Yellow for medium confidence
            else:
                color = (0, 0, 255)  # Red for low confidence
            
            # Draw circle
            cv2.circle(vis_image, (x, y), 8, color, 2)
            
            # Draw confidence text
            cv2.putText(vis_image, f"{confidence:.2f}", (x+10, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Add count text
        cv2.putText(vis_image, f"Walnuts: {len(centers)}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        if save_path:
            cv2.imwrite(save_path, vis_image)
        
        return vis_image
    
    def process_image(self, image_path: str, output_dir: str = None, 
                     cluster: bool = True) -> Dict:
        """Process a single image and return detection results"""
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Detect walnuts
        centers, confidences, confidence_map = self.detect_walnuts(image)
        
        # Cluster detections if requested
        if cluster:
            centers, confidences = self.cluster_detections(centers, confidences)
        
        # Create results
        results = {
            'image_path': image_path,
            'num_walnuts': len(centers),
            'centers': centers,
            'confidences': confidences,
            'mean_confidence': np.mean(confidences) if confidences else 0.0
        }
        
        # Save visualization if output directory provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
            # Create visualization
            vis_image = self.visualize_detections(image, centers, confidences, confidence_map)
            
            # Save visualization
            image_name = Path(image_path).stem
            vis_path = os.path.join(output_dir, f"{image_name}_detections.jpg")
            cv2.imwrite(vis_path, vis_image)
            
            # Save confidence map
            conf_path = os.path.join(output_dir, f"{image_name}_confidence.png")
            cv2.imwrite(conf_path, (confidence_map * 255).astype(np.uint8))
            
            results['visualization_path'] = vis_path
            results['confidence_map_path'] = conf_path
        
        return results

def main():
    parser = argparse.ArgumentParser(description="Detect walnuts in images")
    parser.add_argument("--model_path", required=True, help="Path to trained model")
    parser.add_argument("--image_path", help="Process a single image (mutually exclusive with --image_dir)")
    parser.add_argument("--image_dir", help="Process all images in this directory (png/jpg)")
    parser.add_argument("--output_dir", help="Directory for overlays and detection JSON")
    parser.add_argument("--patch_size", type=int, default=32, help="Patch size")
    parser.add_argument("--stride", type=int, default=16, help="Stride for sliding window")
    parser.add_argument("--threshold", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--cluster", action="store_true", help="Cluster nearby detections")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device to run on: auto, cpu, cuda, or mps (default: auto)",
    )
    parser.add_argument("--cluster_eps", type=float, default=24.0, help="DBSCAN eps in px for merging nearby detections (default: 24)")
    parser.add_argument("--local_max_size", type=int, default=5, help="Local maxima window size (odd, default: 5)")
    parser.add_argument("--nms_radius", type=float, default=12.0, help="NMS radius in px; 0 to disable (default: 12)")
    
    args = parser.parse_args()
    
    print("🥜 Walnut Detector")
    print("=" * 30)
    
    detector = WalnutDetector(
        model_path=args.model_path,
        patch_size=args.patch_size,
        stride=args.stride,
        confidence_threshold=args.threshold,
        device=args.device,
        cluster_eps=args.cluster_eps,
        local_max_size=args.local_max_size,
        nms_radius=args.nms_radius,
    )
    
    # Process single image
    if args.image_path:
        print(f"Processing image: {args.image_path}")
        results = detector.process_image(args.image_path, args.output_dir, args.cluster)
        
        print(f"Detected {results['num_walnuts']} walnuts")
        print(f"Mean confidence: {results['mean_confidence']:.3f}")
        
        if args.output_dir:
            print(f"Results saved to: {args.output_dir}")
    
    # Process directory of images
    elif args.image_dir:
        image_files = (list(Path(args.image_dir).glob("*.png")) + 
                      list(Path(args.image_dir).glob("*.jpg")) +
                      list(Path(args.image_dir).glob("*.JPG")) +
                      list(Path(args.image_dir).glob("*.PNG")))
        print(f"Processing {len(image_files)} images...")
        
        all_results = []
        for image_file in tqdm(image_files, desc="Processing images"):
            try:
                results = detector.process_image(str(image_file), args.output_dir, args.cluster)
                all_results.append(results)
            except Exception as e:
                print(f"Error processing {image_file}: {e}")
        
        # Summary statistics
        if len(all_results) > 0:
            total_walnuts = sum(r['num_walnuts'] for r in all_results)
            mean_confidence = np.mean([r['mean_confidence'] for r in all_results])
            
            print(f"\n📊 Summary:")
            print(f"Total walnuts detected: {total_walnuts}")
            print(f"Average walnuts per image: {total_walnuts / len(all_results):.1f}")
            print(f"Mean confidence: {mean_confidence:.3f}")
        else:
            print(f"\n⚠️  No images processed successfully")
        
        # Save results
        if args.output_dir:
            results_path = os.path.join(args.output_dir, "detection_results.json")
            with open(results_path, 'w') as f:
                json.dump(all_results, f, indent=2)
            print(f"Results saved to: {results_path}")
    
    else:
        print("Please provide either --image_path or --image_dir")

if __name__ == "__main__":
    main()
