#!/usr/bin/env python3
"""
refine_synthetic.py — Apply CycleGAN G_AB to synthetic patches (synthetic → real style).

What it does:
  - Loads Generator32 from cyclegan_models/G_AB_best.pth
  - Reads synthetic_patches/images/, writes refined images to synthetic_patches_refined/
  - Copies labels.json unchanged (same counts/labels for downstream training)
  - Exports Generator32 class used by optimize_synthetic_config.py

Prerequisites:
  - python3 setup.py
  - synthetic_patches/images/ and cyclegan_models/G_AB_best.pth under --output_dir

How to run:
  cd Walnut-detection
  source venv/bin/activate
  python refine_synthetic.py --output_dir .
  python refine_synthetic.py --output_dir . --test   # first 10 images only
"""

import argparse
import json
import shutil
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from tqdm import tqdm


class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3),
            nn.InstanceNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3),
            nn.InstanceNorm2d(dim),
        )

    def forward(self, x):
        return x + self.block(x)


class Generator32(nn.Module):
    """Generator for 32x32 (must match train_cyclegan.py)."""
    def __init__(self):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2),
        )
        self.res = nn.Sequential(
            ResBlock(128),
            ResBlock(128),
        )
        self.up = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.down(x)
        x = self.res(x) + x
        return self.up(x)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default=".", help="Base dir containing synthetic_patches/ and cyclegan_models/")
    parser.add_argument("--input_dir", default=None, help="Override input image dir (for iterative refinement)")
    parser.add_argument("--checkpoint", default="G_AB_best.pth", help="G_AB checkpoint name in cyclegan_models/")
    from device_utils import add_device_argument, resolve_device

    add_device_argument(parser, default="auto")
    parser.add_argument("--test", action="store_true", help="Refine only 10 images")
    args = parser.parse_args()

    base = Path(args.output_dir)
    images_in = Path(args.input_dir) if args.input_dir else base / "synthetic_patches" / "images"
    labels_path = base / "synthetic_patches" / "labels.json"
    models_dir = base / "cyclegan_models"
    out_images = base / "synthetic_patches_refined" / "images"
    out_labels = base / "synthetic_patches_refined" / "labels.json"

    if not images_in.exists():
        raise SystemExit(f"Missing {images_in}. Run generate_synthetic.py first.")
    if not (models_dir / args.checkpoint).exists():
        raise SystemExit(f"Missing {models_dir / args.checkpoint}. Run train_cyclegan.py first.")

    device = resolve_device(args.device)
    print(f"Device: {device}")

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    G = Generator32()
    G.load_state_dict(torch.load(models_dir / args.checkpoint, map_location=device))
    G.to(device)
    G.eval()

    paths = sorted(images_in.glob("*.png"))
    if args.test:
        paths = paths[:10]
    out_images.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for p in tqdm(paths, desc="Refine"):
            img = Image.open(p).convert("RGB")
            x = transform(img).unsqueeze(0).to(device)
            out = G(x)
            # Tanh -> [0,1], then to uint8
            out = (out.clamp(-1, 1) * 0.5 + 0.5).squeeze(0).cpu().permute(1, 2, 0).numpy()
            out = (out * 255).astype("uint8")
            Image.fromarray(out).save(out_images / p.name)

    if labels_path.exists():
        shutil.copy(labels_path, out_labels)
        print(f"Copied {labels_path} -> {out_labels}")
    print(f"Refined {len(paths)} images -> {out_images}")
    print("Use synthetic_patches_refined/ (images + labels.json) for density/count training.")


if __name__ == "__main__":
    main()
