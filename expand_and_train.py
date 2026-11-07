"""
expand_and_train.py
-------------------
Step 1: Expand dataset to 500 paired images (clean + synthetic hazy)
        - Keeps existing 150 images in data/clean and data/hazy
        - Adds 350 more from COCO train2017 images
        - Synthetically generates hazy versions using Koschmieder's model
Step 2: Retrain the model on all 500 pairs
"""

import os
import shutil
import random
import numpy as np
from PIL import Image
import subprocess
import sys

# ── Paths ──────────────────────────────────────────────────────────────────
COCO_DIR   = "c:/Minor project/coco/train2017"
CLEAN_DIR  = "c:/Minor project/data/clean"
HAZY_DIR   = "c:/Minor project/data/hazy"
TARGET     = 500   # total images we want
RESIZE     = (320, 320)

# ── Koschmieder Haze Synthesis ─────────────────────────────────────────────
def add_haze(clean_img, beta=1.5, A=0.9):
    """
    Simulate haze using Koschmieder's atmospheric scattering model:
      I(x) = J(x) * t(x) + A * (1 - t(x))
    where t(x) = exp(-beta * depth)
    Uses a random depth map for realistic spatial haze distribution.
    """
    img_np = np.array(clean_img, dtype=np.float32) / 255.0
    h, w, _ = img_np.shape

    # Random depth map (simulates varying fog density spatially)
    depth = np.random.uniform(0.5, 1.5, (h, w))
    # Denser haze near the top (horizon effect)
    gradient = np.linspace(1.0, 0.3, h).reshape(h, 1)
    depth = depth * gradient

    transmission = np.exp(-beta * depth)
    transmission = np.clip(transmission, 0.0, 1.0)
    transmission = np.stack([transmission] * 3, axis=2)

    hazy_np = img_np * transmission + A * (1 - transmission)
    hazy_np = np.clip(hazy_np * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(hazy_np)


def expand_dataset():
    os.makedirs(CLEAN_DIR, exist_ok=True)
    os.makedirs(HAZY_DIR, exist_ok=True)

    # Count existing images
    existing = sorted([f for f in os.listdir(CLEAN_DIR) if f.endswith('.jpg')])
    existing_count = len(existing)
    print(f"Existing clean images: {existing_count}")

    needed = TARGET - existing_count
    if needed <= 0:
        print(f"Already have {existing_count} images (>= {TARGET}). Skipping dataset expansion.")
        return

    print(f"Need {needed} more images from COCO...")

    # Get available COCO images
    coco_images = sorted([f for f in os.listdir(COCO_DIR) if f.endswith('.jpg')])
    random.seed(42)
    selected = random.sample(coco_images, min(needed, len(coco_images)))

    added = 0
    for i, fname in enumerate(selected):
        img_num = existing_count + i + 1
        new_name = f"image_{img_num:03d}.jpg"

        # Check if already exists (avoid duplicates on re-runs)
        if os.path.exists(os.path.join(CLEAN_DIR, new_name)):
            added += 1
            continue

        src = os.path.join(COCO_DIR, fname)
        dst_clean = os.path.join(CLEAN_DIR, new_name)
        dst_hazy  = os.path.join(HAZY_DIR,  new_name)

        try:
            img = Image.open(src).convert("RGB").resize(RESIZE, Image.LANCZOS)
            img.save(dst_clean, quality=90)

            # Generate synthetic haze with varied beta for diverse densities
            beta_values = [0.8, 1.2, 1.8]  # light, moderate, dense
            beta = random.choice(beta_values)
            A    = random.uniform(0.8, 0.95)
            hazy = add_haze(img, beta=beta, A=A)
            hazy.save(dst_hazy, quality=90)

            added += 1
            if added % 50 == 0:
                print(f"  Added {added}/{needed} images...")

        except Exception as e:
            print(f"  Skipping {fname}: {e}")

    # Also generate hazy images for the first 150 original images if missing
    print("\nChecking original 150 images for matching hazy counterparts...")
    for fname in existing:
        hazy_path = os.path.join(HAZY_DIR, fname)
        if not os.path.exists(hazy_path):
            clean_path = os.path.join(CLEAN_DIR, fname)
            try:
                img  = Image.open(clean_path).convert("RGB").resize(RESIZE, Image.LANCZOS)
                beta = random.choice([0.8, 1.2, 1.8])
                A    = random.uniform(0.8, 0.95)
                hazy = add_haze(img, beta=beta, A=A)
                hazy.save(hazy_path, quality=90)
                print(f"  Generated hazy for: {fname}")
            except Exception as e:
                print(f"  Skipping {fname}: {e}")

    total_clean = len([f for f in os.listdir(CLEAN_DIR) if f.endswith('.jpg')])
    total_hazy  = len([f for f in os.listdir(HAZY_DIR)  if f.endswith('.jpg')])
    print(f"\nDataset ready: {total_clean} clean | {total_hazy} hazy images")


def run_training():
    print("\n" + "="*50)
    print("  Starting model training on expanded dataset...")
    print("="*50 + "\n")
    result = subprocess.run([
        sys.executable,
        "c:/Minor project/train.py",
        "--data_dir",   "c:/Minor project/data",
        "--ckpt_dir",   "c:/Minor project/checkpoints",
        "--epochs",     "50",
        "--batch_size", "8",
        "--lr",         "0.0001"
    ], check=False)
    if result.returncode == 0:
        print("\nTraining completed successfully!")
    else:
        print("\nTraining ended with an error. Check logs above.")


if __name__ == "__main__":
    print("="*50)
    print("  STEP 1: Expanding dataset to 500 image pairs")
    print("="*50)
    expand_dataset()

    print("\n" + "="*50)
    print("  STEP 2: Training DAM-RFRNet on 500 image pairs")
    print("="*50)
    run_training()
