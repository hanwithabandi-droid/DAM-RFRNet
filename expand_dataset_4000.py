"""
===============================================================================
DAM-RFRNet Dataset Expansion Utility: 2000 -> 4000 Image Pairs
===============================================================================
Expands the paired defogging benchmark dataset from 2,000 to 4,000 image pairs
by generating indices image_2001.jpg to image_4000.jpg using Koschmieder's
atmospheric scattering model:
    I(x) = J(x) * t(x) + A * (1 - t(x))
where:
    t(x) = exp(-beta * d(x))
    A ~ Uniform(0.75, 0.92)
    beta ~ Uniform(0.90, 1.70)
===============================================================================
"""

import os
import cv2
import numpy as np
from tqdm import tqdm

CLEAN_DIR = "c:/Minor project/data/clean"
HAZY_DIR = "c:/Minor project/data/hazy"
VOC_DIR = "c:/Minor project/VOC/VOC/JPEGImages"
TARGET_TOTAL = 4000

def synthesize_haze(image_np, seed=42):
    rows, cols, _ = image_np.shape
    rng = np.random.RandomState(seed)
    beta = rng.uniform(0.9, 1.7)
    A = rng.uniform(0.75, 0.92)
    
    # Depth gradient from distant horizon (0.85) to foreground (0.15)
    depth_map = np.linspace(0.85, 0.15, rows)[:, np.newaxis] * np.ones((rows, cols))
    noise = rng.normal(0, 0.03, (rows, cols))
    depth_map = np.clip(depth_map + noise, 0, 1)
    
    t = np.exp(-beta * depth_map)
    t = np.dstack([t, t, t])
    J = image_np / 255.0
    I = J * t + A * (1 - t)
    hazy = np.clip(I * 255, 0, 255).astype(np.uint8)
    return hazy

def prepare_clean_images(voc_dir, needed=2000):
    voc_files = sorted([os.path.join(voc_dir, f) for f in os.listdir(voc_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    total_voc = len(voc_files)
    print(f"[*] Found {total_voc} source VOC images in {voc_dir}")
    
    clean_crops = []
    
    # Pass 1: Fresh 500 images from VOC (indices 500 to 1000)
    for src_path in voc_files[500:1000]:
        img = cv2.imread(src_path)
        if img is not None:
            img_resized = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
            clean_crops.append(img_resized)
            
    print(f"[+] Pass 1 (Unused VOC 500-1000): {len(clean_crops)} images")
    
    # Pass 2: Horizontally flipped center crops from all 1000 VOC images
    for src_path in voc_files:
        if len(clean_crops) >= needed:
            break
        img = cv2.imread(src_path)
        if img is not None:
            h, w = img.shape[:2]
            # Center 80% crop
            ch, cw = int(h * 0.8), int(w * 0.8)
            sy, sx = (h - ch) // 2, (w - cw) // 2
            crop = img[sy:sy+ch, sx:sx+cw]
            crop_flipped = cv2.flip(crop, 1) # Horizontal mirror
            crop_resized = cv2.resize(crop_flipped, (256, 256), interpolation=cv2.INTER_AREA)
            clean_crops.append(crop_resized)
            
    print(f"[+] Pass 2 (Horizontally Flipped Center Crops): Total accumulated = {len(clean_crops)} images")
    
    # Pass 3: Upper-perspective panoramic crops (focusing on depth-rich upper scene)
    for src_path in voc_files:
        if len(clean_crops) >= needed:
            break
        img = cv2.imread(src_path)
        if img is not None:
            h, w = img.shape[:2]
            # Upper 75% crop
            crop = img[0:int(h*0.75), :]
            crop_resized = cv2.resize(crop, (256, 256), interpolation=cv2.INTER_AREA)
            clean_crops.append(crop_resized)
            
    print(f"[+] Pass 3 (Perspective Horizon Crops): Total accumulated = {len(clean_crops)} images")
    
    return clean_crops[:needed]

def main():
    os.makedirs(CLEAN_DIR, exist_ok=True)
    os.makedirs(HAZY_DIR, exist_ok=True)
    
    existing_clean = len([f for f in os.listdir(CLEAN_DIR) if f.lower().endswith(('.jpg', '.png'))])
    existing_hazy = len([f for f in os.listdir(HAZY_DIR) if f.lower().endswith(('.jpg', '.png'))])
    
    print("=" * 65)
    print("     DAM-RFRNet DATASET EXPANSION: 2,000 -> 4,000 PAIRS          ")
    print("=" * 65)
    print(f"[*] Current Clean Images : {existing_clean}")
    print(f"[*] Current Hazy Images  : {existing_hazy}")
    print(f"[*] Target Dataset Total : {TARGET_TOTAL} pairs")
    
    needed = TARGET_TOTAL - existing_clean
    if needed <= 0:
        print(f"\n[+] Dataset already contains {existing_clean} images. No expansion needed!")
        return

    print(f"[*] Generating {needed} new image pairs (image_{existing_clean + 1:04d}.jpg to image_{TARGET_TOTAL:04d}.jpg)...")
    
    clean_images = prepare_clean_images(VOC_DIR, needed=needed)
    assert len(clean_images) == needed, f"Expected {needed} images, got {len(clean_images)}"
    
    pbar = tqdm(total=needed, desc="Synthesizing Hazy Pairs")
    
    for idx, clean_img in enumerate(clean_images):
        new_num = existing_clean + idx + 1
        filename = f"image_{new_num:04d}.jpg"
        
        clean_path = os.path.join(CLEAN_DIR, filename)
        hazy_path = os.path.join(HAZY_DIR, filename)
        
        # Synthesize physical atmospheric haze with deterministic seed
        hazy_img = synthesize_haze(clean_img, seed=new_num)
        
        # Save both clean and hazy images
        cv2.imwrite(clean_path, clean_img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        cv2.imwrite(hazy_path, hazy_img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        
        pbar.update(1)
        
    pbar.close()
    
    final_clean = len([f for f in os.listdir(CLEAN_DIR) if f.lower().endswith(('.jpg', '.png'))])
    final_hazy = len([f for f in os.listdir(HAZY_DIR) if f.lower().endswith(('.jpg', '.png'))])
    
    print("\n" + "=" * 65)
    print("         DATASET EXPANSION SUCCESSFULLY COMPLETED!              ")
    print("=" * 65)
    print(f"  Total Clean Images       : {final_clean} pairs")
    print(f"  Total Hazy Images        : {final_hazy} pairs")
    print(f"  80% Training Pool        : {int(0.80 * final_clean)} pairs")
    print(f"  20% Held-out Test Pool   : {int(0.20 * final_clean)} pairs")
    print("=" * 65)

if __name__ == '__main__':
    main()
