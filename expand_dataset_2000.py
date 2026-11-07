import os
import random
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

CLEAN_DIR = "c:/Minor project/data/clean"
HAZY_DIR = "c:/Minor project/data/hazy"
TARGET_TOTAL = 2000
START_IDX = 501

def synthesize_haze(image_np, seed=42):
    rows, cols, _ = image_np.shape
    np.random.seed(seed)
    beta = np.random.uniform(0.9, 1.7)
    A = np.random.uniform(0.75, 0.92)
    
    depth_map = np.linspace(0.85, 0.15, rows)[:, np.newaxis] * np.ones((rows, cols))
    noise = np.random.normal(0, 0.03, (rows, cols))
    depth_map = np.clip(depth_map + noise, 0, 1)
    
    t = np.exp(-beta * depth_map)
    t = np.dstack([t, t, t])
    J = image_np / 255.0
    I = J * t + A * (1 - t)
    hazy = np.clip(I * 255, 0, 255).astype(np.uint8)
    return hazy

def collect_source_images():
    sources = [
        ("c:/Minor project/coco/train2017", 600),
        ("c:/Minor project/coco/test2017", 300),
        ("c:/Minor project/coco/val2017", 100),
        ("c:/Minor project/VOC/VOC/JPEGImages", 500)
    ]
    all_files = []
    for dir_path, max_count in sources:
        if os.path.exists(dir_path):
            files = sorted([os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            selected = files[:max_count]
            print(f"Collected {len(selected)} images from {dir_path}")
            all_files.extend(selected)
    return all_files

def main():
    os.makedirs(CLEAN_DIR, exist_ok=True)
    os.makedirs(HAZY_DIR, exist_ok=True)
    
    existing_clean = len([f for f in os.listdir(CLEAN_DIR) if f.endswith('.jpg')])
    existing_hazy = len([f for f in os.listdir(HAZY_DIR) if f.endswith('.jpg')])
    print(f"Current dataset: {existing_clean} clean, {existing_hazy} hazy images.")
    
    needed = TARGET_TOTAL - existing_clean
    if needed <= 0:
        print(f"Dataset already has {existing_clean} images (target: {TARGET_TOTAL}).")
        return
        
    print(f"Need to generate {needed} additional image pairs...")
    source_files = collect_source_images()
    
    if len(source_files) < needed:
        print(f"Warning: Found {len(source_files)} source files, needed {needed}.")
        needed = len(source_files)
        
    added = 0
    pbar = tqdm(total=needed, desc="Expanding dataset to 2000 pairs")
    
    for idx, src_path in enumerate(source_files[:needed]):
        new_num = existing_clean + idx + 1
        new_name = f"image_{new_num:04d}.jpg" if new_num >= 1000 else f"image_{new_num:03d}.jpg"
        
        clean_target = os.path.join(CLEAN_DIR, new_name)
        hazy_target = os.path.join(HAZY_DIR, new_name)
        
        try:
            # Read clean image
            img_cv = cv2.imread(src_path)
            if img_cv is None:
                continue
                
            # Resize standard to 256x256
            img_cv = cv2.resize(img_cv, (256, 256), interpolation=cv2.INTER_AREA)
            
            # Synthesize realistic atmospheric haze
            hazy_cv = synthesize_haze(img_cv, seed=new_num)
            
            # Save both clean and hazy
            cv2.imwrite(clean_target, img_cv, [cv2.IMWRITE_JPEG_QUALITY, 92])
            cv2.imwrite(hazy_target, hazy_cv, [cv2.IMWRITE_JPEG_QUALITY, 92])
            
            added += 1
            pbar.update(1)
        except Exception as e:
            print(f"Error on {src_path}: {e}")
            
    pbar.close()
    
    final_clean = len([f for f in os.listdir(CLEAN_DIR) if f.endswith('.jpg')])
    final_hazy = len([f for f in os.listdir(HAZY_DIR) if f.endswith('.jpg')])
    print(f"\nExpansion complete! Total dataset: {final_clean} clean, {final_hazy} hazy images.")

if __name__ == "__main__":
    main()
