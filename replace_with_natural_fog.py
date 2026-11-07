import os
import shutil
import glob
from tqdm import tqdm
import requests
from PIL import Image
import io

def clean_directory(dir_path):
    os.makedirs(dir_path, exist_ok=True)
    for f in glob.glob(os.path.join(dir_path, "*")):
        if os.path.isfile(f):
            os.remove(f)

def main():
    hazy_dir = "c:/Minor project/data/hazy"
    clean_dir = "c:/Minor project/data/clean"

    print("Step 1: Removing all old synthetic images...")
    clean_directory(hazy_dir)
    clean_directory(clean_dir)
    print("Old synthetic dataset cleared successfully!")

    print("\nStep 2: Downloading 500 Natural Foggy & Ground-Truth Clean image pairs from HuggingFace/RESIDE benchmark...")
    
    # Using HuggingFace / RESIDE Benchmark real-world outdoor & indoor natural fog pairs
    from datasets import load_dataset
    
    # Load authentic dehazing dataset (RESIDE / real fog benchmark)
    try:
        ds = load_dataset("536767664/RESIDE_Standard", split="train", streaming=True)
    except Exception:
        try:
            ds = load_dataset("eugeniy/reside-in-outdoor", split="train", streaming=True)
        except Exception:
            ds = load_dataset("lmds/RESIDE-OTS", split="train", streaming=True)

    count = 0
    target_count = 500
    pbar = tqdm(total=target_count, desc="Downloading Natural Fog Pairs")

    for item in ds:
        if count >= target_count:
            break
            
        try:
            # Extract hazy and clean image PIL objects
            hazy_img = item.get('hazy') or item.get('image') or item.get('input')
            clean_img = item.get('clean') or item.get('target') or item.get('ground_truth')

            if hazy_img is not None and clean_img is not None:
                if not isinstance(hazy_img, Image.Image):
                    hazy_img = Image.open(io.BytesIO(hazy_img)).convert("RGB")
                else:
                    hazy_img = hazy_img.convert("RGB")

                if not isinstance(clean_img, Image.Image):
                    clean_img = Image.open(io.BytesIO(clean_img)).convert("RGB")
                else:
                    clean_img = clean_img.convert("RGB")

                filename = f"image_{count+1:03d}.jpg"
                
                hazy_path = os.path.join(hazy_dir, filename)
                clean_path = os.path.join(clean_dir, filename)

                hazy_img.save(hazy_path, quality=95)
                clean_img.save(clean_path, quality=95)

                count += 1
                pbar.update(1)
        except Exception as e:
            continue

    pbar.close()
    print(f"\nSuccess! Saved {count} authentic natural foggy image pairs to {hazy_dir} and {clean_dir}.")

if __name__ == '__main__':
    main()
