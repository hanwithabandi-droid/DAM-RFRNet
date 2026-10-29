"""
===============================================================================
DAM-RFRNet Major Project: Standalone Desktop Dehazing Engine
===============================================================================
Run this script directly:
    python run_major_project.py

Features:
- Select an Image (.jpg, .png, .jpeg, .bmp) OR Video (.mp4, .avi, .mov).
- Runs novel DAM-RFRNet deep learning dehazing.
- Applies adaptive color & contrast post-processing.
- Runs downstream AI Object Detection comparison (Hazy vs Dehazed).
- Saves output file in ./outputs/ directory and displays results on desktop.
"""

import os
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image, ImageOps, ImageEnhance
import matplotlib.pyplot as plt

# Project modules
from model import DAM_RFRNet
from detection_module import compare_detections
from video_processor import VideoDehazeProcessor


def pick_input_file() -> Optional[Path]:
    """Launches a desktop file picker dialog for images or videos."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        
        print("[*] Opening desktop file selector dialog...")
        selected = filedialog.askopenfilename(
            title="Select Hazy Image or Video",
            filetypes=[
                ("All Supported Files", "*.jpg *.jpeg *.png *.bmp *.webp *.mp4 *.avi *.mov *.mkv"),
                ("Image Files", "*.jpg *.jpeg *.png *.bmp *.webp"),
                ("Video Files", "*.mp4 *.avi *.mov *.mkv"),
                ("All Files", "*.*")
            ]
        )
        root.destroy()
        
        if selected:
            return Path(selected)
    except Exception:
        pass

    # CLI fallback prompt
    user_input = input("Enter path to hazy image or video file: ").strip('"\' ')
    if user_input:
        p = Path(user_input)
        if p.exists():
            return p
    return None


def process_image(file_path: Path, model: DAM_RFRNet, device: torch.device):
    """Processes a single image file through DAM-RFRNet + Object Detection."""
    print(f"\n[*] Processing Image: {file_path.name}")
    start_time = time.perf_counter()

    # 1. Read Image
    try:
        raw_pil = Image.open(file_path).convert("RGB")
    except Exception as err:
        print(f"[!] Failed to read image: {err}")
        return

    # 2. Neural Forward Pass
    print("[*] Running DAM-RFRNet deep neural dehazing pass...")
    tensor_in = transforms.ToTensor()(raw_pil).unsqueeze(0).to(device)
    
    with torch.no_grad():
        raw_output_tensor = model(tensor_in)

    raw_output_tensor = torch.clamp(raw_output_tensor.squeeze(0).cpu(), 0.0, 1.0)
    dehazed_pil = transforms.ToPILImage()(raw_output_tensor)

    # 3. Post-Processing Enhancements
    print("[*] Applying contrast & saturation post-processing...")
    dehazed_pil = ImageOps.autocontrast(dehazed_pil, cutoff=1.5)
    dehazed_pil = ImageEnhance.Contrast(dehazed_pil).enhance(1.2)
    dehazed_pil = ImageEnhance.Color(dehazed_pil).enhance(1.35)

    elapsed_sec = time.perf_counter() - start_time
    print(f"[+] Dehazing completed in {elapsed_sec:.3f} seconds.")

    # 4. Save Dehazed Output Image
    out_dir = Path("./outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"dehazed_{file_path.name}"
    dehazed_pil.save(out_file, quality=95)
    print(f"[+] Output image saved to: {out_file.resolve()}")

    # 5. Run Downstream Software AI Object Detection Comparison
    print("[*] Running downstream AI Object Detection comparison...")
    det_results = compare_detections(raw_pil, dehazed_pil)
    print(f"    - Hazy Image Detected:    {det_results['hazy_count']} objects (Avg Conf: {det_results['hazy_avg_conf']*100:.1f}%)")
    print(f"    - Dehazed Image Detected: {det_results['dehazed_count']} objects (Avg Conf: {det_results['dehazed_avg_conf']*100:.1f}%)")

    # 6. Display Desktop Comparison Dashboard
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.canvas.manager.set_window_title("DAM-RFRNet Major Project Desktop Output")

    axes[0].imshow(raw_pil)
    axes[0].set_title("1. Original Hazy Input", fontsize=12, pad=8, color='red')
    axes[0].axis('off')

    axes[1].imshow(dehazed_pil)
    axes[1].set_title(f"2. DAM-RFRNet Output ({elapsed_sec*1000:.1f} ms)", fontsize=12, pad=8, color='green')
    axes[1].axis('off')

    axes[2].imshow(det_results['dehazed_annotated'])
    axes[2].set_title(f"3. AI Object Detection ({det_results['dehazed_count']} Objs)", fontsize=12, pad=8, color='blue')
    axes[2].axis('off')

    plt.suptitle("DAM-RFRNet Major Project: Atmospheric Dehazing & Perception System", fontsize=15, weight='bold')
    plt.tight_layout()
    print("[*] Displaying desktop output window...")
    plt.show()


def process_video(file_path: Path):
    """Processes a video file frame-by-frame with temporal smoothing."""
    print(f"\n[*] Processing Video: {file_path.name}")
    
    out_dir = Path("./outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"dehazed_{file_path.stem}.mp4"

    processor = VideoDehazeProcessor(smoothing_alpha=0.75)
    
    def print_progress(percent, curr, total):
        sys.stdout.write(f"\r[*] Video Progress: Frame {curr}/{total} ({percent*100:.1f}%)")
        sys.stdout.flush()

    stats = processor.process_video_file(
        input_video_path=str(file_path),
        output_video_path=str(out_file),
        progress_callback=print_progress
    )
    print("\n")
    print(f"[+] Video Dehazing Finished!")
    print(f"    - Rendered Frames: {stats['total_frames']}")
    print(f"    - Processing FPS:  {stats['processing_fps']:.2f} FPS")
    print(f"    - Saved Output:    {out_file.resolve()}")


def main():
    print("===============================================================================")
    print("      DAM-RFRNet Major Project: Standalone Desktop Dehazing Engine            ")
    print("===============================================================================")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Hardware Accelerator: {device}")

    # 1. Acquire Input File (Dialog or Prompt)
    input_path = pick_input_file()
    
    if not input_path or not input_path.exists():
        print("[!] No input file selected. Exiting.")
        return

    # 2. Load Model Architecture
    model = DAM_RFRNet().to(device)
    ckpt_path = Path("./checkpoints/dam_rfrnet_best.pth")
    if ckpt_path.exists():
        try:
            ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
            state_dict = ckpt['state_dict'] if isinstance(ckpt, dict) and 'state_dict' in ckpt else ckpt
            model.load_state_dict(state_dict, strict=False)
            print(f"[+] Model weights loaded from {ckpt_path}")
        except Exception as e:
            print(f"[!] Warning loading weights: {e}")
    else:
        print("[!] Checkpoint not found. Running with initialized DAM-RFRNet model weights.")
    
    model.eval()

    # 3. Route based on file type
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv'}
    
    if input_path.suffix.lower() in video_extensions:
        process_video(input_path)
    else:
        process_image(input_path, model, device)


if __name__ == '__main__':
    main()
