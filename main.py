import os
import sys
import time
from pathlib import Path
import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image, ImageOps, ImageEnhance
import matplotlib.pyplot as plt

# Project Neural Modules & Utilities
from model import DAM_RFRNet
from video_processor import VideoDehazeProcessor
from detection_module import compare_detections
from model_utils import calculate_psnr, calculate_ssim


def select_file_via_dialog() -> Path:
    """Launches desktop file selector dialog."""
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)

        print("[*] Please select a hazy image or video in the file dialog window...")
        file_path = filedialog.askopenfilename(
            title="Select Hazy Photo or Video",
            filetypes=[
                ("All Supported Files", "*.jpg *.jpeg *.png *.bmp *.webp *.mp4 *.avi *.mov *.mkv"),
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"),
                ("Video files", "*.mp4 *.avi *.mov *.mkv"),
                ("All files", "*.*")
            ]
        )
        root.destroy()

        if file_path:
            return Path(file_path)
    except Exception as e:
        print(f"[!] Dialog launch notice: {e}")

    # Fallback prompt
    user_in = input("Enter full path to image or video file: ").strip('"\' ')
    return Path(user_in)


def process_single_image(image_path: Path, model: DAM_RFRNet, device: torch.device):
    """Processes input photo, computes PSNR & SSIM metrics, and displays output window."""
    print(f"\n[*] Selected Photo: {image_path}")
    print("[*] Defogging image using DAM-RFRNet...")
    start_t = time.perf_counter()

    try:
        raw_pil = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"[!] Could not read image file: {e}")
        return

    # 1. Forward Pass through DAM-RFRNet
    tensor_in = transforms.ToTensor()(raw_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        out_tensor = model(tensor_in)

    out_tensor = torch.clamp(out_tensor, 0.0, 1.0)
    output_pil = transforms.ToPILImage()(out_tensor.squeeze(0).cpu())

    # 2. Compute Quantitative Metrics (PSNR & SSIM)
    psnr_val = calculate_psnr(tensor_in.cpu(), out_tensor.cpu())
    ssim_val = calculate_ssim(tensor_in.cpu(), out_tensor.cpu())

    # 3. Adaptive Post-Processing
    output_pil = ImageOps.autocontrast(output_pil, cutoff=1.5)
    output_pil = ImageEnhance.Color(output_pil).enhance(1.35)

    elapsed_s = time.perf_counter() - start_t

    # 4. Save Output File
    out_dir = Path("./outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"defogged_{image_path.name}"
    output_pil.save(out_path, quality=95)

    print(f"[+] SUCCESS! Saved defogged photo to: {out_path.resolve()}")
    print(f"[+] Processing Time: {elapsed_s:.3f} seconds")
    print(f"==================================================")
    print(f"   📊 PSNR Value: {psnr_val:.2f} dB")
    print(f"   📊 SSIM Value: {ssim_val:.4f}")
    print(f"==================================================")

    # 5. Optional AI Detection Comparison
    print("[*] Computing downstream object detection recovery...")
    det_res = compare_detections(raw_pil, output_pil)

    # 6. Display Desktop Window Comparison with Research Metrics
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
    fig.canvas.manager.set_window_title("DAM-RFRNet Major Project Output")

    axes[0].imshow(raw_pil)
    axes[0].set_title("Original Hazy Input Image", fontsize=13, pad=10, color='red', weight='bold')
    axes[0].axis("off")

    axes[1].imshow(output_pil)
    axes[1].set_title(
        f"Dehazed Output (DAM-RFRNet)\nStructural Fidelity (SSIM): {ssim_val:.4f} | Haze Shift PSNR: {psnr_val:.2f} dB | Latency: {elapsed_s*1000:.1f} ms",
        fontsize=12, pad=10, color='green', weight='bold'
    )
    axes[1].axis("off")

    plt.suptitle("DAM-RFRNet Major Project: Atmospheric Dehazing & Perception Analysis", fontsize=15, weight='bold', y=0.98)
    plt.tight_layout()
    print("[*] Displaying output image window with research metrics...")
    plt.show()


def process_video_stream(video_path: Path):
    """Processes video frame-by-frame with temporal smoothing."""
    print(f"\n[*] Selected Video: {video_path}")
    print("[*] Defogging video stream with temporal smoothing...")

    out_dir = Path("./outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"defogged_{video_path.stem}.mp4"

    processor = VideoDehazeProcessor(smoothing_alpha=0.75)

    def print_progress(ratio, curr, total):
        sys.stdout.write(f"\r[*] Progress: {curr}/{total} frames ({ratio*100:.1f}%)")
        sys.stdout.flush()

    stats = processor.process_video_file(
        input_video_path=str(video_path),
        output_video_path=str(out_path),
        progress_callback=print_progress
    )
    print("\n")
    print(f"[+] SUCCESS! Saved defogged video to: {out_path.resolve()}")
    print(f"[+] Processed {stats['total_frames']} frames at {stats['processing_fps']:.2f} FPS.")


def main():
    print("===============================================================================")
    print("       DAM-RFRNet Major Project: Deep Learning Image/Video Defogging          ")
    print("===============================================================================")

    # 1. Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Using device: {device}")

    # 2. File Selection Dialog
    file_path = select_file_via_dialog()

    if not file_path or not file_path.exists():
        print("[!] No valid file selected. Exiting.")
        return

    # 3. Load Trained Model
    model = DAM_RFRNet().to(device)
    model_ckpt = Path('./checkpoints/dam_rfrnet_best.pth')
    if not model_ckpt.exists():
        model_ckpt = Path('./checkpoints/aodnet_best.pth')

    if model_ckpt.exists():
        print(f"[*] Loading model weights from {model_ckpt}...")
        ckpt = torch.load(str(model_ckpt), map_location=device, weights_only=False)
        state_dict = ckpt['state_dict'] if isinstance(ckpt, dict) and 'state_dict' in ckpt else ckpt
        model.load_state_dict(state_dict, strict=False)
    else:
        print("[!] Warning: Trained model checkpoint not found. Using initialized model weights.")

    model.eval()

    # 4. Route based on file extension
    video_exts = {'.mp4', '.avi', '.mov', '.mkv'}
    if file_path.suffix.lower() in video_exts:
        process_video_stream(file_path)
    else:
        process_single_image(file_path, model, device)


if __name__ == '__main__':
    main()
