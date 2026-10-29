"""
===============================================================================
MASTER COMPARISON: ALL 3 DEHAZING MODELS SIDE-BY-SIDE
===============================================================================
Executes all 3 models on the same hazy input image:
  1. Proposed DAM-RFRNet (With novel RFRB + DSCA) -> Test Avg PSNR: 20.46 dB
  2. Traditional Baseline (Dark Channel Prior - DCP) -> Test Avg PSNR: 20.31 dB
  3. Reference DehazeAttentionNet (Without novel blocks) -> Test Avg PSNR: 20.24 dB
Generates a side-by-side comparison figure and benchmark card (Pure Raw Mathematics).
===============================================================================
"""

import os
import sys
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

PROJECT_ROOT = "c:/Minor project" if os.path.exists("c:/Minor project") else os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from run_model1_dam_rfrnet import run_model_1, MODEL1_AVG_PSNR, MODEL1_AVG_SSIM
from run_model2_traditional_dcp import run_model_2, MODEL2_AVG_PSNR, MODEL2_AVG_SSIM
from run_model3_baseline_no_novel import run_model_3, MODEL3_AVG_PSNR, MODEL3_AVG_SSIM

def compare_all(hazy_path=None, clean_path=None, output_dir=None):
    if hazy_path is None:
        hazy_path = "c:/Minor project/data/hazy/image_001.jpg" if os.path.exists("c:/Minor project/data/hazy/image_001.jpg") else "./data/hazy/image_001.jpg"
    if clean_path is None:
        fname = os.path.basename(hazy_path)
        clean_path = f"c:/Minor project/data/clean/{fname}" if os.path.exists(f"c:/Minor project/data/clean/{fname}") else f"./data/clean/{fname}"
    if output_dir is None:
        output_dir = "c:/Minor project/outputs" if os.path.exists("c:/Minor project/outputs") else "./outputs"

    print("=" * 75)
    print("      EXECUTING MASTER COMPARISON ACROSS ALL 3 DEHAZING MODELS      ")
    print("=" * 75)

    # 1. Run each model individually
    run_model_1(hazy_path, clean_path, output_dir)
    run_model_2(hazy_path, clean_path, output_dir)
    run_model_3(hazy_path, clean_path, output_dir)

    # 2. Load the 3 generated output images
    m1_img = Image.open(os.path.join(output_dir, "model1_dam_rfrnet_output.jpg"))
    m2_img = Image.open(os.path.join(output_dir, "model2_traditional_dcp_output.jpg"))
    m3_img = Image.open(os.path.join(output_dir, "model3_reference_no_novel_output.jpg"))
    hazy_img = Image.open(hazy_path).convert("RGB")
    clean_img = Image.open(clean_path).convert("RGB") if os.path.exists(clean_path) else None

    # 3. Build Combined 5-Panel High-Resolution Figure
    fig, axes = plt.subplots(1, 5, figsize=(25, 6))

    axes[0].imshow(hazy_img)
    axes[0].set_title("(a) Input Hazy Image", fontsize=11, fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(m1_img)
    axes[1].set_title(f"(b) 1. Proposed DAM-RFRNet\nTest Avg: {MODEL1_AVG_PSNR:.2f} dB | {MODEL1_AVG_SSIM:.4f}", fontsize=11, fontweight='bold', color='darkgreen')
    axes[1].axis('off')

    axes[2].imshow(m2_img)
    axes[2].set_title(f"(c) 2. Traditional (DCP)\nTest Avg: {MODEL2_AVG_PSNR:.2f} dB | {MODEL2_AVG_SSIM:.4f}", fontsize=11, fontweight='bold', color='darkred')
    axes[2].axis('off')

    axes[3].imshow(m3_img)
    axes[3].set_title(f"(d) 3. Reference (No Novel Blocks)\nTest Avg: {MODEL3_AVG_PSNR:.2f} dB | {MODEL3_AVG_SSIM:.4f}", fontsize=11, fontweight='bold', color='darkorange')
    axes[3].axis('off')

    if clean_img is not None:
        axes[4].imshow(clean_img)
        axes[4].set_title("(e) Ground Truth Clean Image", fontsize=11, fontweight='bold')
    axes[4].axis('off')

    plt.suptitle("Quantitative & Qualitative Benchmark across All 3 Dehazing Models", fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    master_fig_path = os.path.join(output_dir, "all_3_models_master_comparison.png")
    plt.savefig(master_fig_path, dpi=160, bbox_inches='tight')
    plt.close()

    print("\n" + "=" * 75)
    print("      SUMMARY BENCHMARK TABLE: ALL 3 MODELS ON 800 TEST DATASET     ")
    print("=" * 75)
    print(f"{'Model Architecture':<36} | {'Test Avg PSNR':<14} | {'Test Avg SSIM':<14} | {'Paper Reference'}")
    print("-" * 75)
    print(f"{'1. Proposed DAM-RFRNet':<36} | {MODEL1_AVG_PSNR:<6.2f} dB     | {MODEL1_AVG_SSIM:<14.4f} | Proposed Network")
    print(f"{'2. Traditional Model (DCP) [1]':<36} | {MODEL2_AVG_PSNR:<6.2f} dB     | {MODEL2_AVG_SSIM:<14.4f} | He et al., 2011")
    print(f"{'3. Reference (No Novel Blocks) [M0]':<36} | {MODEL3_AVG_PSNR:<6.2f} dB     | {MODEL3_AVG_SSIM:<14.4f} | Ablation Backbone")
    print(f"{'Net Gain over Traditional Baseline':<36} | {MODEL1_AVG_PSNR - MODEL2_AVG_PSNR:+5.2f} dB     | {MODEL1_AVG_SSIM - MODEL2_AVG_SSIM:+13.4f} | Edge & Structure Gain")
    print(f"{'Net Gain over Baseline Backbone':<36} | {MODEL1_AVG_PSNR - MODEL3_AVG_PSNR:+5.2f} dB     | {MODEL1_AVG_SSIM - MODEL3_AVG_SSIM:+13.4f} | Novel Blocks Gain (RFRB+DSCA)")
    print("=" * 75)
    print(f"\n[+] Master 5-panel comparison figure saved to: {master_fig_path}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Master comparison of all 3 models")
    parser.add_argument('--hazy_path', type=str, default="c:/Minor project/data/hazy/image_001.jpg", help="Path to hazy image")
    parser.add_argument('--clean_path', type=str, default="c:/Minor project/data/clean/image_001.jpg", help="Path to clean image")
    parser.add_argument('--output_dir', type=str, default="c:/Minor project/outputs", help="Output directory")
    args = parser.parse_args()
    compare_all(args.hazy_path, args.clean_path, args.output_dir)
