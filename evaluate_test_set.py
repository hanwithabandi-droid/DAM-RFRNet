"""
===============================================================================
DAM-RFRNet Benchmark Suite: Test Set Evaluation (800 Unseen Image Pairs)
===============================================================================
Evaluates the model across the 20% held-out test split (800 image pairs from
the 4,000-image dataset partition: 3,200 training / 800 testing).
Computes:
  - Pure Raw Mathematical PSNR (10 * log10(1 / MSE)) & Standard Deviation
  - Pure Raw Mathematical SSIM & Standard Deviation
  - Min / Max raw metrics
  - Average Inference Time per Image (ms)
Saves summary report to ./outputs/test_set_evaluation_report.txt
and per-image metrics to ./outputs/test_metrics_breakdown.csv
"""

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from PIL import Image

from model import DAM_RFRNet, DehazeAttentionNet
from dataset import DefoggingDataset
from model_utils import calculate_psnr, calculate_ssim

def evaluate_test_set(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 65)
    print("       DAM-RFRNet BENCHMARK EVALUATION (20% TEST SET)       ")
    print("=" * 65)
    print(f"[*] Compute Device     : {device}")
    print(f"[*] Model Checkpoint   : {args.model_path}")
    print(f"[*] Dataset Directory  : {args.data_dir}")

    # 1. Load Model Architecture & Checkpoint
    model = DAM_RFRNet().to(device)
    if os.path.exists(args.model_path):
        print(f"[*] Loading checkpoint weights: {args.model_path}")
        ckpt = torch.load(args.model_path, map_location=device, weights_only=False)
        state_dict = ckpt['state_dict'] if isinstance(ckpt, dict) and 'state_dict' in ckpt else ckpt
        model.load_state_dict(state_dict, strict=False)
        print("[+] Checkpoint loaded successfully!")
    else:
        print(f"[!] Warning: Checkpoint {args.model_path} not found. Running initialized weights.")

    model.eval()

    # 2. Prepare 80:20 Test Split (800 Images from 4,000 total)
    print("\n[*] Loading dataset and partitioning into 80:20 split...")
    full_dataset = DefoggingDataset(root_dir=args.data_dir, augment=False)
    total_samples = len(full_dataset)
    train_size = int(0.80 * total_samples)
    test_size = total_samples - train_size

    # Deterministic reproducible 80:20 split matching training
    _, test_dataset = random_split(
        full_dataset,
        [train_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    print(f"[*] Total Dataset Size   : {total_samples} pairs")
    print(f"[*] Training Pool (80%)  : {train_size} pairs")
    print(f"[*] Held-out Test Set (20%): {len(test_dataset)} pairs")
    print("=" * 65 + "\n")

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    # 3. Evaluation Loop
    results = []
    latencies = []

    print(f"[*] Evaluating defogging performance across all {len(test_dataset)} test pairs...")
    pbar = tqdm(test_loader, desc="Evaluating Test Set")

    with torch.no_grad():
        for batch in pbar:
            hazy_t = batch['hazy'].to(device)
            clean_t = batch['clean'].to(device)
            img_name = batch['name'][0]

            # Measure inference latency
            t0 = time.perf_counter()
            out_t = model(hazy_t)
            out_t = torch.clamp(out_t, 0.0, 1.0)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            latencies.append(latency_ms)

            # Pure Raw Tensor Metrics
            psnr_raw = float(calculate_psnr(out_t, clean_t))
            ssim_raw = float(calculate_ssim(out_t, clean_t))

            results.append({
                'image_name': img_name,
                'psnr_db': psnr_raw,
                'ssim': ssim_raw,
                'raw_psnr_db': psnr_raw,
                'raw_ssim': ssim_raw,
                'latency_ms': latency_ms
            })

            pbar.set_postfix({
                'Avg PSNR': f"{np.mean([r['psnr_db'] for r in results]):.2f} dB",
                'Avg SSIM': f"{np.mean([r['ssim'] for r in results]):.4f}"
            })

    # 4. Statistical Aggregation
    df = pd.DataFrame(results)

    avg_psnr = df['psnr_db'].mean()
    std_psnr = df['psnr_db'].std()
    min_psnr = df['psnr_db'].min()
    max_psnr = df['psnr_db'].max()

    avg_ssim = df['ssim'].mean()
    std_ssim = df['ssim'].std()
    min_ssim = df['ssim'].min()
    max_ssim = df['ssim'].max()

    avg_latency = np.mean(latencies)

    # 5. Display Summary Card in Terminal
    print("\n" + "=" * 65)
    print(f"          FINAL TEST SET QUANTITATIVE RESULTS ({len(df)} PAIRS)          ")
    print("=" * 65)
    print(f"  Images Evaluated         : {len(df)} unseen testing pairs")
    print(f"  Average PSNR             : {avg_psnr:.2f} +/- {std_psnr:.2f} dB")
    print(f"  Peak PSNR                : {max_psnr:.2f} dB  (Min: {min_psnr:.2f} dB)")
    print(f"  Average SSIM             : {avg_ssim:.4f} +/- {std_ssim:.4f}")
    print(f"  Peak SSIM                : {max_ssim:.4f}  (Min: {min_ssim:.4f})")
    print(f"  Average Inference Time   : {avg_latency:.2f} ms per image")
    print(f"  Effective Throughput     : {1000.0 / avg_latency:.1f} FPS")
    print("=" * 65)

    # 6. Save Report & CSV to ./outputs/
    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "test_metrics_breakdown.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n[+] Saved detailed per-image breakdown to: {csv_path}")

    report_path = os.path.join(args.output_dir, "test_set_evaluation_report.txt")
    with open(report_path, "w") as f:
        f.write("=================================================================\n")
        f.write("     DAM-RFRNet MINOR PROJECT: TEST SET EVALUATION REPORT        \n")
        f.write("=================================================================\n\n")
        f.write(f"Evaluated Test Samples   : {len(df)} unseen image pairs (20% split)\n")
        f.write(f"Model Architecture       : DAM-RFRNet (Dual Attention + RFRB)\n")
        f.write(f"Checkpoint Loaded        : {args.model_path}\n")
        f.write(f"Execution Device         : {device}\n\n")
        f.write("-----------------------------------------------------------------\n")
        f.write("AGGREGATED PERFORMANCE METRICS\n")
        f.write("-----------------------------------------------------------------\n")
        f.write(f"Average PSNR (dB)        : {avg_psnr:.2f} dB (+/- {std_psnr:.2f} dB)\n")
        f.write(f"Maximum PSNR             : {max_psnr:.2f} dB\n")
        f.write(f"Minimum PSNR             : {min_psnr:.2f} dB\n")
        f.write(f"Average SSIM             : {avg_ssim:.4f} (+/- {std_ssim:.4f})\n")
        f.write(f"Maximum SSIM             : {max_ssim:.4f}\n")
        f.write(f"Minimum SSIM             : {min_ssim:.4f}\n")
        f.write(f"Mean Latency (ms/image)  : {avg_latency:.2f} ms\n")
        f.write(f"Throughput (FPS)         : {1000.0 / avg_latency:.1f} FPS\n\n")
        f.write("=================================================================\n")

    print(f"[+] Saved complete evaluation report to: {report_path}")

    # 7. Optional Distribution Plot
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        axes[0].hist(df['psnr_db'], bins=25, color='#2ea043', edgecolor='black', alpha=0.85)
        axes[0].axvline(avg_psnr, color='darkred', linestyle='dashed', linewidth=2, label=f'Mean: {avg_psnr:.2f} dB')
        axes[0].set_title("Test Set PSNR Distribution", fontsize=12)
        axes[0].set_xlabel("PSNR (dB)")
        axes[0].set_ylabel("Number of Images")
        axes[0].legend()
        axes[0].grid(axis='y', alpha=0.3)

        axes[1].hist(df['ssim'], bins=25, color='#0969da', edgecolor='black', alpha=0.85)
        axes[1].axvline(avg_ssim, color='darkred', linestyle='dashed', linewidth=2, label=f'Mean: {avg_ssim:.4f}')
        axes[1].set_title("Test Set SSIM Distribution", fontsize=12)
        axes[1].set_xlabel("SSIM Score")
        axes[1].set_ylabel("Number of Images")
        axes[1].legend()
        axes[1].grid(axis='y', alpha=0.3)

        plt.tight_layout()
        dist_plot_path = os.path.join(args.output_dir, "test_metrics_distribution.png")
        plt.savefig(dist_plot_path, dpi=150)
        plt.close()
        print(f"[+] Saved metrics distribution chart to: {dist_plot_path}")
    except Exception as e:
        print(f"[!] Chart generation note: {e}")

    print("\nBenchmark Evaluation Complete!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate DAM-RFRNet on the 20% Test Set (800 images)")
    parser.add_argument('--model_path', type=str, default='c:/Minor project/checkpoints/dam_rfrnet_best.pth' if os.path.exists('c:/Minor project/checkpoints/dam_rfrnet_best.pth') else './checkpoints/dam_rfrnet_best.pth', help='Path to model weights')
    parser.add_argument('--data_dir', type=str, default='c:/Minor project/data' if os.path.exists('c:/Minor project/data') else './data', help='Path to data folder')
    parser.add_argument('--output_dir', type=str, default='c:/Minor project/outputs' if os.path.exists('c:/Minor project/outputs') else './outputs', help='Directory to save results')
    args = parser.parse_args()
    evaluate_test_set(args)
