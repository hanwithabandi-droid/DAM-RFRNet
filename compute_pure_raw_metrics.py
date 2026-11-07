import torch
import numpy as np
import pandas as pd
from dataset import DefoggingDataset
from torch.utils.data import DataLoader, random_split
from model import DAM_RFRNet
from run_baseline_without_novel_blocks import DehazeAttentionNetWithoutNovelBlocks
from model_utils import calculate_psnr, calculate_ssim
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[*] Compute Device: {device}")

dataset = DefoggingDataset(root_dir='c:/Minor project/data', augment=False)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
_, test_ds = random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42))
loader = DataLoader(test_ds, batch_size=1, shuffle=False)

ckpt = torch.load('c:/Minor project/checkpoints/dam_rfrnet_best.pth', map_location=device, weights_only=False)
state = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt

# Model With Novel Blocks (DAM-RFRNet)
model_with_novel = DAM_RFRNet().to(device)
model_with_novel.load_state_dict(state, strict=False)
model_with_novel.eval()

# Model Without Novel Blocks (M0 Baseline)
model_without_novel = DehazeAttentionNetWithoutNovelBlocks().to(device)
model_without_novel.load_state_dict(state, strict=False)
model_without_novel.eval()

res = []
print(f"[*] Computing pure raw mathematical metrics for all {len(test_ds)} test images...")
pbar = tqdm(loader, desc="Evaluating All 800 Images")

with torch.no_grad():
    for batch in pbar:
        h = batch['hazy'].to(device)
        c = batch['clean'].to(device)
        name = batch['name'][0]
        
        # 1. Input Hazy Image
        p_hazy = float(calculate_psnr(h, c))
        s_hazy = float(calculate_ssim(h, c))
        
        # 2. Model Without Novel Blocks
        out_no = torch.clamp(model_without_novel(h), 0.0, 1.0)
        p_no = float(calculate_psnr(out_no, c))
        s_no = float(calculate_ssim(out_no, c))
        
        # 3. Model With Novel Blocks
        out_with = torch.clamp(model_with_novel(h), 0.0, 1.0)
        p_with = float(calculate_psnr(out_with, c))
        s_with = float(calculate_ssim(out_with, c))
        
        res.append({
            'name': name,
            'hazy_psnr': p_hazy, 'hazy_ssim': s_hazy,
            'no_novel_psnr': p_no, 'no_novel_ssim': s_no,
            'with_novel_psnr': p_with, 'with_novel_ssim': s_with
        })

df = pd.DataFrame(res)
csv_path = 'c:/Minor project/outputs/all_800_raw_comparison.csv'
df.to_csv(csv_path, index=False)

print("\n" + "=" * 75)
print("   EXACT PURE RAW MATHEMATICAL BENCHMARK METRICS (ALL 800 TEST IMAGES)   ")
print("=" * 75)
print(f"  Input Hazy Image (Untreated) : PSNR = {df['hazy_psnr'].mean():.2f} +/- {df['hazy_psnr'].std():.2f} dB | SSIM = {df['hazy_ssim'].mean():.4f}")
print(f"  Without Novel Blocks (M0)    : PSNR = {df['no_novel_psnr'].mean():.2f} +/- {df['no_novel_psnr'].std():.2f} dB | SSIM = {df['no_novel_ssim'].mean():.4f}")
print(f"  With Novel Blocks (DAM-RFRNet): PSNR = {df['with_novel_psnr'].mean():.2f} +/- {df['with_novel_psnr'].std():.2f} dB | SSIM = {df['with_novel_ssim'].mean():.4f}")
print("-" * 75)
print(f"  Gain of With-Novel over Hazy : {df['with_novel_psnr'].mean() - df['hazy_psnr'].mean():+.2f} dB PSNR | {df['with_novel_ssim'].mean() - df['hazy_ssim'].mean():+.4f} SSIM")
print(f"  Gain of Novel Blocks (RFRB+DSCA) : {df['with_novel_psnr'].mean() - df['no_novel_psnr'].mean():+.2f} dB PSNR | {df['with_novel_ssim'].mean() - df['no_novel_ssim'].mean():+.4f} SSIM")
print("=" * 75)
