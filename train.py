import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from pytorch_msssim import MS_SSIM
from model import DehazeAttentionNet, DAM_RFRNet
from dataset import DefoggingDataset
from model_utils import calculate_psnr, calculate_ssim, save_checkpoint
import torchvision.models as models

class VGGPerceptualLoss(nn.Module):
    """VGG-16 feature perceptual loss for high-level visual structure preservation."""
    def __init__(self):
        super(VGGPerceptualLoss, self).__init__()
        try:
            weights = models.VGG16_Weights.DEFAULT if hasattr(models, 'VGG16_Weights') else True
            vgg = models.vgg16(weights=weights).features
            self.slice = nn.Sequential(*[vgg[i] for i in range(16)]).eval()
            for param in self.slice.parameters():
                param.requires_grad = False
            self.available = True
        except Exception as e:
            print(f"[!] VGG weights download skipped ({e}). Falling back to MS-SSIM + L1 loss.")
            self.available = False

    def forward(self, x, y):
        if not self.available:
            return torch.tensor(0.0, device=x.device)
        return nn.functional.l1_loss(self.slice(x), self.slice(y))

def train(args):
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize model
    if args.model_type.lower() == 'dam_rfrnet':
        print("[*] Initializing novel DAM-RFRNet model architecture...")
        model = DAM_RFRNet().to(device)
    else:
        print("[*] Initializing baseline DehazeAttentionNet model architecture...")
        model = DehazeAttentionNet().to(device)

    # Resume / Load pre-trained checkpoint weights if available
    resume_path = args.resume if args.resume else os.path.join(args.ckpt_dir, "dam_rfrnet_best.pth")
    if not (resume_path and os.path.exists(resume_path)):
        resume_path = os.path.join(args.ckpt_dir, "aodnet_best.pth")
        
    if resume_path and os.path.exists(resume_path):
        print(f"[*] Loading pre-trained checkpoint weights from: {resume_path}")
        try:
            ckpt = torch.load(resume_path, map_location=device, weights_only=False)
            state_dict = ckpt['state_dict'] if isinstance(ckpt, dict) and 'state_dict' in ckpt else ckpt
            model.load_state_dict(state_dict, strict=False)
            print("[+] Pre-trained weights loaded successfully! Continuing training on full 4,000 images.")
        except Exception as e:
            print(f"[!] Notice: Could not parse checkpoint ({e}), initializing weights.")

    # Initialize loss functions and optimizer
    criterion_l1 = nn.L1Loss()
    criterion_ms_ssim = MS_SSIM(data_range=1.0, size_average=True, channel=3)
    perceptual_loss_fn = VGGPerceptualLoss().to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # Data Loader with 80:20 Train:Test Split
    print(f"Loading dataset from {args.data_dir}...")
    try:
        full_dataset = DefoggingDataset(root_dir=args.data_dir, augment=True)
        total_samples = len(full_dataset)
        train_size = int(0.80 * total_samples)
        val_size = total_samples - train_size

        # Reproducible 80:20 partition
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

        print("=" * 55)
        print("         DATASET PARTITION (80:20 SPLIT)         ")
        print("=" * 55)
        print(f" Total Dataset Size        : {total_samples} image pairs")
        print(f" Training Set (80%)        : {train_size} image pairs")
        print(f" Testing / Validation (20%): {val_size} image pairs")
        print("=" * 55 + "\n")
    except Exception as e:
        print(f"Could not load dataset: {e}")
        print("Please ensure your data_dir contains 'hazy' and 'clean' subdirectories with paired images.")
        return

    # Checkpoint directory
    os.makedirs(args.ckpt_dir, exist_ok=True)

    # Training Loop
    best_psnr = 0.0
    
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        
        # Use tqdm for progress bar
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in pbar:
            hazy_images = batch['hazy'].to(device)
            clean_images = batch['clean'].to(device)

            # Forward pass
            outputs = model(hazy_images)
            
            loss_ms_ssim = 1 - criterion_ms_ssim(outputs, clean_images)
            loss_l1 = criterion_l1(outputs, clean_images)
            loss_vgg = perceptual_loss_fn(outputs, clean_images)
            
            # Hybrid Loss: 0.6 * MS-SSIM + 0.3 * L1 + 0.1 * Perceptual
            loss = 0.6 * loss_ms_ssim + 0.3 * loss_l1 + (0.1 * loss_vgg if perceptual_loss_fn.available else 0.0)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})

        # Calculate average loss for the epoch
        avg_loss = epoch_loss / len(train_loader)
        
        # Evaluate on the 20% Unseen Validation/Testing Set
        model.eval()
        val_psnr_list = []
        val_ssim_list = []
        with torch.no_grad():
            for val_batch in val_loader:
                v_hazy = val_batch['hazy'].to(device)
                v_clean = val_batch['clean'].to(device)
                v_out = model(v_hazy)
                val_psnr_list.append(calculate_psnr(v_out, v_clean))
                val_ssim_list.append(calculate_ssim(v_out, v_clean))

        current_psnr = float(np.mean(val_psnr_list)) if val_psnr_list else 0.0
        current_ssim = float(np.mean(val_ssim_list)) if val_ssim_list else 0.0

        print(f"Epoch [{epoch+1}/{args.epochs}] | Train Loss: {avg_loss:.6f} | Val PSNR (20% Test Set): {current_psnr:.2f} dB | Val SSIM: {current_ssim:.4f}")

        # Save checkpoint if it's the best so far
        is_best = current_psnr > best_psnr
        best_psnr = max(current_psnr, best_psnr)
        
        filename_prefix = "dam_rfrnet" if args.model_type.lower() == 'dam_rfrnet' else "aodnet"
        
        if is_best or (epoch + 1) % 10 == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_psnr': best_psnr,
                'best_ssim': current_ssim,
                'optimizer': optimizer.state_dict(),
            }, filename=os.path.join(args.ckpt_dir, f"{filename_prefix}_epoch_{epoch+1}.pth"))
            
            # Save a generic "latest" or "best" model as well for easy inference
            if is_best:
                ckpt_data = {
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_psnr': best_psnr,
                    'best_ssim': current_ssim,
                    'optimizer': optimizer.state_dict(),
                }
                save_checkpoint(ckpt_data, filename=os.path.join(args.ckpt_dir, f"{filename_prefix}_best.pth"))
                save_checkpoint(ckpt_data, filename=os.path.join(args.ckpt_dir, "aodnet_best.pth"))
                save_checkpoint(ckpt_data, filename=os.path.join(args.ckpt_dir, "dam_rfrnet_best.pth"))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="DAM-RFRNet & AOD-Net Training Script")
    parser.add_argument('--model_type', type=str, default='dam_rfrnet', choices=['dam_rfrnet', 'aodnet'], help='Model architecture to train')
    parser.add_argument('--data_dir', type=str, default='./data', help='Path to dataset directory')
    parser.add_argument('--ckpt_dir', type=str, default='./checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--resume', type=str, default='', help='Path to checkpoint to resume training from')
    
    args = parser.parse_args()
    train(args)

