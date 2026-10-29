import os
import argparse
import kagglehub
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import AODNet
from dataset import KaggleLargeDataset
from model_utils import calculate_psnr, save_checkpoint

def run_overnight_training(dataset_path, epochs, batch_size):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Deploying Overnight Pipeline on: {device}")
    
    # Init Model and Loss
    model = AODNet().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

    # Boot up the massive dataset loader
    try:
        train_dataset = KaggleLargeDataset(dataset_path=dataset_path)
        # Using num_workers=4 for faster massive IO, warning: on Windows this sometimes requires __main__ guard
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    except Exception as e:
        print(f"FATAL ERROR Loading Kaggle Dataset: {e}")
        return

    os.makedirs("./checkpoints", exist_ok=True)
    best_psnr = 0.0

    print(f"\n--- INITIATING MASSIVE DATASET TRAINING LOOP ({epochs} Epochs) ---")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        # tqdm for big progress
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch_idx, batch in enumerate(pbar):
            hazy_images = batch['hazy'].to(device)
            clean_images = batch['clean'].to(device)

            optimizer.zero_grad()
            outputs = model(hazy_images)
            loss = criterion(outputs, clean_images)
            
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
            # Save micro-checkpoints every 1000 batches since Kaggle dataset is huge
            if batch_idx % 1000 == 0 and batch_idx > 0:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'batch': batch_idx,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, filename=f"./checkpoints/aodnet_checkpoint_e{epoch+1}_b{batch_idx}.pth")

        avg_loss = epoch_loss / max(len(train_loader), 1)
        
        # Eval subset
        model.eval()
        with torch.no_grad():
            outputs = model(hazy_images)
            current_psnr = calculate_psnr(outputs, clean_images)
            
        print(f"\nEpoch [{epoch+1}/{epochs}] Complete! Loss: {avg_loss:.6f}, Batch PSNR: {current_psnr:.2f} dB")

        # Save main epoch checkpoints
        is_best = current_psnr > best_psnr
        best_psnr = max(current_psnr, best_psnr)
        
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_psnr': best_psnr,
            'optimizer': optimizer.state_dict(),
        }, filename=f"./checkpoints/aodnet_epoch_{epoch+1}_kaggle.pth")
        
        if is_best:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_psnr': best_psnr,
            }, filename="./checkpoints/aodnet_best.pth")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Overnight Kaggle Training Trigger")
    parser.add_argument('--kaggle_handle', type=str, default="mandybr/fogroad-12m-dataset", help='Kaggle dataset repository handle')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs (keep low for massive datasets)')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for GPU RAM control')
    
    args = parser.parse_args()
    
    print("===============================================")
    print(f"Triggering KaggleHub Download: {args.kaggle_handle}")
    print("WARNING: This may take several hours. Do not close the window.")
    print("===============================================")
    
    # 1. Block and Download the 83.4 GB dataset locally
    dataset_path = kagglehub.dataset_download(args.kaggle_handle)
    print(f"\n[OK] Dataset Download Complete. Files located at: {dataset_path}")
    
    # 2. Immediately trigger the new robust data loader and training loop
    run_overnight_training(dataset_path=dataset_path, epochs=args.epochs, batch_size=args.batch_size)
