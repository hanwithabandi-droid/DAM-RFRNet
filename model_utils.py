import torch
import math

try:
    from pytorch_msssim import ssim
except ImportError:
    ssim = None

def calculate_psnr(img1, img2):
    """
    Compute Peak Signal-to-Noise Ratio (PSNR)
    img1, img2: PyTorch tensors, shape (B, C, H, W) or (C, H, W)
    Values range [0, 1]
    """
    if img1.dim() == 3:
        img1 = img1.unsqueeze(0)
    if img2.dim() == 3:
        img2 = img2.unsqueeze(0)
        
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    
    max_pixel = 1.0
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse.item()))
    return psnr

def calculate_ssim(img1, img2):
    """
    Compute Structural Similarity Index Measure (SSIM)
    img1, img2: PyTorch tensors, shape (B, C, H, W) or (C, H, W)
    Values range [0, 1]
    """
    if img1.dim() == 3:
        img1 = img1.unsqueeze(0)
    if img2.dim() == 3:
        img2 = img2.unsqueeze(0)
        
    if ssim is not None:
        try:
            val = ssim(img1, img2, data_range=1.0, size_average=True)
            return val.item()
        except Exception:
            pass
            
    # Simple direct structural similarity estimation fallback
    mu1 = torch.mean(img1)
    mu2 = torch.mean(img2)
    sigma1_sq = torch.var(img1)
    sigma2_sq = torch.var(img2)
    sigma12 = torch.mean((img1 - mu1) * (img2 - mu2))
    
    c1 = (0.01 * 1.0) ** 2
    c2 = (0.03 * 1.0) ** 2
    
    ssim_val = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / ((mu1**2 + mu2**2 + c1) * (sigma1_sq + sigma2_sq + c2))
    return ssim_val.item()

def save_checkpoint(state, filename="checkpoint.pth.tar"):
    """
    Saves a model checkpoint.
    """
    torch.save(state, filename)
    print(f"Checkpoint saved to {filename}")

def load_checkpoint(checkpoint_path, model, optimizer=None):
    """
    Loads a model checkpoint.
    """
    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer'])
    print(f"Checkpoint loaded from {checkpoint_path}")
    return model, optimizer, checkpoint.get('epoch', 0)

