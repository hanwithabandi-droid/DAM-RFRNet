import os
import argparse
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

from model import DehazeAttentionNet

def infer(args):
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load Model
    model = DehazeAttentionNet().to(device)
    
    # Load Checkpoint if provided
    if os.path.exists(args.model_path):
        print(f"Loading weights from {args.model_path}")
        checkpoint = torch.load(args.model_path, map_location=device)
        state_dict = checkpoint['state_dict'] if isinstance(checkpoint, dict) and 'state_dict' in checkpoint else checkpoint
        model.load_state_dict(state_dict, strict=False)
    else:
        print(f"Warning: Model path '{args.model_path}' not found. Using untrained weights for demonstration.")

    model.eval()

    # Image preparation
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    if not os.path.exists(args.image_path):
        print(f"Error: Could not find image at {args.image_path}")
        return

    image = Image.open(args.image_path).convert("RGB")
    
    # Optional: resize for inference to avoid out of memory on very large images
    # We will keep original size by default, but it's good to consider for production
    image_tensor = transform(image).unsqueeze(0).to(device) # Add batch dimension

    # Inference
    print("Running inference...")
    with torch.no_grad():
        output_tensor = model(image_tensor)

    # Convert back to PIL Image
    output_tensor = output_tensor.squeeze(0).cpu() # Remove batch dimension
    output_image = transforms.ToPILImage()(output_tensor)

    # Save Output
    os.makedirs(args.output_dir, exist_ok=True)
    out_filename = os.path.join(args.output_dir, "defogged_" + os.path.basename(args.image_path))
    output_image.save(out_filename)
    print(f"Saved defogged image to {out_filename}")

    # Optional: Display Image
    if args.show:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(image)
        axes[0].set_title("Original Hazy Image")
        axes[0].axis("off")
        
        axes[1].imshow(output_image)
        axes[1].set_title("Defogged Image")
        axes[1].axis("off")
        
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="AOD-Net Inference Script")
    parser.add_argument('--image_path', type=str, required=True, help='Path to hazy image')
    parser.add_argument('--model_path', type=str, default='./checkpoints/aodnet_best.pth', help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='Directory to save output image')
    parser.add_argument('--show', action='store_true', help='Display the original and defogged images')
    
    args = parser.parse_args()
    infer(args)
