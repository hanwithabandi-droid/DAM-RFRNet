"""
===============================================================================
Custom Neural Image Dehazing & Post-Processing Suite
Architecture: DehazeAttentionNet
===============================================================================
This module provides a modular, object-oriented pipeline for deep-learning 
based atmospheric dehazing, including model loading, tensor processing, 
adaptive post-processing enhancement, and statistical visualization.
"""

import os
import sys
import time
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageOps, ImageEnhance
import matplotlib.pyplot as plt

# Import the neural model architecture
from model import DehazeAttentionNet


class AdaptivePostProcessor:
    """
    Applies image enhancement operations to eliminate residual atmospheric haze,
    restore natural color saturation, and adjust brightness range.
    """
    def __init__(self, contrast_cutoff: float = 2.0, color_gain: float = 1.35, sharpness_gain: float = 1.1):
        self.contrast_cutoff = contrast_cutoff
        self.color_gain = color_gain
        self.sharpness_gain = sharpness_gain

    def process(self, img: Image.Image) -> Image.Image:
        """Runs the enhancement chain on a PIL RGB Image."""
        # 1. Histogram stretching (Auto-Contrast) to expand dynamic range
        enhanced = ImageOps.autocontrast(img, cutoff=self.contrast_cutoff)
        
        # 2. Color saturation restoration
        if self.color_gain != 1.0:
            enhanced = ImageEnhance.Color(enhanced).enhance(self.color_gain)
            
        # 3. Fine detail sharpness adjustment
        if self.sharpness_gain != 1.0:
            enhanced = ImageEnhance.Sharpness(enhanced).enhance(self.sharpness_gain)
            
        return enhanced


class DehazeInferenceEngine:
    """
    Neural Inference Engine managing GPU/CPU hardware context, checkpoint parsing,
    and single-pass forward propagation.
    """
    def __init__(self, model_checkpoint: str, device: Optional[str] = None):
        self.device = torch.device(device if device else ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.model_path = Path(model_checkpoint)
        self.model = DehazeAttentionNet().to(self.device)
        self.model_loaded = self._load_checkpoint()
        
        # Image transformation pipeline
        self.to_tensor = transforms.ToTensor()
        self.to_pil = transforms.ToPILImage()

    def _load_checkpoint(self) -> bool:
        """Safely parses state dictionary keys across various saving conventions."""
        if not self.model_path.exists():
            print(f"[!] Warning: Checkpoint file '{self.model_path}' not found. Initializing with default weights.")
            self.model.eval()
            return False

        try:
            print(f"[*] Loading weights from: {self.model_path}")
            checkpoint = torch.load(str(self.model_path), map_location=self.device, weights_only=False)
            
            # Resolve key structures
            if isinstance(checkpoint, dict):
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                elif 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint

            # Strip 'module.' prefix if saved via DistributedDataParallel
            cleaned_state_dict = {
                (k[7:] if k.startswith('module.') else k): v
                for k, v in state_dict.items()
            }

            self.model.load_state_dict(cleaned_state_dict)
            self.model.eval()
            print("[+] Neural network weights successfully loaded.")
            return True
        except Exception as err:
            print(f"[!] Error loading model weights: {err}")
            self.model.eval()
            return False

    @torch.no_grad()
    def defog(self, input_image: Image.Image) -> Tuple[Image.Image, float]:
        """
        Executes model inference on an input RGB Image.
        Returns: (Raw Defogged PIL Image, Inference time in seconds)
        """
        start_time = time.perf_counter()
        
        # Tensor formatting
        tensor_in = self.to_tensor(input_image).unsqueeze(0).to(self.device)
        
        # Forward pass
        raw_output_tensor = self.model(tensor_in)
        
        # Post-processing normalization
        raw_output_tensor = torch.clamp(raw_output_tensor.squeeze(0).cpu(), 0.0, 1.0)
        output_image = self.to_pil(raw_output_tensor)
        
        elapsed_time = time.perf_counter() - start_time
        return output_image, elapsed_time


def acquire_image_file() -> Optional[Path]:
    """
    Attempts to launch a native desktop file dialog.
    Falls back gracefully if Tkinter or GUI context is unavailable.
    """
    try:
        import tkinter as tk
        from tkinter import filedialog
        
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        
        selected = filedialog.askopenfilename(
            title="Select Hazy Target Image",
            filetypes=[("Supported Images", "*.jpg *.jpeg *.png *.bmp *.webp"), ("All Files", "*.*")]
        )
        root.destroy()
        
        if selected:
            return Path(selected)
    except Exception:
        pass

    # CLI fallback prompt if GUI dialog is skipped or unavailable
    user_input = input("Enter path to hazy image: ").strip('"\' ')
    if user_input:
        path = Path(user_input)
        if path.exists():
            return path
    return None


def render_comparison_dashboard(
    original_img: Image.Image,
    enhanced_img: Image.Image,
    metrics: Dict[str, Any]
) -> None:
    """
    Displays a styled before/after visual dashboard using Matplotlib.
    """
    plt.style.use('dark_background') if 'dark_background' in plt.style.available else None
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    fig.canvas.manager.set_window_title("Neural Haze Removal Results")

    # Original Image Frame
    axes[0].imshow(original_img)
    axes[0].set_title(f"Hazy Input ({original_img.width}x{original_img.height})", fontsize=13, pad=10, color='#ff6b6b')
    axes[0].axis('off')

    # Restored Image Frame
    axes[1].imshow(enhanced_img)
    axes[1].set_title(f"Dehazed Output (Latency: {metrics['latency']:.3f}s)", fontsize=13, pad=10, color='#51cf66')
    axes[1].axis('off')

    plt.suptitle("Deep Learning Atmospheric Dehazing & Color Recovery", fontsize=16, y=0.96, weight='bold')
    plt.tight_layout()
    plt.show()


def run_pipeline(
    image_path: Optional[Path] = None,
    checkpoint_path: str = "./checkpoints/aodnet_best.pth",
    output_directory: str = "./outputs",
    display: bool = True
) -> None:
    """
    Full pipeline execution controller.
    """
    # 1. Resolve Target Image File
    if image_path is None or not Path(image_path).exists():
        print("[*] Launching file selection prompt...")
        image_path = acquire_image_file()

    if not image_path or not image_path.is_file():
        print("[!] No valid image selected. Exiting pipeline.")
        return

    print(f"[*] Processing Target Image: {image_path}")

    # 2. Initialize Engine & Enhancer
    engine = DehazeInferenceEngine(model_checkpoint=checkpoint_path)
    enhancer = AdaptivePostProcessor(contrast_cutoff=2.0, color_gain=1.35, sharpness_gain=1.05)

    # 3. Read Input Image
    try:
        raw_hazy_img = Image.open(image_path).convert("RGB")
    except Exception as exc:
        print(f"[!] Failed to read image file: {exc}")
        return

    # 4. Neural Model Inference
    print("[*] Running neural defogging pass...")
    neural_defogged_img, latency = engine.defog(raw_hazy_img)

    # 5. Adaptive Post-Processing
    print("[*] Applying adaptive visual enhancements...")
    final_output_img = enhancer.process(neural_defogged_img)

    # 6. Save Result
    out_dir = Path(output_directory)
    out_dir.mkdir(parents=True, exist_ok=True)
    destination_file = out_dir / f"defogged_{image_path.name}"
    
    final_output_img.save(destination_file, quality=95)
    print(f"[+] Processed image successfully saved to: {destination_file.resolve()}")

    # 7. Visualization
    if display:
        stats = {
            'latency': latency,
            'device': str(engine.device),
            'checkpoint': checkpoint_path
        }
        render_comparison_dashboard(raw_hazy_img, final_output_img, stats)


if __name__ == "__main__":
    # Standard script execution entry point
    run_pipeline()
