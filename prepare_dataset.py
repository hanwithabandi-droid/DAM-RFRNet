import os
import urllib.request
import numpy as np
import cv2
from PIL import Image

def synthesize_haze(image_np):
    """
    Applies synthetic haze to a clear image using the atmospheric scattering model.
    I(x) = J(x)t(x) + A(1 - t(x))
    """
    # Create a dummy transmission map (t) based on a vertical gradient 
    # to roughly simulate depth (further away at top, closer at bottom)
    rows, cols, _ = image_np.shape
    
    # Introduce random parameters for variation
    beta = np.random.uniform(0.8, 2.5) # scattering coefficient
    A = np.random.uniform(0.6, 0.9)    # atmospheric light 
    
    # Depth map: 0 (near/bottom) to 1 (far/top), plus some noise
    depth_map = np.linspace(1.0, 0.2, rows)[:, np.newaxis] * np.ones((rows, cols))
    noise = np.random.normal(0, 0.1, (rows, cols))
    depth_map = np.clip(depth_map + noise, 0, 1)
    
    # calc transmission: t = exp(-beta * depth)
    t = np.exp(-beta * depth_map)
    t = np.dstack([t, t, t]) # broadcasting to 3 channels
    
    # Apply scattering model
    J = image_np / 255.0
    I = J * t + A * (1 - t)
    
    # Ensure final values are clipped properly
    I = np.clip(I * 255, 0, 255).astype(np.uint8)
    return I

def main():
    root_dir = './data'
    hazy_dir = os.path.join(root_dir, 'hazy')
    clean_dir = os.path.join(root_dir, 'clean')
    
    os.makedirs(hazy_dir, exist_ok=True)
    os.makedirs(clean_dir, exist_ok=True)
    
    num_samples = 150 # Increased dataset size for better training
    print(f"Downloading and generating {num_samples} image pairs...")
    
    base_url = "https://picsum.photos/256/256"
    
    for i in range(1, num_samples + 1):
        filename = f"image_{i:03d}.jpg"
        clean_path = os.path.join(clean_dir, filename)
        hazy_path = os.path.join(hazy_dir, filename)
        
        try:
            # Download a random clear image
            urllib.request.urlretrieve(base_url, clean_path)
            
            # Read the image
            img = cv2.imread(clean_path)
            
            # Picsum sometimes fails or returns empty if rate limited, verifying we have an image
            if img is not None:
                # Add synthetic haze
                hazy_img = synthesize_haze(img)
                
                # Save hazy image
                cv2.imwrite(hazy_path, hazy_img)
                if i % 10 == 0:
                    print(f"Generated {i}/{num_samples} pairs...")
            else:
                print(f"Failed to read downloaded image {i}")
        except Exception as e:
            print(f"Error on image {i}: {e}")

    print("\nDataset preparation complete! You can now run the training script.")

if __name__ == '__main__':
    main()
