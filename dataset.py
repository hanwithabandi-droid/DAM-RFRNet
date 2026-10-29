import os
import glob
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch

class DefoggingDataset(Dataset):
    """
    Standard Dataset Loader for small/moderate collections.
    """
    def __init__(self, root_dir, hazy_dir="hazy", clean_dir="clean", transform=None, augment=False):
        self.root_dir = root_dir
        self.hazy_dir = os.path.join(root_dir, hazy_dir)
        self.clean_dir = os.path.join(root_dir, clean_dir)
        
        self.hazy_images = sorted([f for f in os.listdir(self.hazy_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        self.clean_images = sorted([f for f in os.listdir(self.clean_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        
        assert len(self.hazy_images) == len(self.clean_images), "Mismatched number of hazy and clean images"
        
        if transform:
            self.transform = transform
        else:
            if augment:
                self.transform = transforms.Compose([
                    transforms.Resize((256, 256)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor()
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.Resize((256, 256)),
                    transforms.ToTensor()
                ])

    def __len__(self):
        return len(self.hazy_images)

    def __getitem__(self, idx):
        hazy_img_path = os.path.join(self.hazy_dir, self.hazy_images[idx])
        clean_img_path = os.path.join(self.clean_dir, self.clean_images[idx])

        hazy_image = Image.open(hazy_img_path).convert("RGB")
        clean_image = Image.open(clean_img_path).convert("RGB")

        if self.transform:
            hazy_image = self.transform(hazy_image)
            clean_image = self.transform(clean_image)

        return {'hazy': hazy_image, 'clean': clean_image, 'name': self.hazy_images[idx]}


class KaggleLargeDataset(Dataset):
    """
    Robust scaling class built specifically to handle massive 1M+ image datasets 
    via lazy traversal without crashing system RAM.
    """
    def __init__(self, dataset_path, transform=None):
        self.dataset_path = dataset_path
        print(f"Indexing Massive Kaggle Dataset at {dataset_path}...")
        
        # In Kaggle driving datasets, files are sometimes uniquely named or sequentially named in folders.
        # We will attempt to find all commonly known 'fog/haze' and 'clear' structures.
        hazy_patterns = ['*fog*/**/*.png', '*fog*/**/*.jpg', '*hazy*/**/*.png', '*hazy*/**/*.jpg', '**/hazy/*.jpg']
        clean_patterns = ['*clear*/**/*.png', '*clear*/**/*.jpg', '*gt*/**/*.png', '*gt*/**/*.jpg', '**/clean/*.jpg']
        
        self.hazy_paths = []
        self.clean_paths = []
        
        for p in hazy_patterns:
            self.hazy_paths.extend(glob.glob(os.path.join(dataset_path, p), recursive=True))
            
        for p in clean_patterns:
            self.clean_paths.extend(glob.glob(os.path.join(dataset_path, p), recursive=True))
            
        self.hazy_paths = sorted(self.hazy_paths)
        self.clean_paths = sorted(self.clean_paths)
        
        # If the Kaggle dataset is strictly unsupervised (only fog driving, no clear images),
        # we will use a self-fed loop (just generating the output without MSE target support)
        self.unsupervised = False
        if len(self.clean_paths) == 0:
            print("WARNING: This Kaggle Dataset appears to NOT contain paired 'clear/ground truth' images.")
            print("To train AODNet, it theoretically needs a clean reference. We will operate in exploratory mode where loss is synthetic for structural testing only.")
            self.unsupervised = True
            self.clean_paths = self.hazy_paths # Dummy fallback to prevent crashes

        elif len(self.hazy_paths) != len(self.clean_paths):
            print(f"WARNING: Found {len(self.hazy_paths)} hazy vs {len(self.clean_paths)} clean images.")
            print("Truncating the larger set to match sizes by forcing pairs.")
            min_len = min(len(self.hazy_paths), len(self.clean_paths))
            self.hazy_paths = self.hazy_paths[:min_len]
            self.clean_paths = self.clean_paths[:min_len]
            
        print(f"Successfully Indexed {len(self.hazy_paths)} image pairs.")
        
        if transform:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)), # Standardized tensor output
                transforms.ToTensor()
            ])

    def __len__(self):
        return len(self.hazy_paths)

    def __getitem__(self, idx):
        hazy_image = Image.open(self.hazy_paths[idx]).convert("RGB")
        clean_image = Image.open(self.clean_paths[idx]).convert("RGB")

        if self.transform:
            hazy_image = self.transform(hazy_image)
            clean_image = self.transform(clean_image)

        return {'hazy': hazy_image, 'clean': clean_image, 'name': os.path.basename(self.hazy_paths[idx])}
