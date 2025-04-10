import torch
import torch.utils.data
from PIL import Image
import random
from pathlib import Path
from torchvision import transforms
from torch.utils.data import DataLoader
from bcc_detection.models.bcc_model import BCCModel
from bcc_detection.configs.config import Config

class BCCDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None, split='train', num_samples=None, used_samples=None):       
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.split = split
        self.samples = self._load_samples(num_samples, used_samples)
        print(f"Loaded {len(self.samples)} samples for {split} split")

    def _load_samples(self, num_samples=None, used_samples=None):
        samples = []

        # Load BCC (positive) samples
        bcc_dir = self.data_dir / "package" / "bcc" / "data" / "images"
        if bcc_dir.exists():
            for img_path in bcc_dir.glob("*.tif"):
                samples.append({
                    'image_path': str(img_path),
                    'label': 1  # BCC is positive class
                })

        # Load non-malignant (negative) samples
        normal_dir = self.data_dir / "package" / "non-malignant" / "data" / "images"
        if normal_dir.exists():
            for img_path in normal_dir.glob("*.tif"):
                samples.append({
                    'image_path': str(img_path),
                    'label': 0  # Non-malignant is negative class
                })

        print(f"Found {len(samples)} total samples before filtering")

        # Exclude previously used samples
        if used_samples:
            samples = [s for s in samples if s['image_path'] not in used_samples]

        # Randomly sample if num_samples is specified
        if num_samples is not None:
            # Ensure equal number of samples from each class
            bcc_samples = [s for s in samples if s['label'] == 1]
            normal_samples = [s for s in samples if s['label'] == 0]

            print(f"Found {len(bcc_samples)} BCC samples and {len(normal_samples)} normal samples")

            # Take equal number of samples from each class
            num_samples_per_class = num_samples // 2
            bcc_samples = random.sample(bcc_samples, min(num_samples_per_class, len(bcc_samples)))
            normal_samples = random.sample(normal_samples, min(num_samples_per_class, len(normal_samples))) 

            samples = bcc_samples + normal_samples
            print(f"Selected {len(samples)} samples ({len(bcc_samples)} BCC, {len(normal_samples)} normal)")

        # Split into train/val/test
        random.shuffle(samples)
        n = len(samples)
        if self.split == 'train':
            return samples[:int(0.7*n)]
        elif self.split == 'val':
            return samples[int(0.7*n):int(0.85*n)]
        else:  # test
            return samples[int(0.85*n):]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        try:
            # Load and resize the image
            image = Image.open(sample['image_path'])

            # Get original dimensions
            width, height = image.size

            # Calculate new dimensions while maintaining aspect ratio
            target_size = 4096  # Maximum dimension
            if width > height:
                new_width = target_size
                new_height = int(height * (target_size / width))
            else:
                new_height = target_size
                new_width = int(width * (target_size / height))

            # Resize the image
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

            # Convert to RGB
            image = image.convert('RGB')

            # Extract a random 224x224 patch
            if new_width > 224 and new_height > 224:
                left = random.randint(0, new_width - 224)
                top = random.randint(0, new_height - 224)
                image = image.crop((left, top, left + 224, top + 224))
            else:
                # If image is smaller than 224x224, pad it
                image = image.resize((224, 224), Image.Resampling.LANCZOS)

            # Apply transforms if specified
            if self.transform:
                image = self.transform(image)

            return image, sample['label']

        except Exception as e:
            print(f"Error loading image {sample['image_path']}: {str(e)}")
            # Return a black image and label 0 in case of error
            return torch.zeros((3, 224, 224)), 0

def create_data_loaders(data_dir, batch_size, num_samples=None, used_samples=None):
    # Define transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create datasets
    train_dataset = BCCDataset(data_dir, transform=transform, split='train', 
                             num_samples=num_samples, used_samples=used_samples)
    val_dataset = BCCDataset(data_dir, transform=transform, split='val', 
                           num_samples=num_samples, used_samples=used_samples)
    test_dataset = BCCDataset(data_dir, transform=transform, split='test', 
                            num_samples=num_samples, used_samples=used_samples)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader, test_loader 