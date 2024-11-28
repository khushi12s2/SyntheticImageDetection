# src/data_processing.py
import os
import cv2
import torch
from albumentations import Compose, Resize, Flip, Normalize
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

class SyntheticImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        """
        Args:
            image_paths (list): List of image file paths.
            labels (list): Corresponding labels (0 for real, 1 for synthetic).
            transform (callable): Transformation to be applied on a sample.
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image=image)["image"]
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return image, label


def get_transforms():
    """
    Data augmentation and transformation pipeline.
    """
    return Compose([
        Resize(128, 128),  # Resize to 128x128
        Flip(p=0.5),       # Random horizontal flip
        Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),  # Normalize to [-1, 1]
        ToTensorV2()       # Convert to PyTorch tensor
    ])
