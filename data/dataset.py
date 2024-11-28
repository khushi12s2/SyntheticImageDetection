import os
import torch
import cv2
from torch.utils.data import Dataset
from torchvision import transforms

# Define paths
real_dir = "C:\Users\shoum\OneDrive\Desktop\SyntheticImageDetection\SyntheticImageDetection\data\real"
synthetic_dir = "C:\Users\shoum\OneDrive\Desktop\SyntheticImageDetection\SyntheticImageDetection\data\synthetic"

# Load image paths
real_images = [os.path.join(real_dir, f) for f in os.listdir(real_dir)]
synthetic_images = [os.path.join(synthetic_dir, f) for f in os.listdir(synthetic_dir)]

# Assign labels (0 for real, 1 for synthetic)
image_paths = real_images + synthetic_images
labels = [0] * len(real_images) + [1] * len(synthetic_images)

# Define custom dataset
class SyntheticImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return image, label

# Define transformations
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Create dataset
dataset = SyntheticImageDataset(image_paths, labels, transform=transform)

# Example of how to use this dataset
from torch.utils.data import DataLoader
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Iterate over batches
for images, labels in train_loader:
    # Do something with images and labels
    print(images.shape, labels)
