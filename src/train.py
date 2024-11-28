import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn import BCELoss
from src.data_processing import SyntheticDataset, get_transforms
from src.model import CNNModel

# Paths and labels
real_images = ["data/real/" + f for f in os.listdir("data/real/")]
synthetic_images = ["data/synthetic/" + f for f in os.listdir("data/synthetic/")]
image_paths = real_images + synthetic_images
labels = [0] * len(real_images) + [1] * len(synthetic_images)

# Dataset and DataLoader
transform = get_transforms()
dataset = SyntheticDataset(image_paths, labels, transform=transform)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Model, Loss, Optimizer
model = CNNModel()
criterion = BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(10):
    model.train()
    epoch_loss = 0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels.unsqueeze(1))
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}")

# Save the model
torch.save(model, "models/model.pth")
