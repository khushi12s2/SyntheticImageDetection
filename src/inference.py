import torch
from torchvision import transforms
from PIL import Image

# Load model
model = torch.load("models/model.pth")
model.eval()

# Transform pipeline
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Prediction function
def predict(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image)
    return "Synthetic" if output.item() > 0.5 else "Real"
