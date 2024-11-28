from fastapi import FastAPI, File, UploadFile
from PIL import Image
import torch
from torchvision import transforms

app = FastAPI()

# Load model
model = torch.load("models/model.pth")
model.eval()

# Transform pipeline
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(file.file).convert("RGB")
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image)
    confidence = output.item()
    label = "Synthetic" if confidence > 0.5 else "Real"
    return {"label": label, "confidence": confidence}
