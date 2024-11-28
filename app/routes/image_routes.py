from fastapi import APIRouter, File, UploadFile
from PIL import Image
import torch
from app.models.detector_model import load_model, detect_image # type: ignore
from app.utils.preprocess import preprocess_image  # type: ignore

router = APIRouter()

# Load the model at startup
model = load_model()

@router.post("/detect/")
async def detect_synthetic_image(file: UploadFile = File(...)):
    # Load and preprocess image
    img = Image.open(file.file)
    img_tensor = preprocess_image(img)
    
    # Perform prediction
    result = detect_image(model, img_tensor)
    
    return {"result": "Synthetic" if result > 0.5 else "Real"}
