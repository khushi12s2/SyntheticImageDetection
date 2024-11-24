from torchvision import models, transforms
import torch
from PIL import Image

class MetaFeatureExtractor:
    def __init__(self):
        # Load DINO pretrained model
        self.model = torch.hub.load('facebookresearch/dino:main', 'dino_vits8')
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
    
    def extract_features(self, image_path):
        img = Image.open(image_path).convert("RGB")
        input_tensor = self.transform(img).unsqueeze(0)
        with torch.no_grad():
            features = self.model(input_tensor)
        return features

if __name__ == "__main__":
    extractor = MetaFeatureExtractor()
    features = extractor.extract_features("./data/synthetic_images/synthetic_0.png")
    print(f"Extracted features: {features.shape}")
