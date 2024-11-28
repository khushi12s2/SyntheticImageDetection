import torch
from torch import nn

# Define or load a pre-trained model
class SyntheticImageDetector(nn.Module):
    def __init__(self):
        super(SyntheticImageDetector, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.fc(x)

def load_model():
    model = SyntheticImageDetector()
    model.load_state_dict(torch.load("app/models/stylegan_model.pth", map_location="cpu"))
    model.eval()
    return model

def detect_image(model, img_tensor):
    with torch.no_grad():
        output = model(img_tensor)
        return output.item()
