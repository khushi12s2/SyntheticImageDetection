from albumentations import ( # type: ignore
    Compose, RandomCrop, HorizontalFlip, RandomBrightnessContrast, GaussianBlur
)
from albumentations.pytorch import ToTensorV2 # type: ignore
import cv2 # type: ignore
import os

def augment_image(image_path, output_dir):
    augmentations = Compose([
        RandomCrop(width=224, height=224),
        HorizontalFlip(p=0.5),
        RandomBrightnessContrast(p=0.5),
        GaussianBlur(p=0.2),
        ToTensorV2()
    ])

    image = cv2.imread(image_path)
    augmented = augmentations(image=image)["image"]
    output_path = os.path.join(output_dir, os.path.basename(image_path))
    cv2.imwrite(output_path, augmented.permute(1, 2, 0).numpy() * 255)

if __name__ == "__main__":
    augment_image("./data/synthetic_images/synthetic_0.png", "./data/processed/")
