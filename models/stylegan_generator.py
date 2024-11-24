import torch # type: ignore
from torchvision.utils import save_image # type: ignore
import os

# Load a pretrained StyleGAN2 generator
from stylegan2_pytorch import ModelLoader  # type: ignore # Install 'stylegan2-pytorch'

def generate_images(output_dir, num_images=100, seed=None):
    # Set random seed
    if seed is not None:
        torch.manual_seed(seed)
    
    # Load StyleGAN2 model (ensure pretrained weights are downloaded)
    model = ModelLoader(gan_type='stylegan2', image_size=256)
    os.makedirs(output_dir, exist_ok=True)

    # Generate synthetic images
    for i in range(num_images):
        z = torch.randn(1, model.latent_dim).cuda()  # Latent vector
        img = model.generate(z)
        save_image(img, os.path.join(output_dir, f"synthetic_{i}.png"))

    print(f"Generated {num_images} synthetic images in {output_dir}")

if __name__ == "__main__":
    generate_images(output_dir="./data/synthetic_images", num_images=50)
