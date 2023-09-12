

import torch
from diffusion_model import DiffusionModel
import matplotlib.pyplot as plt
import yaml
import argparse
from torchvision import datasets, transforms
from diffusion_model import DiffusionModel

def parse_args():
    parser = argparse.ArgumentParser(description="Diffusion Model Inference Script")
    parser.add_argument("--config", default="config.yaml", help="Path to config file (default: config.yaml)")
    return parser.parse_args()

def main(config_path):
    # Load Config
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # Load the trained model
    model = DiffusionModel().to(torch.device("mps" if torch.backends.mps.is_available() else "cpu"))
    model.load_state_dict(torch.load(config["model_path"]))
    model.eval()  # Set the model to evaluation mode

    # Generate images from random noise
    with torch.no_grad():
        random_noise = torch.randn(config["num_images"], 3, config["image_size"], config["image_size"]).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        generated_images = model(random_noise.to("mps" if torch.backends.mps.is_available() else "cpu"))
        generated_images = generated_images.view(config["num_images"], 3, config["image_size"], config["image_size"])

    # Display the generated images
    fig, axs = plt.subplots(1, config["num_images"], figsize=(50, 50))
    for i in range(config["num_images"]):
        image = transforms.ToPILImage()(generated_images[i].cpu())
        axs[i].imshow(image)
        axs[i].axis('off')
    plt.show()

if __name__ == "__main__":
    args = parse_args()
    main(args.config)
