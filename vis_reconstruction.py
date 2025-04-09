import os
import glob
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import pickle

from models.autoencoder import Autoencoder

def load_image(image_path, img_dim):
    """Load an image from disk, convert to RGB, resize and normalize."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (img_dim, img_dim))
    image = image.astype(np.float32) / 255.0  # Normalize to [0,1]
    return image

def preprocess_for_model(image):
    """Convert image (H,W,3) to tensor (1,3,H,W)."""
    image = np.transpose(image, (2, 0, 1))  # CHW format
    image_tensor = torch.from_numpy(image).float().unsqueeze(0)
    return image_tensor

def visualize_reconstruction(autoencoder, image_paths, img_dim, device, num_samples=5):
    autoencoder.eval()
    fig, axes = plt.subplots(num_samples, 2, figsize=(8, 4*num_samples))

    with torch.no_grad():
        for idx, image_path in enumerate(image_paths[:num_samples]):
            # Load and preprocess image.
            orig_img = load_image(image_path, img_dim)
            input_tensor = preprocess_for_model(orig_img).to(device)
            
            reconstructed = autoencoder(input_tensor)
            reconstructed = reconstructed.squeeze(0).cpu().numpy()
            reconstructed = np.transpose(reconstructed, (1, 2, 0))
            reconstructed = np.clip(reconstructed, 0, 1)
            
            # Plot original image and reconstructed image.
            axes[idx, 0].imshow(orig_img)
            axes[idx, 0].set_title("Original")
            axes[idx, 0].axis('off')
            
            axes[idx, 1].imshow(reconstructed)
            axes[idx, 1].set_title("Reconstructed")
            axes[idx, 1].axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualize Autoencoder Reconstructions")
    parser.add_argument('--img_dim', type=int, default=256, help='Image dimension to resize to')
    parser.add_argument('--weights', type=str, default='./ae_weights/autoencoder_256_epochs_50.pth', help='Path to autoencoder weights file')
    parser.add_argument('--num_samples', type=int, default=5, help='Number of samples to visualize')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Instantiate and load the autoencoder.
    latent_dim = 64
    autoencoder = Autoencoder(latent_dim=latent_dim).to(device)
    autoencoder.load_state_dict(torch.load(args.weights, map_location=device))
    
    # Get a list of image paths.
    image_paths = pickle.load(open('test_image_paths.pkl', 'rb'))
    if len(image_paths) == 0:
        raise ValueError("No images found in the specified directory.")
    
    visualize_reconstruction(autoencoder, image_paths, args.img_dim, device, num_samples=args.num_samples)