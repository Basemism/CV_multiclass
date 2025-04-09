import pickle
import os
import glob
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse

from models.autoencoder import Autoencoder

# Load an image from disk, convert to RGB, resize and normalize
def load_image(image_path, img_dim):
   
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (img_dim, img_dim))
    image = image.astype(np.float32) / 255.0  # Normalize to [0,1]
    return image

# Convert image (H,W,3) to tensor (1,3,H,W).
def preprocess_for_model(image):
    image = np.transpose(image, (2, 0, 1))  # CHW format
    image_tensor = torch.from_numpy(image).float().unsqueeze(0)
    return image_tensor

# Visualize the latent representation of images using the autoencoder.
def visualize_latent(autoencoder, image_paths, img_dim, device, num_samples=5, num_latent_channels=8):
    autoencoder.eval()
    fig, axes = plt.subplots(num_samples, num_latent_channels + 1, figsize=(3 * (num_latent_channels + 1), 3 * num_samples))
    for idx, image_path in enumerate(image_paths[:num_samples]):
        # Load and preprocess the image.
        orig_img = load_image(image_path, img_dim)
        input_tensor = preprocess_for_model(orig_img).to(device)
        
        # Forward pass through the autoencoder.
        with torch.no_grad():
            # Get latent representation from the encoder.
            latent = autoencoder.encoder(input_tensor)
        
        # Process latent representation.
        # latent shape: (1, C, H_latent, W_latent)
        latent = latent.squeeze(0).cpu().numpy()  # shape (C, H_latent, W_latent)
        C, H_latent, W_latent = latent.shape

        # Choose a subset of latent channels to visualize.
        num_channels_to_show = min(num_latent_channels, C)
        latent_channels = latent[:num_channels_to_show]
        
        # Normalize each latent channel for display.
        latent_channels_disp = []
        for ch in latent_channels:
            ch_min, ch_max = ch.min(), ch.max()
            if ch_max - ch_min > 1e-5:
                ch_norm = (ch - ch_min) / (ch_max - ch_min)
            else:
                ch_norm = ch - ch_min
            latent_channels_disp.append(ch_norm)
        
        latent_channels_disp = np.array(latent_channels_disp)  # (num_channels, H_latent, W_latent)
        
        # Plot original image.
        axes[idx, 0].imshow(orig_img)
        axes[idx, 0].set_title("Original")
        axes[idx, 0].axis('off')
        
        # Plot latent channels.
        for i in range(num_channels_to_show):
            axes[idx, i + 1].imshow(latent_channels_disp[i], cmap='viridis')
            axes[idx, i + 1].set_title(f"Channel {i}")
            axes[idx, i + 1].axis('off')
        
        # Hide any extra axes.
        for i in range(num_channels_to_show + 1, len(axes[idx])):
            axes[idx, i].axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualize Autoencoder Latent Representations")
    parser.add_argument('--img_dim', type=int, default=256, help='Image dimension to resize to')
    parser.add_argument('--weights', type=str, default='./ae_weights/autoencoder_256_epochs_50.pth', help='Path to autoencoder weights file')
    parser.add_argument('--data_dir', type=str, default='./trainval_256/images', help='Directory with images for visualization')
    parser.add_argument('--num_samples', type=int, default=5, help='Number of samples to visualize')
    parser.add_argument('--num_latent_channels', type=int, default=8, help='Number of latent channels to show per sample')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Instantiate and load the autoencoder.
    latent_dim = 64  # Adjust as needed.
    autoencoder = Autoencoder(latent_dim=latent_dim).to(device)
    autoencoder.load_state_dict(torch.load(args.weights, map_location=device))
    
    # Get image paths.
    image_paths = pickle.load(open('test_image_paths.pkl', 'rb'))
    if len(image_paths) == 0:
        raise ValueError("No images found in the specified directory.")
    
    visualize_latent(autoencoder, image_paths, args.img_dim, device, num_samples=args.num_samples, num_latent_channels=args.num_latent_channels)
