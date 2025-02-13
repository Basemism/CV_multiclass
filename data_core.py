import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

class myDataset(Dataset):
    def __init__(self, image_paths, mask_paths, img_size):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.img_size = img_size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Image not found at {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.img_size, self.img_size))
        image = image.astype(np.float32) / 255.0  
        image = np.transpose(image, (2, 0, 1))
        
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Mask not found at {mask_path}")
        mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        mask = mask - 1
        
        image_tensor = torch.from_numpy(image).float()
        mask_tensor = torch.from_numpy(mask).long()
        
        return image_tensor, mask_tensor

class CLIPDataset(Dataset):
    def __init__(self, image_paths, mask_paths, target_dim, clip_model, clip_preprocess, device):
        """
        image_paths: list of paths to images.
        mask_paths: list of corresponding mask paths.
        target_dim: desired output segmentation mask resolution (e.g., 256).
        clip_model: the loaded CLIP model.
        clip_preprocess: the CLIP preprocessing transform.
        device: the torch device.
        """
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.target_dim = target_dim
        self.clip_model = clip_model
        self.clip_preprocess = clip_preprocess
        self.device = device

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image using OpenCV and convert to PIL Image
        image = cv2.imread(self.image_paths[idx])
        if image is None:
            raise ValueError(f"Image not found: {self.image_paths[idx]}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image)

        # Use CLIP's own preprocessing (which resizes to 224x224 by default)
        image_tensor = self.clip_preprocess(pil_image).unsqueeze(0).to(self.device)

        # Extract CLIP features (note: the feature map shape depends on the model architecture)
        with torch.no_grad():
            clip_features = self.clip_model.encode_image(image_tensor)
        clip_features = clip_features.squeeze(0)  # Remove the batch dimension.

        # Load and process the corresponding mask.
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Mask not found: {self.mask_paths[idx]}")
        mask = cv2.resize(mask, (self.target_dim, self.target_dim), interpolation=cv2.INTER_NEAREST)
        mask = mask - 1  # Shift labels from {1,2,3} to {0,1,2}
        mask = torch.from_numpy(mask).long()

        return clip_features, mask
