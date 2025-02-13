import cv2
import numpy as np
import torch
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