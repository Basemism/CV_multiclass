import os
import glob
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import cv2
import numpy as np

from models.prompt_unet import PromptUNet

class PromptDataset(Dataset):
    def __init__(self, image_paths, mask_paths, img_size):
        """
        image_paths: List of image file paths.
        mask_paths: List of corresponding segmentation mask file paths.
        img_size: Image size to which to resize images and masks.
        """
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.img_size = img_size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        
        # Load and process image.
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Image not found at {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.img_size, self.img_size))
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))  # (3, H, W)
        
        # Load and process mask.
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Mask not found at {mask_path}")
        mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        # Remap mask:
        # Original trimap: 1 = foreground, 2 = background, 3 = unclassified.
        # We want: background = 0, foreground = (cat:1 or dog:2), unclassified = 255.
        filename = os.path.basename(img_path)
        if filename[0].isupper():
            fg_class = 1  # cat
        else:
            fg_class = 2  # dog
        new_mask = np.full(mask.shape, 255, dtype=np.uint8)
        new_mask[mask == 2] = 0
        new_mask[mask == 1] = fg_class
        
        # Create prompt heat map.
        # We sample a random foreground point (where mask==1 before remapping).
        foreground_indices = np.argwhere(mask == 1)
        prompt = np.zeros((self.img_size, self.img_size), dtype=np.float32)
        if len(foreground_indices) > 0:
            chosen_idx = foreground_indices[np.random.choice(len(foreground_indices))]
            prompt[chosen_idx[0], chosen_idx[1]] = 1.0
            # Smooth the prompt with a Gaussian filter.
            prompt = cv2.GaussianBlur(prompt, (15,15), 0)
        
        # Convert to tensors.
        image_tensor = torch.from_numpy(image).float()    # (3, H, W)
        prompt_tensor = torch.from_numpy(prompt).unsqueeze(0).float()  # (1, H, W)
        # Concatenate image and prompt into a 4-channel input.
        input_tensor = torch.cat([image_tensor, prompt_tensor], dim=0)  # (4, H, W)
        mask_tensor = torch.from_numpy(new_mask).long()  # (H, W)
        
        return input_tensor, mask_tensor



# Training function.
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    epoch_loss = 0.0
    for inputs, masks in loader:
        inputs = inputs.to(device)  # shape: (B, 4, H, W)
        masks = masks.to(device)    # shape: (B, H, W)
        optimizer.zero_grad()
        outputs = model(inputs)     # (B, num_classes, H, W)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * inputs.size(0)
    return epoch_loss / len(loader.dataset)

def evaluate(model, loader, criterion, device):
    model.eval()
    epoch_loss = 0.0
    correct = 0
    total_pixels = 0
    with torch.no_grad():
        for inputs, masks in loader:
            inputs = inputs.to(device)
            masks = masks.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, masks)
            epoch_loss += loss.item() * inputs.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == masks).sum().item()
            total_pixels += torch.numel(masks)
    accuracy = correct / total_pixels
    return epoch_loss / len(loader.dataset), accuracy

# Settings.
dim = 256
num_classes = 3  # 0: background, 1: cat, 2: dog
batch_size = 16
num_epochs = 50
seed = 42

train_images_dir = f'./trainval_{dim}/images'
train_masks_dir  = f'./trainval_{dim}/annotations'

image_paths = sorted(glob.glob(os.path.join(train_images_dir, '*.png')))
mask_paths  = sorted(glob.glob(os.path.join(train_masks_dir, '*.png')))

train_img_paths, val_img_paths, train_mask_paths, val_mask_paths = train_test_split(
    image_paths, mask_paths, test_size=0.1, random_state=seed, shuffle=True)

print(f"Training images: {len(train_img_paths)}")
print(f"Validation images: {len(val_img_paths)}")

# Create dataset instances.
train_dataset = PromptDataset(train_img_paths, train_mask_paths, dim)
val_dataset   = PromptDataset(val_img_paths, val_mask_paths, dim)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device {device}")

model = PromptUNet(num_classes=num_classes, in_channels=4).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss(ignore_index=255)  # Ignore unclassified regions.

best_val_loss = float('inf')
for epoch in tqdm(range(1, num_epochs+1), desc="Training Epochs"):
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)
    tqdm.write(f"Epoch: {epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_path = f"prompt_weights/prompt_unet_model_{dim}_epochs_{num_epochs}.pth"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model.state_dict(), save_path)

print("Training complete.")
