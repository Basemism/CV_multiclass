import os
import cv2
import numpy as np
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

# Create a heatmap with a Gaussian centered at 'point'
def generate_prompt_heatmap(img_size, point, kernel_size=5):

    prompt = np.zeros((img_size, img_size), dtype=np.float32)
    prompt[point[0], point[1]] = 1.0
    prompt = cv2.GaussianBlur(prompt, (kernel_size, kernel_size), 0)
    # Normalize so max is 1
    if prompt.max() > 0:
        prompt = prompt / prompt.max()
    return prompt

class PromptDatasetBinary(Dataset):
    def __init__(self, image_paths, mask_paths, img_size, samples_per_image=5):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.img_size = img_size
        self.samples_per_image = samples_per_image

    def __len__(self):
        return len(self.image_paths) * self.samples_per_image

    def __getitem__(self, idx):
        # Map index to an image.
        image_idx = idx % len(self.image_paths)
        img_path = self.image_paths[image_idx]
        mask_path = self.mask_paths[image_idx]
        
        # Load and process image.
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Image not found at {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.img_size, self.img_size))
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))  # (3, H, W)
        
        # Load and process mask.
        # Original trimap: 1 = foreground, 2 = background, 3 = unclassified.
        # For binary segmentation, we map:
        # foreground (mask==1) -> 1, background (mask==2) -> 0, unclassified (mask==3) -> ignore (255)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Mask not found at {mask_path}")
        mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        binary_mask = np.full(mask.shape, 255, dtype=np.uint8)
        binary_mask[mask == 2] = 0
        binary_mask[mask == 1] = 1
        
        # Sample a random point uniformly over the image.
        # Note: if the pixel is unclassified we treat it as background
        valid_mask = (binary_mask != 255)  # use only valid pixels
        rows, cols = np.where(valid_mask)
        if len(rows) == 0:
            # choose center
            point = (self.img_size // 2, self.img_size // 2)
        else:
            idx_point = np.random.choice(len(rows))
            point = (rows[idx_point], cols[idx_point])
        
        # Determine if this prompt is a foreground or background prompt.
        if binary_mask[point[0], point[1]] == 1:
            # Foreground prompt: target is the original binary_mask.
            target = binary_mask.copy()
        else:
            # Background prompt: target is all background (zeros).
            target = np.zeros_like(binary_mask, dtype=np.uint8)
        
        # Create prompt heat map.
        prompt_heat = generate_prompt_heatmap(self.img_size, point, kernel_size=5)
        prompt_heat = np.expand_dims(prompt_heat, axis=0)  # (1, H, W)
        
        # Concatenate image and prompt heat map to form a 4-channel input.
        input_tensor = np.concatenate([image, prompt_heat], axis=0)  # (4, H, W)
        
        # Convert to torch tensors.
        input_tensor = torch.from_numpy(input_tensor).float()
        target_tensor = torch.from_numpy(target).long()
        
        return input_tensor, target_tensor

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    epoch_loss = 0.0
    for inputs, targets in loader:
        inputs = inputs.to(device)    # (B, 4, H, W)
        targets = targets.to(device)  # (B, H, W)
        optimizer.zero_grad()
        outputs = model(inputs)       # (B, num_classes, H, W)
        loss = criterion(outputs, targets)
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
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            epoch_loss += loss.item() * inputs.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == targets).sum().item()
            total_pixels += torch.numel(targets)
    accuracy = correct / total_pixels
    return epoch_loss / len(loader.dataset), accuracy

if __name__ == '__main__':
    # Settings.
    dim = 256
    num_classes = 2  # Binary segmentation: 0 = background, 1 = object
    batch_size = 16
    num_epochs = 50
    seed = 42
    samples_per_image = 6

    train_images_dir = f'./trainval_{dim}/images'
    train_masks_dir  = f'./trainval_{dim}/annotations'

    image_paths = sorted(glob.glob(os.path.join(train_images_dir, '*.png')))
    mask_paths  = sorted(glob.glob(os.path.join(train_masks_dir, '*.png')))

    train_img_paths, val_img_paths, train_mask_paths, val_mask_paths = train_test_split(
        image_paths, mask_paths, test_size=0.1, random_state=seed, shuffle=True)

    print(f"Training images: {len(train_img_paths)}")
    print(f"Validation images: {len(val_img_paths)}")

    # Create dataset instances.
    train_dataset = PromptDatasetBinary(train_img_paths, train_mask_paths, dim, samples_per_image=samples_per_image)
    val_dataset   = PromptDatasetBinary(val_img_paths, val_mask_paths, dim, samples_per_image=samples_per_image)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}")

    model = PromptUNet(num_classes=num_classes, in_channels=4).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=255)

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