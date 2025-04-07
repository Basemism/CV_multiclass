import os
import glob
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from models.autoencoder import Autoencoder, SegmentationModel

class RawImageDataset(Dataset):
    """
    Dataset for training the autoencoder.
    Loads raw images from a directory, resizes and normalizes them.
    """
    def __init__(self, image_paths, img_size):
        self.image_paths = image_paths
        self.img_size = img_size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Image not found: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.img_size, self.img_size))
        image = image.astype(np.float32) / 255.0
        # Change from HWC to CHW.
        image = np.transpose(image, (2, 0, 1))
        return torch.from_numpy(image).float(), torch.from_numpy(image).float()  # input and target are the same

class AESegDataset(Dataset):
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

        filename = os.path.basename(img_path)
        if filename[0].isupper():
            fg_class = 1  # cat
        else:
            fg_class = 2  # dog
        
        new_mask = np.full(mask.shape, 255, dtype=np.uint8)
        new_mask[mask == 2] = 0
        new_mask[mask == 1] = fg_class
        
        image_tensor = torch.from_numpy(image).float()
        mask_tensor = torch.from_numpy(new_mask).long()
        
        return image_tensor, mask_tensor

def train_autoencoder(model, loader, optimizer, criterion, device):
    model.train()
    epoch_loss = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * inputs.size(0)
    return epoch_loss / len(loader.dataset)

def evaluate_autoencoder(model, loader, criterion, device):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            epoch_loss += loss.item() * inputs.size(0)
    return epoch_loss / len(loader.dataset)

def train_segmentation(model, loader, optimizer, criterion, device):
    model.train()
    epoch_loss = 0
    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)  # (B, num_classes, H, W)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * images.size(0)
    return epoch_loss / len(loader.dataset)

def evaluate_segmentation(model, loader, criterion, device):
    model.eval()
    epoch_loss = 0
    correct, total = 0, 0
    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            epoch_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == masks).sum().item()
            total += torch.numel(masks)
    accuracy = correct / total
    return epoch_loss / len(loader.dataset), accuracy


# Settings.
img_dim = 256
batch_size = 16
num_epochs_ae = 50
num_epochs_seg = 50
num_seg_classes = 3  # e.g., background, cat, dog
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Get image paths for autoencoder training.
image_paths = sorted(glob.glob(os.path.join(f'./trainval_{img_dim}/images', '*.png')))
train_paths, val_paths = train_test_split(image_paths, test_size=0.1, random_state=42)

train_dataset_ae = RawImageDataset(train_paths, img_dim)
val_dataset_ae   = RawImageDataset(val_paths, img_dim)
train_loader_ae = DataLoader(train_dataset_ae, batch_size=batch_size, shuffle=True)
val_loader_ae   = DataLoader(val_dataset_ae, batch_size=batch_size, shuffle=False)

print("Training Autoencoder...")
autoencoder = Autoencoder().to(device)
optimizer_ae = optim.Adam(autoencoder.parameters(), lr=1e-3)
criterion_ae = nn.MSELoss()

for epoch in range(1, num_epochs_ae+1):
    train_loss = train_autoencoder(autoencoder, train_loader_ae, optimizer_ae, criterion_ae, device)
    val_loss = evaluate_autoencoder(autoencoder, val_loader_ae, criterion_ae, device)
    print(f"AE Epoch {epoch}/{num_epochs_ae}  Train Loss: {train_loss:.4f}  Val Loss: {val_loss:.4f}")

# Save autoencoder weights.
os.makedirs("ae_weights", exist_ok=True)
torch.save(autoencoder.state_dict(), f"ae_weights/autoencoder_{img_dim}_epochs_{num_epochs_ae}.pth")
print("Autoencoder training complete.")


# Freeze the autoencoder encoder.
for param in autoencoder.encoder.parameters():
    param.requires_grad = False

seg_image_paths = sorted(glob.glob(os.path.join(f'./trainval_{img_dim}/images', '*.png')))
seg_mask_paths = sorted(glob.glob(os.path.join(f'./trainval_{img_dim}/annotations', '*.png')))
train_img_paths, val_img_paths, train_mask_paths, val_mask_paths = train_test_split(
    seg_image_paths, seg_mask_paths, test_size=0.1, random_state=42, shuffle=True)

train_dataset_seg = AESegDataset(train_img_paths, train_mask_paths, img_dim)
val_dataset_seg   = AESegDataset(val_img_paths, val_mask_paths, img_dim)
train_loader_seg = DataLoader(train_dataset_seg, batch_size=batch_size, shuffle=True)
val_loader_seg   = DataLoader(val_dataset_seg, batch_size=batch_size, shuffle=False)

# Build segmentation model using the frozen encoder.
seg_model = SegmentationModel(encoder=autoencoder.encoder, num_classes=num_seg_classes).to(device)
optimizer_seg = optim.Adam(seg_model.decoder.parameters(), lr=1e-3)  # only train decoder
# Use ignore_index=255 if needed (e.g., for unclassified regions).
criterion_seg = nn.CrossEntropyLoss(ignore_index=255)

print("Training Segmentation Model...")
for epoch in range(1, num_epochs_seg+1):
    train_loss = train_segmentation(seg_model, train_loader_seg, optimizer_seg, criterion_seg, device)
    val_loss, val_acc = evaluate_segmentation(seg_model, val_loader_seg, criterion_seg, device)
    print(f"Seg Epoch {epoch}/{num_epochs_seg}  Train Loss: {train_loss:.4f}  Val Loss: {val_loss:.4f}  Val Acc: {val_acc*100:.2f}%")

# Save segmentation model weights.
os.makedirs("seg_weights", exist_ok=True)
torch.save(seg_model.state_dict(), f"seg_weights/seg_model_{img_dim}_epochs_{num_epochs_seg}.pth")
print("Segmentation training complete.")