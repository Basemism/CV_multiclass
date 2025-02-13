import os
import glob
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import clip 
from data_core import CLIPDataset

from utils import train_one_epoch, evaluate
from models.clip_seg import CLIPSegmentationNet

dim = 256
num_classes = 3  # Trimap labels: {1,2,3} (we'll shift them to {0,1,2} in preprocessing)
batch_size = 16
num_epochs = 50
seed = 42
torch.manual_seed(seed)

train_images_dir = f'./trainval_{dim}/images'
train_masks_dir  = f'./trainval_{dim}/annotations'

image_paths = sorted(glob.glob(os.path.join(train_images_dir, '*.png')))
mask_paths  = sorted(glob.glob(os.path.join(train_masks_dir, '*.png')))

train_img_paths, val_img_paths, train_mask_paths, val_mask_paths = train_test_split(
    image_paths, mask_paths, test_size=0.1, random_state=seed, shuffle=True)

print(f"Training images: {len(train_img_paths)}")
print(f"Validation images: {len(val_img_paths)}")


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
clip_model, clip_preprocess = clip.load("RN50", device=device)
clip_model.eval()
for param in clip_model.parameters():
    param.requires_grad = False


clip_feature_dim = clip_model.visual.output_dim
print(f"Using device: {device}")
print(f"CLIP feature dimension: {clip_feature_dim}")


train_dataset = CLIPDataset(train_img_paths, train_mask_paths, dim, clip_model, device)
val_dataset   = CLIPDataset(val_img_paths, val_mask_paths, dim, clip_model, device)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

model = CLIPSegmentationNet(clip_feature_dim=clip_feature_dim, num_classes=num_classes).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()  # expects logits [B, C, H, W] and target [B, H, W]

best_val_loss = float('inf')

for epoch in tqdm(range(1, num_epochs+1), desc="Training Epochs"):
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)
    tqdm.write(f"Epoch: {epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        os.makedirs("clip_seg_weights", exist_ok=True)
        torch.save(model.state_dict(), f"clip_seg_weights/clip_seg_model_{dim}_epochs_{num_epochs}.pth")

print("Training complete.")