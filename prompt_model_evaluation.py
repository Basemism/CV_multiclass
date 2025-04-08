import os
import pickle
import cv2
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def generate_prompt_heatmap(img_size, point, kernel_size=5):
    """
    Create a heatmap with a Gaussian centered at 'point'
    :param img_size: (H, W)
    :param point: (row, col)
    :param kernel_size: size of Gaussian kernel
    :return: heatmap normalized to [0,1]
    """
    prompt = np.zeros((img_size, img_size), dtype=np.float32)
    prompt[point[0], point[1]] = 1.0
    prompt = cv2.GaussianBlur(prompt, (kernel_size, kernel_size), 0)
    if prompt.max() > 0:
        prompt = prompt / prompt.max()
    return prompt

def preprocess_for_model(image, prompt_heat, target_size=256):
    """
    Resize image to target_size, convert from BGR to RGB, normalize, and concatenate prompt heat map.
    :param image: input image in BGR format
    :param prompt_heat: a (H,W) heat map with values in [0,1]
    :param target_size: desired spatial size (assumed square)
    :return: torch tensor of shape (1,4,target_size,target_size)
    """
    image_resized = cv2.resize(image, (target_size, target_size))

    image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
    image_rgb = image_rgb.astype(np.float32) / 255.0  # normalize
    image_rgb = np.transpose(image_rgb, (2, 0, 1))  # (3, H, W)

    prompt_heat = cv2.resize(prompt_heat, (target_size, target_size))
    prompt_heat = np.expand_dims(prompt_heat, axis=0)  # (1, H, W)

    input_tensor = np.concatenate([image_rgb, prompt_heat], axis=0)  # (4, H, W)
    input_tensor = torch.from_numpy(input_tensor).unsqueeze(0).float().to(device)

    return input_tensor

def postprocess_prediction(pred_logits):
    """
    Convert logits to final binary mask.
    :param pred_logits: tensor of shape (1, num_classes, H, W)
    :return: binary mask (H, W) where pixel is 0 or 1
    """
    pred = torch.argmax(pred_logits, dim=1).squeeze().cpu().numpy()
    return pred

def compute_iou_dice(gt, pred):
    """
    Compute Intersection over Union (IoU) and Dice coefficient for the foreground (label=1).
    Only consider valid pixels (where gt != ignore_index, assumed ignore index is 255).
    :param gt: ground truth mask (H, W)
    :param pred: predicted mask (H, W)
    :return: iou, dice
    """
    valid = (gt != 255)
    gt_valid = gt[valid]
    pred_valid = pred[valid]
    
    # Binary: foreground=1, background=0.
    intersection = np.sum((gt_valid == 1) & (pred_valid == 1))
    union = np.sum((gt_valid == 1) | (pred_valid == 1))
    iou = intersection / union if union != 0 else 0
    
    dice = (2 * intersection) / (np.sum(gt_valid == 1) + np.sum(pred_valid == 1)) if (np.sum(gt_valid == 1) + np.sum(pred_valid == 1)) != 0 else 0
    return iou, dice


import sys

with open('test_image_paths.pkl', 'rb') as f:
    test_images = pickle.load(f)

with open('test_trimap_paths.pkl', 'rb') as f:
    test_trimap_paths = pickle.load(f)

if len(test_images) != len(test_trimap_paths):
    raise ValueError("Number of test images should match number of trimaps.")

from models.prompt_unet import PromptUNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PromptUNet(num_classes=2, in_channels=4).to(device)

model_path = 'prompt_weights/prompt_unet_model_256_epochs_50.pth'
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# -------------------------
# Evaluation Loop
# -------------------------
# For evaluation, we use a simple prompt: the center of the image.
target_size = 256
center_point = (target_size // 2, target_size // 2)  # (row, col)
dummy_prompt_heat = generate_prompt_heatmap(target_size, center_point, kernel_size=5)

pixel_accs = []
precisions = []
recalls = []
f1_scores = []
ious = []
dices = []

for img_path, mask_path in tqdm(zip(test_images, test_trimap_paths), total=len(test_images), desc="Evaluating"):
    # Load image.
    image = cv2.imread(img_path)
    if image is None:
        continue
    # Preprocess with dummy prompt.
    input_tensor = preprocess_for_model(image, dummy_prompt_heat, target_size=target_size)
    
    # Run the model.
    with torch.no_grad():
        logits = model(input_tensor)
    pred_mask = postprocess_prediction(logits)  # (256, 256)

    # Load ground truth mask.
    gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if gt_mask is None:
        continue
    gt_mask = cv2.resize(gt_mask, (target_size, target_size), interpolation=cv2.INTER_NEAREST)
    
    # Remap ground truth:
    # Original: 1 = foreground, 2 = background, 3 = unclassified.
    # For binary segmentation: map foreground (1) -> 1, background (2) -> 0, unclassified (3) -> ignore (255)
    binary_gt = np.full(gt_mask.shape, 255, dtype=np.uint8)
    binary_gt[gt_mask == 2] = 0
    binary_gt[gt_mask == 1] = 1

    # Flatten valid pixels (non 255).
    valid = (binary_gt != 255)
    gt_flat = binary_gt[valid]
    pred_flat = pred_mask[valid]
    
    # Compute pixel accuracy and other metrics.
    if len(gt_flat) == 0:
        continue
    acc = accuracy_score(gt_flat, pred_flat)
    prec = precision_score(gt_flat, pred_flat, average="binary", zero_division=0)
    rec = recall_score(gt_flat, pred_flat, average="binary", zero_division=0)
    f1 = f1_score(gt_flat, pred_flat, average="binary", zero_division=0)
    iou, dice = compute_iou_dice(binary_gt, pred_mask)
    
    pixel_accs.append(acc)
    precisions.append(prec)
    recalls.append(rec)
    f1_scores.append(f1)
    ious.append(iou)
    dices.append(dice)

# Compute means.
mean_acc = np.mean(pixel_accs) * 100
mean_prec = np.mean(precisions) * 100
mean_rec = np.mean(recalls) * 100
mean_f1 = np.mean(f1_scores) * 100
mean_iou = np.mean(ious) * 100
mean_dice = np.mean(dices) * 100

# Print summarized results.
print("\n--- Evaluation Results ---")
print(f"Mean Pixel Accuracy: {mean_acc:.2f}%")
print(f"Mean Precision: {mean_prec:.2f}%")
print(f"Mean Recall: {mean_rec:.2f}%")
print(f"Mean F1 Score: {mean_f1:.2f}%")
print(f"Mean IoU: {mean_iou:.2f}%")
print(f"Mean Dice Coefficient: {mean_dice:.2f}%")

metrics_filename = 'metrics/prompt_unet_model_256_epochs_50.txt'

with open(metrics_filename, 'w') as f:
    f.write("\n--- Evaluation Results (Averaged over multiple prompts per image) ---")
    f.write(f"Mean Pixel Accuracy: {mean_acc:.2f}%")
    f.write(f"Mean Precision: {mean_prec:.2f}%")
    f.write(f"Mean Recall: {mean_rec:.2f}%")
    f.write(f"Mean F1 Score: {mean_f1:.2f}%")
    f.write(f"Mean IoU: {mean_iou:.2f}%")
    f.write(f"Mean Dice Coefficient: {mean_dice:.2f}%")

print(f"Metrics saved to {metrics_filename}")