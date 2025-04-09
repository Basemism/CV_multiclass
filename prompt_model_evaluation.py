import os
import pickle
import cv2
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Create a heatmap with a Gaussian centered at 'point'
def generate_prompt_heatmap(img_size, point, kernel_size=5):
    """

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

# Resize image to target_size, convert from BGR to RGB, normalize, and concatenate prompt heat map.
def preprocess_for_model(image, prompt_heat, target_size=256):

    image_resized = cv2.resize(image, (target_size, target_size))
    image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
    image_rgb = image_rgb.astype(np.float32) / 255.0  # normalize
    image_rgb = np.transpose(image_rgb, (2,0,1))  # (3, H, W)
    prompt_heat = cv2.resize(prompt_heat, (target_size, target_size))
    prompt_heat = np.expand_dims(prompt_heat, axis=0)  # (1, H, W)
    input_tensor = np.concatenate([image_rgb, prompt_heat], axis=0)  # (4, H, W)
    input_tensor = torch.from_numpy(input_tensor).unsqueeze(0).float().to(device)
    return input_tensor


# Post-process the model output to get the predicted mask.
def postprocess_prediction(pred_logits):
    pred = torch.argmax(pred_logits, dim=1).squeeze().cpu().numpy()

    return pred

# Compute Intersection over Union (IoU) and Dice coefficient for the foreground (label=1).
# Only consider valid pixels (where gt != ignore_index, assumed ignore index is 255).
def compute_iou_dice(gt, pred):
    valid = (gt != 255)
    gt_valid = gt[valid]
    pred_valid = pred[valid]
    intersection = np.sum((gt_valid == 1) & (pred_valid == 1))
    union = np.sum((gt_valid == 1) | (pred_valid == 1))
    iou = intersection / union if union != 0 else 0
    dice = (2 * intersection) / (np.sum(gt_valid == 1) + np.sum(pred_valid == 1)) if (np.sum(gt_valid == 1) + np.sum(pred_valid == 1)) != 0 else 0
    return iou, dice

# Given a binary mask (values 0,1,255), sample num_samples points where pixel equals target_value.
def sample_prompt_points(binary_mask, num_samples, target_value):
    indices = np.argwhere(binary_mask == target_value)

    if len(indices) == 0:
        return []  # Return an empty list if no valid points are found
    sampled_indices = indices[np.random.choice(len(indices), size=num_samples, replace=(len(indices) < num_samples))]
    return [tuple(pt) for pt in sampled_indices]

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


# For each test image, we sample 3 foreground prompts and 3 background prompts.
num_fg_prompts = 3
num_bg_prompts = 3
target_size = 256

# Containers for metrics for foreground and background evaluations.
fg_accs = []
fg_precs = []
fg_recs = []
fg_f1s = []
fg_ious = []
fg_dices = []

bg_accs = []
bg_precs = []
bg_recs = []
bg_f1s = []
bg_ious = []
bg_dices = []

# Process each test image.
for img_path, mask_path in tqdm(zip(test_images, test_trimap_paths), total=len(test_images), desc="Evaluating"):
    image = cv2.imread(img_path)
    if image is None:
        continue
    
    # Load and process ground truth trimap.
    gt = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if gt is None:
        continue
    gt = cv2.resize(gt, (target_size, target_size), interpolation=cv2.INTER_NEAREST)
    # Remap ground truth: foreground (1) -> 1, background (2) -> 0, unclassified (3) -> 255.
    binary_gt = np.full(gt.shape, 255, dtype=np.uint8)
    binary_gt[gt == 2] = 0
    binary_gt[gt == 1] = 1

    # Sample foreground and background prompt points.
    fg_points = sample_prompt_points(binary_gt, num_fg_prompts, target_value=1)
    bg_points = sample_prompt_points(binary_gt, num_bg_prompts, target_value=0)

    if len(fg_points) == 0 or len(bg_points) == 0:
        print(f"\nSkipping image {img_path} due to insufficient prompt points.")
        continue

    # For foreground, expected target is the original binary_gt.
    for point in fg_points:
        prompt_heat = generate_prompt_heatmap(target_size, point, kernel_size=5)
        input_tensor = preprocess_for_model(image, prompt_heat, target_size=target_size)
        with torch.no_grad():
            logits = model(input_tensor)
        pred_mask = postprocess_prediction(logits)
        # Use entire binary_gt as expected.
        expected = binary_gt.copy()
        
        valid = (expected != 255)
        if np.sum(valid)==0:
            continue
        gt_flat = expected[valid]
        pred_flat = pred_mask[valid]
        acc = accuracy_score(gt_flat, pred_flat)
        prec = precision_score(gt_flat, pred_flat, average="binary", zero_division=0)
        rec = recall_score(gt_flat, pred_flat, average="binary", zero_division=0)
        f1 = f1_score(gt_flat, pred_flat, average="binary", zero_division=0)
        iou, dice = compute_iou_dice(expected, pred_mask)
        
        fg_accs.append(acc)
        fg_precs.append(prec)
        fg_recs.append(rec)
        fg_f1s.append(f1)
        fg_ious.append(iou)
        fg_dices.append(dice)
        
    # For background prompts, expected target is all zeros.
    expected_bg = np.zeros_like(binary_gt, dtype=np.uint8)
    for point in bg_points:
        prompt_heat = generate_prompt_heatmap(target_size, point, kernel_size=5)
        input_tensor = preprocess_for_model(image, prompt_heat, target_size=target_size)
        with torch.no_grad():
            logits = model(input_tensor)
        pred_mask = postprocess_prediction(logits)
        valid = (expected_bg != 255)  # In a blank target, all pixels are valid.
        gt_flat = expected_bg[valid]
        pred_flat = pred_mask[valid]
        acc = accuracy_score(gt_flat, pred_flat)
        
        # Handle cases where ground truth and predictions are all zeros
        if np.sum(gt_flat) == 0 and np.sum(pred_flat) == 0:
            prec, rec, f1, iou, dice = 1.0, 1.0, 1.0, 1.0, 1.0
        else:
            prec = precision_score(gt_flat, pred_flat, average="binary", zero_division=0)
            rec = recall_score(gt_flat, pred_flat, average="binary", zero_division=0)
            f1 = f1_score(gt_flat, pred_flat, average="binary", zero_division=0)
            iou, dice = compute_iou_dice(expected_bg, pred_mask)
        
        bg_accs.append(acc)
        bg_precs.append(prec)
        bg_recs.append(rec)
        bg_f1s.append(f1)
        bg_ious.append(iou)
        bg_dices.append(dice)

# Compute mean metrics per test set.
mean_fg_acc = np.mean(fg_accs) * 100
mean_fg_prec = np.mean(fg_precs) * 100
mean_fg_rec = np.mean(fg_recs) * 100
mean_fg_f1 = np.mean(fg_f1s) * 100
mean_fg_iou = np.mean(fg_ious) * 100
mean_fg_dice = np.mean(fg_dices) * 100

mean_bg_acc = np.mean(bg_accs) * 100
mean_bg_prec = np.mean(bg_precs) * 100
mean_bg_rec = np.mean(bg_recs) * 100
mean_bg_f1 = np.mean(bg_f1s) * 100
mean_bg_iou = np.mean(bg_ious) * 100
mean_bg_dice = np.mean(bg_dices) * 100

print("\n--- Evaluation Results for Foreground Prompts ---")
print(f"Mean Pixel Accuracy: {mean_fg_acc:.2f}%")
print(f"Mean Precision: {mean_fg_prec:.2f}%")
print(f"Mean Recall: {mean_fg_rec:.2f}%")
print(f"Mean F1 Score: {mean_fg_f1:.2f}%")
print(f"Mean IoU: {mean_fg_iou:.2f}%")
print(f"Mean Dice Coefficient: {mean_fg_dice:.2f}%")

print("\n--- Evaluation Results for Background Prompts ---")
print(f"Mean Pixel Accuracy: {mean_bg_acc:.2f}%")
print(f"Mean Precision: {mean_bg_prec:.2f}%")
print(f"Mean Recall: {mean_bg_rec:.2f}%")
print(f"Mean F1 Score: {mean_bg_f1:.2f}%")
print(f"Mean IoU: {mean_bg_iou:.2f}%")
print(f"Mean Dice Coefficient: {mean_bg_dice:.2f}%")

metrics_filename = 'metrics/prompt_unet_model_256_epochs_50.txt'
with open(metrics_filename, 'w') as f:
    f.write("\n--- Evaluation Results for Foreground Prompts ---\n")
    f.write(f"Mean Pixel Accuracy: {mean_fg_acc:.2f}%\n")
    f.write(f"Mean Precision: {mean_fg_prec:.2f}%\n")
    f.write(f"Mean Recall: {mean_fg_rec:.2f}%\n")
    f.write(f"Mean F1 Score: {mean_fg_f1:.2f}%\n")
    f.write(f"Mean IoU: {mean_fg_iou:.2f}%\n")
    f.write(f"Mean Dice Coefficient: {mean_fg_dice:.2f}%\n")
    
    f.write("\n--- Evaluation Results for Background Prompts ---\n")
    f.write(f"Mean Pixel Accuracy: {mean_bg_acc:.2f}%\n")
    f.write(f"Mean Precision: {mean_bg_prec:.2f}%\n")
    f.write(f"Mean Recall: {mean_bg_rec:.2f}%\n")
    f.write(f"Mean F1 Score: {mean_bg_f1:.2f}%\n")
    f.write(f"Mean IoU: {mean_bg_iou:.2f}%\n")
    f.write(f"Mean Dice Coefficient: {mean_bg_dice:.2f}%\n")
    
print(f"Metrics saved to {metrics_filename}")
