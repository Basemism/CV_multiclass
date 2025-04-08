import os
import pickle
import cv2
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import argparse

from models.prompt_unet import PromptUNet

def preprocess_image(image, dim, device):
    """
    Preprocesses an image: converts to RGB, resizes to (dim,dim), scales to [0,1],
    and converts to a tensor with shape (3, H, W).
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (dim, dim))
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))
    image_tensor = torch.from_numpy(image).float().to(device)
    return image_tensor

def postprocess_output(output, image_dims):
    """
    Resizes the predicted mask to the original image dimensions.
    """
    output = output.astype(np.uint8)
    output = cv2.resize(output, (image_dims[1], image_dims[0]), interpolation=cv2.INTER_NEAREST)
    return output

def generate_prompt(mask, sigma=5):
    """
    Given a binary ground-truth mask (0: non-object, 1: object, 255: ignore),
    sample a random valid pixel (where mask != 255) and generate a prompt heat map.
    
    If the sampled pixel belongs to the object (1), the prompt value is +1;
    if background (0), the value is -1.
    A Gaussian blur is applied to spread the prompt.
    
    Returns:
      prompt: a float32 numpy array of shape (H, W).
    """
    valid_idx = np.argwhere(mask != 255)
    if len(valid_idx) == 0:
        # If no valid pixel, sample uniformly.
        valid_idx = np.argwhere(np.ones_like(mask, dtype=bool))
    chosen_idx = valid_idx[np.random.choice(len(valid_idx))]
    click_value = 1.0 if mask[chosen_idx[0], chosen_idx[1]] == 1 else -1.0

    prompt = np.zeros(mask.shape, dtype=np.float32)
    prompt[chosen_idx[0], chosen_idx[1]] = click_value

    # Ensure the kernel size is odd.
    ksize = (sigma | 1, sigma | 1)
    prompt = cv2.GaussianBlur(prompt, ksize, 0)
    return prompt

def remap_mask(mask):
    """
    Remap the original trimap mask to binary:
      - Original trimap: 1 = foreground, 2 = background, 3 = unclassified.
      - After remapping: foreground (mask==1) becomes 1 (object),
                        background (mask==2) becomes 0 (non-object),
                        unclassified (mask==3) becomes 255 (ignore).
    """
    mask = mask.copy()
    new_mask = np.full(mask.shape, 255, dtype=np.uint8)
    new_mask[mask == 2] = 0
    new_mask[mask == 1] = 1
    return new_mask

parser = argparse.ArgumentParser()
parser.add_argument('--dim', type=int, default=256, help='Image dimension for resizing')
parser.add_argument('--weights', type=str, default='./prompt_weights/prompt_unet_model_256_epochs_100.pth',
                    help='Path to prompt-based model weights')
parser.add_argument('--metrics', type=str, default='./metrics/prompt_model_256_epochs_100.txt',
                    help='Path to save metrics')
args = parser.parse_args()

dim = args.dim
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
weight_file = args.weights
metrics_filename = args.metrics

# For prompt-based segmentation, our model outputs 2 classes:
# 0: non-object, 1: object.
model = PromptUNet(num_classes=2, in_channels=4).to(device)
model.load_state_dict(torch.load(weight_file, map_location=device))
model.eval()

# Load test image and mask paths.
with open('test_image_paths.pkl', 'rb') as f:
    test_images = pickle.load(f)
with open('test_trimap_paths.pkl', 'rb') as f:
    test_mask_paths = pickle.load(f)

if len(test_images) != len(test_mask_paths):
    raise ValueError("Number of test images should match number of masks.")

# Containers for metrics.
pixel_accs = []
precisions = []
recalls = []
f1_scores = []

# For IoU, accumulate intersections and unions per class.
intersection_obj = 0
union_obj = 0
intersection_bg = 0
union_bg = 0

print(f"Found {len(test_images)} test images.")

for i in tqdm(range(len(test_images)), desc="Evaluating"):
    # Load image.
    image = cv2.imread(test_images[i])
    if image is None:
        print(f"Warning: Could not load image: {test_images[i]}")
        continue
    image_dims = image.shape[:2]  # original dimensions

    # Preprocess image.
    image_tensor = preprocess_image(image, dim=dim, device=device)  # shape (3, dim, dim)

    # Load and remap ground truth mask.
    gt_mask_orig = cv2.imread(test_mask_paths[i], cv2.IMREAD_GRAYSCALE)
    if gt_mask_orig is None:
        raise ValueError(f"Mask not found at {test_mask_paths[i]}")
    # Resize ground truth to (dim, dim) for evaluation.
    gt_mask_resized = cv2.resize(gt_mask_orig, (dim, dim), interpolation=cv2.INTER_NEAREST)
    gt_mask = remap_mask(gt_mask_resized)  # binary mask: 0 (non-object), 1 (object), 255 (ignore)

    # Generate prompt heat map from the remapped mask.
    prompt = generate_prompt(gt_mask, sigma=15)  # shape (dim, dim)

    # Create 4-channel input: concatenate image and prompt.
    image_np = image_tensor.cpu().numpy()  # (3, dim, dim)
    prompt_np = np.expand_dims(prompt, axis=0)  # (1, dim, dim)
    input_tensor = torch.from_numpy(np.concatenate([image_np, prompt_np], axis=0)).unsqueeze(0).float().to(device)
    # input_tensor shape: (1, 4, dim, dim)

    # Run the model.
    with torch.no_grad():
        output = model(input_tensor)  # (1, 2, dim, dim)
        pred = torch.argmax(output, dim=1).squeeze().cpu().numpy()  # shape (dim, dim)

    # Optionally, resize prediction back to original image size.
    pred_resized = postprocess_output(pred, image_dims)
    gt_mask_full = cv2.resize(gt_mask, (image_dims[1], image_dims[0]), interpolation=cv2.INTER_NEAREST)

    # Only evaluate on valid pixels (mask != 255).
    valid_pixels = gt_mask_full != 255
    pred_valid = pred_resized[valid_pixels]
    gt_valid = gt_mask_full[valid_pixels]

    if gt_valid.size == 0:
        print(f"Warning: No valid pixels for {test_images[i]}")
        continue

    pixel_accs.append(accuracy_score(gt_valid, pred_valid))
    precisions.append(precision_score(gt_valid, pred_valid, average='weighted', zero_division=0))
    recalls.append(recall_score(gt_valid, pred_valid, average='weighted', zero_division=0))
    f1_scores.append(f1_score(gt_valid, pred_valid, average='weighted', zero_division=0))

    # Compute IoU per class.
    # For background (class 0):
    inter_bg = np.logical_and(pred_resized == 0, gt_mask_full == 0).sum()
    union_bg_val = np.logical_or(pred_resized == 0, gt_mask_full == 0).sum()
    intersection_bg += inter_bg
    union_bg += union_bg_val
    # For object (class 1):
    inter_obj = np.logical_and(pred_resized == 1, gt_mask_full == 1).sum()
    union_obj_val = np.logical_or(pred_resized == 1, gt_mask_full == 1).sum()
    intersection_obj += inter_obj
    union_obj += union_obj_val

avg_pixel_acc = np.mean(pixel_accs)
avg_precision = np.mean(precisions)
avg_recall = np.mean(recalls)
avg_f1_score = np.mean(f1_scores)
iou_bg = intersection_bg / union_bg if union_bg > 0 else 0
iou_obj = intersection_obj / union_obj if union_obj > 0 else 0
avg_iou_overall = (iou_bg + iou_obj) / 2

print(f"Pixel Accuracy: {avg_pixel_acc*100:.2f}%")
print(f"Precision: {avg_precision*100:.2f}%")
print(f"Recall: {avg_recall*100:.2f}%")
print(f"F1 Score: {avg_f1_score*100:.2f}%")
print(f"Mean IoU: {avg_iou_overall*100:.2f}%")
print(f"IoU Background: {iou_bg*100:.2f}%")
print(f"IoU Object: {iou_obj*100:.2f}%")

with open(metrics_filename, 'w') as f:
    f.write(f"Pixel Accuracy: {avg_pixel_acc*100:.2f}%\n")
    f.write(f"Precision: {avg_precision*100:.2f}%\n")
    f.write(f"Recall: {avg_recall*100:.2f}%\n")
    f.write(f"F1 Score: {avg_f1_score*100:.2f}%\n")
    f.write(f"Mean IoU: {avg_iou_overall*100:.2f}%\n")
    f.write(f"IoU Background: {iou_bg*100:.2f}%\n")
    f.write(f"IoU Object: {iou_obj*100:.2f}%\n")

print(f"Metrics saved to {metrics_filename}")
