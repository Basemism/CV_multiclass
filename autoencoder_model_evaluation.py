import os
import pickle
import cv2
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import argparse

# Import the segmentation model.
# This model is defined as SegmentationModel in our pretraining-finetuning pipeline.
from models.autoencoder import SegmentationModel, Autoencoder
# Note: models/segmentation_model.py should contain the SegmentationDecoder and SegmentationModel classes.

# Preprocess the input image to match the model's input dimensions and format.
def preprocess_image(image, dim, device):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (dim, dim))
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))
    image_tensor = torch.from_numpy(image).float().unsqueeze(0).to(device)
    return image_tensor

# Postprocess the model output to obtain the final segmentation mask equivalent to the input image resolution.
def postprocess_output(output, image_dims):
    output = output.astype(np.uint8)
    output = cv2.resize(output, (image_dims[1], image_dims[0]), interpolation=cv2.INTER_NEAREST)
    return output

parser = argparse.ArgumentParser()
parser.add_argument('--dim', type=int, default=256, help='Image dimension')
parser.add_argument('--weights', type=str, default='./seg_weights/seg_model_256_epochs_50.pth', help='Path to segmentation weights file')
parser.add_argument('--metrics', type=str, default='./metrics/seg_model_256_epochs_50.txt', help='Path to metrics file')
args = parser.parse_args()

dim = args.dim
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
weight_file = args.weights
metrics_filename = args.metrics


num_seg_classes = 3  # 0: background, 1: cat, 2: dog

autoencoder = Autoencoder().to(device)
autoencoder.load_state_dict(torch.load('./ae_weights/autoencoder_256_epochs_50.pth', map_location=device))
autoencoder.eval()


model = SegmentationModel(encoder=autoencoder.encoder, num_classes=num_seg_classes).to(device)
# The model's encoder and decoder should have been defined in models/segmentation_model.py.
model.load_state_dict(torch.load(weight_file, map_location=device))
model.eval()

# Load test image and trimap paths.
with open('test_image_paths.pkl', 'rb') as f:
    test_images = pickle.load(f)
with open('test_trimap_paths.pkl', 'rb') as f:
    test_trimap_paths = pickle.load(f)

if len(test_images) != len(test_trimap_paths):
    raise ValueError("Number of test images should match number of trimaps.")

# Containers for metrics.
pixel_accs = []
precisions = []
recalls = []
f1_scores = []
iou_scores_bg = []    # background (class 0)
iou_scores_cat = []   # cat (class 1)
iou_scores_dog = []   # dog (class 2)

print(f"Found {len(test_images)} test images.")

for i in tqdm(range(len(test_images)), desc="Predicting and Evaluating"):
    # Load and preprocess the image.
    image = cv2.imread(test_images[i])
    if image is None:
        print(f"Warning: Could not load image: {test_images[i]}")
        continue
    image_dims = image.shape[:2]
    image_tensor = preprocess_image(image, dim=dim, device=device)

    # Get model prediction.
    with torch.no_grad():
        output = model(image_tensor)
        pred = torch.argmax(output, dim=1).squeeze().cpu().numpy()
    pred = postprocess_output(pred, image_dims)

    # Load ground truth trimap.
    gt_mask = cv2.imread(test_trimap_paths[i], cv2.IMREAD_GRAYSCALE)
    if gt_mask is None:
        raise ValueError(f"Image not found at {test_trimap_paths[i]}")
    if gt_mask.shape != image_dims:
        gt_mask = cv2.resize(gt_mask, (image_dims[1], image_dims[0]), interpolation=cv2.INTER_NEAREST)

    # Remap ground truth.
    # Original trimap: 1 = foreground, 2 = background, 3 = unclassified.
    # After subtracting 1, we have: 0 = foreground, 1 = background, 2 = unclassified.
    # Determine image type from filename (cat if first letter uppercase, else dog).
    filename = os.path.basename(test_images[i])
    if filename[0].isupper():
        fg_class = 1  # cat
    else:
        fg_class = 2  # dog

    # Create remapped ground truth: background (original 1) → 0, foreground (original 0) → fg_class,
    # and unclassified (original 2) → 255 (to be ignored).
    gt_remapped = np.full(gt_mask.shape, 255, dtype=np.uint8)
    gt_remapped[gt_mask == 2] = 0       # original value 2 → background becomes 0
    gt_remapped[gt_mask == 1] = fg_class  # original value 1 → foreground becomes fg_class
    # Pixels with original value 3 remain 255 (ignored).

    valid_pixels = gt_remapped != 255
    pred_valid = pred[valid_pixels]
    gt_valid = gt_remapped[valid_pixels]
    if gt_valid.size == 0:
        print(f"Warning: No valid pixels for {test_images[i]}")
        continue

    pixel_accs.append(accuracy_score(gt_valid, pred_valid))
    precisions.append(precision_score(gt_valid, pred_valid, average='weighted', zero_division=0))
    recalls.append(recall_score(gt_valid, pred_valid, average='weighted', zero_division=0))
    f1_scores.append(f1_score(gt_valid, pred_valid, average='weighted', zero_division=0))

    # Compute IoU for each class (0: background, 1: cat, 2: dog).
    for cls, iou_list in zip([0, 1, 2], [iou_scores_bg, iou_scores_cat, iou_scores_dog]):
        gt_cls = (gt_valid == cls)
        pred_cls = (pred_valid == cls)
        intersection = np.logical_and(gt_cls, pred_cls).sum()
        union = np.logical_or(gt_cls, pred_cls).sum()
        iou = intersection / union if union != 0 else 0
        iou_list.append(iou)

avg_pixel_acc = np.mean(pixel_accs)
avg_precision = np.mean(precisions)
avg_recall = np.mean(recalls)
avg_f1_score = np.mean(f1_scores)
avg_iou_bg = np.mean(iou_scores_bg)
avg_iou_cat = np.mean(iou_scores_cat)
avg_iou_dog = np.mean(iou_scores_dog)
avg_iou_overall = np.mean(iou_scores_bg + iou_scores_cat + iou_scores_dog)

print(f"Pixel Accuracy: {avg_pixel_acc*100:.2f}%")
print(f"Precision: {avg_precision*100:.2f}%")
print(f"Recall: {avg_recall*100:.2f}%")
print(f"F1 Score: {avg_f1_score*100:.2f}%")
print(f"Mean IoU: {avg_iou_overall*100:.2f}%")
print(f"IoU Background: {avg_iou_bg*100:.2f}%")
print(f"IoU Cat: {avg_iou_cat*100:.2f}%")
print(f"IoU Dog: {avg_iou_dog*100:.2f}%")

with open(metrics_filename, 'w') as f:
    f.write(f"Pixel Accuracy: {avg_pixel_acc*100:.2f}%\n")
    f.write(f"Precision: {avg_precision*100:.2f}%\n")
    f.write(f"Recall: {avg_recall*100:.2f}%\n")
    f.write(f"F1 Score: {avg_f1_score*100:.2f}%\n")
    f.write(f"Mean IoU: {avg_iou_overall*100:.2f}%\n")
    f.write(f"IoU Background: {avg_iou_bg*100:.2f}%\n")
    f.write(f"IoU Cat: {avg_iou_cat*100:.2f}%\n")
    f.write(f"IoU Dog: {avg_iou_dog*100:.2f}%\n")

print(f"Metrics saved to {metrics_filename}")
