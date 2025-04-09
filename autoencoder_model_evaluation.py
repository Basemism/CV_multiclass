import os
import pickle
import cv2
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import argparse

from models.autoencoder import SegmentationModel, Autoencoder

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

intersection_bg = 0   # background (class 0)
union_bg = 0  # background (class 0)

intersection_cat = 0   # background (class 1)
union_cat = 0   # cat (class 1)

intersection_dog = 0   #  (class 2)
union_dog = 0   # dog (class 2)


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
    filename = os.path.basename(test_images[i])
    if filename[0].isupper():
        fg_class = 1  # cat
    else:
        fg_class = 2  # dog

    # Remap the trimap: 0 (background), 1 (cat), 2 (dog), 255 (ignored).
    gt_remapped = np.full(gt_mask.shape, 255, dtype=np.uint8)
    gt_remapped[gt_mask == 2] = 0
    gt_remapped[gt_mask == 1] = fg_class
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

    for j in range(num_seg_classes):
        intersection = np.logical_and(pred == j, gt_remapped == j).sum()
        union = np.logical_or(pred == j, gt_remapped == j).sum()
        if j == 0:
            intersection_bg += intersection
            union_bg += union
        elif j == 1:
            intersection_cat += intersection
            union_cat += union
        elif j == 2:
            intersection_dog += intersection
            union_dog += union
    
avg_pixel_acc = np.mean(pixel_accs)
avg_precision = np.mean(precisions)
avg_recall = np.mean(recalls)
avg_f1_score = np.mean(f1_scores)
avg_iou_overall = (intersection_bg + intersection_cat + intersection_dog) / (union_bg + union_cat + union_dog)
avg_iou_bg = intersection_bg / union_bg if union_bg > 0 else 0
avg_iou_cat = intersection_cat / union_cat if union_cat > 0 else 0
avg_iou_dog = intersection_dog / union_dog if union_dog > 0 else 0

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
