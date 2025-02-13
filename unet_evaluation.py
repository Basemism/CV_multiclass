import pickle
import cv2
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from unet import UNet
import argparse


parser = argparse.ArgumentParser()

parser.add_argument('--dim', type=int, default=128, help='image dimension')
parser.add_argument('--weights', type=str, default='./unet_weights/unet_model_128_epochs_500.pth', help='path to weights file')
parser.add_argument('--metrics', type=str, default='./metrics/unet_128_epochs_500.txt', help='path to metrics file')

args = parser.parse_args()

dim = args.dim
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
weight_file = args.weights
metrics_filename= args.metrics

def preprocess_image(image, dim=dim):
    # Preprocess the image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (dim, dim))
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))
    
    image_tensor = torch.from_numpy(image).float().unsqueeze(0).to(device)

    return image_tensor

def postprocess_output(output, image_dims):
    output = output + 1 # Trimap {0,1,2} -> {1,2,3}
    output = output.astype(np.uint8)
    output = cv2.resize(output, (image_dims[1], image_dims[0]), interpolation=cv2.INTER_NEAREST)

    return output

model = UNet(num_classes=3).to(device)
model.load_state_dict(torch.load(weight_file, map_location=torch.device(device)))
model.eval()

# Load test image paths
with open('test_image_paths.pkl', 'rb') as f:
    test_images = pickle.load(f)

# Load test trimap paths
with open('test_trimap_paths.pkl', 'rb') as f:
    test_trimap_paths = pickle.load(f)

if len(test_images) != len(test_trimap_paths):
    raise ValueError("Number of test images should match number of trimaps.")

pixel_accs = []
precisions = []
recalls = []
f1_scores = []

print(f"Found {len(test_images)} test images.")

for i in tqdm(range(len(test_images)), desc="Predicting and Evaluating"):
    # Read the image
    image = cv2.imread(test_images[i])
    if image is None:
        print(f"Warning: Could not load image: {test_images[i]}")
        continue

    image_dims = image.shape[:2]

    image_tensor = preprocess_image(image, dim=dim)

    with torch.no_grad():
        output = model(image_tensor)
        output = torch.argmax(output, dim=1).squeeze().cpu().numpy()

    output = postprocess_output(output, image_dims)

    # Load ground truth trimap
    gt_mask = cv2.imread(test_trimap_paths[i], cv2.IMREAD_GRAYSCALE)
    if gt_mask is None:
        raise ValueError(f"Image not found at {test_trimap_paths[i]}")
    
    valid_pixels = gt_mask != 0
    pred_mask_valid = output[valid_pixels]
    gt_mask_valid = gt_mask[valid_pixels]

    pixel_accs.append(accuracy_score(gt_mask_valid, pred_mask_valid))
    precisions.append(precision_score(gt_mask_valid, pred_mask_valid, average='weighted', zero_division=0))
    recalls.append(recall_score(gt_mask_valid, pred_mask_valid, average='weighted', zero_division=0))
    f1_scores.append(f1_score(gt_mask_valid, pred_mask_valid, average='weighted', zero_division=0))

avg_pixel_acc = np.mean(pixel_accs)
avg_precision = np.mean(precisions)
avg_recall = np.mean(recalls)
avg_f1_score = np.mean(f1_scores)

print(f"Pixel Accuracy: {avg_pixel_acc*100:.2f}%")
print(f"Precision: {avg_precision*100:.2f}%")
print(f"Recall: {avg_recall*100:.2f}%")
print(f"F1 Score: {avg_f1_score*100:.2f}%")

with open(metrics_filename, 'w') as f:
    f.write(f"Pixel Accuracy: {avg_pixel_acc*100:.2f}%\n")
    f.write(f"Precision: {avg_precision*100:.2f}%\n")
    f.write(f"Recall: {avg_recall*100:.2f}%\n")
    f.write(f"F1 Score: {avg_f1_score*100:.2f}%\n")

print(f"Metrics saved to {metrics_filename}")
