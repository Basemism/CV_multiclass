import os
import argparse
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from utils import preprocess_image, postprocess_output

from models.unet import UNet


parser = argparse.ArgumentParser()

parser.add_argument('--input', type=str,  default='./example.jpg', help='input image')
parser.add_argument('--gt', type=str, help='ground truth image')
parser.add_argument('--output', type=str, help='output filename')

parser.add_argument('--weights', type=str, default='./unet_weights/unet_model_256_epochs_50.pth', help='path to weights file')
parser.add_argument('--dim', type=int, default=256, help='image dimension')
parser.add_argument('--gpu', type=int, default=0, help='default cuda:0 (will convert to cpu if not available)')

args = parser.parse_args()

device = 'cuda:%d' % args.gpu if args.gpu >= 0 else 'cpu'
dim = args.dim

# Load the model
model = UNet(3).to(device)
model.load_state_dict(torch.load(args.weights, map_location=torch.device(device)))
model.eval()

# Load the image
image = cv2.imread(args.input)
image_dims = image.shape[:2]

image_tensor = preprocess_image(image, dim=dim, device=device)

# Perform inference
with torch.no_grad():
    output = model(image_tensor)
    output = torch.argmax(output, dim=1).cpu().numpy().squeeze()

output = postprocess_output(output, image_dims)

print(np.unique(output))

if args.output:
    cv2.imwrite(args.output, output)

include_gt = args.gt is not None

# Load the ground truth image
if include_gt:
    gt = cv2.imread(args.gt, cv2.IMREAD_GRAYSCALE)

# Display the images
fig, ax = plt.subplots(1, 3 if include_gt else 2, figsize=(12 if include_gt else 8, 4))

ax[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
ax[0].set_title('Input Image')
ax[0].axis('off')

ax[1].imshow(output)
ax[1].set_title('Output Image')
ax[1].axis('off')

if include_gt:
    ax[2].imshow(gt)
    ax[2].set_title('Ground Truth')
    ax[2].axis('off')

plt.show()

