import os
import argparse
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from models.unet import UNet
from models.autoencoder import Autoencoder, SegmentationModel

def preprocess_image(image, dim, device):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (dim, dim))
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))
    image_tensor = torch.from_numpy(image).float().unsqueeze(0).to(device)
    return image_tensor

def postprocess_output(output, image_dims):
    output = output.astype(np.uint8)
    output = cv2.resize(output, (image_dims[1], image_dims[0]), interpolation=cv2.INTER_NEAREST)
    return output

parser = argparse.ArgumentParser()

parser.add_argument('--input', type=str,  default='./Example.jpg', help='input image')
parser.add_argument('--gt', type=str, help='ground truth image')
parser.add_argument('--category', type=int, default=1, help='1 = cat 2 = dog')
parser.add_argument('--output', type=str, help='output filename')
parser.add_argument('--model', type=str, default='unet', help='model name')


parser.add_argument('--weights', type=str, default='./unet_weights/unet_model_256_epochs_50.pth', help='path to weights file')
parser.add_argument('--dim', type=int, default=256, help='image dimension')
parser.add_argument('--gpu', type=int, default=0, help='default cuda:0 (will convert to cpu if not available)')

args = parser.parse_args()

device = 'cpu'

if torch.cuda.is_available():
    device = f'cuda:{args.gpu}'

dim = args.dim

# Load the model

if args.model == 'unet':
    model = UNet(3).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=torch.device(device)))
    model.eval()
elif args.model == 'ae':

    autoencoder = Autoencoder().to(device)
    autoencoder.load_state_dict(torch.load('./ae_weights/autoencoder_256_epochs_50.pth', map_location=device))
    autoencoder.eval()

    model = SegmentationModel(encoder=autoencoder.encoder, num_classes=3).to(device)
    # The model's encoder and decoder should have been defined in models/segmentation_model.py.
    model.load_state_dict(torch.load('./seg_weights/seg_model_256_epochs_50.pth', map_location=device))
    model.eval()

elif args.model == 'clip':
    from models.clip import CLIPSegmentation
    model = CLIPSegmentation(num_classes=3).to(device)

    model.load_state_dict(torch.load('./clip_weights/clip_model_256_epochs_50.pth', map_location=device))

    model.eval()

    dim = 224



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

# Create a custom colormap
cmap = ListedColormap(['black', 'red', 'blue', 'white'])

# Display the images
fig, ax = plt.subplots(1, 3 if include_gt else 2, figsize=(12 if include_gt else 8, 4))

ax[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
ax[0].set_title('Input Image')
ax[0].axis('off')

ax[1].imshow(output, cmap=cmap, vmin=0, vmax=3)
ax[1].set_title('Output Image')
ax[1].axis('off')

if include_gt:
    ground_truth = cv2.imread(args.gt, cv2.IMREAD_GRAYSCALE)
    
    gt_remapped = np.full(ground_truth.shape, 3 , dtype=np.uint8)
    gt_remapped[ground_truth == 2] = 0
    if args.category is not None:
        gt_remapped[ground_truth == 1] = args.category
    else:
        raise ValueError("The '--category' argument must be provided.")

    ax[2].imshow(gt_remapped, cmap=cmap, vmin=0, vmax=3)
    ax[2].set_title('Ground Truth')
    ax[2].axis('off')

plt.show()
