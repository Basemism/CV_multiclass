import os
import argparse
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from models.prompt_unet import PromptUNet

def preprocess_image_and_prompt(image, prompt_coords, dim, device, prompt_kernel=15):
    """
    Preprocesses an image and creates a prompt heat map.
    
    Args:
        image (np.array): Input BGR image.
        prompt_coords (tuple): (x, y) coordinates in original image space.
        dim (int): Desired dimension (image resized to dim x dim).
        device: torch.device.
        prompt_kernel (int): Gaussian kernel size for smoothing the prompt.
    
    Returns:
        input_tensor (torch.Tensor): 4-channel tensor (RGB+prompt) of shape (1,4,dim,dim).
        prompt_heatmap (np.array): The prompt heat map (for visualization) resized to (dim,dim).
    """
    # Convert image to RGB, resize, and normalize.
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = image.shape[:2]
    resized_img = cv2.resize(image_rgb, (dim, dim))
    resized_img = resized_img.astype(np.float32) / 255.0
    # Convert to CHW.
    image_tensor = np.transpose(resized_img, (2, 0, 1))
    
    # Map prompt coordinate from original to resized dimensions.
    scale_x = dim / orig_w
    scale_y = dim / orig_h
    prompt_x = int(prompt_coords[0] * scale_x)
    prompt_y = int(prompt_coords[1] * scale_y)
    
    # Create a blank heat map and place a 1 at the prompt coordinate.
    prompt_heatmap = np.zeros((dim, dim), dtype=np.float32)
    # Clip coordinates to valid range.
    prompt_x = np.clip(prompt_x, 0, dim-1)
    prompt_y = np.clip(prompt_y, 0, dim-1)
    prompt_heatmap[prompt_y, prompt_x] = 1.0
    
    # Apply Gaussian blur to spread the prompt signal.
    prompt_heatmap = cv2.GaussianBlur(prompt_heatmap, (prompt_kernel, prompt_kernel), 0)
    
    # Normalize prompt heatmap between 0 and 1.
    prompt_heatmap = prompt_heatmap / (prompt_heatmap.max() + 1e-8)
    
    # Stack image and prompt (4 channels).
    input_array = np.concatenate([image_tensor, np.expand_dims(prompt_heatmap, axis=0)], axis=0)
    input_tensor = torch.from_numpy(input_array).float().unsqueeze(0).to(device)
    
    return input_tensor, prompt_heatmap

def postprocess_output(output, image_dims):
    # Convert prediction (assumed to be a 2D numpy array) to original dimensions.
    output = output.astype(np.uint8)
    output = cv2.resize(output, (image_dims[1], image_dims[0]), interpolation=cv2.INTER_NEAREST)
    return output

def remap_gt_mask(gt_mask, class_id):
    """
    Remaps the ground truth mask to the format used by the model.
    Original trimap: 1 = foreground, 2 = background, 3 = unclassified.
    We want: background = 0, foreground = (cat:1 or dog:2), unclassified = 3.
    """
    new_mask = np.full(gt_mask.shape, 3, dtype=np.uint8)
    new_mask[gt_mask == 2] = 0
    new_mask[gt_mask == 1] = class_id
    return new_mask

# Setup argument parser.
parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, default='./Example.jpg', help='Path to input image')
parser.add_argument('--prompt_x', type=int, default=None, help='Prompt x coordinate in original image')
parser.add_argument('--prompt_y', type=int, default=None, help='Prompt y coordinate in original image')
parser.add_argument('--gt', type=str, default=None, help='Path to ground truth mask (optional)')
parser.add_argument('--output', type=str, default=None, help='Path to save output segmentation image')
parser.add_argument('--weights', type=str, default='./prompt_weights/prompt_unet_model_256_epochs_50.pth', help='Path to model weights')
parser.add_argument('--dim', type=int, default=256, help='Image dimension (resize)')
parser.add_argument('--gpu', type=int, default=0, help='GPU id (if available), defaults to cpu if gpu is not available, and 0 if multiple GPUs are available')
args = parser.parse_args()

device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
dim = args.dim

# Load input image.
image = cv2.imread(args.input)
if image is None:
    raise ValueError(f"Could not load image at {args.input}")
orig_dims = image.shape[:2]

# Determine prompt coordinate. If not provided, use center.
if args.prompt_x is not None and args.prompt_y is not None:
    prompt_coords = (args.prompt_x, args.prompt_y)
else:
    prompt_coords = (image.shape[1]//2, image.shape[0]//2)

# Preprocess image and prompt.
input_tensor, prompt_heatmap = preprocess_image_and_prompt(image, prompt_coords, dim, device)

# Load the prompt segmentation model.
model = PromptUNet(num_classes=3, in_channels=4).to(device)
model.load_state_dict(torch.load(args.weights, map_location=device))
model.eval()

# Run inference.
with torch.no_grad():
    output_logits = model(input_tensor)
    pred = torch.argmax(output_logits, dim=1).cpu().numpy().squeeze()

# Resize prediction to original dimensions.
pred_resized = postprocess_output(pred, orig_dims)

print("Unique prediction labels:", np.unique(pred_resized))

# Visualization using matplotlib.
# Define a custom colormap: (0: background, 1: cat, 2: dog, 3: for overlay if needed)
cmap = ListedColormap(['black', 'red', 'blue', 'white'])

# Create figure with 3 subplots: input image, prompt heatmap, segmentation output.

f = 3 if args.gt is None else 4

fig, ax = plt.subplots(1, f, figsize=(f*5, 5))

# Input image.
ax[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
ax[0].set_title('Input Image')
ax[0].axis('off')

# Prompt heat map (resized to same dim as input image if desired).
ax[1].imshow(prompt_heatmap, cmap='hot')
ax[1].set_title('Prompt Heat Map')
ax[1].axis('off')

# Output segmentation.
ax[2].imshow(pred_resized, cmap=cmap, vmin=0, vmax=3)
ax[2].set_title('Segmentation Output')
ax[2].axis('off')

if args.gt:
    # Load and display ground truth mask.
    gt_mask = cv2.imread(args.gt, cv2.IMREAD_GRAYSCALE)
    if gt_mask is None:
        raise ValueError(f"Could not load ground truth mask at {args.gt}")
    
    gt_mask = cv2.resize(gt_mask, (dim, dim), interpolation=cv2.INTER_NEAREST)

    class_id = 1 if os.path.basename(args.input)[0].isupper() else 2  # cat or dog

    print(class_id)
    print(os.path.basename(args.input))


    gt_mask = remap_gt_mask(gt_mask, class_id)  # Assuming class_id for cat is 1.

    ax[3].imshow(gt_mask, cmap=cmap, vmin=0, vmax=3)
    ax[3].set_title('Ground Truth Mask')
    ax[3].axis('off')


plt.tight_layout()
plt.show()

# Optionally, save output segmentation image.
if args.output:
    cv2.imwrite(args.output, pred_resized)
    print(f"Output segmentation saved to {args.output}")
