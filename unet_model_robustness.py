import os
import pickle
import cv2
import numpy as np
import torch
from tqdm import tqdm
from models.unet import UNet
import argparse
from skimage.util import random_noise
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

"""
Parsing for Command Line Arguments
"""
num_seg_classes = 3  # 0: background, 1: cat, 2: dog
parser = argparse.ArgumentParser()
parser.add_argument('--dim', type=int, default=256, help='image dimension')
parser.add_argument('--weights', type=str, default='./unet_weights/unet_model_256_epochs_50.pth',
                    help='path to weights file')
parser.add_argument('--metrics', type=str, default='./robustness/unet_256_epochs_50.txt', help='path to metrics file')
args = parser.parse_args()

dim = args.dim
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
weight_file = args.weights
metrics_filename = args.metrics

model = UNet(num_classes=3).to(device)
model.load_state_dict(torch.load(weight_file, map_location=device))
model.eval()

"""
Loading the Test Images and Masks 
"""

# Load test image and trimap paths.
with open('test_image_paths.pkl', 'rb') as f:
    test_images = pickle.load(f)
with open('test_trimap_paths.pkl', 'rb') as f:
    test_trimap_paths = pickle.load(f)

if len(test_images) != len(test_trimap_paths):
    raise ValueError("Number of test images should match number of trimaps.")

print(f"Found {len(test_images)} test images.")

"""
Helper Functions for Pre-processing and Post-processing
"""
#  Preprocess the input image to match the model's input dimensions and format.
def preprocess_image(image:np.ndarray, dim, device):
    # Preprocess the image
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


"""
!!! Perturbation Methods !!!
"""

def add_gaussian_pixel_noise(image:np.ndarray, idx:int) -> np.ndarray:
    noise_array = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
    noise = np.random.normal(loc=0.0, scale=noise_array[idx], size=image.shape)
    noisy_image = image.astype(np.float32) + noise
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    return noisy_image

def add_gaussian_blur(image:np.ndarray, idx:int) -> np.ndarray:
    # create a 3x3 approximate Guassian kernel
    gaussian_kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16.0
    # apply the kernel to the image idx times
    blurred_image = image
    for _ in range(idx):
        blurred_image = cv2.filter2D(blurred_image, -1, gaussian_kernel)

    return blurred_image

def increase_contrast(image: np.ndarray, idx: int) -> np.ndarray:
    contrasts = [1.0, 1.01, 1.02, 1.03, 1.04, 1.05, 1.1, 1.15, 1.2, 1.25]
    contrast = contrasts[idx]
    # Multiply each pixel by the contrast factor.
    new_image = image.astype(np.float32) * contrast
    # Clip values to the valid range [0, 255] and convert to uint8.
    new_image = np.clip(new_image, 0, 255).astype(np.uint8)
    return new_image

def decrease_contrast(image: np.ndarray, idx: int) -> np.ndarray:
    contrasts = [1.0, 0.95, 0.9, 0.85, 0.8, 0.6, 0.4, 0.3, 0.2, 0.1]
    contrast = contrasts[idx]
    # Multiply each pixel by the contrast factor.
    new_image = image.astype(np.float32) * contrast
    # Clip values to the valid range [0, 255] and convert to uint8.
    new_image = np.clip(new_image, 0, 255).astype(np.uint8)
    return new_image

def increase_brightness(image: np.ndarray, idx: int) -> np.ndarray:
    brightnesses = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]
    brightness = brightnesses[idx]

    # Add brightness to each pixel and ensure the values remain in the range 0..255.
    new_image = image.astype(np.float32) + brightness
    new_image = np.clip(new_image, 0, 255).astype(np.uint8)
    return new_image

def decrease_brightness(image: np.ndarray, idx: int) -> np.ndarray:
    brightnesses = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]
    brightness = brightnesses[idx]

    # Subtract brightness from each pixel and ensure the values remain in the range 0..255.
    new_image = image.astype(np.float32) - brightness
    new_image = np.clip(new_image, 0, 255).astype(np.uint8)
    return new_image

def increase_occlusion(image: np.ndarray, idx: int) -> np.ndarray:
    square_edge_lengths = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]
    square_edge_length = square_edge_lengths[idx]

    # If the square edge length is 0, no occlusion is applied.
    if square_edge_length == 0:
        return image.copy()

    # Assume the image is in H x W x C format.
    H, W = image.shape[:2]

    # Ensure the square fits within the image dimensions.
    if square_edge_length > H or square_edge_length > W:
        square_edge_length = min(H, W)

    # Choose a random top-left coordinate for the square.
    top = np.random.randint(0, H - square_edge_length + 1)
    left = np.random.randint(0, W - square_edge_length + 1)

    # Create a copy of the image and set the selected region to black (0).
    occluded_image = image.copy()
    occluded_image[top:top + square_edge_length, left:left + square_edge_length] = 0

    return occluded_image

def add_salt_and_pepper(image: np.ndarray, idx: int) -> np.ndarray:
    noises = [0.00, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18]
    noise = noises[idx]

    # Add salt and pepper noise, returns float64 image in range [0.0, 1.0]
    noisy_image = random_noise(image, mode='s&p', amount=noise)

    # Convert back to uint8 in range [0, 255]
    noisy_image = (noisy_image * 255).astype(np.uint8)

    return noisy_image

"""
Calculating Dice Score Over Entire Test Set
"""

def get_valid_masks(idx, perturbation_func, p_index):
    # Load the image.
    image = cv2.imread(test_images[idx])
    if image is None:
        raise ValueError("Image not found")

    image_dims = image.shape[:2]  # original dimensions

    # Perturb the image BEFORE preprocessing, if a function is provided.
    image = perturbation_func(image, p_index)

    # now preprocess the image
    image_tensor = preprocess_image(image, dim=dim, device=device)

    # Get model prediction.
    with torch.no_grad():
        output = model(image_tensor)
        pred = torch.argmax(output, dim=1).squeeze().cpu().numpy()

    pred = postprocess_output(pred, image_dims)

    # Load the ground truth trimap.
    gt_mask = cv2.imread(test_trimap_paths[idx], cv2.IMREAD_GRAYSCALE)

    if gt_mask is None:
        raise "Mask not found"

    # Resize ground truth to original dimensions if necessary.
    if gt_mask.shape != image_dims:
        gt_mask = cv2.resize(gt_mask, (image_dims[1], image_dims[0]), interpolation=cv2.INTER_NEAREST)

    # Determine image class.
    filename = os.path.basename(test_images[idx])
    if filename[0].isupper():
        fg_class = 1  # cat
    else:
        fg_class = 2  # dog

    gt_remapped = np.full(gt_mask.shape, 255, dtype=np.uint8)
    gt_remapped[gt_mask == 2] = 0
    gt_remapped[gt_mask == 1] = fg_class

    valid_pixels = gt_remapped != 255
    pred_valid = pred[valid_pixels]
    gt_valid = gt_remapped[valid_pixels]

    if gt_valid.size == 0:
        raise f"Warning: No valid pixels for {test_images[idx]}"

    return gt_valid, fg_class, pred_valid

def calculate_average_dice(perturbation_func, p_index):
    dice_scores = []
    # Randomly sample 100 indices without replacement.
    np.random.seed(0)
    N = int(len(test_images) / 10)
    sample_indices = np.random.choice(len(test_images), size=N, replace=False)
    for i in tqdm(sample_indices, desc="Predicting and Evaluating"):
        gt_valid, fg_class, pred_valid = get_valid_masks(i, perturbation_func=perturbation_func, p_index=p_index)
        # calculate the dice score using sklearn metrics f1 score
        dice = f1_score(gt_valid, pred_valid, average='weighted', zero_division=0)
        dice_scores.append(dice)
    return np.mean(dice_scores)

"""
Calculates the average dice score over the entire test set and saves the metrics to a file under /robustness
"""
def main():
    gaussian_noise_dice = []
    gaussian_blur_dice = []
    image_contrast_increase_dice = []
    image_contrast_decrease_dice = []
    image_brightness_increase_dice = []
    image_brightness_decrease_dice = []
    image_occlusion_dice = []
    image_salt_and_pepper_dice = []

    for i in range(10):
        # Gaussian Noise
        val = calculate_average_dice(perturbation_func=add_gaussian_pixel_noise, p_index=i)
        gaussian_noise_dice.append(val)
        print(f"Gaussian Noise {i}: {val:.4f}")

        # Gaussian Blur
        val = calculate_average_dice(perturbation_func=add_gaussian_blur, p_index=i)
        gaussian_blur_dice.append(val)
        print(f"Gaussian Blur {i}: {val:.4f}")

        # Increase Contrast
        val = calculate_average_dice(perturbation_func=increase_contrast, p_index=i)
        image_contrast_increase_dice.append(val)
        print(f"Increase Contrast {i}: {val:.4f}")

        # Decrease Contrast
        val = calculate_average_dice(perturbation_func=decrease_contrast, p_index=i)
        image_contrast_decrease_dice.append(val)
        print(f"Decrease Contrast {i}: {val:.4f}")

        # Increase Brightness
        val = calculate_average_dice(perturbation_func=increase_brightness, p_index=i)
        image_brightness_increase_dice.append(val)
        print(f"Increase Brightness {i}: {val:.4f}")

        # Decrease Brightness
        val = calculate_average_dice(perturbation_func=decrease_brightness, p_index=i)
        image_brightness_decrease_dice.append(val)
        print(f"Decrease Brightness {i}: {val:.4f}")

        # Increase Occlusion
        val = calculate_average_dice(perturbation_func=increase_occlusion, p_index=i)
        image_occlusion_dice.append(val)
        print(f"Increase Occlusion {i}: {val:.4f}")

        # Salt and Pepper Noise
        val = calculate_average_dice(perturbation_func=add_salt_and_pepper, p_index=i)
        image_salt_and_pepper_dice.append(val)
        print(f"Salt and Pepper Noise {i}: {val:.4f}")

    # Convert all lists to native Python floats before writing
    def to_float_list(arr):
        return [float(x) for x in arr]

    # Save to file
    with open(metrics_filename, 'w') as f:
        f.write(f"Gaussian Noise: {to_float_list(gaussian_noise_dice)}\n")
        f.write(f"Gaussian Blur: {to_float_list(gaussian_blur_dice)}\n")
        f.write(f"Increase Contrast: {to_float_list(image_contrast_increase_dice)}\n")
        f.write(f"Decrease Contrast: {to_float_list(image_contrast_decrease_dice)}\n")
        f.write(f"Increase Brightness: {to_float_list(image_brightness_increase_dice)}\n")
        f.write(f"Decrease Brightness: {to_float_list(image_brightness_decrease_dice)}\n")
        f.write(f"Increase Occlusion: {to_float_list(image_occlusion_dice)}\n")
        f.write(f"Salt and Pepper Noise: {to_float_list(image_salt_and_pepper_dice)}\n")

def plot_graph(metric_name, metrics_filename):
    """
    Reads the robustness metrics file, extracts the dice scores for the specified metric,
    and plots the dice score versus perturbation level.

    Parameters:
        metric_name (str): The key for the perturbation (e.g., "Gaussian Noise")
        metrics_filename (str): Path to the file with saved metrics.
    """
    import matplotlib.pyplot as plt

    # Read the file and find the line that starts with the metric name.
    with open(metrics_filename, 'r') as f:
        lines = f.readlines()

    metric_line = None
    for line in lines:
        if line.startswith(metric_name + ":"):
            metric_line = line
            break
    if metric_line is None:
        raise ValueError(f"Metric {metric_name} not found in the file.")

    # Extract the list of values. Expected format: "Metric: [v1, v2, ..., vN]"
    try:
        # Split on colon, then remove brackets and whitespace.
        values_str = metric_line.split(":", 1)[1].strip().strip("[]")
        # Split on commas and convert to floats.
        dice_scores = [float(val.strip()) for val in values_str.split(",") if val.strip() != '']
    except Exception as e:
        raise ValueError(f"Error parsing metric values: {e}")

    # Create x-axis values corresponding to perturbation levels.
    perturbation_levels = list(range(len(dice_scores)))

    # Set LaTeX style for plotting.
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    # Plotting with enhanced aesthetics.
    plt.figure(figsize=(8, 5))
    plt.plot(perturbation_levels, dice_scores, marker='o', linestyle='-',
             color='darkred', linewidth=2, markersize=6)
    plt.xlabel(r'\textbf{Perturbation Intensity Level}', fontsize=14)
    plt.ylabel(r'\textbf{DICE-S{\o}rensen Score}', fontsize=14)
    plt.title(r'\textbf{UNet DICE-S{\o}rensen Score: ' + metric_name + '}', fontsize=16)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Set x-axis ticks to show every integer perturbation level.
    plt.xticks(perturbation_levels)

    plt.tight_layout()
    # Save the plot into robustness folder with high resolution.
    plt.savefig(f"./robustness/{metric_name}.png", dpi=300)

def plot_all():
    """
    Plot all metrics from the robustness file.
    """
    with open(metrics_filename, 'r') as f:
        lines = f.readlines()

    for line in lines:
        metric_name = line.split(":")[0]
        plot_graph(metric_name, metrics_filename)

def plot_example_images():
    """
    For each of the perturbation methods, show the perturbed image.
    """
    image = cv2.imread(test_images[2500])
    # plot the image as the "original image"
    plot_img(image, title="Original Image")
    #
    # # For each perturbation, display only the perturbed image with a title.
    # perturbed_image = add_gaussian_pixel_noise(image, 9)
    # plot_img(perturbed_image, title="Gaussian Noise")
    #
    # perturbed_image = add_gaussian_blur(image, 9)
    # plot_img(perturbed_image, title="Gaussian Blur")
    #
    # perturbed_image = increase_contrast(image, 9)
    # plot_img(perturbed_image, title="Increase Contrast")
    #
    # perturbed_image = decrease_contrast(image, 9)
    # plot_img(perturbed_image, title="Decrease Contrast")
    #
    # perturbed_image = increase_brightness(image, 9)
    # plot_img(perturbed_image, title="Increase Brightness")
    #
    # perturbed_image = decrease_brightness(image, 9)
    # plot_img(perturbed_image, title="Decrease Brightness")
    #
    # perturbed_image = increase_occlusion(image, 9)
    # plot_img(perturbed_image, title="Increase Occlusion")
    #
    # perturbed_image = add_salt_and_pepper(image, 9)
    # plot_img(perturbed_image, title="Salt and Pepper Noise")

def plot_img_img(img_a, img_b, title_a="Original", title_b="Perturbed"):
    """
    `   Plot two images side by side.
    """
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(cv2.cvtColor(img_a, cv2.COLOR_BGR2RGB))
    axs[0].set_title(title_a)
    axs[0].axis('off')

    axs[1].imshow(cv2.cvtColor(img_b, cv2.COLOR_BGR2RGB))
    axs[1].set_title(title_b)
    axs[1].axis('off')

    plt.tight_layout()

    # save the file in the robustness folder
    plt.savefig(f"./robustness/{title_a}_{title_b}.png")

def plot_img(img, title="Perturbed"):
    """
    Plot one image.
    """
    fig, axs = plt.subplots(1, 1, figsize=(10, 5))
    axs.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axs.set_title(title)
    axs.axis('off')

    plt.tight_layout()

    # Save the file with reduced whitespace around the image
    plt.savefig(f"./robustness/{title}.png", bbox_inches='tight', pad_inches=0.1)

if __name__ == "__main__":
    #main()
    #plot_all()
    #retrive a random image and its mask
    # show the image
    plot_example_images()
