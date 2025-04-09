# Multiclass Segmentation Framework

This repository contains a framework for training, evaluating, and visualizing deep learning models for multiclass image segmentation tasks. The project supports multiple architectures, including UNet, Autoencoder-based segmentation, and CLIP-based segmentation.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Directory Structure](#directory-structure)
- [Setup](#setup)
- [Usage](#usage)
  - [Training](#training)
  - [Evaluation](#evaluation)
  - [Visualisation](#visualisation)
- [Models](#models)


## Overview

This framework is designed for multiclass segmentation tasks, where the goal is to classify each pixel in an image into one of several predefined classes. The supported classes in this project are:
- Background
- Cat
- Dog

## Features

- **Training**: Train models like UNet, Autoencoder-based segmentation, and CLIP-based segmentation.
- **Evaluation**: Evaluate models using metrics such as pixel accuracy, precision, recall, F1 score, IoU, and Dice coefficient.
- **Visualisation**: Segmentations of model output 
- **Robustness Testing**: Test model robustness under noisy conditions.

## Directory Structure
```
CV_multiclass/ 
    ├── autoencoder_model_evaluation.py # Evaluation script for Autoencoder-based segmentation 
    ├── autoencoder_training.py # Training script for Autoencoder 
    ├── clip_model_evaluation.py # Evaluation script for CLIP-based segmentation 
    ├── clip_training.py # Training script for CLIP-based segmentation 
    ├── metrics/ # Directory for storing evaluation metrics 
    ├── models/ # Contains model definitions 
    │ ├── autoencoder.py # Autoencoder model 
    │ ├── clip.py # CLIP-based segmentation model 
    │ ├── unet.py # UNet model 
    ├── prompt_model_evaluation.py # Evaluation script for PromptUNet 
    ├── prompt_unet_training.py # Training script for PromptUNet 
    ├── unet_model_evaluation.py # Evaluation script for UNet 
    ├── unet_training.py # Training script for UNet 
    ├── vis_latent.py # Script for visualizing latent representations 
    ├── README.md # Project documentation
```


## Setup

1. Clone the repository:
```bash
git clone https://github.com/Basemism/CV_multiclass.git
cd CV_multiclass
```
2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Ensure the dataset is prepared in the following structure:
```
trainval_<dim>/
├── images/       # Input images
├── annotations/  # Corresponding segmentation masks
```

## Usage
### Training:
- UNet: `python unet_training.py`
- Autoencoder: `python autoencoder_training.py`
- Clip-based Segmentation: `python clip_training.py`
- Prompt-Unet: `python prompt_unet_training.py`

### Evaluation:
- UNet: `python unet_model_evaluation.py --weights <path_to_weights>`
- Autoencoder: `python autoencoder_model_evaluation.py --weights <path_to_weights>`
- Clip-based Segmentation: `python clip_model_evaluation.py --weights <path_to_weights>`
- Prompt-Unet: `python prompt_model_evaluation.py --weights <path_to_weights>`

### Visualisation
### Visualisation:
- UNet, Autoencoder, CLIP: `python model_ui.py --input <path_to_input_image> --gt <path_to_ground_truth_image> --category <category_id> --output <path_to_output_image> --model <model_name> --weights <path_to_weights_file> --dim <image_dimension> --gpu <gpu_id>`
- Prompt-Based: `python prompt_gui.py`
- Prompt-Based: `python prompt_gui.py`

## Models
- UNet: A popular architecture for image segmentation tasks.
- Autoencoder-based Segmentation: Uses an autoencoder's encoder for feature extraction.
- CLIP-based Segmentation: Combines CLIP's visual encoder with a segmentation decoder.
- PromptUNet: Incorporates prompt-based inputs for segmentation.