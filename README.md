# Dataset Preparation
Download dataset (images, and annotations) from here; https://www.robots.ox.ac.uk/%7Evgg/data/pets/ .

Extract and place the files inside the project directory as shown below:

```
Computer_Vision_Project/
│ ├── dataset/ 
| │ ├── annotations/ 
| │ │ ├── trimaps/ 
| │ │ ├── xmls/ 
| │ │ ├── ._trimaps 
| │ │ ├── README 
| │ │ ├── test.txt 
| │ │ ├── trainval.txt 
| │ ├── images/
| | │ ├──Abyssinian_1.jpg 
| | | ├──...
```
# UNet Model

The UNet model is implemented in the `unet.py` file. It consists of an encoder-decoder architecture with skip connections. The encoder downsamples the input image, capturing context, while the decoder upsamples the feature maps to produce the final segmentation map.

## UNet Architecture

- **DoubleConv**: A helper class that performs two consecutive convolution operations, each followed by batch normalization and ReLU activation.
- **UNet**: The main UNet class that defines the architecture, including the encoder, decoder, and skip connections.

## Training the UNet

The training script is provided in the `unet_training.ipynb` notebook. It includes data loading, model training, and validation steps.

## Inference with UNet

The `unet_ui.py` script allows you to perform inference using a trained UNet model. It takes an input image, processes it through the model, and saves or displays the output.

### Usage

```bash
python unet_ui.py --input <input_image> --weights <path_to_weights> --output <output_image> (optionally --gt <path_to_ground_truth>)
```

## Evaluation

The `unet_evaluation.py` script evaluates the performance of the trained UNet model on a test dataset. It calculates metrics such as pixel accuracy, precision, recall, and F1 score.

### Usage

```bash
python unet_evaluation.py --dim <image_dimension> --weights <path_to_weights> --metrics <path_to_save_file>
```

This script outputs the evaluation metrics and saves them to the specified file.