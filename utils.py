import torch
import cv2
import numpy as np

"""
Preprocess the input image to match the model's input dimensions and format.
"""
def preprocess_image(image, dim, device):
    # Preprocess the image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (dim, dim))
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))
    
    image_tensor = torch.from_numpy(image).float().unsqueeze(0).to(device)

    return image_tensor


"""
Postprocess the model output to obtain the final segmentation mask equivalent to the input image resolution.
"""
def postprocess_output(output, image_dims):
    output = output + 1 # Trimap {0,1,2} -> {1,2,3}
    output = output.astype(np.uint8)
    output = cv2.resize(output, (image_dims[1], image_dims[0]), interpolation=cv2.INTER_NEAREST)

    return output

"""
Train the model for one epoch. Returns the average loss.
"""
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    epoch_loss = 0.0
    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)  # (B, num_classes, H, W)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item() * images.size(0)
    return epoch_loss / len(loader.dataset)


"""
Evaluate the model on the validation set. Returns the average loss and accuracy.
"""
def evaluate(model, loader, criterion, device):
    model.eval()
    epoch_loss = 0.0
    correct = 0
    total_pixels = 0
    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            epoch_loss += loss.item() * images.size(0)
            
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == masks).sum().item()
            total_pixels += torch.numel(masks)
    accuracy = correct / total_pixels
    return epoch_loss / len(loader.dataset), accuracy
