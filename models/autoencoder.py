import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, latent_dim=256):
        super(Autoencoder, self).__init__()
        # Encoder: Similar to UNet encoder (simpler version)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),  # [B,16,H,W]
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # [B,16,H/2,W/2]
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1), # [B,32,H/2,W/2]
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # [B,32,H/4,W/4]
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1), # [B,64,H/4,W/4]
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # [B,64,H/8,W/8]
        )
        # Decoder: Mirror the encoder.
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),  # [B,32,H/4,W/4]
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),  # [B,16,H/2,W/2]
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(16, 3, kernel_size=2, stride=2),   # [B,3,H,W]
            nn.Sigmoid()  # Output in [0,1]
        )
        
    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed

class SegmentationDecoder(nn.Module):
    def __init__(self, num_classes):
        super(SegmentationDecoder, self).__init__()
        # A simple decoder that upsamples from the encoder output resolution to full size.
        # Here, we assume the encoder output is of size (B,64,H/8,W/8).
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)  # H/4, W/4
        self.conv1 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.up2 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)  # H/2, W/2
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.up3 = nn.ConvTranspose2d(16, 16, kernel_size=2, stride=2)  # H, W
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.out_conv = nn.Conv2d(16, num_classes, kernel_size=1)
    
    def forward(self, x):
        x = self.up1(x)
        x = self.conv1(x)
        x = self.up2(x)
        x = self.conv2(x)
        x = self.up3(x)
        x = self.conv3(x)
        logits = self.out_conv(x)
        return logits

class SegmentationModel(nn.Module):
    def __init__(self, encoder, num_classes):
        super(SegmentationModel, self).__init__()
        self.encoder = encoder  # Pre-trained and frozen
        self.decoder = SegmentationDecoder(num_classes)
    
    def forward(self, x):
        with torch.no_grad():
            features = self.encoder(x)
        logits = self.decoder(features)
        return logits