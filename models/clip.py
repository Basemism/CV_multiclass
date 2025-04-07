from torch import nn
import torch
import clip

device = "cuda" if torch.cuda.is_available() else "cpu"

class CLIPEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.clip_model, self.preprocess = clip.load("ViT-B/32", device=device) # frozen CLIP backbone

    def forward(self, x):
        # Check if input is a batch of images
        if x.dim() == 3:
            x = x.unsqueeze(0)
        return self.clip_model.encode_image(x)

class SegmentationDecoder(nn.Module):
    def __init__(self, num_classes):
        super(SegmentationDecoder, self).__init__()
        # This decoder assumes the input is of size [B, 64, H/8, W/8]
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)  # from H/8 to H/4
        self.conv1 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.up2 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)  # from H/4 to H/2
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.up3 = nn.ConvTranspose2d(16, 16, kernel_size=2, stride=2)  # from H/2 to H
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

class LatentSegmentationDecoder(nn.Module):
    def __init__(self, num_classes, latent_dim=512, image_size=224):
        """
        Projects a latent vector of shape [B, latent_dim] into a spatial feature map,
        then decodes it into a segmentation map with num_classes channels.
        Assumes the final segmentation map is image_size x image_size.
        """
        super().__init__()
        self.image_size = image_size
        # Calculate spatial dimensions after dividing by 8.
        spatial_dim = image_size // 8  # For image_size=224, spatial_dim=28
        # Fully connected layer to project latent vector to a flattened feature map
        self.fc = nn.Linear(latent_dim, 64 * spatial_dim * spatial_dim)
        # Use the provided segmentation decoder, which expects input shape [B,64,spatial_dim,spatial_dim]
        self.decoder = SegmentationDecoder(num_classes)

    def forward(self, latent):
        # latent: [B, latent_dim]
        x = self.fc(latent)  # now x has shape [B, 64 * spatial_dim * spatial_dim]
        spatial_dim = self.image_size // 8
        x = x.view(-1, 64, spatial_dim, spatial_dim)  # reshape to [B, 64, spatial_dim, spatial_dim]
        logits = self.decoder(x)
        return logits


class CLIPSegmentation(nn.Module):
    name = "clip_segmentation"
    type = "segmentation"

    def __init__(self, num_classes=3):
        super().__init__()
        self.encoder = CLIPEncoder()
        self.decoder = LatentSegmentationDecoder(num_classes)

    def forward(self, x):
        with torch.no_grad():
            features = self.encoder(x)
        features = features.float()
        logits = self.decoder(features)
        return logits