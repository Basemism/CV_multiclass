import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    """2 x (Convolution => BatchNorm => ReLU)"""
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)


class UNet(nn.Module):
    def __init__(self, num_classes):
        super(UNet, self).__init__()
        # Initial convolution block
        self.inc = DoubleConv(3, 16)
        
        # Downsampling layers
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(16, 32))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(32, 64))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        
        # Upsampling layers
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(256, 128)
        
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(128, 64)
        
        self.up3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(64, 32)
        
        self.up4 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.conv4 = DoubleConv(32, 16)
        
        # Output layer
        self.outc = nn.Conv2d(16, num_classes, kernel_size=1)
    
    def forward(self, x):
        x1 = self.inc(x)       # (B, 16, H, W)
        x2 = self.down1(x1)    # (B, 32, H/2, W/2)
        x3 = self.down2(x2)    # (B, 64, H/4, W/4)
        x4 = self.down3(x3)    # (B, 128, H/8, W/8)
        x5 = self.down4(x4)    # (B, 256, H/16, W/16)
        
        u1 = self.up1(x5)      # (B, 128, H/8, W/8)
        # Concatenate skip connection from x4.
        u1 = torch.cat([u1, x4], dim=1)
        u1 = self.conv1(u1)
        
        u2 = self.up2(u1)      # (B, 64, H/4, W/4)
        u2 = torch.cat([u2, x3], dim=1)
        u2 = self.conv2(u2)
        
        u3 = self.up3(u2)      # (B, 32, H/2, W/2)
        u3 = torch.cat([u3, x2], dim=1)
        u3 = self.conv3(u3)
        
        u4 = self.up4(u3)      # (B, 16, H, W)
        u4 = torch.cat([u4, x1], dim=1)
        u4 = self.conv4(u4)
        
        logits = self.outc(u4)  # (B, num_classes, H, W)
        return logits
    