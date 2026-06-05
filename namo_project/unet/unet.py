import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class AttentionGate(nn.Module):
    """
    Attention Gate to filter skip connection features.
    g: gating signal (coarser feature map from decoder)
    x: skip connection signal (finer feature map from encoder)
    """
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # Downsample skip connection feature if dimensions don't match (for pooling steps)
        g_in = self.W_g(g)
        x_in = self.W_x(x)
        
        # Align spatial dimensions by upsampling the gating signal to match skip features
        if g_in.shape[2:] != x_in.shape[2:]:
            g_in = F.interpolate(g_in, size=x_in.shape[2:], mode='bilinear', align_corners=True)
            
        out = self.relu(g_in + x_in)
        out = self.psi(out)
        return x * out

class AttentionUNet(nn.Module):
    """
    Attention U-Net architecture tailored for NAMO occupancy grids.
    Input: (batch_size, 4, H, W) -> Channels: [Walls, Movable Box, Robots, Goals]
    Output: (batch_size, 1, H, W) -> Deadlock Risk Heatmap (probabilities 0-1)
    """
    def __init__(self, in_channels=4, out_channels=1):
        super().__init__()
        
        # Encoder (contracting path)
        self.inc = DoubleConv(in_channels, 32)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(32, 64))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        
        # Bridge
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        
        # Decoder (expanding path) with Attention Gates
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.att1 = AttentionGate(F_g=128, F_l=128, F_int=64)
        self.conv_up1 = DoubleConv(256, 128)
        
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.att2 = AttentionGate(F_g=64, F_l=64, F_int=32)
        self.conv_up2 = DoubleConv(128, 64)
        
        self.up3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.att3 = AttentionGate(F_g=32, F_l=32, F_int=16)
        self.conv_up3 = DoubleConv(64, 32)
        
        # Final prediction layer
        self.outc = nn.Sequential(
            nn.Conv2d(32, out_channels, kernel_size=1),
            nn.Sigmoid() # Risk mapping as probability 0.0 - 1.0
        )

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        
        # Bridge
        x4 = self.down3(x3)
        
        # Decoder 1
        g1 = self.up1(x4)
        att1 = self.att1(g=g1, x=x3)
        # Handle odd-sized dimensions during upsampling alignment
        if g1.shape[2:] != att1.shape[2:]:
            g1 = F.interpolate(g1, size=att1.shape[2:], mode='bilinear', align_corners=True)
        x_up1 = torch.cat([g1, att1], dim=1)
        x_dec1 = self.conv_up1(x_up1)
        
        # Decoder 2
        g2 = self.up2(x_dec1)
        att2 = self.att2(g=g2, x=x2)
        if g2.shape[2:] != att2.shape[2:]:
            g2 = F.interpolate(g2, size=att2.shape[2:], mode='bilinear', align_corners=True)
        x_up2 = torch.cat([g2, att2], dim=1)
        x_dec2 = self.conv_up2(x_up2)
        
        # Decoder 3
        g3 = self.up3(x_dec2)
        att3 = self.att3(g=g3, x=x1)
        if g3.shape[2:] != att3.shape[2:]:
            g3 = F.interpolate(g3, size=att3.shape[2:], mode='bilinear', align_corners=True)
        x_up3 = torch.cat([g3, att3], dim=1)
        x_dec3 = self.conv_up3(x_up3)
        
        return self.outc(x_dec3)

if __name__ == "__main__":
    # Test network with multiple sizes to ensure fully-convolutional dynamic scaling
    model = AttentionUNet()
    
    print("\nAttention U-Net Dynamic Input Scaling Test:")
    print("------------------------------------------")
    for size in [10, 15, 20, 50]:
        test_input = torch.randn(2, 4, size, size)
        test_output = model(test_input)
        print(f"Grid Size: {size:2d}x{size:2d} | Input: {list(test_input.shape)} | Output: {list(test_output.shape)}")
        assert test_output.shape == (2, 1, size, size), f"Failed scaling test for size {size}"
    
    print("\n>>> SUCCESS: Model successfully supports arbitrary sizes (including 20x20 and expansion up to 50x50).")
