import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention

class VAE_AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, channels)
        self.attention = SelfAttention(1, channels)
        
    def forward(self, x):
        residual = x
        b, c, h, w = x.shape
        x = x.view((b, c, h*w))
        x = x.transpose(-1, -2)
        x = self.attention(x)
        x = x.transpose(-1, -2)
        x = x.view((b, c, h, w))
        x += residual
        return x

class VAE_ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.groupnorm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        
        self.groupnorm2 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
            
    def forward(self, x):
        residual = x
        x = self.groupnorm1(x)
        x = F.silu(x)
        x = self.conv1(x)
        x = self.groupnorm2(x)
        x = F.silu(x)
        x = self.conv2(x)
        return x + self.residual_layer(residual)
    
class VAE_Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.initial_conv = nn.Sequential(
            nn.Conv2d(4, 4, kernel_size=1, padding=0),
            nn.Conv2d(4, 512, kernel_size=3, padding=1)
        )
        
        self.initial_residual_attention_blocks = nn.Sequential(
            VAE_ResidualBlock(512, 512),
            VAE_AttentionBlock(512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512)
        )
        
        self.upsampling = nn.ModuleList([
            nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                VAE_ResidualBlock(512, 512),
                VAE_ResidualBlock(512, 512),
                VAE_ResidualBlock(512, 512)
            ), 
            nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                VAE_ResidualBlock(512, 256),
                VAE_ResidualBlock(256, 256),
                VAE_ResidualBlock(256, 256)
            ),
            nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                VAE_ResidualBlock(256, 128),
                VAE_ResidualBlock(128, 128),
                VAE_ResidualBlock(128, 128),
            )
        ])
        
        self.final_block = nn.Sequential(
            nn.GroupNorm(32, 128),
            nn.SiLU(),
            nn.Conv2d(128, 3, kernel_size=3, padding=1)
        )
        
    def forward(self, x):
        x /= 0.18215
        x = self.initial_conv(x)
        x = self.initial_residual_attention_blocks(x)
        for upsample in self.upsampling:
            x = upsample(x)
        x = self.final_block(x)
        return x
