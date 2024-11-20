import torch
import torch.nn as nn
import torch.nn.functional as F 
from decoder import VAE_AttentionBlock, VAE_ResidualBlock

class VAE_Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.initial_conv = nn.Conv2d(3, 128, kernel_size=3, padding=1)
        
        self.initial_residual_blocks = nn.Sequential(
            VAE_ResidualBlock(128, 128),
            VAE_ResidualBlock(128, 128)
        )
        
        self.downsampling = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),
                VAE_ResidualBlock(128, 256),
                VAE_ResidualBlock(256, 256)
            ),
            nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0),
                VAE_ResidualBlock(256, 512),
                VAE_ResidualBlock(512, 512)
            ),
            nn.Sequential(
                nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0),
                VAE_ResidualBlock(512, 512),
                VAE_ResidualBlock(512, 512),
                VAE_ResidualBlock(512, 512)
            )
        ])
        
        self.final_block = nn.Sequential(
            VAE_AttentionBlock(512),
            VAE_ResidualBlock(512, 512),
            nn.GroupNorm(32, 512),
            nn.SiLU(),
            nn.Conv2d(512, 8, kernel_size=3, padding=1),
            nn.Conv2d(8, 8, kernel_size=1, padding=0)
        )
        
    def forward(self, x, noise):
        x = self.initial_conv(x)
        x = self.initial_residual_blocks(x)
        
        for downsampling in self.downsampling:
            #Asymmetric padding
            x = F.pad(x, (0, 1, 0, 1))
            x = downsampling(x)
        
        x = self.final_block(x)
        mean, log_variance = torch.chunk(x, 2, dim=1)
        log_variance = torch.clamp(log_variance, -30, 20)
        variance = log_variance.exp()
        sd = variance.sqrt()
        x = mean + sd*noise
        x *= 0.18215 #scaling constant
        
        return x  
        