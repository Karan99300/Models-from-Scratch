import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from attention import SelfAttention, CrossAttention

class TimeEmbedding(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, 4*embed_dim)
        self.linear2 = nn.Linear(4*embed_dim, 4*embed_dim)
        
    def forward(self, x):
        x = self.linear1(x)
        x = F.silu(x)
        x = self.linear2(x)
        return x
    
class UNet_ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_times=1280):
        super().__init__()
        self.groupnorm_feature = nn.GroupNorm(32, in_channels)
        self.conv_feature = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.linear_time = nn.Linear(n_times, out_channels)
        
        self.groupnorm_merged = nn.GroupNorm(32, out_channels)
        self.conv_merged = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        
    def forward(self, feature, time):
        residual = feature
        feature = self.groupnorm_feature(feature)
        feature = F.silu(feature)
        feature = self.conv_feature(feature)
        
        time = F.silu(time)
        time = self.linear_time(time)
        
        merged = feature + time.unsqueeze(-1).unsqueeze(-1)
        merged = self.groupnorm_merged(merged)
        merged = F.silu(merged)
        merged = self.conv_merged(merged)
        
        return merged + self.residual_layer(residual)
    
class UNet_AttentionBlock(nn.Module):
    def __init__(self, num_heads, embed_dim, context_dim=768):
        super().__init__()
        channels = num_heads*embed_dim
        self.groupnorm = nn.GroupNorm(32, channels, eps=1e-6)
        self.conv_input = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
        self.layernorm1 = nn.LayerNorm(channels)
        self.attention1 = SelfAttention(num_heads, channels, in_proj_bias=False)
        self.layernorm2 = nn.LayerNorm(channels)
        self.attention2 = CrossAttention(num_heads, channels, context_dim, in_proj_bias=False)
        self.layernorm3 = nn.LayerNorm(channels)
        self.linear1 = nn.Linear(channels, 4*channels*2)
        self.linear2 = nn.Linear(4*channels, channels)
        self.conv_output = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
    
    def forward(self, x, context):
        residual_long = x
        x = self.groupnorm(x)
        x = self.conv_input(x)
        n, c, h, w = x.shape
        x = x.view((n, c, h*w))
        x = x.transpose(-1,-2)
        
        residual_short = x
        x = self.layernorm1(x)
        x = self.attention1(x)
        x += residual_short
        
        residual_short = x
        x = self.layernorm2(x)
        x = self.attention2(x, context)
        x += residual_short
        
        residual_short = x
        x = self.layernorm3(x)
        
        x, gate = self.linear1(x).chunk(2, dim=-1)
        x = x * F.gelu(gate)
        x = self.linear2(x)
        x += residual_short
        
        x = x.transpose(-1,-2)
        x = x.view((n, c, h, w))
        return self.conv_output(x) + residual_long
    
class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)
    
class SwitchSequential(nn.Sequential):
    def forward(self, x, context, time):
        for layer in self:
            if isinstance(layer, UNet_AttentionBlock):
                x = layer(x, context) 
            elif isinstance(layer, UNet_ResidualBlock):
                x = layer(x, time)
            else:
                x = layer(x) 
        return x   

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.ModuleList([
            SwitchSequential(nn.Conv2d(4, 320, kernel_size=3, padding=1)),
            SwitchSequential(UNet_ResidualBlock(320, 320), UNet_AttentionBlock(8, 40)),
            SwitchSequential(UNet_ResidualBlock(320, 320), UNet_AttentionBlock(8, 40)),
            SwitchSequential(nn.Conv2d(320, 320, kernel_size=3, stride=2, padding=1)),
            SwitchSequential(UNet_ResidualBlock(320, 640), UNet_AttentionBlock(8, 80)),
            SwitchSequential(UNet_ResidualBlock(320, 640), UNet_AttentionBlock(8, 80)),
            SwitchSequential(nn.Conv2d(640, 640, kernel_size=3, stride=2, padding=1)),
            SwitchSequential(UNet_ResidualBlock(640, 1280), UNet_AttentionBlock(8, 160)),
            SwitchSequential(UNet_ResidualBlock(640, 1280), UNet_AttentionBlock(8, 160)),
            SwitchSequential(nn.Conv2d(1280, 1280, kernel_size=3, stride=2, padding=1)),
            SwitchSequential(UNet_ResidualBlock(1280, 1280)),
            SwitchSequential(UNet_ResidualBlock(1280, 1280)),
        ])
        
        self.bottleneck_layer = SwitchSequential(
            UNet_ResidualBlock(1280, 1280),
            UNet_AttentionBlock(8, 160),
            UNet_ResidualBlock(1280, 1280)
        )
        
        self.decoders = nn.ModuleList([
            SwitchSequential(UNet_ResidualBlock(2560, 1280)),
            SwitchSequential(UNet_ResidualBlock(2560, 1280)), 
            SwitchSequential(UNet_ResidualBlock(2560, 1280), Upsample(1280)),
            SwitchSequential(UNet_ResidualBlock(2560, 1280), UNet_AttentionBlock(8, 160)),
            SwitchSequential(UNet_ResidualBlock(2560, 1280), UNet_AttentionBlock(8, 160)),
            SwitchSequential(UNet_ResidualBlock(1920, 1280), UNet_AttentionBlock(8, 160), Upsample(1280)),
            SwitchSequential(UNet_ResidualBlock(1920, 640), UNet_AttentionBlock(8, 80)),
            SwitchSequential(UNet_ResidualBlock(1280, 640), UNet_AttentionBlock(8, 80)),
            SwitchSequential(UNet_ResidualBlock(960, 640), UNet_AttentionBlock(8, 80), Upsample(640)),
            SwitchSequential(UNet_ResidualBlock(960, 320), UNet_AttentionBlock(8, 40)),
            SwitchSequential(UNet_ResidualBlock(640, 320), UNet_AttentionBlock(8, 40)),
            SwitchSequential(UNet_ResidualBlock(640, 320), UNet_AttentionBlock(8, 40)),
        ])
        
class UNet_OutputLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        
    def forward(self, x):
        x = self.groupnorm(x)
        x = F.silu(x)
        x = self.conv(x)
        return x

class Diffusion(nn.Module):
    def __init__(self):
        self.time_embedding = TimeEmbedding(320)
        self.unet = UNet()
        self.final = UNet_OutputLayer(320, 4)
    
    def forward(self, latent, context, time):
        time = self.time_embedding(time)
        output = self.unet(latent, context, time)
        output = self.final(output)
        return output     