import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import random
import math
import numpy as np

class PositionalEmbeddings(nn.Module):
    def __init__(self,embed_dim,time_steps):
        super().__init__()
        pos=torch.arange(time_steps).unsqueeze(1).float()
        div=torch.exp(torch.arange(0,embed_dim,2).float()*-(math.log(10000.0)/embed_dim))
        embeddings=torch.zeros(time_steps,embed_dim,requires_grad=False)
        embeddings[:,0::2]=torch.sin(pos*div)
        embeddings[:,1::2]=torch.cos(pos*div)
        self.embeddings=embeddings
        
    def forward(self,x,t):
        embeddings=self.embeddings[t].to(x.device)
        return embeddings[:,:,None,None]

class ResidualBlock(nn.Module):
    def __init__(self,channels,num_groups,dropout):
        super().__init__()
        self.group_norm1=nn.GroupNorm(num_groups,channels)
        self.group_norm2=nn.GroupNorm(num_groups,channels)
        self.relu=nn.ReLU(inplace=True)
        self.conv1=nn.Conv2d(channels,channels,kernel_size=3,padding=1)
        self.conv2=nn.Conv2d(channels,channels,kernel_size=3,padding=1)
        self.dropout=nn.Dropout(dropout,inplace=True)
    
    def forward(self,x,embeddings):
        x=x+embeddings[:,:x.shape[1],:,:]
        out=self.conv1(self.relu(self.group_norm1(x)))
        out=self.dropout(out)
        out=self.conv2(self.relu(self.group_norm2(out)))
        return out+x


class Attention(nn.Module):
    def __init__(self,embed_dim,num_heads,dropout_prob):
        super().__init__()
        self.embed_dim=embed_dim
        self.proj1=nn.Linear(embed_dim,embed_dim*3)
        self.proj2=nn.Linear(embed_dim,embed_dim)
        self.num_heads=num_heads
        self.dropout_prob=dropout_prob
        self.layer_norm=nn.LayerNorm(embed_dim)

    def forward(self, x):
        h,w=x.shape[2:]
        x=rearrange(x, 'b c h w -> b (h w) c')
        x=self.layer_norm(x)
        x=self.proj1(x)
        x=rearrange(x, 'b L (C H K) -> K b H L C',K=3,H=self.num_heads,C=self.C//self.num_heads)
        q,k,v=x[0], x[1], x[2]
        x=F.scaled_dot_product_attention(q,k,v,is_causal=False,dropout_p=self.dropout_prob)
        x=rearrange(x,'b H (h w) C -> b h w (C H)',h=h,w=w,C=self.C//self.num_heads)
        x=self.proj2(x)
        return rearrange(x,'b h w C -> b C h w')
    
class UNetLayer(nn.Module):
    def __init__(self,upscale,attention,num_groups,dropout,num_heads,embed_dim):
        super().__init__()
        self.residual_block1=ResidualBlock(embed_dim,num_groups,dropout)
        self.residual_block2=ResidualBlock(embed_dim,num_groups,dropout)
        if upscale:
            self.conv=nn.ConvTranspose2d(embed_dim,embed_dim//2,kernel_size=4,stride=2,padding=1)
        else:
            self.conv=nn.Conv2d(embed_dim,embed_dim*2,kernel_size=3,stride=2,padding=1)
        if attention:
            self.attention_layer=Attention(embed_dim,num_heads,dropout)
        
    def forward(self,x,embeddings):
        x=self.residual_block1(x,embeddings)
        if hasattr(self,'attention_layer'):
            x=self.attention_layer(x)
        x=self.residual_block2(x,embeddings)
        return self.conv(x),x

class UNet(nn.Module):
    def __init__(self,channels=[64,128,256,512,512,384],attentions=[False, True, False, False, False, True],upscales=[False, False, False, True, True, True],
                 num_groups=32,dropout=0.1,num_heads=8,input_channels=1,output_channels=1,time_steps=1000):
        super().__init__()
        self.num_layers=len(channels)
        self.single_conv=nn.Conv2d(input_channels,channels[0],kernel_size=3,padding=1)
        out_channels=(channels[-1]//2)+channels[0]
        self.late_conv=nn.Conv2d(out_channels,out_channels//2,kernel_size=3,padding=1)
        self.output_conv=nn.Conv2d(out_channels//2,output_channels,kernel_size=1)
        self.relu=nn.ReLU(inplace=True)
        self.embeddings=PositionalEmbeddings(max(channels),time_steps)
        for i in range(self.num_layers):
            layer=UNetLayer(upscales[i],attentions[i],num_groups,dropout,num_heads,channels[i])
            setattr(self,f'layer{i+1}',layer)
    
    def forward(self,x,t):
        x=self.single_conv(x)
        residuals=[]
        for i in range(self.num_layers//2):
            layer=getattr(self,f'layer{i+1}')
            embeddings=self.embeddings(x,t)
            x,r=layer(x,embeddings)
            residuals.append(r)
        for i in range(self.num_layers//2,self.num_layers):
            layer=getattr(self,f'layer{i+1}')
            x=torch.concat((layer(x,embeddings)[0],residuals[self.num_layers-i-1]),dim=1)
        return self.output_conv(self.relu(self.late_conv(x)))