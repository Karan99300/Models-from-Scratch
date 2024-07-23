import torch
import torch.nn as nn
import math
from einops import rearrange

#Implementing Convolutional Projections
class ConvolutionalEmbeddings(nn.Module):
    def __init__(self,embed_size,patch_size,stride,in_ch=3):
        super().__init__()
        self.conv_embeddings=nn.Conv2d(in_ch,embed_size,kernel_size=patch_size,stride=stride)
        self.layer_norm=nn.LayerNorm(embed_size)
    
    def forward(self,x):
        b,c,h,w=x.shape
        x=self.conv_embeddings(x)
        x=rearrange(x,'b c h w -> b (h w) c')        
        x=self.layer_norm(x)
        return x
    
class ConvMHA(nn.Module):
    def __init__(self,input_dim,num_heads,kernel_size=3,dropout=0.1,cls_token=False):
        super().__init__()
        self.head_dim=input_dim//num_heads
        padding=(kernel_size-1)//2
        self.forward_conv=self.forward_conv
        self.num_heads=num_heads
        self.cls_token=cls_token
        self.conv=nn.Sequential(
            nn.Conv2d(input_dim,input_dim,kernel_size=kernel_size,padding=padding,stride=1),
            nn.BatchNorm2d(input_dim),
        )
        self.dropout=nn.Dropout(dropout)
        
    def forward_conv(self,x):
        b,hw,c=x.shape
        
        if self.cls_token:
            cls_token,x=torch.split(x,[1,hw-1],1)
            
        height=width=int(math.sqrt(x.shape[1]))
        
        x=rearrange(x,'b (h w) c -> b c h w',h=height,w=width)
        q,k,v=self.conv(x),self.conv(x),self.conv(x)
        
        q=rearrange(q,'b c h w -> b (h w) c')
        k=rearrange(k,'b c h w -> b (h w) c')
        v=rearrange(v,'b c h w -> b (h w) c')
        
        if self.cls_token:
            q=torch.cat((cls_token, q), dim=1)
            k=torch.cat((cls_token, k), dim=1)
            v=torch.cat((cls_token, v), dim=1)
            
        return q,k,v
    
    def forward(self,x):
        q,k,v=self.forward_conv(x)
        
        q=rearrange(q, 'b t (d H) -> b H t d', H=self.num_heads)
        k=rearrange(k, 'b t (d H) -> b H t d', H=self.num_heads)
        v=rearrange(v, 'b t (d H) -> b H t d', H=self.num_heads)
        
        attention_scores=torch.matmul(q,k.transpose(-1,-2))
        attention_scores=attention_scores/math.sqrt(self.head_dim)
        attention_probs=nn.functional.softmax(attention_scores,dim=-1)
        attention_probs=torch.matmul(attention_probs,v)
        attention_probs=rearrange(attention_probs,'b H t d -> b t (H d)')
        
        return attention_probs

class MLP(nn.Module):
    def __init__(self,embed_size):
        super().__init__()
        self.ff=nn.Sequential(
            nn.Linear(embed_size,4*embed_size),
            nn.GELU(),
            nn.Linear(4*embed_size,embed_size)
        )
    
    def forward(self,x):
        return self.ff(x)

class ConvTransformerBlock(nn.Module):
    def __init__(self,embed_size,num_heads,cls_token=False,dropout=0.1):
        super().__init__()
        self.cmha=ConvMHA(embed_size,num_heads,cls_token=cls_token)
        self.mlp=MLP(embed_size)
        self.layer_norm=nn.LayerNorm(embed_size)
        self.dropout=nn.Dropout(dropout)
    
    def forward(self,x):
        x=x+self.dropout(self.cmha(self.layer_norm(x)))
        x=x+self.dropout(self.mlp(self.layer_norm(x)))
        return x
    
class Stage(nn.Module):
    def __init__(self,depth,embed_size,num_heads,patch_size,stride,cls_token=False,in_ch=3):
        super().__init__()
        self.stride=stride
        self.cls_token=cls_token
        self.embeddings=ConvolutionalEmbeddings(embed_size,patch_size,stride,in_ch)
        self.layers=nn.Sequential(*[ConvTransformerBlock(embed_size,num_heads,cls_token) for _ in range(depth)])
        
        if cls_token:
            self.cls_tokens=nn.Parameter(torch.randn(1,1,384))
    
    def forward(self,x,ch_out=False):
        b,c,h,w=x.shape
        x=self.embeddings(x)
        if self.cls_token:
            cls_tokens=self.cls_tokens.expand(b,-1,-1)
            x = torch.cat([cls_tokens,x],dim=1)
            
        x=self.layers(x)
        
        if not ch_out:
            x = rearrange(x, 'b (h w) c -> b c h w', h=(h-1)//self.stride, w=(w-1)//self.stride)
        return x

class CvT(nn.Module):
    def __init__(self, embed_size, num_class):
        super().__init__()
        self.stage1=Stage(depth=1,embed_size=64,num_heads=1,patch_size=7,stride=4)
        self.stage2=Stage(depth=2,embed_size=192,num_heads=3,patch_size=3,stride=2,in_ch = 64)
        self.stage3=Stage(depth=10,embed_size=384,num_heads=6,patch_size=3,stride=2,in_ch=192,cls_token=True)
        self.ff = nn.Sequential(
            nn.Linear(6*embed_size,embed_size),
            nn.ReLU(),
            nn.Linear(embed_size,num_class)
        )

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x, ch_out=True)
        x = x[:, 1, :]
        x = self.ff(x)
        return x  