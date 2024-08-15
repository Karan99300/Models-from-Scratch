import torch
import torch.nn as nn
import torchvision.transforms as T
import numpy as np
import einops
from einops import rearrange
import math

class PositionalEmbedding(nn.Module):
    def __init__(self,embed_dim,max_len=2**12):
        super().__init__()
        pos=torch.arange(max_len).unsqueeze(1)
        i=torch.arange(embed_dim//2).unsqueeze(0)
        angle=pos/(10000**(2*i/embed_dim))
        self.pe_matrix=torch.zeros(size=(max_len,embed_dim))
        self.pe_matrix[:,0::2]=torch.sin(angle)
        self.pe_matrix[:,1::2]=torch.cos(angle)
        self.register_buffer("pe_matrix",self.pe_matrix)
    
    def forward(self,x):
        return x + einops.repeat(
            self.pe_matrix.to(x.device), pattern="l d -> b l d", b=x.size(0),
        )[:, : x.size(1), :]
        
class Attention(nn.Module):
    def __init__(self,embed_dim,num_heads,dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim//num_heads
        
        self.q_proj=nn.Linear(embed_dim,embed_dim,bias=False)
        self.k_proj=nn.Linear(embed_dim,embed_dim,bias=False)
        self.v_proj=nn.Linear(embed_dim,embed_dim,bias=False)
        self.dropout=nn.Dropout(dropout)
        self.output_proj=nn.Linear(embed_dim,embed_dim,bias=False)
    
    def forward(self,x,mask=None):
        q=self.q_proj(x)
        k=self.k_proj(x)
        v=self.v_proj(x)
        
        q=rearrange(q,'b n (h d) -> b h n d',h=self.num_heads)
        k=rearrange(k,'b n (h d) -> b h n d',h=self.num_heads)
        v=rearrange(v,'b n (h d) -> b h n d',h=self.num_heads)
        
        attention_scores=torch.matmul(q,k.transpose(-2,-1))
        attention_scores=attention_scores/math.sqrt(self.head_dim)
        
        if mask is not None:
            mask=rearrange(mask,'b n -> b 1 1 n')
            attention_scores=attention_scores.masked_fill(mask==0,float('-inf'))
        
        attention_probs=nn.functional.softmax(attention_scores,dim=-1)
        attention_probs=self.dropout(attention_probs)
        attention_output=torch.matmul(attention_probs,v)
        return self.output_proj(attention_output)
    
class TransformerEncoder(nn.Module):
    def __init__(self,embed_dim,num_heads,dropout=0.0):
        super().__init__()
        self.layer_norm1=nn.LayerNorm(embed_dim)
        self.attention=Attention(embed_dim,num_heads,dropout)
        self.layer_norm2=nn.LayerNorm(embed_dim)
        self.mlp=nn.Sequential(
            nn.Linear(embed_dim,4*embed_dim),
            nn.GELU(),
            nn.Linear(4*embed_dim,embed_dim)
        )
        
    def forward(self,x,mask=None):
        x=x+self.attention(self.layer_norm1(x),mask)
        x=x+self.mlp(self.layer_norm2(x))
        return x
    
class TextEncoder(nn.Module):
    def __init__(self,vocab_size,embed_dim,max_seq_len,num_heads,num_layers,proj_dim,dropout=0.0):
        super().__init__()
        self.embedding=nn.Parameter(vocab_size,embed_dim)
        self.positional_embedding=PositionalEmbedding(embed_dim,max_seq_len)
        self.encoder=nn.ModuleList([
            TransformerEncoder(embed_dim,num_heads,dropout) for _ in range(num_layers)
        ])
        self.projection=nn.Parameter(torch.randn(embed_dim,proj_dim))
        
    def forward(self,text,mask=None):
        x=self.embedding(text)
        x=self.positional_embedding(x)
        
        for encoder_layer in self.encoder:
            x=encoder_layer(x,mask)
            
        x = x[torch.arange(text.shape[0]), torch.sub(torch.sum(mask[:,0],dim=1),1)]
        
        x=torch.matmul(x,self.projection)
        x = x / torch.norm(x, dim=-1, keepdim=True)
        return x
        
class ImageEncoder(nn.Module):
    def __init__(self,embed_dim,image_size,patch_size,num_channels,num_heads,num_layers,proj_dim,dropout=0.0):
        super().__init__()
        assert image_size[0] % patch_size[0] == 0 and image_size[1] % patch_size[1] == 0, "img_size dimensions must be divisible by patch_size dimensions"
        
        self.num_patches=(image_size[0] * image_size[1]) // (patch_size[0] * patch_size[1])
        self.max_seq_len=self.num_patches+1
        self.conv=nn.Conv2d(num_channels,embed_dim,kernel_size=patch_size,stride=patch_size)
        self.cls_token=nn.Parameter(torch.randn(1,1,embed_dim))
        self.positional_embedding=PositionalEmbedding(embed_dim,self.max_seq_len)
        self.encoder=nn.ModuleList([
            TransformerEncoder(embed_dim,num_heads,dropout) for _ in range(num_layers)
        ])
        self.projection=nn.Parameter(torch.randn(embed_dim,proj_dim))
    
    def forward(self,x):
        x=self.conv(x)
        x=x.flatten(2).transpose(1,2)
        x = torch.cat((self.cls_token.expand(x.size()[0], -1, -1),x), dim=1)
        x = self.positional_embedding(x)
        
        for encoder_layer in self.encoder:
            x=encoder_layer(x)
            
        x = x[:, 0, :]
        x=torch.matmul(x,self.projection)
        x = x / torch.norm(x, dim=-1, keepdim=True)
        return x
            
class CLIP(nn.Module):
    def __init__(self,proj_dim,image_encoder_embed_dim,image_size,patch_size,num_channels,image_encoder_num_heads,image_encoder_num_layers,
                 vocab_size,text_encoder_embed_dim,max_seq_len,text_encoder_num_heads,text_encoder_num_layers):
        self.image_encoder=ImageEncoder(image_encoder_embed_dim,image_size,patch_size,num_channels,image_encoder_num_heads,image_encoder_num_layers,proj_dim,0.1)
        self.text_encoder=TextEncoder(vocab_size,text_encoder_embed_dim,max_seq_len,text_encoder_num_heads,text_encoder_num_layers,proj_dim,0.1)
        self.temperature = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
    
    def forward(self,image,text,mask=None):
        I_e = self.image_encoder(image)
        T_e = self.text_encoder(text, mask=mask)
        
        logits = (I_e @ T_e.transpose(-2,-1)) * torch.exp(self.temperature)
        labels = torch.arange(logits.shape[0]).to(self.device)

        loss_i = nn.functional.cross_entropy(logits.transpose(-2,-1), labels)
        loss_t = nn.functional.cross_entropy(logits, labels)

        loss = (loss_i + loss_t) / 2
        return loss
        