import torch
import torch.nn as nn
import torch.nn.functional as F 
import math 
from einops import rearrange

class SelfAttention(nn.Module):
    def __init__(self, num_heads, embed_dim, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        self.qkv_proj = nn.Linear(embed_dim, 3*embed_dim, bias=in_proj_bias)
        self.output_proj = nn.Linear(embed_dim, embed_dim, bias=out_proj_bias)
        self.num_heads = num_heads
        self.head_dim = embed_dim//num_heads
    
    def forward(self, x, causal_mask=False):
        b, s, _ = x.shape
        qkv = self.qkv_proj(x)
        qkv = rearrange(qkv, 'b s (three h d) -> b h s d three', three=3, h=self.num_heads)
        q, k, v = qkv[..., 0], qkv[..., 1], qkv[..., 2]
        
        attention_scores = torch.matmul(q, k.transpose(-1,-2))
        if causal_mask:
            mask = torch.ones_like(attention_scores, dtype=torch.bool).triu(1)
            attention_scores.masked_fill_(mask, -torch.inf)
        
        attention_scores = attention_scores / math.sqrt(self.head_dim)
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_output = torch.matmul(attention_probs, v)
        attention_output = rearrange(attention_output, 'b h s d -> b s (h d)')
        return self.output_proj(attention_output)
    