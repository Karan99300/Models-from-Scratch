import torch
import torch.nn as nn
import torch.nn.functional as F 
from attention import SelfAttention

class CLIP_Embedding(nn.Module):
    def __init__(self, vocab_dim, embed_dim, n_tokens):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_dim, embed_dim)
        self.position_embedding = nn.Parameter(torch.zeros(n_tokens, embed_dim))
    
    def forward(self, tokens):
        x = self.token_embedding(tokens)
        x += self.position_embedding
        return x
    
class CLIP_Layer(nn.Module):
    def __init__(self, num_heads, embed_dim):
        super().__init__()
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.attention = SelfAttention(num_heads, embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, 4*embed_dim),
            nn.GELU(),
            nn.Linear(4*embed_dim, embed_dim)
        )
    
    def forward(self, x):
        residual = x
        x = self.layernorm1(x)
        x = self.attention(x, causal_mask=True)
        x += residual
        residual = x
        x = self.layernorm2(x)
        x = self.ff(x)
        x += residual
        return x

class CLIP_Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = CLIP_Embedding(49408, 768, 77)
        self.layers = nn.Module([
            CLIP_Layer(12, 768) for i in range(12)
        ])
        self.layernorm = nn.LayerNorm(768)
    
    def forward(self, tokens):
        tokens = tokens.type(torch.long) 
        state = self.embedding(tokens)
        for layer in self.layers:
            state = layer(state)
        output = self.layernorm(state)
        return output