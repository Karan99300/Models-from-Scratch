import torch 
import torch.nn as nn 

class MLPBlock(nn.Module):
    def __init__(self, embedding_dim, mlp_dim, act):
        super().__init__()
        self.linear1 = nn.Linear(embedding_dim, mlp_dim)
        self.linear2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act
        
    def forward(self, x):
        return self.linear2(self.act(self.linear1(x)))
    
class LayerNorm2d(nn.Module):
    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps
        
    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x