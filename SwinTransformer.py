import torch 
import torch.nn as nn
import math
from einops import rearrange
import numpy as np

class SwinEmbeddings(nn.Module):
    def __init__(self, patch_size=4, embed_size=96):
        super().__init__()
        self.linear_embedding = nn.Conv2d(3,embed_size,kernel_size=patch_size,stride=patch_size)
        
    def forward(self, x):
        x = self.linear_embedding(x)
        x = rearrange(x,'b c h w -> b (h w) c')
        return x
    
class PatchMerging(nn.Module):
    def __init__(self,embed_size):
        super().__init__()
        self.linear=nn.Linear(4*embed_size,2*embed_size)
        self.layer_norm=nn.LayerNorm(2*embed_size)
    
    def forward(self,x):
        height=width=int(math.sqrt(x.shape[1])/2)
        x=rearrange(x,'b (h s1 w s2) c -> b (h w) (s2 s1 c)',s1=2,s2=2,h=height,w=width)
        x=self.linear(x)
        x=self.layer_norm(x)
        return x #(b,H/2,W/2,2C)

class ShiftedWindowMSA(nn.Module):
    def __init__(self,embed_size,num_heads,window_size,shifted=True):
        super().__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.window_size = window_size
        self.shifted = shifted
        self.linear1 = nn.Linear(embed_size, 3*embed_size)
        self.linear2 = nn.Linear(embed_size, embed_size)

        self.pos_embeddings = nn.Parameter(torch.randn(window_size*2 - 1, window_size*2 - 1))
        self.indices = torch.tensor(np.array([[x, y] for x in range(window_size) for y in range(window_size)]))
        self.relative_indices = self.indices[None, :, :] - self.indices[:, None, :]
        self.relative_indices += self.window_size - 1
        
    def forward(self,x):
        h_dim=self.embed_size/self.num_heads
        height=width=int(math.sqrt(x.shape[1]))
        x=self.linear1(x)
        
        x=rearrange(x,'b (h w) (c k) -> b h w c k',h=height,w=width,k=3,c=self.embed_size)
        
        if self.shifted:
            x=torch.roll(x,(-self.window_size//2,-self.window_size//2),dims=(1,2))
        
        x=rearrange(x,'b (Wh w1) (Ww w2) (e H) k -> b H Wh Ww (w1 w2) e k',w1=self.window_size,w2=self.window_size,H=self.num_heads)
        
        Q,K,V=x.chunk(3,dim=6)
        Q, K, V = Q.squeeze(-1), K.squeeze(-1), V.squeeze(-1)
        window_attention_scores=torch.matmul(Q,K.transpose(4,5))
        window_attention_scores=window_attention_scores/math.sqrt(h_dim)
        
        rel_pos_embedding=self.pos_embeddings[self.relative_indices[:,:,0],self.relative_indices[:,:,1]]
        window_attention_scores+=rel_pos_embedding
        
        if self.shifted:
            row_mask=torch.zeros((self.window_size**2,self.window_size**2))
            row_mask[-self.window_size*(self.window_size//2):,0:-self.window_size*(self.window_size//2)]=float('-inf')
            row_mask[0:-self.window_size * (self.window_size//2), -self.window_size * (self.window_size//2):] = float('-inf')
            column_mask=rearrange(row_mask,'(r w1) (c w2) -> (w1 r) (w2 c)',w1=self.window_size,w2=self.window_size)
            window_attention_scores[:,:,-1,:]+=row_mask
            window_attention_scores[:,:,:,-1]+=column_mask
        
        window_attention_probs=nn.functional.softmax(window_attention_scores,dim=-1)
        window_attention_probs=torch.matmul(window_attention_probs,V)
        
        window_attention_probs=rearrange(window_attention_probs,'b H Wh Ww (w1 w2) e -> b (Wh w1) (Ww w2) (H e)',w1=self.window_size,w2=self.window_size,H=self.num_heads)
        window_attention_probs=rearrange(window_attention_probs,'b h w c -> b (h w) c')
        window_attention_probs=self.linear2(window_attention_probs)
        return window_attention_probs
 
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
    
class SwinTransformerBlock(nn.Module):
    def __init__(self,embed_size,num_heads,window_size=7):
        super().__init__()
        self.embed_size=embed_size
        self.num_heads=num_heads
        self.WMSA=ShiftedWindowMSA(embed_size,num_heads,window_size,shifted=False)
        self.SWMSA=ShiftedWindowMSA(embed_size,num_heads,window_size,shifted=True)
        self.MLP=MLP(self.embed_size)
        self.layer_norm=nn.LayerNorm(self.embed_size)
    
    def forward(self,x):
        x=x+self.WMSA(self.layer_norm(x))
        x=x+self.MLP(self.layer_norm(x))
        
        x=x+self.SWMSA(self.layer_norm(x))
        x=x+self.MLP(self.layer_norm(x))
        
        return x
    
class SwinTransformer(nn.Module):
    def __init__(self,num_classes):
        super().__init__()
        self.Embedding=SwinEmbeddings()
        self.PatchMerging=nn.ModuleList()
        embed_size=96
        
        for i in range(3):
            self.PatchMerging.append(PatchMerging(embed_size))
            embed_size *= 2
               
        self.stage1=SwinTransformerBlock(96,3)
        self.stage2 = SwinTransformerBlock(192, 6)
        self.stage3 = nn.ModuleList([SwinTransformerBlock(384, 12),
                                     SwinTransformerBlock(384, 12),
                                     SwinTransformerBlock(384, 12) 
                                    ])
        self.stage4 = SwinTransformerBlock(768, 24)
        
        self.avgpool1d = nn.AdaptiveAvgPool1d(output_size = 1)
        self.avg_pool_layer = nn.AvgPool1d(kernel_size=49)
        
        self.layer = nn.Linear(768, num_classes)
        
    def forward(self, x):
        x = self.Embedding(x)
        x = self.stage1(x)
        x = self.PatchMerging[0](x)
        x = self.stage2(x)
        x = self.PatchMerging[1](x)
        for stage in self.stage3:
            x = stage(x)
        x = self.PatchMerging[2](x)
        x = self.stage4(x)
        x = self.layer(self.avgpool1d(x.transpose(1, 2)).squeeze(2))
        return x
    
if __name__=="__main__":
    x=torch.randn(1,3,224,224)
    model=SwinTransformer(5)
    out=model(x)
    print(out.shape)