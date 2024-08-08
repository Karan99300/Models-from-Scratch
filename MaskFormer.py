import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
import torch.nn.functional as F
import einops
from einops import rearrange
import math

class BackBone(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        layers = list(backbone.children())
        
        self.layer0 = nn.Sequential(*layers[:4])  
        self.layer1 = layers[4]  
        self.layer2 = layers[5]  
        self.layer3 = layers[6]  
        self.layer4 = layers[7]  
    
    def forward(self, x):
        x0 = self.layer0(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        
        return x0, x1, x2, x3, x4
    
class ProjectionLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.group_norm = nn.GroupNorm(32, out_channels)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.group_norm(x)
        output = F.relu(x)
        return output

class FusionLayer(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.group_norm = nn.GroupNorm(32, in_channels)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.group_norm(x)
        output = F.relu(x)
        return output

class PixelDecoder(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()
        self.embed_dim = embed_dim
        
        self.proj4 = ProjectionLayer(2048, embed_dim)
        self.proj3 = ProjectionLayer(1024, embed_dim)
        self.proj2 = ProjectionLayer(512, embed_dim)
        self.proj1 = ProjectionLayer(256, embed_dim)
        self.proj0 = ProjectionLayer(64, embed_dim)
        
        self.fuse4 = FusionLayer(embed_dim)
        self.fuse3 = FusionLayer(embed_dim)
        self.fuse2 = FusionLayer(embed_dim)
        self.fuse1 = FusionLayer(embed_dim)
    
    def forward(self, x0, x1, x2, x3, x4):
        x = self.proj4(x4)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x3 = self.proj3(x3)
        x = x + x3
        x = self.fuse4(x)
        
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x2 = self.proj2(x2)
        x = x + x2
        x = self.fuse3(x)
        
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x1 = self.proj1(x1)
        x = x + x1
        x = self.fuse2(x)
        
        x0 = self.proj0(x0)
        x = x + x0
        x = self.fuse1(x)
        return x
    
class PixelLevelModule(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()
        self.backbone = BackBone()
        self.pixel_decoder = PixelDecoder(embed_dim)
        self.final_conv = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
    
    def forward(self, x):
        x0, x1, x2, x3, x4 = self.backbone(x)
        output = self.pixel_decoder(x0, x1, x2, x3, x4)
        output = self.final_conv(output)
        output = F.interpolate(output, scale_factor=4, mode='bilinear', align_corners=True)
        return x4, output
    
class PositionalEncoding(nn.Module):
    def __init__(self,embed_dim,max_len=2**12):
        super().__init__()
        pos=torch.arange(max_len).unsqueeze(1)
        i=torch.arange(embed_dim//2).unsqueeze(0)
        angle=pos/(10000*(2*i/embed_dim))
        self.pe_matrix=torch.zeros(size=(max_len,embed_dim))
        self.pe_matrix[:,0::2]=torch.sin(angle)
        self.pe_matrix[:,1::2]=torch.cos(angle)
        self.register_buffer("pe_matrix_maskformer",self.pe_matrix)
    
    def forward(self,x):
        return x+einops.repeat(
            self.pe_matrix.to(x.device), pattern="l d -> b l d", b=x.size(0),
        )[:, : x.size(1), :]
        
class Attention(nn.Module):
    def __init__(self,embed_dim,num_heads,dropout=0.0):
        super().__init__()
        self.embed_dim=embed_dim
        self.num_heads=num_heads
        self.head_dim=embed_dim//num_heads
        
        self.q_proj=nn.Linear(embed_dim,embed_dim,bias=False) 
        self.k_proj=nn.Linear(embed_dim,embed_dim,bias=False)   
        self.v_proj=nn.Linear(embed_dim,embed_dim,bias=False)     
        self.dropout=nn.Dropout(dropout)
        self.output_proj=nn.Linear(embed_dim,embed_dim,bias=False)
    
    def forward(self,q,k,v):
        q=self.q_proj(q)
        k=self.k_proj(k)
        v=self.v_proj(v)
        
        q=rearrange(q,'b i (h d) -> b h i d',h=self.num_heads)
        k=rearrange(k,'b j (h d) -> b h j d',h=self.num_heads)
        v=rearrange(v,'b j (h d) -> b h j d',h=self.num_heads)
        
        attention_scores=torch.matmul(q,k.transpose(-2,-1))
        attention_scores=attention_scores/math.sqrt(self.head_dim)
        attention_probs=F.softmax(attention_scores,dim=-1)
        attention_probs=self.dropout(attention_probs)
        attention_output=torch.matmul(attention_probs,v)
        attention_output=rearrange(attention_output,'b h i d -> b i (h d)')
        return self.output_proj(attention_output)

class FeedForward(nn.Module):
    def __init__(self,embed_dim,latent_dim,dropout=0.0):
        super().__init__()
        self.layers=nn.Sequential(
            nn.Linear(embed_dim,latent_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim,embed_dim)
        )
        
    def forward(self,x):
        return self.layers(x)

class ResidualConnection(nn.Module):
    def __init__(self,fn,embed_dim,dropout=0.0):
        super().__init__()
        self.fn=fn
        self.dropout=nn.Dropout(dropout)
        self.layer_norm=nn.LayerNorm(embed_dim)
    
    def forward(self,skip,**kwargs):
        x=self.fn(**kwargs)
        x=self.dropout(x)
        x=x+skip
        x=self.layer_norm(x)
        return x
    
class DecoderLayer(nn.Module):
    def __init__(self,embed_dim,num_heads,latent_dim,dropout=0.0):
        super().__init__()
        self.positional_encoding=PositionalEncoding(embed_dim)
        self.self_attention=Attention(embed_dim,num_heads,dropout)
        self.features_decoder_attention=Attention(embed_dim,num_heads,dropout)
        self.feedforward=FeedForward(embed_dim,latent_dim,dropout)
        
        self.self_attention_residual_connection=ResidualConnection(
            lambda x,output_positional_encoding : self.self_attention(
                q=x+output_positional_encoding,k=x+output_positional_encoding,v=x
            )[0],
            embed_dim,dropout
        )
        
        self.features_decoder_attention_residual_connection=ResidualConnection(
            lambda x,output_positional_encoding,image_features : self.features_decoder_attention(
                q=x+output_positional_encoding,k=self.positional_encoding(image_features),v=image_features
            )[0],
            embed_dim,dropout
        )
        
        self.feedforward_residual_connection=ResidualConnection(
            self.feedforward,embed_dim,dropout
        )
        
    def forward(self,q,output_positional_encoding,image_features):
        x=self.self_attention_residual_connection(skip=q,x=q,output_positional_encoding=output_positional_encoding)
        x=self.features_decoder_attention_residual_connection(skip=x,x=x,output_positional_encoding=output_positional_encoding,image_features=image_features)
        x=self.feedforward_residual_connection(skip=x,x=x)
        return x
    
class Decoder(nn.Module):
    def __init__(self,embed_dim,num_heads,num_layers,latent_dim=None,dropout=0.0):
        super().__init__()
        self.latent_dim=latent_dim if latent_dim is not None else embed_dim*4
        self.decoder=nn.ModuleList([
            DecoderLayer(embed_dim,num_heads,self.latent_dim,dropout)
            for _ in range(num_layers)
        ])
        
        self.feature_projection=nn.Conv2d(embed_dim,embed_dim,kernel_size=1)
        
    def forward(self,q,output_positional_encoding,image_features):
        projected_features=self.feature_projection(image_features)
        projected_features = projected_features.flatten(2).transpose(1, 2)
        for decoder_layer in self.decoder:
            q=decoder_layer(q,output_positional_encoding,image_features)
        return q
        
class Classifier(nn.Module):
    def __init__(self,embed_dim,num_classes):
        self.linear=nn.Linear(embed_dim,num_classes+1)
    
    def forward(self,x):
        logits=self.linear(x)
        probabilities=F.softmax(logits,dim=-1)
        return probabilities

class MLP(nn.Module):
    def __init__(self,input_dim,hidden_dim,output_dim):
        super().__init__()
        self.fc1=nn.Linear(input_dim,hidden_dim)
        self.fc2=nn.Linear(hidden_dim,output_dim)
        
    def forward(self,x):
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        return x

class MaskPredictor(nn.Module):
    def __init__(self, segment_embedding_dim, hidden_dim, mask_embedding_dim, num_masks):
        super().__init__()
        self.mlp = MLP(segment_embedding_dim, hidden_dim, mask_embedding_dim * num_masks)
        self.mask_embedding_dim = mask_embedding_dim
        self.num_masks = num_masks
    
    def forward(self,mask_embeddings,per_pixel_embeddings):
        b,_,_,_=per_pixel_embeddings.size()
        mask_embeddings=self.mlp(mask_embeddings)
        mask_embeddings=mask_embeddings.view(b,self.num_masks,self.mask_embedding_dim)
        
        mask_predictions = []
        for i in range(self.num_masks):
            mask_embedding = mask_embeddings[:, i, :]  
            mask_embedding = mask_embedding.unsqueeze(-1).unsqueeze(-1)  
            mask_prediction = torch.sum(mask_embedding *per_pixel_embeddings, dim=1)  
            mask_prediction = torch.sigmoid(mask_prediction)  
            mask_predictions.append(mask_prediction)
        
        mask_predictions = torch.stack(mask_predictions, dim=1)  
        
        return mask_predictions 
    
#For training bipartite matching