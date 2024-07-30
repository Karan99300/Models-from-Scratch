import torch.nn as nn  
import torch
import einops
from einops import rearrange
import math
from torchvision.ops import box_convert
from torchvision.models import resnet50, ResNet50_Weights
from scipy.optimize import linear_sum_assignment
from einops.layers.torch import Rearrange

class BackBone(nn.Module):
    def freeze_bn2d_params(self):
        for module in self.cnn.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.weight.requires_grad = False
                module.bias.requires_grad = False
    
    def __init__(self):
        super().__init__()
        self.cnn=resnet50(weight=ResNet50_Weights.DEFAULT)
        self.freeze_bn2d_params()
        self.cnn.avgpool=nn.Identity()
        self.cnn.fc=nn.Identity()
    
    def forward(self,x):
        return self.cnn(x)

class PositionalEncoding(nn.Module):
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
            self.pe_mat.to(x.device), pattern="l d -> b l d", b=x.size(0),
        )[:, : x.size(1), :]

class Attention(nn.Module):
    def __init__(self,embed_dim,num_heads,dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
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
        attention_probs=nn.functional.softmax(attention_scores,dim=-1)
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

class EncoderLayer(nn.Module):
    def __init__(self,embed_dim,num_heads,latent_dim,dropout=0.0):
        super().__init__()
        self.spatial_positional_encoding=PositionalEncoding(embed_dim)
        self.self_attention=Attention(embed_dim,num_heads,dropout)
        self.feedforward=FeedForward(embed_dim,latent_dim,dropout)
        
        self.self_attention_residual_connection=ResidualConnection(
            lambda x: self.self_attention(
                q=self.spatial_positional_encoding(x),k=self.spatial_positional_encoding(x),v=x
            )[0],
            embed_dim,dropout
        )
        
        self.feedforward_residual_connection=ResidualConnection(
            self.feedforward,embed_dim,dropout
        )
        
    def forward(self,x):
        x=self.self_attention_residual_connection(skip=x,x=x)
        x=self.feedforward_residual_connection(skip=x,x=x)
        return x
    
class Encoder(nn.Module):
    def __init__(self,embed_dim,num_heads,num_layers,latent_dim=None,dropout=0.0): 
        super().__init__()
        self.latent_dim=latent_dim if latent_dim is not None else embed_dim*4
        self.encoder=nn.ModuleList([
            EncoderLayer(embed_dim,num_heads,self.latent_dim,dropout)
            for _ in range(num_layers)
        ])
        
    def forward(self,x):
        for encoder_layer in self.encoder:
            x=encoder_layer(x)
        return x

class DecoderLayer(nn.Module):
    def __init__(self,embed_dim,num_heads,latent_dim,dropout=0.0):
        super().__init__()
        self.spatial_positional_encoding=PositionalEncoding(embed_dim)
        self.self_attention=Attention(embed_dim,num_heads,dropout)
        self.encoder_decoder_attention=Attention(embed_dim,num_heads,dropout)
        self.feedforward=FeedForward(embed_dim,latent_dim,dropout)
        
        self.self_attention_residual_connection=ResidualConnection(
            lambda x,output_positional_encoding : self.self_attention(
                q=x+output_positional_encoding,k=x+output_positional_encoding,v=x
            )[0],
            embed_dim,dropout
        )
        
        self.encoder_decoder_attention_residual_connection=ResidualConnection(
            lambda x,output_positional_encoding,encoder_memory : self.encoder_decoder_attention(
                q=x+output_positional_encoding,k=self.spatial_positional_encoding(encoder_memory),v=encoder_memory
            )[0],
            embed_dim,dropout
        )
        
        self.feedforward_residual_connection=ResidualConnection(
            self.feedforward,embed_dim,dropout
        )
    
    def forward(self,q,encoder_memory,output_positional_encoding):
        x=self.self_attention_residual_connection(skip=q,x=q,output_positional_encoding=output_positional_encoding)
        x=self.encoder_decoder_attention_residual_connection(skip=x,x=x,output_positional_encoding=output_positional_encoding,encoder_memory=encoder_memory)
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
        
    def forward(self,q,output_positional_encoding,encoder_memory):
        for decoder_layer in self.decoder:
            q=decoder_layer(q,output_positional_encoding,encoder_memory)
        return q

class Transformer(nn.Module):
    def __init__(self,embed_dim,num_encoder_heads=8,num_encoder_layers=6,num_decoder_heads=8,num_decoder_layers=6,dropout=0.0):
        super().__init__()
        self.encoder=Encoder(embed_dim,num_encoder_heads,num_encoder_layers,dropout)
        self.decoder=Decoder(embed_dim,num_decoder_heads,num_decoder_layers,dropout)
        
    def forward(self,image_features,q,output_positional_encoding):
        encoder_memory=self.encoder(image_features)
        return self.decoder(q,output_positional_encoding,encoder_memory)

class GeneralizedIOU(nn.Module):
    @staticmethod
    def get_area(coordinates):
        return torch.clip(coordinates[...,2]-coordinates[...,0],min=0)*torch.clip(coordinates[...,3]-coordinates[...,1],min=0)
    
    @staticmethod
    def get_intersection_area(coordinates1,coordinates2):
        left=torch.maximum(coordinates1[...,0][...,None],coordinates2[...,0][...,None])
        top=torch.maximum(coordinates1[...,1][...,None],coordinates2[...,1][...,None]) 
        right=torch.minimum(coordinates1[...,2][...,None],coordinates2[...,2][...,None])
        bottom=torch.minimum(coordinates1[...,3][...,None],coordinates2[...,3][...,None])
        return torch.clip(right-left,min=0)*torch.clip(bottom-top,min=0)
    
    @staticmethod
    def get_smallest_enclosing_area(coordinates1,coordinates2):
        left=torch.minimum(coordinates1[...,0][...,None],coordinates2[...,0][...,None])
        top=torch.minimum(coordinates1[...,1][...,None],coordinates2[...,1][...,None]) 
        right=torch.maximum(coordinates1[...,2][...,None],coordinates2[...,2][...,None])
        bottom=torch.maximum(coordinates1[...,3][...,None],coordinates2[...,3][...,None])
        return torch.clip(right-left,min=0)*torch.clip(bottom-top,min=0)
    
    def get_giou(self,coordinates1,coordinates2):
        area1=self.get_area(coordinates1)
        area2=self.get_area(coordinates2)
        intersection_area=self.get_intersection_area(coordinates1,coordinates2)
        union_area=area1[:,None]+area2[None,:]-intersection_area
        smallest_enclosing_area=self.get_smallest_enclosing_area(coordinates1,coordinates2)
        iou=torch.where(union_area==0,0,intersection_area/union_area)
        return torch.where(smallest_enclosing_area==0,-1,iou-((smallest_enclosing_area-union_area)/smallest_enclosing_area))
    
    def __call__(self,coordinates1,coordinates2):
        giou=self.get_giou(coordinates1,coordinates2)
        return 1-giou

class DETR(nn.Module):
    def __init__(self,N=100,num_classes=80,embed_dim=512,num_encoder_heads=8,num_decoder_heads=8,num_encoder_layers=6,num_decoder_layers=6,image_size=512,stride=32,latent_dim=2048):
        super().__init__()
        self.N=N
        self.num_classes=num_classes
        self.image_size=image_size
        self.stride=stride
        self.latent_dim=latent_dim
        
        self.backbone=BackBone()
        self.giou_loss=GeneralizedIOU()
        self.to_sequence=nn.Sequential(
            nn.Conv2d(latent_dim,embed_dim,1,1,0),
            Rearrange('b l h w -> b (h w) l')
        )
        self.transformer=Transformer(embed_dim,num_encoder_heads,num_encoder_layers,num_decoder_heads,num_decoder_layers,0.1)
        self.q=torch.zeros((self.N,embed_dim))
        self.object_q=nn.Embedding(self.N,embed_dim).weight
        self.bbox_feedforward=nn.Sequential(
            nn.Linear(embed_dim,embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim,embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 4),
            nn.Sigmoid(),
        )
        self.class_feedforward=nn.Sequential(
            nn.Linear(embed_dim,num_classes+1),
            nn.Softmax(dim=-1)
        )
    
    @staticmethod
    def change_box_format(norm_xywh):
        return box_convert(norm_xywh,in_fmt='cxcywh',out_fmt='xyxy')
    
    def forward(self,image):
        x=self.backbone(image)
        x=x.view(x.size(0),self.latent_dim,self.image_size//self.stride,self.image_size//self.stride)
        x=self.to_sequence(x)
        x=self.transformer(x,
                           einops.repeat(self.q.to(image.device),pattern='n d -> b n d',b=image.size(0)),
                           einops.repeat(self.object_q.to(image.device),pattern="n d -> b n d",b=image.size(0))
                        )
        output_norm_xywh=self.bbox_feedforward(x)
        output_class=self.class_feedforward(x)
        
        output_norm_xyxy=self.change_box_format(output_norm_xywh)
        return output_norm_xyxy,output_class
    
    def bipartite_matching(self,pred_norm_bbox,pred_prob,gt_bbox,label,l1_weight,iou_weight):
        iou_loss=self.giou_loss(pred_norm_bbox,gt_bbox)       
        l1_loss = torch.abs(pred_norm_bbox[:,None,:]-gt_bbox[None,:,:]).sum(dim=-1) 
        box_loss=l1_weight*l1_loss+iou_weight*iou_loss
        label_prob=pred_prob[:,label]
        match_loss=box_loss-label_prob
        pred_indices,gt_indices=linear_sum_assignment(match_loss.detach().cpu().numpy())
        return pred_indices,gt_indices,box_loss,label_prob
    
    def get_loss(self,image,gt_norm_bboxs,labels,no_object_weight=0.1,l1_weight=5,iou_weight=2):
        output_norm_xyxy,output_class_prob=self(image)
        sum_losses = torch.zeros((1,), dtype=torch.float32, device=image.device)
        for pred_norm_bbox,pred_prob,gt_bbox,label in zip(output_norm_xyxy,output_class_prob,gt_norm_bboxs,labels):
            (pred_indices,gt_indices,box_loss,label_prob,)=self.bipartite_matching(pred_norm_bbox,pred_prob,gt_bbox,label,l1_weight,iou_weight)
            class_loss = -torch.log(label_prob[pred_indices, gt_indices]).sum()
            no_object_mask = ~torch.isin(torch.arange(self.N),torch.from_numpy(pred_indices),)
            no_object_class_loss = -torch.log(pred_prob[no_object_mask, self.num_classes],).sum()
            hungrian_cls_loss = class_loss + no_object_weight * no_object_class_loss
            hungarian_box_loss = box_loss[pred_indices, gt_indices].sum()
            loss = hungrian_cls_loss + hungarian_box_loss
            sum_losses += loss
        num_objs = sum([label.size(0) for label in labels])
        if num_objs != 0:
            sum_losses /= num_objs
        return sum_losses