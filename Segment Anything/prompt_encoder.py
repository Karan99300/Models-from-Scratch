import numpy as np
import torch
import torch.nn as nn
from utils import LayerNorm2d

class PositionEmbeddingRandom(nn.Module):
    def __init__(self, num_pos_feats=64, scale=None):
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer("positional_encoding_gaussian_matrix", scale * torch.randn((2, num_pos_feats)))
        
    def pe_encoding(self, coords):
        coords = 2 * coords - 1
        coords = coords * self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)
    
    def forward(self, size): 
        h, w = size
        device = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w

        pe = self.pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)
    
    def forward_with_coords(self, coords, size):
        coords[:, :, 0] = coords[:, :, 0] / size[1]
        coords[:, :, 1] = coords[:, :, 1] / size[0]
        return self.pe_encoding(coords.to(torch.float))
    
class PromptEncoder(nn.Module):
    def __init__(self, embed_dim, image_embedding_size, input_image_size, mask_in_chans, act=nn.GELU):
        super().__init__()
        self.embed_dim = embed_dim
        self.image_embedding_size = image_embedding_size
        self.input_image_size = input_image_size
        
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)
        self.num_point_embeddings: int = 4  # pos/neg point + 2 box corners
        point_embeddings = [nn.Embedding(1, embed_dim) for i in range(self.num_point_embeddings)]
        self.point_embeddings = nn.ModuleList(point_embeddings)
        self.not_a_point_embed = nn.Embedding(1, embed_dim)
        
        self.mask_input_size = (4 * image_embedding_size[0], 4 * image_embedding_size[1])
        self.mask_downscaling = nn.Sequential(
            nn.Conv2d(1, mask_in_chans // 4, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_chans // 4),
            act(),
            nn.Conv2d(mask_in_chans // 4, mask_in_chans, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_chans),
            act(),
            nn.Conv2d(mask_in_chans, embed_dim, kernel_size=1),
        )
        self.no_mask_embed = nn.Embedding(1, embed_dim)
        
    def get_dense_pe(self):
        return self.pe_layer(self.image_embedding_size).unsqueeze(0)
    
    def embed_points(self, points, labels, pad):
        points = points + 0.5 # Shift to center of pixel
        if pad:
            padding_point = torch.zeros((points.shape[0], 1, 2), device=points.device)
            padding_label = -torch.ones((labels.shape[0], 1), device=labels.device)
            points = torch.cat([points, padding_point], dim=1)
            labels = torch.cat([labels, padding_label], dim=1)
        point_embedding = self.pe_layer.forward_with_coords(points, self.input_image_size)
        point_embedding[labels == -1] = 0.0
        point_embedding[labels == -1] += self.not_a_point_embed.weight
        point_embedding[labels == 0] += self.point_embeddings[0].weight
        point_embedding[labels == 1] += self.point_embeddings[1].weight
        return point_embedding
    
    def embed_boxes(self, boxes):
        boxes = boxes + 0.5  # Shift to center of pixel
        coords = boxes.reshape(-1, 2, 2)
        corner_embedding = self.pe_layer.forward_with_coords(coords, self.input_image_size)
        corner_embedding[:, 0, :] += self.point_embeddings[2].weight
        corner_embedding[:, 1, :] += self.point_embeddings[3].weight
        return corner_embedding
    
    def embed_masks(self, masks):
        return self.mask_downscaling(masks)
    
    def get_batch_size(self, points=None, boxes=None, masks=None):
        if points is not None:
            return points[0].shape[0]
        elif boxes is not None:
            return boxes.shape[0]
        elif masks is not None:
            return masks.shape[0]
        else:
            return 1

    def get_device(self):
        return self.point_embeddings[0].weight.device
    
    def forward(self, points=None, boxes=None, masks=None):
        B = self.get_batch_size(points, boxes, masks)
        sparse_embeddings = torch.empty((B, 0, self.embed_dim), device=self.get_device())
        
        if points is not None:
            coords, labels = points
            point_embeddings = self.embed_points(coords, labels, pad=(boxes is None))
            sparse_embeddings = torch.cat([sparse_embeddings, point_embeddings], dim=1)
            
        if boxes is not None:
            box_embeddings = self.embed_boxes(boxes)
            sparse_embeddings = torch.cat([sparse_embeddings, box_embeddings], dim=1)
            
        if masks is not None:
            dense_embeddings = self.embed_masks(masks)
        else: 
            dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(B, -1, self.image_embedding_size[0], self.image_embedding_size[1])
            
        return sparse_embeddings, dense_embeddings