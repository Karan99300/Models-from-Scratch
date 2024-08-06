import torch
import torch.nn as nn
import torch.nn.functional as F 
from timm.models.layers import trunc_normal_,DropPath

class ConvNextBlock(nn.Module):
    def __init__(self,input_dim,drop_path=0.0,layer_scale_initial_value=1e-6):
        super().__init__()
        self.depth_wise_conv=nn.Conv2d(input_dim,input_dim,kernel_size=7,padding=3,groups=input_dim)
        self.layer_norm=nn.LayerNorm(input_dim)
        self.point_wise_conv1=nn.Linear(input_dim,4*input_dim)
        self.gelu=nn.GELU()
        self.point_wise_conv2=nn.Linear(4*input_dim,input_dim)
        self.gamma=nn.Parameter(layer_scale_initial_value*torch.ones((input_dim)),requires_grad=True) if layer_scale_initial_value>0 else None
        self.drop_path=DropPath(drop_path) if drop_path>0 else nn.Identity()
    
    def forward(self,x):
        skip=x
        x=self.depth_wise_conv(x)
        x=x.permute(0,2,3,1)
        x=self.layer_norm(x)
        x=self.point_wise_conv1(x)
        x=self.gelu(x)
        x=self.point_wise_conv2(x)
        if self.gamma is not None:
            x=x*self.gamma
        x=x.permute(0,3,1,2)
        x=skip+self.drop_path(x)
        return x

class ConvNext(nn.Module):
    def __init__(self,in_channels=3,num_classes=10,depths=[3,3,9,3],dims=[96,192,384,768],drop_path=0.0,layer_scale_initial_value=1e-6,head_initial_value=1):
        super().__init__()
        self.downsampling_layers=nn.ModuleList()
        stem=nn.Sequential(
            nn.Conv2d(in_channels,dims[0],kernel_size=4,stride=4),
            nn.LayerNorm(dims[0])
        )
        self.downsampling_layers.append(stem)
        
        for i in range(3):
            downsampling_layer=nn.Sequential(
                nn.LayerNorm(dims[i]),
                nn.Conv2d(dims[i],dims[i+1],kernel_size=2,stride=2)
            )
            self.downsampling_layers.append(downsampling_layer)
            
        self.stages=nn.ModuleList()
        drop_path_rates=[x.item() for x in torch.linspace(0, drop_path, sum(depths))] 
        current=0
        
        for i in range(4):
            stage=nn.Sequential(
                *[ConvNextBlock(dims[i],drop_path=drop_path_rates[current+j],layer_scale_initial_value=layer_scale_initial_value)
                  for j in range(depths[i])]
            )
            self.stages.append(stage)
            current+=depths[i]
        
        self.layer_norm=nn.LayerNorm(dims[-1])
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_initial_value)
        self.head.bias.data.mul_(head_initial_value)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)
    
    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([-2, -1]))
    
    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x