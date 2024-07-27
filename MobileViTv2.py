import torch 
import torch.nn as nn
from einops import rearrange
import math

def conv_2d(input_dim,output_dim,kernel_size=3,stride=1,padding=0,groups=1,bias=False,norm=True,act=True):
    conv=nn.Sequential()
    conv.add_module('conv', nn.Conv2d(input_dim,output_dim,kernel_size,stride,padding,bias=bias,groups=groups))
    if norm:
        conv.add_module('BatchNorm2d',nn.BatchNorm2d(output_dim))
    if act:
        conv.add_module('Activation',nn.SiLU())
    return conv

class MobileNetV2Block(nn.Module):
    def __init__(self,input_dim,output_dim,stride,expand_ratio):
        super().__init__()
        self.stride=stride
        hidden_dim=int(round(input_dim*expand_ratio))
        self.block=nn.Sequential()
        if expand_ratio != 1:
            self.block.add_module('exp_1x1', conv_2d(input_dim, hidden_dim, kernel_size=1, stride=1, padding=0))
        self.block.add_module('conv_3x3', conv_2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim))
        self.block.add_module('red_1x1', conv_2d(hidden_dim, output_dim, kernel_size=1, stride=1, padding=0, act=False))
        self.use_res_connect = self.stride == 1 and input_dim == output_dim
    
    def forward(self,x):
        if self.use_res_connect:
            return x + self.block(x)
        else:
            return self.block(x)  

class Attention(nn.Module):
    def __init__(self, embed_dim, dropout=0):
        super().__init__()
        self.ikv_proj=conv_2d(embed_dim,1+2*embed_dim,kernel_size=1,bias=True,norm=False,act=False)
        self.dropout=nn.Dropout(dropout)
        self.output_projection=conv_2d(embed_dim,embed_dim,kernel_size=1,bias=True,norm=False,act=False)
        self.embed_dim=embed_dim
        self.relu=nn.ReLU(embed_dim)
        
    def forward(self,x):
        ikv=self.ikv_proj(x)
        i,k,v=torch.split(ikv,split_size_or_sections=[1,self.embed_dim,self.embed_dim],dim=1)
        context_scores=nn.functional.softmax(i,dim=-1)
        context_scores=self.dropout(context_scores)
        context_vector=k*context_scores
        context_vector=torch.sum(context_vector,dim=-1,keepdim=True)
        output=self.relu(v)*context_vector.expand_as(v)
        output=self.output_projection(output)
        return output
        
                    
class Transformer(nn.Module):
    def __init__(self,embed_dim,latent_dim,dropout=0,attention_dropout=0):
        super().__init__()
        
        self.mha=nn.Sequential(
            nn.GroupNorm(num_channels=embed_dim, eps=1e-5, affine=True, num_groups=1),
            Attention(embed_dim,attention_dropout),
            nn.Dropout(dropout)
        )
        
        self.ff=nn.Sequential(
            nn.GroupNorm(num_channels=embed_dim, eps=1e-5, affine=True, num_groups=1),
            conv_2d(embed_dim, latent_dim, kernel_size=1, stride=1, bias=True, norm=False, act=True),
            nn.Dropout(dropout),
            conv_2d(latent_dim, embed_dim, kernel_size=1, stride=1, bias=True, norm=False, act=True),
            nn.Dropout(dropout)
        )
        
    def forward(self,x):
        x=self.mha(x)+x
        x=self.ff(x)+x
        return x
        
class MobileViTBlockv2(nn.Module):
    def __init__(self,input_dim,embed_dim,ff_multiplier,num_attention_blocks,patch_size):
        super().__init__()
        self.patch_h,self.patch_w=patch_size
        self.patch_area=int(self.patch_h*self.patch_w)
        
        self.local_rep = nn.Sequential()
        self.local_rep.add_module('conv_3x3', conv_2d(input_dim, input_dim, kernel_size=3, stride=1, padding=1, groups=input_dim))
        self.local_rep.add_module('conv_1x1', conv_2d(input_dim, embed_dim, kernel_size=1, stride=1, norm=False, act=False))

        self.global_rep = nn.Sequential()
        ff_dims=[int((ff_multiplier*embed_dim)//16*16)]*num_attention_blocks
        for i in range(num_attention_blocks):
            ff_dim=ff_dims[i]
            self.global_rep.add_module(f'Transformer{i}',Transformer(embed_dim,ff_dim))
        self.global_rep.add_module('LayerNorm',nn.GroupNorm(num_channels=embed_dim, eps=1e-5, affine=True, num_groups=1))
        
        self.conv_proj=conv_2d(embed_dim,input_dim,kernel_size=1,stride=1,padding=0,act=False)
        
    def unfolding_pytorch(self, feature_map):
        batch_size, in_channels, h, w = feature_map.shape
        patches = nn.functional.unfold(
            feature_map,
            kernel_size=(self.patch_h, self.patch_w),
            stride=(self.patch_h, self.patch_w),
        )
        patches = patches.reshape(
            batch_size, in_channels, self.patch_h * self.patch_w, -1
        )
        return patches, (h, w)

    def folding_pytorch(self, patches, output_size):
        batch_size, in_dim, patch_size, n_patches = patches.shape
        patches = patches.reshape(batch_size, in_dim * patch_size, n_patches)
        feature_map = nn.functional.fold(
            patches,
            output_size=output_size,
            kernel_size=(self.patch_h, self.patch_w),
            stride=(self.patch_h, self.patch_w),
        )
        return feature_map
                
    def forward(self, x):
        x = self.local_rep(x)
        x, output_size = self.unfolding_pytorch(x)
        x = self.global_rep(x)
        x = self.folding_pytorch(patches=x, output_size=output_size)
        x = self.conv_proj(x)
        return x
    
class MobileViT(nn.Module):
    def __init__(self,image_size,width_multiplier,num_classes,patch_size=(2,2)):
        super().__init__()
        h,w=image_size
        self.patch_h,self.patch_w=patch_size
        assert h % self.patch_h == 0 and w % self.patch_w == 0 
        assert width_multiplier in [0.5, 0.75, 1, 1.25, 1.5, 1.75, 2]
        
        mv2_exp_mult = 2
        ff_multiplier = 2
        
        channels = []
        channels.append(int(max(16, min(64, 32 * width_multiplier))))
        channels.append(int(64 * width_multiplier))
        channels.append(int(128 * width_multiplier))
        channels.append(int(256 * width_multiplier))
        channels.append(int(384 * width_multiplier))
        channels.append(int(512 * width_multiplier))
        attention_dim = []
        attention_dim.append(int(128 * width_multiplier))
        attention_dim.append(int(192 * width_multiplier))
        attention_dim.append(int(256 * width_multiplier))

        
        self.conv_0 = conv_2d(3, channels[0], kernel_size=3, stride=2)

        self.layer_1 = nn.Sequential(
            MobileNetV2Block(channels[0], channels[1], stride=1, expand_ratio=mv2_exp_mult)
        )
        self.layer_2 = nn.Sequential(
            MobileNetV2Block(channels[1], channels[2], stride=2, expand_ratio=mv2_exp_mult),
            MobileNetV2Block(channels[2], channels[2], stride=1, expand_ratio=mv2_exp_mult)
        )
        self.layer_3 = nn.Sequential(
            MobileNetV2Block(channels[2], channels[3], stride=2, expand_ratio=mv2_exp_mult),
            MobileViTBlockv2(channels[3], attention_dim[0], ff_multiplier, 2, patch_size=patch_size)
        )
        self.layer_4 = nn.Sequential(
            MobileNetV2Block(channels[3], channels[4], stride=2, expand_ratio=mv2_exp_mult),
            MobileViTBlockv2(channels[4], attention_dim[1], ff_multiplier, 4, patch_size=patch_size)
        )
        self.layer_5 = nn.Sequential(
            MobileNetV2Block(channels[4], channels[5], stride=2, expand_ratio=mv2_exp_mult),
            MobileViTBlockv2(channels[5], attention_dim[2], ff_multiplier, 3, patch_size=patch_size)
        )
        self.out=nn.Linear(channels[-1],num_classes,bias=True)
        
    def forward(self,x):
        x = self.conv_0(x)
        x = self.layer_1(x)
        x = self.layer_2(x) 
        x = self.layer_3(x)
        x = self.layer_4(x)
        x = self.layer_5(x)
        x = torch.mean(x, dim=[-2, -1])
        x = self.out(x)
        
        return x
        
if __name__=='__main__':
    batch_size = 1
    channels = 3  
    height = 256 
    width = 256  

    sample_input = torch.randn(batch_size, channels, height, width)
    num_classes = 1000  
    model = MobileViT(image_size=(height, width), width_multiplier=1,num_classes=num_classes)

    with torch.no_grad():
        output = model(sample_input)

    print(f"Input shape: {sample_input.shape}")
    print(f"Output shape: {output.shape}")