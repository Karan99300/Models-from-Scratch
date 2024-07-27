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
    def __init__(self, embed_dim, num_heads=4, head_dim=8, dropout=0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        
        self.qkv_proj = nn.Linear(embed_dim, 3 * num_heads * head_dim)
        self.dropout = nn.Dropout(dropout)
        self.output_projection = nn.Linear(num_heads * head_dim, embed_dim)
        
    def forward(self, x):
        b, s, _ = x.shape
        qkv = self.qkv_proj(x)
        qkv = rearrange(qkv, 'b s (three h d) -> b h s d three', three=3, h=self.num_heads)
        q, k, v = qkv[..., 0], qkv[..., 1], qkv[..., 2]
        
        attention_scores = torch.matmul(q, k.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.head_dim)
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        attention_output = torch.matmul(attention_probs, v)
        
        attention_output = rearrange(attention_output, 'b h s d -> b s (h d)')
        return self.output_projection(attention_output)
    
class Transformer(nn.Module):
    def __init__(self,embed_dim,latent_dim,num_heads,head_dim,dropout=0,attention_dropout=0):
        super().__init__()
        
        self.mha=nn.Sequential(
            nn.LayerNorm(embed_dim),
            Attention(embed_dim,num_heads,head_dim,attention_dropout),
            nn.Dropout(dropout)
        )
        
        self.ff=nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim,latent_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim,embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self,x):
        x=self.mha(x)+x
        x=self.ff(x)+x
        return x
        
class MobileViTBlock(nn.Module):
    def __init__(self,input_dim,embed_dim,ff_multiplier,num_heads,head_dim,num_attention_blocks,patch_size):
        super().__init__()
        self.patch_h,self.patch_w=patch_size
        self.patch_area=int(self.patch_h*self.patch_w)
        
        self.local_rep = nn.Sequential()
        self.local_rep.add_module('conv_3x3', conv_2d(input_dim, input_dim, kernel_size=3, stride=1, padding=1))
        self.local_rep.add_module('conv_1x1', conv_2d(input_dim, embed_dim, kernel_size=1, stride=1, norm=False, act=False))

        self.global_rep = nn.Sequential()
        ff_dims=[int((ff_multiplier*embed_dim)//16*16)]*num_attention_blocks
        for i in range(num_attention_blocks):
            ff_dim=ff_dims[i]
            self.global_rep.add_module(f'Transformer{i}',Transformer(embed_dim,ff_dim,num_heads,head_dim))
        self.global_rep.add_module('LayerNorm',nn.LayerNorm(embed_dim))
        
        self.conv_proj=conv_2d(embed_dim,input_dim,kernel_size=1,stride=1)
        self.fusion=conv_2d(2*input_dim,input_dim,kernel_size=3,stride=1)
        
    def unfolding(self, feature_map):
        b, c, h, w = feature_map.shape
        new_h = math.ceil(h / self.patch_h) * self.patch_h
        new_w = math.ceil(w / self.patch_w) * self.patch_w
        
        interpolate = (new_w != w) or (new_h != h)
        if interpolate:
            feature_map = nn.functional.interpolate(feature_map, size=(new_h, new_w), mode="bilinear", align_corners=False)
        
        patches = rearrange(feature_map, 'b c (nh ph) (nw pw) -> (b ph pw) (nh nw) c',
                            ph=self.patch_h, pw=self.patch_w)
        
        info_dict = {
            "orig_size": (h, w),
            "batch_size": b,
            "interpolate": interpolate,
            "total_patches": (new_h // self.patch_h) * (new_w // self.patch_w),
            "num_patches_w": new_w // self.patch_w,
            "num_patches_h": new_h // self.patch_h,
        }
        return patches, info_dict
    
    def folding(self, patches, info_dict):
        b, ph, pw, nh, nw, c = (
            info_dict["batch_size"], self.patch_h, self.patch_w,
            info_dict["num_patches_h"], info_dict["num_patches_w"], patches.size(-1)
        )
        
        feature_map = rearrange(patches, '(b ph pw) (nh nw) c -> b c (nh ph) (nw pw)',
                                b=b, ph=ph, pw=pw, nh=nh, nw=nw)
        
        if info_dict["interpolate"]:
            feature_map = nn.functional.interpolate(
                feature_map,
                size=info_dict["orig_size"],
                mode="bilinear",
                align_corners=False,
            )
        
        return feature_map
                
    def forward(self,x):
        x_clone=x.clone()
        x=self.local_rep(x)
        x,info_dict=self.unfolding(x)
        x=self.global_rep(x)
        x=self.folding(x,info_dict)
        x=self.conv_proj(x)
        x=self.fusion(torch.cat((x_clone,x),dim=1))
        return x
    
class MobileViT(nn.Module):
    def __init__(self,image_size,num_classes,patch_size=(2,2)):
        super().__init__()
        h,w=image_size
        self.patch_h,self.patch_w=patch_size
        assert h % self.patch_h == 0 and w % self.patch_w == 0 
        
        mv_exp_mult = 2
        ff_multiplier = 2
        last_layer_exp_factor = 4
        channels = [16, 16, 24, 48, 64, 80]
        attention_dim = [64, 80, 96]
        
        self.conv_0 = conv_2d(3, channels[0], kernel_size=3, stride=2)
        
        self.layer_1=nn.Sequential(
            MobileNetV2Block(channels[0],channels[1],stride=1,expand_ratio=mv_exp_mult)
        )
        
        self.layer_2=nn.Sequential(
            MobileNetV2Block(channels[1],channels[2],stride=2,expand_ratio=mv_exp_mult),
            MobileNetV2Block(channels[2],channels[2],stride=1,expand_ratio=mv_exp_mult),
            MobileNetV2Block(channels[2],channels[2],stride=1,expand_ratio=mv_exp_mult)
        )
        
        self.layer_3=nn.Sequential(
            MobileNetV2Block(channels[2],channels[3],stride=2,expand_ratio=mv_exp_mult),
            MobileViTBlock(channels[3],attention_dim[0],ff_multiplier,4,8,2,patch_size)
        )
        
        self.layer_4=nn.Sequential(
            MobileNetV2Block(channels[3],channels[4],stride=2,expand_ratio=mv_exp_mult),
            MobileViTBlock(channels[4],attention_dim[1],ff_multiplier,4,8,4,patch_size)
        )
        
        self.layer_5=nn.Sequential(
            MobileNetV2Block(channels[4],channels[5],stride=2,expand_ratio=mv_exp_mult),
            MobileViTBlock(channels[5],attention_dim[2],ff_multiplier,4,8,3,patch_size)
        )
        
        self.conv_1x1_exp = conv_2d(channels[-1], channels[-1]*last_layer_exp_factor, kernel_size=1, stride=1)
        self.out = nn.Linear(channels[-1]*last_layer_exp_factor, num_classes, bias=True)
        
    def forward(self,x):
        x = self.conv_0(x)
        x = self.layer_1(x)
        x = self.layer_2(x) 
        x = self.layer_3(x)
        x = self.layer_4(x)
        x = self.layer_5(x)
        x = self.conv_1x1_exp(x)
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
    model = MobileViT(image_size=(height, width), num_classes=num_classes)

    with torch.no_grad():
        output = model(sample_input)

    print(f"Input shape: {sample_input.shape}")
    print(f"Output shape: {output.shape}")