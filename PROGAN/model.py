import torch
import torch.nn as nn
import torch.nn.functional as F
from math import log2

factors=[1,1,1,1,1/2,1/4,1/8,1/16,1/32]

#For the equalized learining rate
class WSConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=3,stride=1,padding=1,gain=2):
        super().__init__()
        self.conv=nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding)
        self.scale=(gain/(in_channels*kernel_size**2))**0.5
        self.bias=self.conv.bias
        self.conv.bias=None
        nn.init.normal_(self.conv.weight)
        nn.init.zeros_(self.bias)
        
    def forward(self,x):
        return self.conv(x*self.scale)+self.bias.view(1,self.bias.shape[0],1,1)

class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.eps=1e-8
        
    def forward(self,x):
        return x/torch.sqrt(torch.mean(x**2,dim=1,keepdim=True)+self.eps)

class ConvBlock(nn.Module):
    def __init__(self,in_channels,out_channels,use_pixelnorm=True):
        super().__init__()
        self.conv1=WSConv2d(in_channels,out_channels)
        self.conv2=WSConv2d(out_channels,out_channels)
        self.act=nn.LeakyReLU(0.2)
        self.pixel_norm=PixelNorm()
        self.use_pixelnorm=use_pixelnorm
    
    def forward(self,x):
        x=self.act(self.conv1(x))
        x=self.pixel_norm(x) if self.use_pixelnorm else x
        x=self.act(self.conv2(x))
        x=self.pixel_norm(x) if self.use_pixelnorm else x
        return x

class Generator(nn.Module):
    def __init__(self,z_dim,in_channels,img_channels=3):
        super().__init__()
        self.initial=nn.Sequential(
            PixelNorm(),
            nn.ConvTranspose2d(z_dim,in_channels,4,1,0),
            nn.LeakyReLU(),
            WSConv2d(in_channels,in_channels),
            nn.LeakyReLU(),
            PixelNorm()
        )
        
        self.initial_rgb=WSConv2d(in_channels,img_channels,kernel_size=1,stride=1,padding=0)
        
        self.prog_blocks,self.rgb_layers=nn.ModuleList(),nn.ModuleList([self.initial_rgb])
        
        for i in range(len(factors)-1):
            conv_in_channels=int(in_channels*factors[i])
            conv_out_channels=int(in_channels*factors[i+1])
            self.prog_blocks.append(ConvBlock(conv_in_channels,conv_out_channels))
            self.rgb_layers.append(WSConv2d(conv_out_channels,img_channels,kernel_size=1,stride=1,padding=0))
        
    def fade_in(self,alpha,upscaled_image,generated_image):
        return torch.tanh(alpha*generated_image+(1-alpha)*upscaled_image)
    
    def forward(self,x,alpha,steps): #steps=0 -> 4*4 , steps=1 -> 8*8
        out=self.initial(x)
        if steps==0:
            return self.initial_rgb(out)
        
        for step in range(steps):
            upscaled=F.interpolate(out,scale_factor=2,mode="nearest")
            out=self.prog_blocks[step](upscaled)
        
        final_upscaled=self.rgb_layers[steps-1](upscaled)
        final_out=self.rgb_layers[steps](out)
        return self.fade_in(alpha,final_upscaled,final_out)
        

class Discriminator(nn.Module):
    def __init__(self,in_channels,img_channels):
        super().__init__()
        
        self.prog_blocks,self.rgb_layers=nn.ModuleList(),nn.ModuleList()
        self.act=nn.LeakyReLU(0.2)
        for i in range(len(factors)-1,0,-1):
            conv_in_channels=int(in_channels*factors[i])
            conv_out_channels=int(in_channels*factors[i-1])
            self.prog_blocks.append(ConvBlock(conv_in_channels,conv_out_channels,use_pixelnorm=False))
            self.rgb_layers.append(WSConv2d(img_channels,conv_in_channels,kernel_size=1,stride=1,padding=0))
            
        self.final_rgb=WSConv2d(img_channels,in_channels,kernel_size=1,stride=1,padding=0)
        self.rgb_layers.append(self.final_rgb)
        self.avg_pool=nn.AvgPool2d(kernel_size=2,stride=2)
        self.final=nn.Sequential(
            WSConv2d(in_channels+1,in_channels),
            nn.LeakyReLU(0.2),
            WSConv2d(in_channels,in_channels,kernel_size=4,stride=1,padding=0),
            nn.LeakyReLU(0.2),
            WSConv2d(in_channels,1,kernel_size=1,stride=1,padding=0),
        )

    def fade_in(self,alpha,out_image,downscaled_image):
        return alpha*out_image+(1-alpha)*downscaled_image
    
    def minibatch_std(self,x):
        batch_stats=torch.std(x,dim=0).mean().repeat(x.shape[0],1,x.shape[2],x.shape[3])
        return torch.cat([x,batch_stats],dim=1)
    
    def forward(self,x,alpha,steps): #steps=0 -> 4*4 , steps=1 -> 8*8
        cur_step=len(self.prog_blocks)-steps
        out=self.act(self.rgb_layers[cur_step](x))
        
        if steps==0:
            out=self.minibatch_std(out) 
            return self.final(out).view(out.shape[0],-1)
        
        downscaled=self.act(self.rgb_layers[cur_step + 1](self.avg_pool(x)))
        out=self.avg_pool(self.prog_blocks[cur_step](out))
        out=self.fade_in(alpha, downscaled, out)

        for step in range(cur_step+1,len(self.prog_blocks)):
            out=self.prog_blocks[step](out)
            out=self.avg_pool(out)

        out=self.minibatch_std(out)
        return self.final(out).view(out.shape[0], -1)

if __name__ == "__main__":
    Z_DIM = 512
    IN_CHANNELS = 512
    gen = Generator(Z_DIM,IN_CHANNELS, img_channels=3)
    critic = Discriminator(IN_CHANNELS, img_channels=3)

    for img_size in [4, 8, 16, 32, 64, 128, 256, 512, 1024]:
        num_steps = int(log2(img_size / 4))
        x = torch.randn((1, Z_DIM, 1, 1))
        z = gen(x, 0.5, steps=num_steps)
        assert z.shape == (1, 3, img_size, img_size)
        out = critic(z, alpha=0.5, steps=num_steps)
        assert out.shape == (1, 1)
        print(f"Success! At img size: {img_size}")