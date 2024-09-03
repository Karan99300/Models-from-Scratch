import torch 
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self,in_channels,out_channels,disc=False,use_activation=True,use_batchnorm=True,**kwargs):
        super().__init__()
        self.cnn=nn.Conv2d(in_channels,out_channels,**kwargs,bias=not use_batchnorm)
        self.batch_norm=nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity()
        self.use_activation=use_activation
        self.activation=(
            nn.LeakyReLU(0.2,inplace=True) if disc else nn.PReLU(num_parameters=out_channels)
        )
    
    def forward(self,x):
        return self.activation(self.batch_norm(self.cnn(x))) if self.use_activation else self.batch_norm(self.cnn(x))        

class UpsampleBlock(nn.Module):
    def __init__(self,in_channels,scale_factor=2):
        super().__init__()
        self.conv=nn.Conv2d(in_channels,in_channels*scale_factor**2,kernel_size=3,stride=1,padding=1)
        self.pixel_shuffle=nn.PixelShuffle(scale_factor) #in_c*4,h,w --> in_c,h*2,w*2 
        self.activation=nn.PReLU(num_parameters=in_channels)
    
    def forward(self,x):
        return self.activation(self.pixel_shuffle(self.conv(x)))

class ResidualBlock(nn.Module):
    def __init__(self,in_channels):
        super().__init__()
        self.block1=ConvBlock(in_channels,in_channels,kernel_size=3,stride=1,padding=1)
        self.block2=ConvBlock(in_channels,in_channels,kernel_size=3,stride=1,padding=1,use_activation=False)
        
    def forward(self,x):
        out=self.block1(x)
        out=self.block2(x)
        return out+x

class Generator(nn.Module):
    def __init__(self,in_channels=3,num_channels=64,num_blocks=16):
        super().__init__()
        self.initial=ConvBlock(in_channels,num_channels,kernel_size=9,stride=1,padding=4,use_batchnorm=False)
        self.residuals=nn.Sequential(*[ResidualBlock(num_channels) for _ in range(num_blocks)])
        self.conv_block=ConvBlock(num_channels,num_channels,kernel_size=3,stride=1,padding=1,use_activation=False)
        self.upsamples=nn.Sequential(
            UpsampleBlock(num_channels,scale_factor=2),
            UpsampleBlock(num_channels,scale_factor=2)
        )
        self.final=nn.Conv2d(num_channels,in_channels,kernel_size=9,stride=1,padding=4)
        
    def forward(self,x):
        initial=self.initial(x)
        x=self.residuals(initial)
        x=self.conv_block(x)+initial
        x=self.upsamples(x) 
        return torch.tanh(self.final(x))
    
class Discriminator(nn.Module):
    def __init__(self,in_channels=3,features=[64,64,128,128,256,256,512,512]):
        super().__init__()
        blocks=[]
        for idx,feature in enumerate(features):
            blocks.append(ConvBlock(in_channels,feature,kernel_size=3,stride=1+idx%2,padding=1,disc=True,use_activation=True,use_batchnorm=False if idx==0 else True))
            in_channels=feature
        
        self.blocks=nn.Sequential(*blocks)
        self.classifier=nn.Sequential(
            nn.AdaptiveAvgPool2d((6,6)),
            nn.Flatten(),
            nn.Linear(512*6*6,1024),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(1024,1)
        )
        
    def forward(self,x):
        x=self.blocks(x)
        return self.classifier(x) #not applying sigmoid as we will use bce with logits
    
def test():
    low_resolution = 24  # 96x96 -> 24x24
    x = torch.randn((5, 3, low_resolution, low_resolution))
    gen = Generator()
    gen_out = gen(x)
    disc = Discriminator()
    disc_out = disc(gen_out)

    print(gen_out.shape)
    print(disc_out.shape)

if __name__=="__main__":
    test()
