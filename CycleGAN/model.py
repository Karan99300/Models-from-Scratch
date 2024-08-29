import torch 
import torch.nn as nn

class DiscBlock(nn.Module):
    def __init__(self,in_channels,out_channels,stride):
        super().__init__()
        self.conv=nn.Sequential(
            nn.Conv2d(in_channels, out_channels,4,stride,1,bias=True,padding_mode="reflect"),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )
    
    def forward(self,x):
        return self.conv(x)  
    
class Discriminator(nn.Module):
    def __init__(self,in_channels=3,features=[64,128,256,512]):
        super().__init__()
        self.initial=nn.Sequential(
            nn.Conv2d(in_channels, features[0],4,2,1,padding_mode="reflect"),
            nn.LeakyReLU(0.2)
        )
        
        layers=[]
        in_channels=features[0]
        for feature in features[1:]:
            layers.append(DiscBlock(in_channels,feature,stride=1 if feature==features[-1] else 2))
            in_channels=feature
        
        layers.append(nn.Conv2d(in_channels,1,4,1,1,padding_mode="reflect"))
        self.model=nn.Sequential(*layers)
    
    def forward(self,x):
        x=self.initial(x)
        return torch.sigmoid(self.model(x))

class ConvBlock(nn.Module):
    def __init__(self,in_channels,out_channels,downsampling=True,use_activation=True,**kwargs):
        super().__init__()
        self.conv=nn.Sequential(
            nn.Conv2d(in_channels,out_channels,padding_mode="reflect",**kwargs) if downsampling else nn.ConvTranspose2d(in_channels,out_channels,**kwargs),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True) if use_activation else nn.Identity()
        )
    
    def forward(self,x):
        return self.conv(x)
    
class ResidualBlock(nn.Module):
    def __init__(self,channels):
        super().__init__()
        self.block=nn.Sequential(
            ConvBlock(channels,channels,kernel_size=3,padding=1),
            ConvBlock(channels,channels,use_activation=False,kernel_size=3,padding=1)
        )
    
    def forward(self,x):
        return x+self.block(x)

class Generator(nn.Module):
    def __init__(self, image_channels,features=64,num_residual_blocks=9):
        super().__init__()
        self.initial=nn.Sequential(
            nn.Conv2d(image_channels,features,7,1,3,padding_mode="reflect"),
            nn.ReLU(inplace=True)
        )
        
        self.down_blocks=nn.ModuleList([
            ConvBlock(features,features*2,kernel_size=3,stride=2,padding=1),
            ConvBlock(features*2,features*4,kernel_size=3,stride=2,padding=1),
        ])
        
        self.residual_blocks=nn.Sequential(
            *[ResidualBlock(features*4) for _ in range(num_residual_blocks)]
        )
        
        self.up_blocks=nn.ModuleList([
            ConvBlock(features*4,features*2,downsampling=False,kernel_size=3,stride=2,padding=1,output_padding=1),
            ConvBlock(features*2,features,downsampling=False,kernel_size=3,stride=2,padding=1,output_padding=1)
        ])
        
        self.last_block=nn.Conv2d(features,image_channels,7,1,3,padding_mode="reflect")
        
    def forward(self,x):
        x=self.initial(x)
        for layer in self.down_blocks:
            x=layer(x)
            
        x=self.residual_blocks(x)
        
        for layer in self.up_blocks:
            x=layer(x)
        return torch.tanh(self.last_block(x))
    
def test():
    x=torch.randn((2,3,256,256))
    disc=Discriminator()
    gen=Generator(3)
    print(disc(x).shape)
    print(gen(x).shape)

if __name__=="__main__":
    test()