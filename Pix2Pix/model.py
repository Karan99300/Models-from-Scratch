import torch
import torch.nn as nn

class DiscBlock(nn.Module):
    def __init__(self,in_channels,out_channels,stride=2):
        super().__init__()
        self.conv=nn.Sequential(
            nn.Conv2d(in_channels, out_channels,4,stride,bias=False,padding=1,padding_mode="reflect"),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )
    
    def forward(self,x):
        return self.conv(x) 

class Discriminator(nn.Module):
    def __init__(self,in_channels=3,features=[64,128,256,512]):
        super().__init__()
        self.initial=nn.Sequential(
            nn.Conv2d(in_channels*2,features[0],4,2,1,padding_mode="reflect"),
            nn.LeakyReLU(0.2)
        )
        
        layers=[]
        in_channels=features[0]
        for feature in features[1:]:
            layers.append(DiscBlock(in_channels,feature,stride=1 if feature==features[-1] else 2))
            in_channels=feature
        
        layers.append(
            nn.Conv2d(in_channels,1,4,1,1,padding_mode="reflect")
        )
        
        self.model=nn.Sequential(*layers)
        
    def forward(self,x,y):
        x=torch.cat([x,y],dim=1)
        x=self.initial(x)
        return self.model(x)

class GenBlock(nn.Module):
    def __init__(self,in_channels,out_channels,downsampling=True,activation="relu",use_dropout=False):
        super().__init__()
        self.conv=nn.Sequential(
            nn.Conv2d(in_channels,out_channels,4,2,1,bias=False,padding_mode="reflect") if downsampling else nn.ConvTranspose2d(in_channels,out_channels,4,2,1,bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU() if activation=="relu" else nn.LeakyReLU(0.2)
        )
        
        self.use_dropout=use_dropout
        self.dropout=nn.Dropout(0.5)
    
    def forward(self,x):
        x=self.conv(x)
        return self.dropout(x) if self.use_dropout else x

class Generator(nn.Module):
    def __init__(self,in_channels=3,features=64):
        super().__init__()
        self.initial_down=nn.Sequential(
            nn.Conv2d(in_channels,features,4,2,1,padding_mode="reflect"),
            nn.LeakyReLU(0.2)
        )
        
        self.down1=GenBlock(features,features*2,downsampling=True,activation="leaky",use_dropout=False)
        self.down2=GenBlock(features*2,features*4,downsampling=True,activation="leaky",use_dropout=False)
        self.down3=GenBlock(features*4,features*8,downsampling=True,activation="leaky",use_dropout=False)
        self.down4=GenBlock(features*8,features*8,downsampling=True,activation="leaky",use_dropout=False)
        self.down5=GenBlock(features*8,features*8,downsampling=True,activation="leaky",use_dropout=False)
        self.down6=GenBlock(features*8,features*8,downsampling=True,activation="leaky",use_dropout=False)
    
        self.bottleneck=nn.Sequential(
            nn.Conv2d(features*8,features*8,4,2,1,padding_mode="reflect"),
            nn.ReLU()
        )
        
        self.up1=GenBlock(features*8,features*8,downsampling=False,activation="relu",use_dropout=True)
        self.up2=GenBlock(features*8*2,features*8,downsampling=False,activation="relu",use_dropout=True)
        self.up3=GenBlock(features*8*2,features*8,downsampling=False,activation="relu",use_dropout=True)
        self.up4=GenBlock(features*8*2,features*8,downsampling=False,activation="relu",use_dropout=False)
        self.up5=GenBlock(features*8*2,features*4,downsampling=False,activation="relu",use_dropout=False)
        self.up6=GenBlock(features*4*2,features*2,downsampling=False,activation="relu",use_dropout=False)
        self.up7=GenBlock(features*2*2,features,downsampling=False,activation="relu",use_dropout=False)
        self.final_up=nn.Sequential(
            nn.ConvTranspose2d(features*2,in_channels,4,2,1),
            nn.Tanh()
        )
        
    def forward(self,x):
        d1=self.initial_down(x)
        d2=self.down1(d1)
        d3=self.down2(d2)
        d4=self.down3(d3)
        d5=self.down4(d4)
        d6=self.down5(d5)
        d7=self.down6(d6)
        
        bottleneck=self.bottleneck(d7)
        
        u1=self.up1(bottleneck)
        u2=self.up2(torch.cat([u1,d7],dim=1))
        u3=self.up3(torch.cat([u2,d6],dim=1))
        u4=self.up4(torch.cat([u3,d5],dim=1))
        u5=self.up5(torch.cat([u4,d4],dim=1))
        u6=self.up6(torch.cat([u5,d3],dim=1))
        u7=self.up7(torch.cat([u6,d2],dim=1))
        return self.final_up(torch.cat([u7,d1],dim=1))
