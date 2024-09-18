import torch
import torch.nn as nn
import torch.nn.functional as F 

class Block(nn.Module):
    def __init__(self,in_channels,out_channels,stride=1,expansion=4,is_first_block=False):
        super().__init__()
        self.conv1=nn.Conv2d(in_channels,out_channels,1,1,0)
        self.bn1=nn.BatchNorm2d(out_channels)
        self.conv2=nn.Conv2d(out_channels,out_channels,3,stride,1)
        self.bn2=nn.BatchNorm2d(out_channels)
        self.conv3=nn.Conv2d(out_channels,out_channels*expansion,1,1,0)
        self.bn3=nn.BatchNorm2d(out_channels*expansion)
        self.downsample=None
        if is_first_block:
            self.downsample=nn.Sequential(
                nn.Conv2d(in_channels,out_channels**expansion,1,stride,0),
                nn.BatchNorm2d(out_channels*expansion)
            )
            
    def forward(self,x):
        identity=x.clone()
        x=F.relu(self.bn1(self.conv1(x)))
        x=F.relu(self.bn2(self.conv2(x)))
        x=self.bn3(self.conv3(x))
        
        if self.downsample:
            identity=self.downsample(identity)
        
        x+=identity
        x=F.relu(x)
        return x
    
class ResNet(nn.Module):
    def __init__(self,Block,num_blocks_list=[3,4,6,3],out_channels_list=[64,128,256,512],num_channels=3,expansion=4):
        self.initial_conv=nn.Sequential(
            nn.Conv2d(num_channels,64,7,2,3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3,2,1)
        )
        
        self.expansion=expansion
        in_channels=64
        self.layer1=self.create_layer(Block,num_blocks_list[0],in_channels,out_channels_list[0],stride=1)
        self.layer2=self.create_layer(Block,num_blocks_list[1],out_channels_list[0]*expansion,out_channels_list[1],stride=2)
        self.layer3=self.create_layer(Block,num_blocks_list[2],out_channels_list[1]*expansion,out_channels_list[2],stride=2)
        self.layer4=self.create_layer(Block,num_blocks_list[3],out_channels_list[2]*expansion,out_channels_list[3],stride=2)
        
    def create_layer(self,Block,num_blocks,in_channels,out_channels,stride=1):
        layer=[]
        for i in range(num_blocks):
            if i==0:
                layer.append(Block(in_channels=in_channels,out_channels=out_channels,stride=stride,is_first_block=True))
            else:
                layer.append(Block(in_channels=out_channels*self.expansion,out_channels=out_channels,stride=stride))
            
        return nn.Sequential(*layer)
    
    def forward(self,x):
        x=self.initial_conv(x)
        C2=self.layer1(x)
        C3=self.layer2(x)
        C4=self.layer3(x)
        C5=self.layer4(x)
        
        return C2,C3,C4,C5

class FPNBlock(nn.Module):
    def __init__(self,in_channels,out_channels,is_highest_block=False):
        super().__init__()
        self.conv1=nn.Conv2d(in_channels,out_channels,1,1,0)
        self.conv2=nn.Conv2d(out_channels,out_channels,3,1,1)
        self.is_highest_block=is_highest_block
        
    def forward(self,current,upper):
        x=self.conv1(x)
        if not self.is_highest_block:
            x+=F.interpolate(upper,scale_factor=2,mode='bilinear',align_corners=True)
        
        out=self.conv2(x)
        
        return x,out
    
class FPN(nn.Module):
    def __init__(self,expansion=4,in_channels_list=[64,128,256,512],out_channels=256):
        super().__init__()
        self.P2=FPNBlock(in_channels_list[0]*expansion,out_channels)
        self.P3=FPNBlock(in_channels_list[1]*expansion,out_channels)
        self.P4=FPNBlock(in_channels_list[2]*expansion,out_channels)
        self.P5=FPNBlock(in_channels_list[3]*expansion,out_channels,True)
        self.P6=nn.MaxPool2d(1,2,0)
        
    def forward(self,C2,C3,C4,C5):
        x,P5=self.P5(C5,None)
        x,P4=self.P4(C4,x)
        x,P3=self.P3(C3,x)
        _,P2=self.P2(C2,x)
        P6=self.P6(P5)
        
        return P2,P3,P4,P5,P6
    
class ResNetFPN(nn.Module):
    def __init__(self):
        self.resnet=ResNet(Block)
        self.fpn=FPN()
    
    def forward(self,x):
        C2,C3,C4,C5=self.resnet(x)
        P2,P3,P4,P5,P6=self.fpn(C2,C3,C4,C5)
        return P2,P3,P4,P5,P6
    
