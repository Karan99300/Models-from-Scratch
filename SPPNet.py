import torch
import torch.nn as nn
import math

class SpatialPyramidPool(nn.Module):
    def __init__(self,out_pool_size):
        super().__init__()
    
    def forward(self,feature_map,batch_size,feature_map_size):
        for i in range(self.out_pool_size):
            h_stride=int(math.ceil(feature_map_size[0]/self.out_pool_size[i]))
            w_stride=int(math.ceil(feature_map_size[0]/self.out_pool_size[i]))
            
            h_padding=(h_stride*self.out_pool_size[i]-feature_map_size[i]+1)//2
            w_padding=(w_stride*self.out_pool_size[i]-feature_map_size[i]+1)//2
            
            maxpool=nn.MaxPool2d((h_stride,w_stride),stride=(h_stride,w_stride),padding=(h_padding,w_padding))
            x=maxpool(feature_map)
            
            if i==0:
                spp=x.view(batch_size,-1)
            else:
                spp=torch.cat((spp,x.view(batch_size,-1)),1)
        return spp

class SPP_NET(nn.Module):
    def __init__(self,input_nc, ndf=64):
        super(SPP_NET, self).__init__()
        self.output_num=[4,2,1]
        
        self.conv1=nn.Conv2d(input_nc,ndf,4,2,1,bias=False)
        self.leaky_relu=nn.LeakyReLU(0.2,inplace=True)
        
        self.conv2=nn.Conv2d(ndf,ndf*2,4,1,1,bias=False)
        self.bn1=nn.BatchNorm2d(ndf*2)
        self.conv3=nn.Conv2d(ndf*2,ndf*4,4,1,1,bias=False)
        self.bn2=nn.BatchNorm2d(ndf*4)
        self.conv4=nn.Conv2d(ndf*4,ndf*8,4,1,1,bias=False)
        self.bn3=nn.BatchNorm2d(ndf*8)
        self.conv5=nn.Conv2d(ndf*8,64,4,1,0,bias=False)
        self.fc1=nn.Linear(10752, 4096)
        self.fc2=nn.Linear(4096, 1000)
        self.sigmoid=nn.Sigmoid()
        
        self.spp=SpatialPyramidPool(self.output_num)

    def forward(self, x):
        x=self.leaky_relu(self.conv1(x))
        x=self.leaky_relu(self.bn1(self.conv2(x)))
        x=self.leaky_relu(self.bn2(self.conv3(x)))
        x=self.conv4(x)
        
        spp=self.spp(x,1,[int(x.size(2)),int(x.size(3))])
        fc1=self.fc1(spp)
        fc2=self.fc2(fc1)
        output=self.sigmoid(fc2)
        return output