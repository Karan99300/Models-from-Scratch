import torch 
import torch.nn as nn

"""
architecture congfig
Tuple is structured by (out_channels,kernel size,stride, padding) 
"maxpool(2,2)" is simply maxpooling with stride 2x2 and kernel 2x2
List is structured by tuples and lastly int with number of repeats
"""
architecture_config=[
    (64,7,2,3),
    "maxpool(2,2)",
    (192,3,1,1),
    "maxpool(2,2)",
    (128,1,1,0),
    (256,3,1,1),
    (256,1,1,0),
    (512,3,1,1),
    "maxpool(2,2)",
    [(256,1,1,0),(512,3,1,1),4],
    (512,1,1,0),
    (1024,3,1,1),
    "maxpool(2,2)",
    [(512,1,1,0),(1024,3,1,1),2],
    (1024,3,1,1),
    (1024,3,2,1),
    (1024,3,1,1),
    (1024,3,1,1)
]

class CNNBlock(nn.Module):
    def __init__(self,in_channels,out_channels,**kwargs):
        super().__init__()
        self.conv=nn.Conv2d(in_channels,out_channels,bias=False,**kwargs)
        self.batch_norm=nn.BatchNorm2d(out_channels)
        self.act=nn.LeakyReLU(0.1)
        
    def forward(self,x):
        return self.act(self.batch_norm(self.conv(x)))
    
class YOLO(nn.Module):
    def __init__(self,in_channels,**kwargs):
        super().__init__()
        self.architecture=architecture_config
        self.in_channels=in_channels
        self.conv_layers=self.create_conv_layers(self.architecture)
        self.fc_layers=self.create_fc_layers(**kwargs)
        
    def create_conv_layers(self,architecture):
        layers=[]
        in_channels=self.in_channels
        
        for x in architecture:
            if type(x)==tuple:
                layers+=[
                    CNNBlock(in_channels,x[0],kernel_size=x[1],stride=x[2],padding=x[3])
                ]
                in_channels=x[0]
                
            elif type(x)==str:
                layers+=[
                    nn.MaxPool2d(2,2)
                ]
            
            elif type(x)==list:
                conv1=x[0]
                conv2=x[1]
                num_repeats=x[2]
                
                for _ in range(num_repeats):
                    layers+=[
                        CNNBlock(in_channels,conv1[0],kernel_size=conv1[1],stride=conv1[2],padding=conv1[3]),
                        CNNBlock(conv1[0],conv2[0],kernel_size=conv2[1],stride=conv2[2],padding=conv2[3])
                    ]
                    in_channels=conv2[0]
                
                
        return nn.Sequential(*layers)
    
    def create_fc_layers(self,S,B,N):
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024*S*S,4096),
            nn.Dropout(0.2),
            nn.LeakyReLU(0.1),
            nn.Linear(4096,S*S*(N+B*5))
        )
        
    def forward(self,x):
        x=self.conv_layers(x)
        return self.fc_layers(torch.flatten(x,dim=1))
    
    