import torch.nn as nn
from torchvision.models import vgg19
import config

class VGGLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg=vgg19(pretrained=True).features[:36].eval().to(config.device)
        self.loss=nn.MSELoss()
        
        for parameter in self.vgg.parameters():
            parameter.requires_grad=False
            
    def forward(self,input,target):
        vgg_input_features=self.vgg(input)
        vgg_target_features=self.vgg(target)
        return self.loss.forward(vgg_input_features,vgg_target_features)