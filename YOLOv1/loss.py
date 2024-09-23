import torch
import torch.nn as nn
from utils import iou

class YOLOLoss(nn.Module):
    def __init__(self,S=7,B=2,N=20):
        super().__init__()
        self.mse=nn.MSELoss(reduction="sum")
        self.S=S
        self.B=B
        self.N=N
        self.lambda_noobj=0.5
        self.lambda_coord=5 
        
    def forward(self,predictions,targets):
        predictions=predictions.reshape(-1,self.S,self.S,self.N+self.B*5)
        
        iou_bbox1=iou(predictions[...,21:25],targets[...,21:25])
        iou_bbox2=iou(predictions[...,26:30],targets[...,21:25])
        ious=torch.cat([iou_bbox1.unsqueeze(0),iou_bbox2.unsqueeze(0)],dim=0)
        
        iou_best,best_bbox=torch.max(ious,dim=0)
        exists_bbox=targets[...,20].unsqueeze(3)
        
        #box loss
        box_predictions=exists_bbox*(
            (
                best_bbox*predictions[...,26:30]+(1-best_bbox)*predictions[...,21:25]
            )
        )
        
        box_targets=exists_bbox*targets[...,21:25]
        
        box_predictions[...,2:4]=torch.sign(box_predictions[...,2:4])*torch.sqrt(torch.abs(box_predictions[..., 2:4]+1e-6))
        box_targets[...,2:4]=torch.sqrt(box_targets[...,2:4])
        
        box_loss=self.mse(torch.flatten(box_predictions, end_dim=-2),torch.flatten(box_targets, end_dim=-2))
        
        #object loss
        pred_bbox=(best_bbox*predictions[...,25:26],(1-best_bbox)*predictions[...,20:21])
        object_loss=self.mse(torch.flatten(exists_bbox*pred_bbox),torch.flatten(exists_bbox*targets[...,20:21]))
        
        #no object loss
        no_object_loss=self.mse(torch.flatten((1-exists_bbox)*predictions[...,20:21],start_dim=1),torch.flatten((1-exists_bbox)*targets[...,20:21], start_dim=1))
        no_object_loss+=self.mse(torch.flatten((1-exists_bbox)*predictions[..., 25:26],start_dim=1),torch.flatten((1-exists_bbox)*targets[..., 20:21],start_dim=1))
        
        #class loss
        class_loss=self.mse(torch.flatten(exists_bbox*predictions[..., :20],end_dim=-2),torch.flatten(exists_bbox*targets[..., :20],end_dim=-2))
        
        loss=self.lambda_coord*box_loss+object_loss+self.lambda_noobj*no_object_loss+class_loss
    
