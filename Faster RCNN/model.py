import torch
import torchvision
from torchvision import ops
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from utils import *

class BackBone(nn.Module):
    def __init__(self):
        super().__init__()
        model=torchvision.models.resnet50(pretrained=True)
        req_layers=list(model.children())[:8]
        self.backbone=nn.Sequential(*req_layers)
        for param in self.backbone.named_parameters():
            param[1].requires_grad=True
        
    def forward(self,x):
        return self.backbone(x)

"""
We consider each point in the feature map as an anchor point. 
So anchor points would just be an array representing coordinates along the width and height dimensions.
"""

"""
every point in the feature map is considered an anchor, and every anchor generates boxes of different sizes and shapes. 
We want to classify each of these boxes as object or background. 
Moreover, we want to predict their offsets from the corresponding ground truth boxes
"""

class ProposalModule(nn.Module):
    def __init__(self,in_features,hidden_dim,num_anc_boxes=9,dropout=0.1):
        super().__init__()
        self.num_anc_boxes=num_anc_boxes
        self.conv=nn.Conv2d(in_features,hidden_dim,kernel_size=3,padding=1)
        self.dropout=nn.Dropout(dropout)
        self.conf_head=nn.Conv2d(hidden_dim,num_anc_boxes,kernel_size=1)
        self.reg_head=nn.Conv2d(hidden_dim,num_anc_boxes*4,kernel_size=1)
        
    def forward(self,feature_map,positive_anc_idx=None,negative_anc_idx=None,positive_anc_coords=None):
        if positive_anc_idx is None or negative_anc_idx is None or positive_anc_coords is None:
            mode='eval'
        else:
            mode='train'
        
        out=self.conv(feature_map)
        out=F.relu(self.dropout(out))
        
        conf_scores_pred=self.conf_head(out)
        reg_offsets_pred=self.reg_head(out)
        
        if mode=='eval':
            return conf_scores_pred,reg_offsets_pred
        elif mode=='train':
            conf_scores_positive=conf_scores_pred.flatten()[positive_anc_idx]
            conf_scores_negative=conf_scores_pred.flatten()[negative_anc_idx]
            
            reg_offsets_positive=reg_offsets_pred.contiguous().view(-1,4)[positive_anc_idx]
            proposals=generate_proposals(positive_anc_coords,reg_offsets_positive)
            
            return conf_scores_positive,conf_scores_negative,reg_offsets_positive,proposals
        
class RegionalProposalNetwork(nn.Module):
    def __init__(self,img_size,out_size,out_channels):
        super().__init__()
        self.h,self.w=img_size
        self.out_h,self.out_w=out_size
        
        self.h_scale_factor=self.h//self.out_h
        self.w_scale_factor=self.w//self.out_w
        
        self.anc_scales=[2, 4, 6]
        self.anc_ratios=[0.5, 1, 1.5]
        self.num_anc_boxes=self.anc_ratios*self.anc_scales
        
        self.pos_thresh = 0.7
        self.neg_thresh = 0.3
        
        self.w_conf=1
        self.w_reg=5
        
        self.backbone=BackBone()
        self.proposal_module=ProposalModule(out_channels,num_anc_boxes=self.num_anc_boxes)
        
    def forward(self,images,gt_bboxes,gt_classes):
        batch_size=images[0]
        feature_map=self.backbone(images)
        
        anc_points_x,anc_points_y=get_anc_centers(out_size=(self.out_h,self.out_w))
        anc_base=get_anc_base(anc_points_x,anc_points_y,self.anc_scales,self.anc_ratios,(self.out_h,self.out_w))
        anc_boxes=anc_base.repeat(batch_size,1,1,1,1)
        
        gt_bboxes_proj=project_bboxes(gt_bboxes,self.w_scale_factor,self.h_scale_factor,mode='p2a')
        
        positive_anc_idx,negative_anc_idx,gt_conf_scores,gt_offsets,gt_class_positive,positive_anc_coords,negative_anc_coords,positive_anc_idx_sep=get_anc_boxes(anc_boxes,gt_bboxes_proj,gt_classes,self.pos_thresh,self.neg_thresh)
        
        conf_scores_positive,conf_scores_negative,reg_offsets_positive,proposals=self.proposal_module(feature_map,positive_anc_idx,negative_anc_idx,positive_anc_coords)
        
        cls_loss=calc_cls_loss(conf_scores_positive,conf_scores_negative,batch_size)
        reg_loss=calc_bbox_reg_loss(gt_offsets,reg_offsets_positive)
        
        total_rpn_loss=self.w_conf*cls_loss+self.w_reg*reg_loss
        
        return total_rpn_loss,feature_map,proposals,positive_anc_idx_sep,gt_class_positive
    
    def inference(self,images,conf_thresh=0.5,nms_thresh=0.7):
        with torch.no_grad():
            batch_size=images[0]
            feature_map=self.backbone(images)
            
            anc_points_x,anc_points_y=get_anc_centers(out_size=(self.out_h,self.out_w))
            anc_base=get_anc_base(anc_points_x,anc_points_y,self.anc_scales,self.anc_ratios,(self.out_h,self.out_w))
            anc_boxes=anc_base.repeat(batch_size,1,1,1,1)
            anc_boxes_flat=anc_boxes.reshape(batch_size,-1,4)
            
            conf_scores_pred,offsets_pred=self.proposal_module(feature_map)
            conf_scores_pred=conf_scores_pred.reshape(batch_size,-1)
            offsets_pred=offsets_pred.reshape(batch_size,-1,4)
            
            proposals_final=[]
            conf_scores_final=[]
            
            for i in range(batch_size):
                conf_scores=torch.sigmoid(conf_scores_pred[i])
                offsets=offsets_pred[i]
                anc_boxes=anc_boxes_flat[i]
                proposals=generate_proposals(anc_boxes,offsets)
                conf_idx=torch.where(conf_scores>=conf_thresh)[0]
                conf_scores_positive=conf_scores[conf_idx]
                proposals_positive=proposals[conf_idx]
                nms_idx=ops.nms(proposals_positive,conf_scores_positive,nms_thresh)
                conf_scores_positive=conf_scores_positive[nms_idx]
                proposals_positive=proposals_positive[nms_idx]
                
                proposals_final.append(proposals_positive)
                conf_scores_final.append(conf_scores_positive)
                
        return proposals_final,conf_scores_final,feature_map
    
"""
In second stage we receive region proposals and predict the category of the object in the proposals. 
This can be done by a simple convolutional network, but theres a catch: all proposals do not have the same size. 
Now, you may think of resizing the proposals before feeding into the model like how we usually resize images in image classification tasks, 
but the problem is resizing is not a differentiable operation, and so backpropagation cannot happen through this operation.
Heres a smarter way to resize: we divide the proposals into roughly equal subregions and apply a max pooling operation on each of them to produce outputs of same size. 
This is called ROI pooling
"""

class ClassificationModule(nn.Module):
    def __init__(self,out_channels,num_classes,roi_size,hidden_dim=512,dropout=0.1):
        super().__init__()
        self.roi_size=roi_size
        self.avgpool=nn.AvgPool2d(self.roi_size)
        self.fc=nn.Linear(out_channels,hidden_dim)
        self.dropout=nn.Dropout(dropout)
        self.cls_head=nn.Linear(hidden_dim,num_classes)
        
    def forward(self,feature_map,proposals_list,gt_classes=None):
        if gt_classes is None:
            mode='eval'
        else:
            mode='train'
            
        roi_out=ops.roi_pool(feature_map,proposals_list,self.roi_size)
        roi_out=self.avgpool(roi_out)
        
        roi_out=roi_out.squeeze(-1).squeeze(-1)
        
        out=self.fc(roi_out)
        out=F.relu(self.dropout(out))
        
        cls_scores=self.cls_head(out)
        
        if mode=='eval':
            return cls_scores
        
        cls_loss=F.cross_entropy(cls_scores,gt_classes.long())
        
        return cls_loss
    
class FasterRCNN(nn.Module):
    def __init__(self,img_size,out_size,out_channels,num_classes,roi_size):
        super().__init__()
        self.rpn=RegionalProposalNetwork(img_size,out_size,out_channels)
        self.classifier=ClassificationModule(out_channels,num_classes,roi_size)
        
    def forward(self,images,gt_bboxes,gt_classes):
        total_rpn_loss,feature_map,proposals,positive_anc_idx_sep,gt_class_positive=self.rpn(images,gt_bboxes,gt_classes)
        
        positive_proposals_list=[]
        batch_size=images.size(dim=0)
        for idx in range(batch_size):
            proposal_idxs=torch.where(positive_anc_idx_sep==idx)[0]
            proposals_sep=proposals[proposal_idxs].detach().clone()
            positive_proposals_list.append(proposals_sep)
            
        cls_loss=self.classifier(feature_map,positive_proposals_list,gt_class_positive)
        total_loss=cls_loss+total_rpn_loss
        
        return total_loss
    
    def inference(self,images,conf_thresh=0.5,nms_thresh=0.7):
        batch_size=images.size(dim=0)
        proposals_final,conf_scores_final,feature_map=self.rpn.inference(images,conf_thresh,nms_thresh)
        cls_scores=self.classifier(feature_map, proposals_final)
        
        cls_probs=F.softmax(cls_scores,dim=-1)
        classes=torch.argmax(cls_probs,dim=-1)
        
        classes_final=[]
        
        c = 0
        for i in range(batch_size):
            n_proposals=len(proposals_final[i]) 
            classes_final.append(classes[c:c+n_proposals])
            c += n_proposals
            
        return proposals_final,conf_scores_final,classes_final