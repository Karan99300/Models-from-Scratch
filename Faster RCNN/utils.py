import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm
import torch
from torchvision import ops
import torch.nn.functional as F
import torch.optim as optim

def get_anc_centers(out_size):
    out_h,out_w=out_size
    anc_points_x=torch.arange(0,out_w)+0.5
    anc_points_y=torch.arange(0,out_h)+0.5
    
    return anc_points_x,anc_points_y

"""
For each anchor point, we generate nine bounding boxes of different shapes and sizes. 
We choose the size and shape of these boxes such that they enclose all the objects in the image. 
The selection of anchor boxes usually depends on the dataset.
"""

def get_anc_base(anc_points_x,anc_points_y,anc_scales,anc_ratios,out_size):
    num_anc_boxes=len(anc_scales)*len(anc_ratios)
    anc_base=torch.zeros(1,anc_points_x.size(dim=0),anc_points_y.size(dim=0),num_anc_boxes,4)
    
    for index_x,x in enumerate(anc_points_x):
        for index_y,y in enumerate(anc_points_y):
            anc_boxes=torch.zeros((num_anc_boxes,4))
            c=0
            for _,scale in enumerate(anc_scales):
                for _,ratio in enumerate(anc_ratios):
                    w=scale*ratio
                    h=scale
                    
                    xmin=x-w/2
                    ymin=y-h/2
                    xmax=x+w/2
                    ymax=y+h/2
                    
                    anc_boxes[c,:]=torch.tensor([xmin,ymin,xmax,ymax])
                    c+=1
            
            anc_base[:,index_x,index_y,:]=ops.clip_boxes_to_image(anc_boxes,size=out_size)
    return anc_base

def get_iou(batch_size,anc_boxes,gt_bboxes):
    anc_boxes=anc_boxes.reshape(batch_size,-1,4)
    num_anc_boxes=anc_boxes.size(dim=1)
    ious=torch.zeros((batch_size,num_anc_boxes,gt_bboxes.size(dim=1)))
    
    for i in range(batch_size):
        gt_bbox=gt_bboxes[i]
        anc_box=anc_boxes[i]
        ious[i,:]=ops.box_iou(anc_box,gt_bbox)
    
    return ious

"""
Its timportant to remember that the IoU is computed in the feature space between the generated anchor boxes and the projected ground truth boxes. 
To project a ground truth box onto the feature space, we simply divide its coordinates by the scale factor as shown in the below function:
"""

def project_bboxes(bboxes,width_scale_factor,height_scale_factor,mode='a2p'):
    assert mode in ['a2p','p2a']
    
    batch_size=bboxes.size(dim=0)
    proj_bboxes=bboxes.clone().reshape(batch_size,-1,4)
    invalid_bbox_mask=(proj_bboxes==-1)
    
    if mode=='a2p':
        proj_bboxes[:,:,[0,2]]*=width_scale_factor
        proj_bboxes[:,:,[1,3]]*=height_scale_factor
    else:
        proj_bboxes[:,:,[0,2]]/=width_scale_factor
        proj_bboxes[:,:,[1,3]]/=height_scale_factor
    
    proj_bboxes.masked_fill_(invalid_bbox_mask,-1)
    proj_bboxes.resize_as_(bboxes)
    
"""
The postive anchor boxes do not exactly align with the ground truth boxes. 
So we compute offsets between the positive anchor boxes and the ground truth boxes and train a neural network to learn these offsets. 
"""

def calc_gt_offsets(anc_box,gt_box):
    anc_box=ops.box_convert(anc_box,in_fmt="xyxy",out_fmt="cxcywh")
    gt_box=ops.box_convert(gt_box,in_fmt="xyxy",out_fmt="cxcywh")
    
    gt_cx,gt_cy,gt_w,gt_h=gt_box[:,0],gt_box[::,1],gt_box[::,2],gt_box[:,3]
    anc_cx,anc_cy,anc_w,anc_h=anc_box[:,0],anc_box[:,1],anc_box[:,2],anc_box[:,3]
    
    x=(gt_cx-anc_cx)/anc_w
    y=(gt_cy-anc_cy)/anc_h
    w=torch.log(gt_w/anc_w)
    h=torch.log(gt_h/anc_h)
    
    return torch.stack([x,y,w,h],dim=-1)

def get_anc_boxes(anc_boxes,gt_bboxes,gt_classes,pos_thresh=0.7,neg_thresh=0.2):
    B,w_amap,h_amap,num_anc_boxes,_=anc_boxes.shape
    N=gt_bboxes[1]
    
    total_anc_boxes=num_anc_boxes*w_amap*h_amap
    
    ious=get_iou(B,anc_boxes,gt_bboxes)
    max_iou_per_gt_box,_=ious.max(dim=1,keepdim=True)
    
    positive_anc_box=torch.logical_and(ious==max_iou_per_gt_box,max_iou_per_gt_box>0)
    positive_anc_box=torch.logical_or(positive_anc_box,ious>pos_thresh)
    positive_anc_idx_sep=torch.where(positive_anc_box)[0]
    positive_anc_box=positive_anc_box.flatten(start_dim=0,end_dim=1)
    positive_anc_idx=torch.where(positive_anc_box)[0]
    
    max_iou_per_anc,max_iou_per_anc_idx=ious.max(dim=-1)
    max_iou_per_anc = max_iou_per_anc.flatten(start_dim=0, end_dim=1)
    
    gt_conf_scores=max_iou_per_anc[positive_anc_idx]
    
    gt_classes_expand=gt_classes.view(B,1,N).expand(B,total_anc_boxes,N)
    gt_classes=torch.gather(gt_classes_expand,-1,max_iou_per_anc_idx.unsqueeze(-1)).squeeze(-1)
    gt_class=gt_class.flatten(start_dim=0,end_dim=1)
    gt_class_positive=gt_class[positive_anc_idx]
    
    gt_bboxes_expand=gt_bboxes.view(B,1,N,4).expand(B,total_anc_boxes,N,4)
    gt_bboxes=torch.gather(gt_bboxes_expand,-2,max_iou_per_anc_idx.reshape(B,total_anc_boxes,1,1).repeat(1,1,1,4))
    gt_bboxes=gt_bboxes.flatten(start_dim=0,end_dim=2)
    gt_bboxes_positive=gt_bboxes[positive_anc_idx]
    
    anc_boxes=anc_boxes.flatten(start_dim=0,end_dim=2)
    positive_anc_coords=anc_boxes[positive_anc_idx]
    
    gt_offsets=calc_gt_offsets(positive_anc_coords,gt_bboxes_positive)
    
    negative_anc_box=(max_iou_per_anc < neg_thresh)
    negative_anc_idx=torch.where(negative_anc_box)[0]
    
    negative_anc_idx=negative_anc_idx[torch.randint(0, negative_anc_idx.shape[0], (positive_anc_idx.shape[0],))]
    negative_anc_coords=anc_boxes[negative_anc_idx]
    
    return positive_anc_idx,negative_anc_idx,gt_conf_scores,gt_offsets,gt_class_positive,positive_anc_coords,negative_anc_coords,positive_anc_idx_sep

def generate_proposals(anchors,offsets):
    anchors=ops.box_convert(anchors,in_fmt="xyxy",out_fmt="cxcywh")
    proposals=torch.zeros_like(anchors)
    proposals[:,0]=anchors[:,0]+offsets[:,0]*anchors[:,2]
    proposals[:,1]=anchors[:,1]+offsets[:,1]*anchors[:,3]
    proposals[:,2]=anchors[:,2]*torch.exp(offsets[:,2])
    proposals[:,3]=anchors[:,3]*torch.exp(offsets[:,3])
    proposals=ops.box_convert(proposals,in_fmt="xyxy",out_fmt="cxcywh")
    
def calc_cls_loss(conf_scores_positive,conf_scores_negative,batch_size):
    targets_positive=torch.ones_like(conf_scores_positive)
    targets_negative=torch.zeros_like(conf_scores_negative)
    
    target=torch.cat((targets_positive,targets_negative))
    inputs=torch.cat((conf_scores_positive,conf_scores_negative))
    
    loss=F.binary_cross_entropy_with_logits(inputs,target,reduction='sum')*1./batch_size
    return loss

def calc_bbox_reg_loss(gt_offsets,reg_offsets_positive,batch_size):
    assert gt_offsets.size()==reg_offsets_positive.size()
    
    loss=F.smooth_l1_loss(reg_offsets_positive,gt_offsets,reduction='sum')*1./batch_size
    return loss

