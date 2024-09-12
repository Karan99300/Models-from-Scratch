import torch
import numpy as np

def iou(bb1,bb2):
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']
    
    x_left=max(bb1['x1'],bb2['x1'])
    y_top=max(bb1['y1'],bb2['y1'])
    x_right=min(bb1['x2'],bb2['x2'])
    y_bottom=min(bb1['y2'],bb2['y2'])
    
    if x_right<x_left or y_bottom<y_top:
        return 0.0
    
    intersection_area=(x_right-x_left)*(y_bottom-y_top)
    
    bb1_area=(bb1['x2']-bb1['x1'])*(bb1['y2']-bb1['y1'])
    bb2_area=(bb2['x2']-bb2['x1'])*(bb2['y2']-bb2['y1'])
    
    iou=intersection_area/(bb1_area+bb2_area-intersection_area)
    assert iou>=0.0
    assert iou<=1.0
    return iou

def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)

def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location="cuda")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
        
def non_max_suppression(boxes,scores,iou_threshold=0.5):
    if len(boxes)==0:
        return 0
    
    boxes=np.array(boxes)
    scores=np.array(scores)
    
    final_boxes=[]
    
    sorted_indices=np.argsort(scores)[::-1]
    
    while len(sorted_indices)>0:
        current_index=sorted_indices[0]
        current_box=boxes[current_index]
        final_boxes.append(current_box)
        
        sorted_indices=sorted_indices[1:]
        remaining_boxes=boxes[sorted_indices]
        
        keep_indices=[]
        
        for i,box in enumerate(remaining_boxes):
            iou_value = iou({'x1': current_box[0], 'y1': current_box[1], 'x2': current_box[2], 'y2': current_box[3]}, 
                            {'x1': box[0], 'y1': box[1], 'x2': box[2], 'y2': box[3]})
            
            if iou_value<iou_threshold:
                keep_indices.append(i)
                
        sorted_indices=sorted_indices[keep_indices]
        
    return final_boxes