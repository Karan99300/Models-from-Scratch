import torch 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import Counter

def iou(boxes_preds,boxes_labels,box_format="midpoint"):
    if box_format=="midpoint":
        box1_x1=boxes_preds[...,0:1]-boxes_preds[...,2:3]/2
        box1_y1=boxes_preds[..., 1:2]-boxes_preds[..., 3:4]/2
        box1_x2=boxes_preds[..., 0:1]+boxes_preds[..., 2:3]/2
        box1_y2=boxes_preds[..., 1:2]+boxes_preds[..., 3:4]/2
        box2_x1=boxes_labels[..., 0:1]-boxes_labels[..., 2:3]/2
        box2_y1=boxes_labels[..., 1:2]-boxes_labels[..., 3:4]/2
        box2_x2=boxes_labels[..., 0:1]+boxes_labels[..., 2:3]/2
        box2_y2=boxes_labels[..., 1:2]+boxes_labels[..., 3:4]/2
        
    if box_format == "corners":
        box1_x1=boxes_preds[...,0:1]
        box1_y1=boxes_preds[...,1:2]
        box1_x2=boxes_preds[...,2:3]
        box1_y2=boxes_preds[...,3:4]  
        box2_x1=boxes_labels[...,0:1]
        box2_y1=boxes_labels[...,1:2]
        box2_x2=boxes_labels[...,2:3]
        box2_y2=boxes_labels[...,3:4]
        
    x1=torch.max(box1_x1,box2_x1)
    y1=torch.max(box1_y1,box2_y1)
    x2=torch.min(box1_x2,box2_x2)
    y2=torch.min(box1_y2,box2_y2)
    
    intersection=(x2-x1).clamp(0)*(y2-y1).clamp(0)
    
    box1_area=abs((box1_x2-box1_x1)*(box1_y2-box1_y1))
    box2_area=abs((box2_x2-box2_x1)*(box2_y2-box2_y1))
    
    return intersection/(box1_area+box2_area-intersection)

import torch

def convert_cellboxes(predictions, S=7):
    predictions = predictions.to("cpu")
    batch_size = predictions.shape[0]
    predictions = predictions.reshape(batch_size, 7, 7, 30)
    bboxes1 = predictions[..., 21:25]
    bboxes2 = predictions[..., 26:30]
    scores = torch.cat(
        (predictions[..., 20].unsqueeze(0), predictions[..., 25].unsqueeze(0)), dim=0
    )
    best_box = scores.argmax(0).unsqueeze(-1)
    best_boxes = bboxes1 * (1 - best_box) + best_box * bboxes2
    cell_indices = torch.arange(7).repeat(batch_size, 7, 1).unsqueeze(-1)
    x = 1 / S * (best_boxes[..., :1] + cell_indices)
    y = 1 / S * (best_boxes[..., 1:2] + cell_indices.permute(0, 2, 1, 3))
    w_y = 1 / S * best_boxes[..., 2:4]
    converted_bboxes = torch.cat((x, y, w_y), dim=-1)
    predicted_class = predictions[..., :20].argmax(-1).unsqueeze(-1)
    best_confidence = torch.max(predictions[..., 20], predictions[..., 25]).unsqueeze(
        -1
    )
    converted_preds = torch.cat(
        (predicted_class, best_confidence, converted_bboxes), dim=-1
    )

    return converted_preds

import torch

def convert_cellboxes_intuitive(predictions, S=7):
    batch_size = predictions.shape[0]
    predictions = predictions.reshape(batch_size, S, S, 30) 
    
    converted_preds = []
    
    for b in range(batch_size):
        image_preds = []  
        for i in range(S):
            row_preds = []  
            for j in range(S):
                cell_pred = predictions[b, i, j]
                
                class_probs = cell_pred[:20]
                predicted_class = torch.argmax(class_probs).unsqueeze(0)
                
                confidence1, confidence2 = cell_pred[20], cell_pred[25]
                box1, box2 = cell_pred[21:25], cell_pred[26:30]
                
                if confidence1 > confidence2:
                    best_box = box1
                    best_confidence = confidence1
                else:
                    best_box = box2
                    best_confidence = confidence2
                
                x, y, w, h = best_box
                x = (x + j) / S
                y = (y + i) / S  
                w = w / S        
                h = h / S        
                cell_result = torch.cat([
                    predicted_class.float(),
                    best_confidence.unsqueeze(0),
                    torch.tensor([x, y, w, h])
                ])
                
                row_preds.append(cell_result) 
            
            row_preds = torch.stack(row_preds)
            image_preds.append(row_preds)
        image_preds = torch.stack(image_preds)
        converted_preds.append(image_preds)
    
    return torch.stack(converted_preds)

def cellboxes_to_boxes(out,S=7):
    #intuitive function is easier to understand but takes more time
    converted_preds=convert_cellboxes_intuitive(out).reshape(out.shape[0],S*S,-1)
    converted_preds[...,0]=converted_preds[...,0].long()
    all_bboxes=[]
    
    for sample_idx in range(out.shape[0]):
        bboxes=[]

        for bbox_idx in range(S*S):
            bboxes.append([x.item() for x in converted_preds[sample_idx,bbox_idx,:]])
        all_bboxes.append(bboxes)

    return all_bboxes

def nms(bboxes,iou_threshold,threshold,box_format="corners"):
    assert type(bboxes)==list
    bboxes=[box for box in bboxes if box[1]>threshold]
    bboxes=sorted(bboxes,key=lambda x:x[1],reverse=True)
    bboxes_after_nms=[]
    
    while bboxes:
        selected_box=bboxes.pop(0)
        bboxes=[
            box for box in bboxes
            if box[0]!=selected_box[0] or iou(torch.tensor(selected_box[2:],torch.tensor(box[2:])),box_format=box_format)<iou_threshold
        ]
        
        bboxes_after_nms.append(selected_box)
        
    return bboxes_after_nms

def plot_image(image, boxes):
    """Plots predicted bounding boxes on the image"""
    im = np.array(image)
    height, width, _=im.shape
    fig,ax=plt.subplots(1)
    ax.imshow(im)
    for box in boxes:
        box = box[2:]
        assert len(box) == 4, "Got more values than in x, y, w, h, in a box!"
        upper_left_x = box[0] - box[2] / 2
        upper_left_y = box[1] - box[3] / 2
        rect = patches.Rectangle(
            (upper_left_x * width, upper_left_y * height),
            box[2] * width,
            box[3] * height,
            linewidth=1,
            edgecolor="r",
            facecolor="none",
        )
        ax.add_patch(rect)

    plt.show()
    
def get_bboxes(loader,model,iou_threshold,threshold,pred_format="cell",box_format="midpoint",device="cuda"):
    all_pred_bboxes=[]
    all_target_bboxes=[]
    
    model.eval()
    train_idx=0
    
    for batch_idx,(x,labels) in enumerate(loader):
        x=x.to(device)
        labels=labels.to(device)
        
        with torch.no_grad():
            predictions=model(x)
            
        batch_size=x.shape[0]
        target_bboxes=cellboxes_to_boxes(labels)
        pred_bboxes=cellboxes_to_boxes(predictions)
        
        for idx in range(batch_size):
            nms_boxes=nms(pred_bboxes[idx],iou_threshold,threshold,box_format)
            
            for nms_box in nms_boxes:
                all_pred_bboxes.append([train_idx]+nms_box)
                
            for box in target_bboxes[idx]:
                if box[1]>threshold:
                    all_target_bboxes.append([train_idx]+box)
                
            train_idx+=1
        
    model.train()
    return all_pred_bboxes,all_target_bboxes

def mAP(pred_bboxes,target_bboxes,iou_threshold=0.5,box_format="midpoint",num_classes=20):
    avg_precisions=[]
    eps=1e-6
    
    for c in range(num_classes):
        detections=[]
        ground_truths=[]
        
        for detection in pred_bboxes:
            if detection[1]==c:
                detections.append(detection)
                
        for target in target_bboxes:
            if target[1]==c:
                ground_truths.append(target)
                
        amount_bboxes=Counter([gt[0] for gt in ground_truths])
        
        for key,val in amount_bboxes.items():
            amount_bboxes[key]=torch.zeros(val)
            
        detections.sort(key=lambda x:x[2],reverse=True)
        TP=torch.zeros(len(detections))
        FP=torch.zeros(len(detections))
        total_targets=len(ground_truths)
        
        if total_targets==0:
            continue
        
        for detection_idx,detection in enumerate(detections):
            ground_truth_img=[bbox for bbox in ground_truths if bbox[0]==detection[0]]
            num_ground_truths=len(ground_truth_img)
            best_iou=0.0
            
            for idx,gt in enumerate(ground_truth_img):
                IOU=iou(torch.tensor(detection[3:]),torch.tensor(gt[3:]),box_format=box_format)
                if IOU>best_iou:
                    best_iou=IOU
                    best_gt_idx=idx 
            
            if best_iou>iou_threshold:
                if amount_bboxes[detections[0]][best_gt_idx]==0:
                    TP[detection_idx]=1
                    amount_bboxes[detections[0]][best_gt_idx]=1
                else:
                    FP[detection_idx]=1
            
            else:
                FP[detection_idx]=1
                
        TP_cumsum=torch.cumsum(TP,dim=0)
        FP_cumsum=torch.cumsum(FP,dim=0)
        recall=TP_cumsum/(total_targets+eps)
        precision=torch.divide(TP_cumsum,(TP_cumsum+FP_cumsum+eps))
        recall=torch.cat((torch.tensor([0]),recall))
        precision=torch.cat((torch.tensor([1]),precision))
        avg_precisions.append(torch.trapz(precision,recall))
        
    return sum(avg_precisions)/len(avg_precisions)
        
def save_checkpoint(model,optimizer,filename="my_checkpoint.pth"):
    print("=> Saving checkpoint")
    checkpoint={
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint,filename)

def load_checkpoint(checkpoint_file,model,optimizer,lr):
    print("=> Loading checkpoint")
    checkpoint=torch.load(checkpoint_file,map_location="cuda")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    for param_group in optimizer.param_groups:
        param_group["lr"]=lr
    
    
if __name__ == "__main__":
    batch_size = 2  
    S = 7         
    predictions = torch.randn((batch_size, S * S * 30))  

    output_original = convert_cellboxes(predictions, S)
    output_intuitive = convert_cellboxes_intuitive(predictions, S)
    
    print(output_original.shape)
    print(output_intuitive.shape)

    if torch.allclose(output_original, output_intuitive, atol=1e-5):
        print("The outputs of both functions are nearly identical.")
    else:
        print("The outputs differ.")

    print("Difference between outputs:")
    print(torch.abs(output_original - output_intuitive))
