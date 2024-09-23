import torch
import torch.optim as optim
from tqdm import tqdm
from model import YOLO
from utils import *
from loss import YOLOLoss

learning_rate=2e-5
device="cuda" if torch.cuda.is_available else "cpu"
batch_size=64
weight_decay=0
num_epochs=1000
num_workers=2
pin_memory=True
load_model=False
save_model=True
checkpoint_file="model.pth"
img_dir=None
label_dir=None

def train(loader,model,optimizer,criterion):
    loop=tqdm(loader,leave=True)
    train_loss=[]
    
    for _,(x,y) in enumerate(loop):
        x=x.to(device)
        y=y.to(device)
        
        out=model(x)
        loss=criterion(out,y)
        train_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loop.set_postfix(loss=loss.item())
        
    print(f"Mean loss was {sum(train_loss)/len(train_loss)}")
  
def main():
    """
    train_dataset = VOCDataset(
        "data/100examples.csv",
        transform=transform,
        img_dir=img_dir,
        label_dir=label_dir,
    )

    test_dataset = VOCDataset(
        "data/test.csv", transform=transform, img_dir=img_dir, label_dir=label_dir,
    )
    """

    train_loader=None
    test_loader=None
    
    model=YOLO(in_channels=3,S=7,B=2,N=20).to(device)
    optimizer=optim.Adam(model.parameters(),lr=learning_rate,weight_decay=weight_decay)
    criterion=YOLOLoss()

    if load_model:
        load_checkpoint(checkpoint_file,model,optimizer,learning_rate)
    
    for epoch in range(num_epochs):
        best_mean_avg_prec=0.0
        for x,y in train_loader:
            x=x.to(device)
            for idx in range(8):
                bboxes=cellboxes_to_boxes(model(x))
                bboxes=nms(bboxes[idx],iou_threshold=0.5,threshold=0.4,box_format="midpoint")
                plot_image(x[idx].permute(1,2,0).to("cpu"),bboxes)
            
        import sys 
        sys.exit()
        
        pred_bboxes,target_bboxes=get_bboxes(train_loader,model,iou_threshold=0.5,threshold=0.4)
        mean_avg_prec=mAP(pred_bboxes,target_bboxes,iou_threshold=0.5,box_format="midpoint")
        print(f"Train mAP: {mean_avg_prec}")
        
        if mean_avg_prec>best_mean_avg_prec and save_model:
            best_mean_avg_prec=mean_avg_prec
            save_checkpoint(model,optimizer,checkpoint_file)
        
        train(train_loader,model,optimizer,criterion)
        
if __name__ == '__main__':
    main()