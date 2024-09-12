import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.utils.data import DataLoader, random_split
from dataset import RCNNDataset
from utils import save_checkpoint,load_checkpoint
from tqdm import tqdm

checkpoint="rcnn_model.pth"
device="cuda" if torch.cuda.is_available() else "cpu"
save_model=True
load_model=False
learning_rate=1e-3
batch_size=64
num_workers=4
num_epochs=100
image_path=""
annotations_path=""
best_val_acc=0.0

def get_model():
    vgg=models.vgg16(pretrained=True)
    for param in vgg.features.parameters():
        param.requires_grad=False
        
    num_features=vgg.classifier[6].in_features
    vgg.classifier[6]=nn.Linear(num_features,1)
    return vgg

def train(model,train_loader,val_loader,optimizer,criterion): 
    loop1=tqdm(train_loader,leave=True)
    total_loss=0.0
    total=0
    correct=0
    
    model.train()
    for batch_idx,(img,labels) in enumerate(loop1):
        img=img.to(device)
        labels=labels.to(device)
        labels=2*labels-1
        
        optimizer.zero_grad()
        outputs=model(img)
        loss=criterion(outputs,labels)
        loss.backward()
        optimizer.step()
        
        total_loss+=loss.item()
        predicted=(outputs>0).float()
        total+=labels.size(0)
        correct+=(predicted==labels).sum().item()
        
    train_loss=total_loss/len(train_loader)
    train_acc=correct/total
    
    loop2=tqdm(val_loader,leave=True)
    total_loss=0.0
    total=0
    correct=0
    
    model.eval()
    with torch.no_grad():
        for batch_idx,(img,labels) in enumerate(loop2):
            img=img.to(device)
            labels=labels.to(device)
            labels=2*labels-1
            
            outputs=model(img)
            loss=criterion(outputs,labels)
            
            total_loss+=loss.item()
            predicted=(outputs>0).float()
            total+=labels.size(0)
            correct+=(predicted==labels).sum().item()
        
    val_loss=total_loss/len(train_loader)
    val_acc=correct/total    
    
    return train_loss,train_acc,val_loss,val_acc

def main():
    dataset=RCNNDataset(image_path, annotations_path)
    train_size=int(0.95*len(dataset))
    val_size=len(dataset)-train_size
    train_dataset,val_dataset=random_split(dataset,[train_size, val_size])

    train_loader=DataLoader(train_dataset,batch_size=32,shuffle=True)
    val_loader=DataLoader(val_dataset,batch_size=32,shuffle=False)

    model=get_model().to(device)
    criterion=nn.HingeEmbeddingLoss()
    optimizer=optim.Adam(model.parameters())
    
    if load_model:
        load_checkpoint(checkpoint,model,optimizer,learning_rate)
    
    for epoch in range(num_epochs):
        train_loss,train_acc,val_loss,val_acc=train(model,train_loader,val_loader,optimizer,criterion)
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        
        if save_model and val_acc>best_val_acc:
            best_val_acc=val_acc
            save_checkpoint(model,optimizer,checkpoint)

if __name__ == '__main__':
    main()