import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from utils import save_checkpoint,load_checkpoint,DDPMScheduler
from model import UNet
from tqdm import tqdm

checkpoint="ddpm_diffusion_model.pth"
device="cuda" if torch.cuda.is_available() else "cpu"
save_model=True
load_model=False
learning_rate=1e-3
batch_size=64
num_workers=4
channels=[64,128,256,512,512,384]
attentions=[False, True, False, False, False, True]
upscales=[False, False, False, True, True, True],
num_groups=32
dropout=0.1
num_heads=8
input_channels=1
output_channels=1
time_steps=1000
num_epochs=100

def train(loader,model,scheduler,optimizer,criterion):
    loop=tqdm(loader,leave=True)
    total_loss=0.0
    total=0
    for batch_idx,(img,_) in enumerate(loop):
        img=img.to(device)
        image=F.pad(img,(2,2,2,2))
        t=torch.randint(0,time_steps,(batch_size,))
        noise_img=torch.randn_like(img,requires_grad=False)
        a=scheduler.alpha[t].view(batch_size,1,1,1).to(device)
        img=(torch.sqrt(a)*img)+(torch.sqrt(1-a)*noise_img)
        output=model(img,t)
        optimizer.zero_grad()
        loss=criterion(output,noise_img)
        total_loss+=loss.item()
        loss.backward() 
        optimizer.step()   
    print(f'Loss: {total_loss}')

def main():
    train_dataset = datasets.MNIST(root='./data', train=True, download=False,transform=transforms.ToTensor())
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers)
    
    model=UNet(channels,attentions,upscales,num_groups,dropout,num_heads,input_channels,output_channels,time_steps).to(device)
    scheduler=DDPMScheduler(time_steps)
    optimizer=optim.Adam(model.parameters(),lr=learning_rate)
    
    if load_model:
        load_checkpoint(checkpoint,model,optimizer,learning_rate)
    
    criterion=nn.MSELoss(reduction='mean')
    
    for epoch in range(num_epochs):
        train(train_loader,model,scheduler,optimizer,criterion)
        
        if save_model:
            save_checkpoint(model,optimizer,checkpoint)

if __name__ == "__main__":
    main()