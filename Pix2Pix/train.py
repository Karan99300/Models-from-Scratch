import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from model import Generator,Discriminator
from dataset import MapDataset
from utils import save_checkpoint,load_checkpoint
from tqdm import tqdm

device="cuda" if torch.cuda.is_available() else "cpu"
train_dir="train_dir"
lr=2e-4
batch_size=16
num_workers=2
image_size=256
image_channels=3
l1_lambda=100
lambda_gp=10
num_epochs=500
load_model=False
save_model=False
checkpoint_disc="disc.pth"
checkpoint_gen="gen.pth"

def train(disc,gen,loader,opt_disc,opt_gen,l1_loss, bce, g_scaler, d_scaler,):
    loop=tqdm(loader,leave=True)
    
    for idx,(X,y) in enumerate(loop):
        X=X.to(device)
        y=y.to(device)
        
        with torch.cuda.amp.autocast():
            y_fake=gen(X)
            D_real=disc(X,y)
            D_real_loss=bce(D_real,torch.ones_like(D_real))
            D_fake=disc(X,y_fake)
            D_fake_loss=bce(D_fake,torch.zeros_like(D_fake))
            D_loss=(D_real_loss+D_fake_loss)/2
            
        disc.zero_grad()
        d_scaler.scale(D_loss).backward(retain_graph=True)
        d_scaler.update(opt_disc)
        d_scaler.update()
        
        with torch.cuda.amp.autocast():
            D_fake=disc(X,y_fake)
            G_fake_loss=bce(D_fake,torch.ones_like(D_fake))
            l1=l1_loss(y_fake,y)*l1_lambda
            G_loss=G_fake_loss+l1
        
        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()
        
        if idx % 10 == 0:
            loop.set_postfix(
                D_real=torch.sigmoid(D_real).mean().item(),
                D_fake=torch.sigmoid(D_fake).mean().item(),
            )


def main():
    disc=Discriminator(image_channels).to(device)
    gen=Generator(image_channels,features=64).to(device)
    opt_disc=optim.Adam(disc.parameters(),lr=lr,betas=(0.5, 0.999))
    opt_gen=optim.Adam(gen.parameters(),lr=lr,betas=(0.5, 0.999))
    bce=nn.BCEWithLogitsLoss()
    l1_loss=nn.L1Loss()
    
    if load_model:
        load_checkpoint(checkpoint_disc,device,disc,opt_disc,lr)
        load_checkpoint(checkpoint_gen,device,gen,opt_gen,lr)
    
    train_dataset=MapDataset(train_dir)
    train_loader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=num_workers)
    
    #Float 16 training uses less VRAM
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()
    
    for epoch in range(num_epochs):
        train(disc,gen,train_loader,opt_disc,opt_gen,l1_loss,bce,g_scaler,d_scaler)
        
        if save_model and epoch%5==0:
            save_checkpoint(disc,opt_disc,checkpoint_disc)
            save_checkpoint(gen,opt_gen,checkpoint_gen)
        
if __name__=="__main__":
    main()