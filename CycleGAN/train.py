import torch
from dataset import HorseZebraDataset
import sys
from utils import save_checkpoint, load_checkpoint
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torchvision.utils import save_image
from model import Discriminator,Generator
import albumentations as A
from albumentations.pytorch import ToTensorV2

device = "cuda" if torch.cuda.is_available() else "cpu"
train_dir = "data/train"
batch_size = 1
lr = 1e-5
lambda_identity = 0.0
lambda_cycle = 10
num_workers = 4
num_epochs = 10
load_model = False
save_model = True
checkpoint_gen_h = "genh.pth"
checkpoint_gen_z = "genz.pth"
checkpoint_disc_h = "disch.pth"
checkpoint_disc_z = "discz.pth"

transforms = A.Compose(
    [
        A.Resize(width=256, height=256),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
        ToTensorV2(),
    ],
    additional_targets={"image0": "image"},
)


def train(disc_h,disc_z,gen_h,gen_z,loader,opt_disc,opt_gen,l1,mse,d_scaler,g_scaler):
    loop=tqdm(loader,leave=True)
    
    for idx,(zebra,horse) in enumerate(loop):
        zebra=zebra.to(device)
        horse=horse.to(device)
        
        with torch.cuda.amp.autocast():
            fake_horse=gen_h(zebra)
            d_h_real=disc_h(horse)
            d_h_fake=disc_h(fake_horse.detach())
            d_h_real_loss=mse(d_h_real,torch.ones_like(d_h_real))
            d_h_fake_loss=mse(d_h_fake,torch.zeros_like(d_h_fake))
            d_h_loss=d_h_real_loss+d_h_fake_loss 
            
            fake_zebra=gen_z(horse)
            d_z_real=disc_z(zebra)
            d_z_fake=disc_z(fake_zebra.detach())
            d_z_real_loss=mse(d_z_real,torch.ones_like(d_z_real))
            d_z_fake_loss=mse(d_z_fake,torch.zeros_like(d_z_fake))
            d_z_loss=d_z_real_loss+d_z_fake_loss 
            
            d_loss=(d_h_loss+d_z_loss)/2
        
        opt_disc.zero_grad()
        d_scaler.scale(d_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()
        
        with torch.cuda.amp.autocast():
            d_h_fake=disc_h(fake_horse)
            d_z_fake=disc_z(fake_zebra)
            loss_g_h=mse(d_h_fake,torch.ones_like(d_h_fake))
            loss_g_z=mse(d_z_fake,torch.ones_like(d_z_fake))
            
            cycle_zebra=gen_z(fake_horse)
            cycle_horse=gen_h(fake_zebra)
            cycle_h_loss=l1(horse,cycle_horse)
            cycle_z_loss=l1(zebra,cycle_zebra)
            
            identity_h=gen_h(horse)
            identity_z=gen_z(zebra)
            identity_h_loss=l1(horse,identity_h)
            identity_z_loss=l1(zebra,identity_z)
            
            g_loss=(loss_g_h+loss_g_z+
                    cycle_h_loss*lambda_cycle+cycle_z_loss*lambda_cycle+
                    identity_h_loss*lambda_identity+identity_z_loss*lambda_identity)
           
        opt_gen.zero_grad()
        g_scaler.scale(g_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()         

def main():
    disc_h=Discriminator(in_channels=3).to(device)
    disc_z=Discriminator(in_channels=3).to(device)
    gen_h=Generator(image_channels=3).to(device)
    gen_z=Generator(image_channels=3).to(device)
    opt_disc=optim.Adam(list(disc_h.parameters()+disc_z.parameters()),lr=lr,betas=(0.5,0.999))
    opt_gen=optim.Adam(list(gen_h.parameters()+gen_z.parameters()),lr=lr,betas=(0.5,0.999))
    
    l1_loss=nn.L1Loss()
    mse_loss=nn.MSELoss()
    
    if load_model:
        load_checkpoint(checkpoint_gen_h,device,gen_h,opt_gen,lr)
        load_checkpoint(checkpoint_gen_z,device,gen_z,opt_gen,lr)
        load_checkpoint(checkpoint_disc_h,device,disc_h,opt_disc,lr)
        load_checkpoint(checkpoint_disc_z,device,disc_z,opt_disc,lr)
    
    dataset=HorseZebraDataset(train_dir+"zebras",train_dir+"/horses",transforms)
    train_loader = DataLoader(dataset,batch_size=batch_size,shuffle=True,num_workers=num_workers,pin_memory=True)
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()
    
    for epoch in range(num_epochs):
        train(disc_h,disc_z,gen_h,gen_z,train_loader,opt_disc,opt_gen,l1_loss,mse_loss,d_scaler,g_scaler)
        
        if save_model:
            save_checkpoint(disc_h,opt_disc,checkpoint_disc_h)
            save_checkpoint(disc_z,opt_disc,checkpoint_disc_z)
            save_checkpoint(gen_h,opt_gen,checkpoint_gen_h)
            save_checkpoint(gen_z,opt_gen,checkpoint_gen_z)
        
    
if __name__=="__main__":
    main()