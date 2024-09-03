import torch
import config
from torch import nn
from torch import optim
from utils import load_checkpoint,save_checkpoint
from loss import VGGLoss
from torch.utils.data import DataLoader
from model import Generator, Discriminator
from tqdm import tqdm
from dataset import MyImageFolder

torch.backends.cudnn.benchmark = True

def train(loader,disc,gen,opt_gen,opt_disc,mse,bce,vgg_loss):
    loop=tqdm(loader,leave=True)
    
    for idx,(low_res,high_res) in enumerate(loop):
        low_res=low_res.to(config.device)
        high_res=high_res.to(config.device)
        
        fake=gen(low_res)
        disc_real=disc(high_res)
        disc_fake=disc(fake.detach())
        disc_loss_real=bce(disc_real,torch.ones_like(disc_real)-0.1*torch.rand_like(disc_real))
        disc_loss_fake=bce(disc_fake,torch.zeros_like(disc_fake))
        disc_loss=disc_loss_real+disc_loss_fake
        
        opt_disc.zero_grad()
        disc_loss.backward()
        opt_disc.step()
        
        disc_fake=disc(fake)
        adverserial_loss=1e-3*bce(disc_fake,torch.ones_like(disc_fake))
        content_loss=0.006*vgg_loss(fake,high_res)
        gen_loss=content_loss+adverserial_loss
        
        opt_gen.zero_grad()
        gen_loss.backward()
        opt_gen.step()
        
def main():
    dataset=MyImageFolder(root_dir="new_data/")
    loader=DataLoader(dataset,batch_size=config.batch_size,shuffle=True,pin_memory=True,num_workers=config.num_workers)
    gen=Generator(in_channels=3).to(config.device)
    disc=Discriminator(img_channels=3).to(config.device)
    opt_gen=optim.Adam(gen.parameters(),lr=config.learning_rate,betas=(0.9,0.999))
    opt_disc=optim.Adam(disc.parameters(),lr=config.learning_rate,betas=(0.9,0.999))
    mse = nn.MSELoss()
    bce = nn.BCEWithLogitsLoss()
    vgg_loss = VGGLoss()
    
    if config.load_model:
        load_checkpoint(config.checkpoint_disc,disc,opt_disc,config.learning_rate)
        load_checkpoint(config.checkpoint_gen,gen,opt_gen,config.learning_rate)
    
    for epoch in range(config.NUM_EPOCHS):
        train(loader,disc,gen,opt_gen,opt_disc,mse,bce,vgg_loss)

        if config.save_model:
            save_checkpoint(gen,opt_gen,filename=config.checkpoint_gen)
            save_checkpoint(disc,opt_disc,filename=config.checkpoint_disc)

if __name__ == "__main__":
    main()