import torch
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils import gradient_penalty,save_checkpoint,load_checkpoint,plot_to_tensorboard
from model import Discriminator, Generator
from math import log2
from tqdm import tqdm

torch.backends.cudnn.benchmarks = True

start_training_at_img_size=128
dataset='celeb_dataset'
checkpoint_gen="generator.pth"
checkpoint_disc="disc.pth"
device="cuda" if torch.cuda.is_available() else "cpu"
save_model=True
load_model=False
learning_rate=1e-3
batch_sizes=[32, 32, 32, 16, 16, 16, 16, 8, 4]
img_channels= 3
z_dim=512
w_dim=512
in_channels=512
disc_iterations=1
lambda_gp=10
progressive_epochs=[30] * len(batch_sizes)
fixed_noise=torch.randn(8, z_dim, 1, 1).to(device)
num_workers=4

def get_loader(img_size):
    pass

def train(disc,gen,loader,dataset,step,alpha,opt_disc,opt_gen,tensorboard_step,writer,scaler_gen,scaler_disc):
    loop = tqdm(loader, leave=True)
    for batch_idx, (real, _) in enumerate(loop):
        real = real.to(device)
        cur_batch_size = real.shape[0]

        noise = torch.randn(cur_batch_size, z_dim, 1, 1).to(device)

        with torch.cuda.amp.autocast():
            fake = gen(noise, alpha, step)
            disc_real = disc(real, alpha, step)
            disc_fake = disc(fake.detach(), alpha, step)
            gp = gradient_penalty(disc, real, fake, alpha, step, device=device)
            loss_disc = (-(torch.mean(disc_real) - torch.mean(disc_fake))+ lambda_gp * gp+ (0.001 * torch.mean(disc_real ** 2)))

        opt_disc.zero_grad()
        scaler_disc.scale(loss_disc).backward()
        scaler_disc.step(opt_disc)
        scaler_disc.update()

        with torch.cuda.amp.autocast():
            gen_fake = disc(fake, alpha, step)
            loss_gen = -torch.mean(gen_fake)

        opt_gen.zero_grad()
        scaler_gen.scale(loss_gen).backward()
        scaler_gen.step(opt_gen)
        scaler_gen.update()

        alpha += cur_batch_size / ((progressive_epochs[step] * 0.5) * len(dataset))
        alpha = min(alpha, 1)

        if batch_idx % 500 == 0:
            with torch.no_grad():
                fixed_fakes = gen(fixed_noise, alpha, step) * 0.5 + 0.5
            plot_to_tensorboard(writer,loss_disc.item(),loss_gen.item(),real.detach(),fixed_fakes.detach(),tensorboard_step,)
            tensorboard_step += 1

        loop.set_postfix(gp=gp.item(),loss_critic=loss_disc.item(),)

    return tensorboard_step, alpha


def main():
    gen = Generator(z_dim,w_dim,in_channels).to(device)
    disc = Discriminator(in_channels).to(device)

    opt_gen = optim.Adam(gen.parameters(), lr=learning_rate, betas=(0.0, 0.99))
    opt_disc = optim.Adam(disc.parameters(), lr=learning_rate, betas=(0.0, 0.99))
    scaler_disc = torch.cuda.amp.GradScaler()
    scaler_gen = torch.cuda.amp.GradScaler()

    writer = SummaryWriter(f"logs")

    if load_model:
        load_checkpoint(checkpoint_gen,gen,opt_gen,learning_rate)
        load_checkpoint(checkpoint_disc,disc,opt_disc,learning_rate)

    gen.train()
    disc.train()
    
    tensorboard_step=0 
    step=int(log2(start_training_at_img_size/4))
    
    for num_epochs in progressive_epochs[step:]:
        alpha=0
        train_loader,train_dataset=get_loader(4*2**step)
        print(f"Image size:{4*2**step}")
        
        for epoch in range(num_epochs):
            tensorboard_step,alpha=train(disc,gen,train_loader,train_dataset,step,alpha,opt_disc,opt_gen,tensorboard_step,writer,scaler_gen,scaler_disc)

            if save_model:
                save_checkpoint(gen,opt_gen,checkpoint_gen)
                save_checkpoint(disc,opt_disc,checkpoint_disc)
        
        steps+=1
                        
if __name__ == '__main__':
    main()