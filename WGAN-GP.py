import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm

class Critic(nn.Module):
    def __init__(self,image_channels,features_d):
        super().__init__()
        self.disc=nn.Sequential(
            nn.Conv2d(image_channels,features_d,kernel_size=4,stride=2,padding=1),
            nn.LeakyReLU(0.2),
            self.block(features_d,features_d*2,4,2,1),
            self.block(features_d*2,features_d*4,4,2,1),
            self.block(features_d*4,features_d*8,4,2,1),
            nn.Conv2d(features_d*8,1,kernel_size=4,stride=2,padding=0),
        )
        
    def block(self,in_channels,out_channels,kernel_size,stride,padding):
        return nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding,bias=False),
            nn.InstanceNorm2d(out_channels,affine=True),
            nn.LeakyReLU(0.2)
        )
    
    def forward(self,x):
        return self.disc(x)
    
class Generator(nn.Module):
    def __init__(self,z_dim,image_channels,features_g):
        super().__init__()
        self.gen=nn.Sequential(
            self.block(z_dim,features_g*16,4,1,0),
            self.block(features_g*16,features_g*8,4,2,1),
            self.block(features_g*8,features_g*4,4,2,1),
            self.block(features_g*4,features_g*2,4,2,1),
            nn.ConvTranspose2d(features_g*2,image_channels,kernel_size=4,stride=2,padding=1),
            nn.Tanh()
        )
        
    def block(self,in_channels,out_channels,kernel_size,stride,padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels,out_channels,kernel_size,stride,padding,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        
    def forward(self,x):
        return self.gen(x)
    
device="cuda" if torch.cuda.is_available() else "cpu"
lr=1e-4
z_dim=100
batch_size=64
epochs=50
image_channels=1
features_d=64
features_g=64
image_size=64
critic_iteration=5
lambda_gp=10

critic=Critic(image_channels,features_d).to(device)
gen=Generator(z_dim,image_channels,features_g).to(device)
transforms = transforms.Compose(
    [
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5 for _ in range(image_channels)], [0.5 for _ in range(image_channels)]
        ),
    ]
)

dataset = datasets.MNIST(root="dataset/", transform=transforms, download=True)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
opt_gen = optim.Adam(gen.parameters(), lr=lr, betas=(0.1, 0.9))
opt_critic = optim.Adam(critic.parameters(), lr=lr, betas=(0.1, 0.9))

gen.train()
critic.train()

def gradient_penalty(critic,real,fake,device="cpu"):
    b,c,h,w,=real.shape
    epsilon=torch.rand((b,1,1,1)).repeat(1,c,h,w).to(device)
    interpolated_images=real*epsilon+fake*(1-epsilon)
    mixed_scores=critic(interpolated_images)
    gradient=torch.autograd.grad(inputs=interpolated_images,outputs=mixed_scores,grad_outputs=torch.ones_like(mixed_scores),create_graph=True,retain_graph=True)[0]
    gradient=gradient.view(gradient.shape[0],-1)
    gradient_norm=gradient.norm(2,dim=-1)
    gradient_penalty=torch.mean((gradient_norm-1)**2)
    return gradient_penalty

for epoch in tqdm(range(epochs), desc='Epochs'):
    for batch_idx, (real, _) in tqdm(enumerate(loader), desc="Batches", leave=False):
        real=real.to(device)
        b_size=real.shape[0]
        
        for _ in range(critic_iteration):
            noise=torch.randn(b_size,z_dim,1,1).to(device)
            fake = gen(noise)
            critic_real=critic(real).reshape(-1)
            critic_fake=critic(fake).reshape(-1)
            gp=gradient_penalty(critic,real,fake,device)
            loss_critic=-(torch.mean(critic_real)-torch.mean(critic_fake))+lambda_gp*gp
            critic.zero_grad()
            loss_critic.backward(retain_graph=True)
            opt_critic.step()
            
        output=critic(fake).reshape(-1)
        loss_gen=-torch.mean(output)
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()