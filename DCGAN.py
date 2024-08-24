import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm

class Discriminator(nn.Module):
    def __init__(self,image_channels,features_d):
        super().__init__()
        self.disc=nn.Sequential(
            nn.Conv2d(image_channels,features_d,kernel_size=4,stride=2,padding=1),
            nn.LeakyReLU(0.2),
            self.block(features_d,features_d*2,4,2,1),
            self.block(features_d*2,features_d*4,4,2,1),
            self.block(features_d*4,features_d*8,4,2,1),
            nn.Conv2d(features_d*8,1,kernel_size=4,stride=2,padding=0),
            nn.Sigmoid()
        )
        
    def block(self,in_channels,out_channels,kernel_size,stride,padding):
        return nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding,bias=False),
            nn.BatchNorm2d(out_channels),
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
lr=3e-4
z_dim=100
batch_size=32
epochs=50
image_channels=1
features_d=64
features_g=64
image_size=64

disc=Discriminator(image_channels,features_d).to(device)
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
opt_gen = optim.Adam(gen.parameters(), lr=lr, betas=(0.5, 0.999))
opt_disc = optim.Adam(disc.parameters(), lr=lr, betas=(0.5, 0.999))
criterion = nn.BCELoss()

gen.train()
disc.train()

for epoch in tqdm(range(epochs), desc='Epochs'):
    for batch_idx, (real, _) in tqdm(enumerate(loader), desc="Batches", leave=False):
        real=real.to(device)
        b_size=real.shape[0]
        noise=torch.randn(b_size,z_dim,1,1).to(device)
        
        fake = gen(noise)
        disc_real = disc(real).view(-1)
        lossD_real = criterion(disc_real, torch.ones_like(disc_real))
        disc_fake = disc(fake).view(-1)
        lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
        lossD = (lossD_real + lossD_fake) / 2
        disc.zero_grad()
        lossD.backward()
        opt_disc.step()
        
        output = disc(fake).view(-1)
        lossG = criterion(output, torch.ones_like(output))
        gen.zero_grad()
        lossG.backward()
        opt_gen.step()
        