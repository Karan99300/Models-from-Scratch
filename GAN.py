import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm

class Discriminator(nn.Module):
    def __init__(self,image_dim):
        super().__init__()
        self.discriminator=nn.Sequential(
            nn.Linear(image_dim,128),
            nn.LeakyReLU(),
            nn.Linear(128,32),
            nn.LeakyReLU(),
            nn.Linear(32,1),
            nn.Sigmoid()
        )
    
    def forward(self,x):
        return self.discriminator(x)
    
class Generator(nn.Module):
    def __init__(self,z_dim,image_dim):
        super().__init__()
        self.generator=nn.Sequential(
            nn.Linear(z_dim,256),
            nn.LeakyReLU(),
            nn.Linear(256,512),
            nn.LeakyReLU(),
            nn.Linear(512,image_dim),
            nn.Tanh()
        )
        
    def forward(self,x):
        return self.generator(x)

device="cuda" if torch.cuda.is_available() else "cpu"
lr=3e-4
z_dim=64
image_dim=1*28*28
batch_size=32
epochs=50

disc=Discriminator(image_dim).to(device)
gen=Generator(z_dim,image_dim).to(device)
transforms = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)

transforms = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)

dataset = datasets.MNIST(root="dataset/", transform=transforms, download=True)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
opt_disc = optim.Adam(disc.parameters(), lr=lr)
opt_gen = optim.Adam(gen.parameters(), lr=lr)
criterion = nn.BCELoss()

for epoch in tqdm(range(epochs), desc='Epochs'):
    for batch_idx, (real, _) in tqdm(enumerate(loader), desc="Batches", leave=False):
        real=real.view(-1,784).to(device)
        b_size=real.shape[0]
        noise=torch.randn(b_size,z_dim).to(device)
        
        noise = torch.randn(batch_size, z_dim).to(device)
        fake = gen(noise)
        disc_real = disc(real).view(-1)
        lossD_real = criterion(disc_real, torch.ones_like(disc_real))
        disc_fake = disc(fake).view(-1)
        lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
        lossD = (lossD_real + lossD_fake) / 2
        disc.zero_grad()
        lossD.backward(retain_graph=True)
        opt_disc.step()
        
        output = disc(fake).view(-1)
        lossG = criterion(output, torch.ones_like(output))
        gen.zero_grad()
        lossG.backward()
        opt_gen.step()
        