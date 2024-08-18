import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

lr=0.0005
batch_size=256
num_epochs=30
num_classes=1

def get_dataloaders_mnist(batch_size, num_workers=0, train_transforms=None, test_transforms=None):
    if train_transforms is None:
        train_transforms = transforms.ToTensor()
    if test_transforms is None:
        test_transforms = transforms.ToTensor()
    train_dataset = datasets.MNIST(root='data',train=True,transform=train_transforms,download=True)
    valid_dataset = datasets.MNIST(root='data',train=True,transform=test_transforms)
    test_dataset = datasets.MNIST(root='data',train=False,transform=test_transforms)
    train_loader = DataLoader(dataset=train_dataset,batch_size=batch_size,num_workers=num_workers,shuffle=True)
    valid_loader = DataLoader(dataset=valid_dataset,batch_size=batch_size,num_workers=num_workers,shuffle=False)
    test_loader = DataLoader(dataset=test_dataset,batch_size=batch_size,num_workers=num_workers,shuffle=False)
    return train_loader, valid_loader, test_loader

train_loader, valid_loader, test_loader = get_dataloaders_mnist(batch_size,num_workers=2)

class VariationalAutoEncoder(nn.Module):
    def __init__(self,device,input_dim=784,hidden_dim=400,latent_dim=200):
        self.device = device
        self.encoder = nn.Sequential(
            nn.Linear(input_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,latent_dim),
            nn.ReLU()
        )
        
        self.mean_layer=nn.Linear(latent_dim,2)
        self.std_layer=nn.Linear(latent_dim,2)
        
        self.decoder= nn.Sequential(
            nn.Linear(2,latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,input_dim),
            nn.Sigmoid()
        )
        
    def encode(self,x):
        x=self.encoder(x)
        mean,std=self.mean_layer(x),self.std_layer(x)
        return mean,std
    
    def reparameterization(self,mean,std):
        epsilon=torch.randn(std).to(self.device)
        z=mean+std*epsilon
        return z
    
    def decode(self,x):
        return self.decoder(x)
    
    def forward(self,x):
        mean,std=self.encode(x)
        z=self.reparameterization(mean,std)
        x_hat=self.decode(z)
        return x_hat, mean, std

def loss_function(x, x_hat, mean, std):
    reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
    KLD = - 0.5 * torch.sum(1+ std - mean.pow(2) - std.exp())

    return reproduction_loss + KLD

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = VariationalAutoEncoder(device).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

train_losses = []
valid_losses = []
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    train_loader_iter = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} (Training)', leave=False)
    
    for images, _ in train_loader_iter:
        images = images.to(device)
        outputs, mean, std = model(images)
        loss = loss_function(images, outputs, mean, std)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        train_loader_iter.set_postfix({'Loss': loss.item()})
        
    train_loss = running_loss / len(train_loader.dataset)
    train_losses.append(train_loss)

    model.eval()
    running_loss = 0.0
    valid_loader_iter = tqdm(valid_loader, desc=f'Epoch {epoch+1}/{num_epochs} (Validation)', leave=False)
    
    with torch.no_grad():
        for images, _ in valid_loader_iter:
            images = images.to(device)
            outputs, mean, std = model(images)
            loss = loss_function(images, outputs, mean, std)

            running_loss += loss.item() * images.size(0)
            valid_loader_iter.set_postfix({'Loss': loss.item()})
            
    valid_loss = running_loss / len(valid_loader.dataset)
    valid_losses.append(valid_loss)

    print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}')

plt.plot(train_losses, label='Training Loss')
plt.plot(valid_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
