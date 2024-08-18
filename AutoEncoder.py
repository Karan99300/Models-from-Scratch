import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import time
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler
from torch.utils.data import sampler
import numpy as np
import matplotlib.colors as mcolors

lr=0.0005
batch_size=256
num_epochs=30
num_classes=1

class Reshape(nn.Module):
    def __init__(self, *shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(x.size(0), *self.shape)
    
class Trim(nn.Module):
    def __init__(self, target_size=(28, 28)):
        super().__init__()
        self.target_size = target_size

    def forward(self, x):
        h, w = x.shape[2], x.shape[3]
        target_h, target_w = self.target_size
        start_h = (h - target_h) // 2
        start_w = (w - target_w) // 2
        return x[:, :, start_h:start_h + target_h, start_w:start_w + target_w]

class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=nn.Sequential(
            nn.Conv2d(1,32,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv2d(32,64,kernel_size=3,stride=2,padding=1),
            nn.ReLU(),
            nn.Conv2d(64,64,kernel_size=3,stride=2,padding=1),
            nn.ReLU(),
            nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1),
            nn.Flatten(),
            nn.Linear(3136,2)
        )
        
        self.decoder=nn.Sequential(
            nn.Linear(2,3136),
            Reshape(-1,64,7,7),
            nn.ConvTranspose2d(64,64,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64,64,kernel_size=3,stride=2,padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64,32,kernel_size=3),
            nn.ReLU(),
            nn.ConvTranspose2d(32,1,kernel_size=3),
            Trim(),
            nn.Sigmoid()
        )

    def forward(self, x):
        x=self.encoder(x)
        x=self.decoder(x)
        return x
    
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

from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AutoEncoder().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

train_losses = []
valid_losses = []
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    train_loader_iter = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} (Training)', leave=False)
    
    for images, _ in train_loader_iter:
        images = images.to(device)
        outputs = model(images)
        loss = criterion(outputs, images)

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
            outputs = model(images)
            loss = criterion(outputs, images)

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
