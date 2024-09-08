import torch
import torch.nn as nn

class DDPMScheduler(nn.Module):
    def __init__(self,num_time_steps=1000):
        super().__init__()
        self.beta=torch.linspace(1e-4,0.02,num_time_steps,requires_grad=False)
        alpha=1-self.beta
        self.alpha=torch.cumprod(alpha,dim=0).requires_grad_(False)
    
    def forward(self,t):
        return self.beta[t],self.alpha[t]

def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)

def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location="cuda")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr