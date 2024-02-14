import torch
import torch.nn as nn
def linear_beta_schedule(timesteps):
    beta_start = 0.0001 # the lower bound of beta, when timesteps == 1000
    beta_end = 0.02 # the upper bound of beta, when timesteps == 1000
    scale = 1000.0 / timesteps # adjust the range of beta, if timesteps != 1000
    # in linear schedule, the distance of neighboring betas is a constant
    return torch.linspace(start=beta_start * scale, end=beta_end * scale, 
                          steps=timesteps, dtype=torch.float64)
def cosine_beta_schedule(timesteps):
    pass
class GaussianDiffusion(nn.Module):
    def __init__(self):
        super(GaussianDiffusion, self).__init__()
        
    def forward(self, img, *args, **kwargs):
        pass