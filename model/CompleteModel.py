import math
import torch
import torch.nn as nn
def linear_beta_schedule(timesteps):
    beta_start = 0.0001 # the lower bound of beta, when timesteps == 1000
    beta_end = 0.02 # the upper bound of beta, when timesteps == 1000
    scale = 1000.0 / timesteps # adjust the range of beta, if timesteps != 1000
    # in linear schedule, the distance of neighboring betas is a constant
    return torch.linspace(start=beta_start * scale, end=beta_end * scale, 
                          steps=timesteps, dtype=torch.float64)
def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    ts = torch.linspace(start=0, end=timesteps, steps=steps, dtype=torch.float64)
    f_of_ts = torch.cos((ts / timesteps + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_bar = f_of_ts / f_of_ts[0]
    betas = 1 - (alphas_bar[1:] / alphas_bar[:-1])
    return torch.clip(input=betas, min=0, max=0.999)
class GaussianDiffusion(nn.Module):
    def __init__(self, model: nn.Module, image_size, timesteps=1000, 
                 sampling_timesteps=None, loss_type='l1', 
                 objective='pred_noise', beta_schedule: str='cosine', 
                 p2_loss_weight_gamma=0, p2_loss_weight_k=1, 
                 ddim_sampling_eta=1):
        super(GaussianDiffusion, self).__init__()
        # epsilon model
        self.model = model
        
        self.image_size = image_size
        # prediction object of the model
        self.objective = objective
        assert objective in {'pred_noise', 'pred_x0', 'pred_v'}
        # manner of adding noise
        if beta_schedule == 'linear':
            self.betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            self.betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError('Unknown beta schedule method: ' + beta_schedule)
    def forward(self, img, *args, **kwargs):
        pass