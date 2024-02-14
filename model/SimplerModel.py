import math
import torch
import torch.nn.functional as F
import torch.utils.data
from torch import nn

from labml_nn.diffusion.ddpm.utils import gather
from typing import Tuple, Optional

# The overall class of denoising diffusion
class DenoiseDiffusion:
    def __init__(self, eps_model: nn.Module, step_num: int, 
                 device: torch.device):
        # the epsilon_theta(x_t, t) model
        self.eps_model = eps_model
        self.step_num = step_num
        # beta, linearly increasing
        # divide the interval [0.0001, 0.02] into 1000 points
        # the distances of neighbouring points are equal
        self.betas = torch.linspace(0.0001, 0.02, step_num).to(device)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.sigma_squares = self.betas
        
    def get_q_xt_x0_distribution(self, x0: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = math.sqrt(gather(consts=self.alpha_bars, t=t)) * x0
        var = 1 - gather(consts=self.alpha_bars, t=t)
        # return mean and varience of the distribution
        return mean, var
    
    def sample_from_q_xt_x0_distribution(self, x0: torch.Tensor, t: torch.Tensor, 
                            epsilon: Optional[torch.Tensor]=None):
        # if noise is not provided, sample it from normal distribution
        if epsilon is None:
            epsilon = torch.randn_like(x0)
        # get mean and variance of distribution q(x_t|x_0)
        mean, var = self.get_q_xt_x0_distribution(x0, t)
        # apply the noise(epsilon) to get a sample from distribution q(x_t|x_0)
        return mean + math.sqrt(var) * epsilon
    
    def sample_from_p_theta_distribution(self, xt: torch.Tensor, t: torch.Tensor):
        epsilon_theta = self.eps_model(xt, t)
        alpha_bar = gather(self.alpha_bars, t)
        alpha = gather(self.alphas, t)
        beta = gather(self.betas, t)
        mean = (xt - beta * epsilon_theta / torch.sqrt(1 - alpha_bar)) / torch.sqrt(alpha)
        var = gather(self.sigma_squares, t)
        return mean, var
    
    def simple_loss(self, x0: torch.Tensor, epsilon: Optional[torch.Tensor]=None):
        # get batch size
        batch_size = x0.shape[0]
        # for each sample in the training batch, obtain the timestep t independently and randomly
        t = torch.randint(low=0, high=self.step_num, size=(batch_size, ), 
                          device=x0.device, dtype=torch.long) 
        # if noise is not provided, then sample it from normal distribution
        if epsilon is None:
            epsilon = torch.randn_like(x0)
        # obtain xt, by sampling from q distribution
        xt = self.sample_from_q_xt_x0_distribution(x0, t, epsilon)
        # obtain the denoising result
        epsilon_theta = self.eps_model(xt, t)
        # return the discrete mean square error
        return F.mse_loss(epsilon, epsilon_theta)
        