import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm
def default(value, default_value):
    if value is None:
        return default_value
    else:
        return value
def extract(consts: torch.Tensor, t, x_shape):
    batch_size, *_ = x_shape
    out = consts.gather(dim=-1, index=t)
    return out.reshape(batch_size, *((1, ) * len(x_shape) - 1))
def linear_beta_schedule(timesteps) -> torch.Tensor:
    beta_start = 0.0001 # the lower bound of beta, when timesteps == 1000
    beta_end = 0.02 # the upper bound of beta, when timesteps == 1000
    scale = 1000.0 / timesteps # adjust the range of beta, if timesteps != 1000
    # in linear schedule, the distance of neighboring betas is a constant
    return torch.linspace(start=beta_start * scale, end=beta_end * scale, 
                          steps=timesteps, dtype=torch.float64)
def cosine_beta_schedule(timesteps, s=0.008) -> torch.Tensor:
    steps = timesteps + 1
    ts = torch.linspace(start=0, end=timesteps, steps=steps, dtype=torch.float64)
    f_of_ts = torch.cos((ts / timesteps + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_bar = f_of_ts / f_of_ts[0]
    betas = 1 - (alphas_bar[1:] / alphas_bar[:-1])
    return torch.clip(input=betas, min=0, max=0.999)
class GaussianDiffusion(nn.Module):
    def __init__(self, model: nn.Module, image_size, loss_fn, timesteps=1000, 
                 sampling_timesteps=None, loss_type='l1', 
                 objective='pred_noise', beta_schedule: str='cosine', 
                 p2_loss_weight_gamma=0, p2_loss_weight_k=1, 
                 ddim_sampling_eta=1):
        super(GaussianDiffusion, self).__init__()
        # The following code defines the class parameters
        # epsilon model
        self.model = model
        self.channels = model.channels
        # whether to use conditional diffusion model
        self.self_condition = model.self_condition
        self.image_size = image_size
        self.timesteps = timesteps
        self.loss_type = loss_type
        self.loss_fn = loss_fn
        # prediction object of the model
        self.objective = objective
        # number of timesteps in sampling
        self.sampling_timesteps = default(sampling_timesteps, timesteps)
        # whether use DDIM sampling method or DDPM sampling method
        self.is_ddim_sampling = (self.sampling_timesteps < self.timesteps)
        # the eta parameter used in ddim_sampling
        self.ddim_sampling_eta = ddim_sampling_eta
        # the loss weight when using L2 norm and formula (10) in DDPM paper
        
        assert objective in {'pred_noise', 'pred_x0'}
        # assert the input and output dimension of the model to be equal
        assert not (type(self) == GaussianDiffusion and model.channels != model.out_dim)
        # assert number of sampling timesteps <= training timesteps
        assert self.sampling_timesteps <= self.timesteps
       
        # THE FOLLOWING CODE DEFINE CONSTANT PARAMETERS INVOLVED IN COMPUTATION
        # WE USE REGISTER BUFFER TO RECORD THEM
        # manner of adding noise(beta)  
        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError('Unknown beta schedule method: ' + beta_schedule)
        # Compute the alphas and alphas_bar, according to Formula (4) in the DDPM paper
        # This part is also contained in the simpler version
        alphas = 1.0 - betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        # alpha bars in (t - 1) timestep
        # pad a 1 at the left end, and pad no value in the right end
        alphas_bar_prev = F.pad(input=alphas_bar[: -1], pad=(1, 0), value=1.)
        sigma_squares = betas
        # Parameters used in forward Markov transition kernel q(x_t | x_{t - 1})  
        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))
        register_buffer('betas', betas)
        register_buffer('alphas', alphas)
        register_buffer('alphas_bar', alphas_bar)
        register_buffer('alphas_bar_prev', alphas_bar_prev)
        register_buffer('sigma_squares', sigma_squares)
        register_buffer('sqrt_alphas_bar', torch.sqrt(alphas_bar))
        register_buffer('sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))
        register_buffer('log_one_minus_alphas_bar', torch.log(1. - alphas_bar))
        register_buffer('sqrt_reciprocal_alphas_bar', torch.sqrt(1. / alphas_bar))
        register_buffer('sqrt_reciprocal_of_alphas_bar_minus_1', torch.sqrt(1. / alphas_bar - 1)) 
        # The posterior variance in reverse process\
        posterior_variance = betas * (1. - alphas_bar_prev) / (1. - alphas_bar)
        register_buffer('posterior_variance', posterior_variance)
        # Logarithm of posterier variance
        # need to be clipped, as the posterior variance at t = 0 is zero
        clipped_posterior_variance = posterior_variance.clip(min=1e-20)
        log_clipped_posterior_variance = torch.log(clipped_posterior_variance)
        register_buffer('log_clipped_posterior_variance', log_clipped_posterior_variance)
        # Coefficients in calculating the mean value of posterior variance
        register_buffer('x0_coeff_in_posterior_mean', torch.sqrt(alphas_bar_prev) / (1. - alphas_bar))
        register_buffer('xt_coeff_in_posterior_mean', torch.sqrt(alphas) * (1. - alphas_bar_prev) 
                        / (1. - alphas_bar))
        register_buffer('forward_loss_weights', 0.5 / sigma_squares)
    def q_sample(self, x0, t, noise=None):
        # sampling from distribution q, adding noise to the original images, to obtain images at timestep t
        # input: the original images x_{0}
        # noise: standard Gaussian noise which has the same dimension as the input images x0
        noise = default(value=noise, default_value=lambda: torch.randn_like(x0))
        # to sample from the distribution
        # we get its mean value and variance
        # then apply the noise
        
        # the alphas_bar and standard deviation at all the timesteps
        alphas_bar = self.sqrt_alphas_bar
        standard_deviations = self.sqrt_one_minus_alphas_bar
        # the mean value and standard deviaion at timestep t
        mean_t = extract(consts=alphas_bar, t=t, x_shape=x0.shape) * x0
        standard_deviation_t = extract(consts=standard_deviations, t=t, x_shape=x0.shape)  
        return mean_t + standard_deviation_t * noise
    def forward_p_loss(self, x0: torch.Tensor, t, noise=None):
        # x0: the original image
        # noise: standard Gaussian noise, which has the same shape as x0
        batch_size, channels, height, width = x0.shape
        noise = default(noise, lambda: torch.randn_like(x0))
        # sample from q distribution, to get image with noise
        xt = self.q_sample(x0, t, noise)
        # the prediction of model at timestep t
        out = self.model(x0, t, self.self_condition)
        # calculate the real value  
        if self.objective == 'pred_noise':
            target = noise 
        elif self.objective == 'pred_x0':
            target = x0
        else:
            raise ValueError("Unknown prediction objective " + self.objective)
        # calculate the loss
        loss = self.loss_fn(out, target, reduction='None')
        # calculate the weighted mean loss
        loss = loss * extract(self.forward_loss_weights, t, loss.shape)
        return loss.mean()
        
    def forward(self, img: torch.Tensor, *args, **kwargs):
        # shape of the training image: (N, C, H, W)
        batch_size, channels, height, width = img.shape
        device = img.device
        img_size = self.image_size
        # obtain a sequence of random integers in range [0, T]
        t = torch.randint(low=0, high=self.timesteps, size=(batch_size, ), device=device).long()
        # normalize the original image to [-1, 1]
        normalize = transforms.Normalize(mean=(0., 0., 0.), 
                                         std=(1., 1., 1.))
        img = transforms.normalize(img)
        return self.forward_p_loss(x0=img, t=t, *args, **kwargs)
    
    # At timestep t, use the model to predict the noise
    # and obtain the denoised image x_{t - 1}
    def reverse_model_prediction(self, xt, t,  x_self_conditioned=None, clip_denoised=True):
        # xt: xt with more noise at the former step
        # t: the current timestep
        # x_self_conditioned: conditional diffusion input
        # clip_denoised: whether to clip the range of generated result
        pass
    
    # This function returns the mean and variance of distribution p
    def get_p_distribution(self, xt, t, x_self_conditioned=None, clip_denoised=True):
        # xt: xt with more noise at the former step
        # t: the current timestep
        # x_self_conditioned: conditional diffusion input
        # clip_denoised: whether to clip the range of generated result
        pass
    
    @torch.no_grad()
    def p_sample(self, xt: torch.Tensor, t: torch.Tensor, x_self_conditioned=None, clip_denoised=True):
        # xt: xt with more noise at the former step
        # t: the current timestep
        # x_self_conditioned: conditional diffusion input
        # clip_denoised: whether to clip the range of generated result
        batch_size = xt.shape[0]
        device = xt.device
        
        pass
    
    # denoise from a noisy image that obeys Gaussian distribution
    # to obtain generative result
    def denoise(self, shape):
        # shape: the shape of output image 
        # [batch_size, channels, height, width]
        batch_size = shape[0]
        device = self.betas.device
        # Get a noisy image which obeys the Gaussian distribution
        noisy_imgs = torch.randn(size=shape, device=device)
        # The final generated image by denoising
        x_denoised = None
        # iteratively denoise from step T to step 0
        # to generate the denoised image
        for t in tqdm(iterable=reversed(range(0, self.timesteps)), 
                      total=self.timesteps):
            pass
        normalize = transforms.Normalize(mean=(0.5, 0.5, 0.5), 
                                         std=(0.5, 0.5, 0.5))
        img = transforms.normalize(img)
        return img
    
    