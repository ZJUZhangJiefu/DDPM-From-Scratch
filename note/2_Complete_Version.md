# DDPM(A Complete Version)  
## 1 Libraries and Basic Functions
### 1.1 Libraries  
- Implementation
    ``` python 
    import math
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision import transforms
    ```
### 1.2 `default` Function  
If `value` is `None`, then return the default value; else return `value`.  
- Implementation  
    ``` python
    def default(value, default_value):
        if value is None:
            return default_value
        else:
            return value
    ```
### 1.3 `extract` Function  
- `consts`: The constants at all timesteps, such as $\alpha_t$, etc.  
- This function extracts the constant at timestep $t$.  
- Implementation  
    ``` python  
    def extract(consts: torch.Tensor, t, x_shape):
        batch_size, *_ = x_shape
        out = consts.gather(dim=-1, index=t)
        return out.reshape(batch_size, *((1, ) * len(x_shape) - 1))
    ```
## 2 Two Kinds of Forward Process Variance $\beta_{t}$  
- $\beta_{t}$ is called "forward process variance" in the paper, and it represents the **intensity of the noise added in the forward steps**.  
- We use $T$ to denote the **number of timesteps**.   
### 2.1 Linear schedule  
- When $T = 1000$, $\beta_0 = 0.0001, \beta_T = 0.02$.  
- $\beta_{t}$ increases linearly with the timestep $t$($0 \leq t \leq T$).  
- If $T \neq 1000$, then the lower and upper bounds of $\beta_t$ should be adjusted, by multiplying a factor $\frac{1000}{T}$.  
- Implementation  
    ``` python
    def linear_beta_schedule(timesteps):
        beta_start = 0.0001 # the lower bound of beta, when timesteps == 1000
        beta_end = 0.02 # the upper bound of beta, when timesteps == 1000
        scale = 1000.0 / timesteps # adjust the range of beta, if timesteps != 1000
        # in linear schedule, the distance of neighboring betas is a constant
        return torch.linspace(start=beta_start * scale, end=beta_end * scale, 
                            steps=timesteps, dtype=torch.float64)
    ```
### 2.2 Cosine Schedule  
This kind of $\beta$ schedule is proposed in [Improved Denoising Diffusion Probabilistic Models](https://openreview.net/pdf?id=-NEXDKk8gZ), and it behaves better than linear schedule.  
- Formulae  
    $\bar{\alpha} _{t} = \frac{f(t)}{f(0)} , f(t) = cos^{2} (\frac{\frac{t}{T} + s}{1 + s} \cdot \frac{\pi}{2})$

    $\beta_{t} = 1 - \frac{\bar{\alpha}_t}{\bar{\alpha} _{t - 1}}$  
    - Note  
    The square($^2$) operator applies on the whole cosine function to guarantee that $\beta_t$ is always positive, and its style of writing in the above paper should be corrected as $f(t) = cos^2(\frac{\frac{t}{T} + s}{1 + s} \cdot \frac{\pi}{2})$.  
- Hyperparameter Settings  
    - Clip $\beta_{t}$ to be no larger than $0.999$, to prevent singularities at the end of the diffusion process near $t = T$.  
    - Use a small offset $s = 0.008$, to prevent $\beta_t$ from being too small near $t = 0$.  
- Comprehension  
    - Variable `timesteps` corresponds to $T$ in the formulae.  
    - Prepare the $t$($0 \leq t \leq T$) for function $f(t)$.  
        ``` python
        steps = timesteps + 1
        ts = torch.linspace(start=0, end=timesteps, steps=steps, dtype=torch.float64)
        ```
    - $f(t) = cos^2(\frac{\frac{t}{T} + s}{1 + s} \cdot \frac{\pi}{2})$
        ``` python
        f_of_ts = torch.cos((ts / timesteps + s) / (1 + s) * math.pi * 0.5) ** 2
        ```
    - $\bar{\alpha}_{t} = \frac{f(t)}{f(0)}$  
        ``` python
        alphas_bar = f_of_ts / f_of_ts[0]
        ```
    - $\beta_{t} = 1 - \frac{\bar{\alpha}_t}{\bar{\alpha} _{t - 1}}$
        ``` python
        betas = 1 - (alphas_bar[1:] / alphas_bar[:-1])
        ```
    - Clip $\beta_{t}$ to be no larger than $0.999$
        ``` python
        return torch.clip(input=betas, min=0, max=0.999)
        ```
- Implementation  
    ``` python
    def cosine_beta_schedule(timesteps, s = 0.008):
        steps = timesteps + 1
        ts = torch.linspace(start=0, end=timesteps, steps=steps, dtype=torch.float64)
        f_of_ts = torch.cos((ts / timesteps + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_bar = f_of_ts / f_of_ts[0]
        betas = 1 - (alphas_bar[1:] / alphas_bar[:-1])
        return torch.clip(input=betas, min=0, max=0.999)
    ```
## 3 Class Initialization  
### 3.1 Input Parameters Specification  
- Input Parameters  
    ``` python  
    def __init__(self, model, image_size, loss_fn, timesteps=1000, 
                 sampling_timesteps=None, loss_type='l1', 
                 objective='pred_noise', beta_schedule='cosine', 
                 p2_loss_weight_gamma=0, p2_loss_weight_k=1, 
                 ddim_sampling_eta=1):
    ```
- Comments  
    - `model`: The deep model to predict the noise $\epsilon$ or velocity $v$ or original image $x_0$, denoted as $\epsilon_{\theta}(x_{t}, t)$ in the DDPM paper.  
    - `image_size`: The size of input images.  
    - `loss_fn`: The loss function.  
    - `timesteps`: $T$, total number of steps in forward/reverse process.  
    - `sampling_timesteps`:  
    - `loss_type`:  
    - `objective`: The prediction objective of the model.  
        - `pred_noise`: Predict the noise added in current timestep $t$.  
        - `pred_x0`: Predict the original image $x_0$.  
        - `pred_v`:  
    - `beta_schedule`: The manner of scheduling the forward process variance(the intensity of noise) $\beta_t$.  
        - `linear`: Use linear schedule.  
        - `cosine`: Use cosine schedule.  
    - `p2_loss_weight_gamma`:  
    - `p2_loss_weight_k`:  
    - `ddim_sampling_eta`:  
### 3.2 Class Parameters Configuration  
- Class Parameters(Not directly contained in the computation graph)  
    - The deep neural model to predict noise $\epsilon$, original image $x_{0}$, or the velocity $v$. (The model to predict $\epsilon$ is designated as ${\epsilon}_{\theta}(x_t, t)$ in the DDPM paper.)  
        ``` python
        self.model = model
        ```
    - Number of channels, whether to use conditional diffusion model, image size, number of timesteps and whether to use DDIM sampling method.  
        ``` python
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
        ```
    - The parameter $\eta$ in DDIM sampling formula $\sigma_{\tau_{i}} = \eta \sqrt{\frac{1 - \alpha_{\tau_{i - 1}}}{1 - \alpha_{\tau_{i}}}}\sqrt{1 - \frac{\alpha_{\tau_{i}}}{\alpha_{\tau_{i - 1}}}}$  
        ``` python
        # the eta parameter used in ddim_sampling
        self.ddim_sampling_eta = ddim_sampling_eta
        assert objective in {'pred_noise', 'pred_x0', 'pred_v'}
        ```
    - Assertion of legal prediction objective; for the `GaussianDiffusion` class, the dimension of input and output should be the same; the number of sampling steps should be no greater than the total number of timesteps.  
        ``` python
        assert objective in {'pred_noise', 'pred_x0'}
        # assert the input and output dimension of the model to be equal
        assert not (type(self) == GaussianDiffusion and model.channels != model.out_dim)
        # assert number of sampling timesteps <= training timesteps
        assert self.sampling_timesteps <= self.timesteps
        ```
- Class Constant Parameters(Directly involved in the computation graph)  
    - Get the forward process variance(the intensity of noise) $\beta_t$ in each timestep $t$, according to the specified noise schedule manner.  
        - `betas`: $\beta$   
        ``` python
        # manner of adding noise(beta)  
        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError('Unknown beta schedule method: ' + beta_schedule)
        ```
    - Compute $\alpha = 1 - \beta,\bar{\alpha}_t = \Pi _{s=1}^t{\alpha}_s$ and $\sigma_t^2 = \beta_t$ in the DDPM paper. 
        - `alphas`: $\alpha_t$  
        - `alphas_bar`: $\bar{\alpha}_{t}$  
        - `alphas_bar_prev`: $\bar{\alpha}_{t - 1}$  
        ``` python
        # Compute the alphas and alphas_bar, according to Formula (4) in the DDPM paper
        # This part is also contained in the simpler version
        alphas = 1.0 - betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        # alpha bars in (t - 1) timestep
        # pad a 1 at the left end, and pad no value in the right end
        alphas_bar_prev = F.pad(input=alphas_bar[: -1], pad=(1, 0), value=1.)
        sigma_squares = betas
        ```
    - Since these parameters do not need gradients, we use `self.register_buffer` to record them.  
        ``` python
        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))
        ```
    - Register the parameters involved in  the forward Markov transition kernel $q(x_t | x_{t - 1})$.  
    Equation (5) in the paper uses KL divergence to directly compare $p_{\theta}(x_{t - 1} | x_{t})$ against forward process posteriors, and the forward posteriors are tractable when conditioned on $x_0$:  
    $q(x_{t - 1}|x_t, x_0) = \mathscr{N} (x_{t - 1}; \tilde {\mu} _ {t} (x_t, x_0), \tilde{\beta} _ t \mathbf{I})$.  
    The mean value: $\tilde{\mu} _ {t} (x_t, x_0) = \frac{\sqrt{\bar{\alpha} _ {t - 1}}\beta_{t}}{1 - \bar{\alpha} _ {t}}x_0 + \frac{\sqrt{\alpha_{t}}(1 - \bar{\alpha} _ {t - 1})}{1 - \bar{\alpha} _ {t}}x_t$.  
    The variance: $\tilde{\beta} _ {t} = \frac{1 - \bar{\alpha} _ {t - 1}}{1 - \bar{\alpha} _ {t}}\beta_{t}$.  
    The forward loss:  
    $L_{t - 1} - C = \mathbf{E} _ {\mathbf{x} _ 0, \epsilon}[\frac{1}{2\sigma _ t ^ 2}||\frac{1}{\sqrt{\alpha _ t}}(\mathbf{x} _ {t} (\mathbf{x} _ 0, \epsilon) - \frac{\beta _ {t}}{\sqrt{1 - \bar{\alpha} _ {t}}}\epsilon) - \mu _ {\theta}(\mathbf{x} _ t(\mathbf{x} _ 0, \epsilon), t)||^2]$  
    The loss weights in forward steps: $\frac{1}{2\sigma_{t}^2}$.  
        - `betas`: $\beta_t$
        - `alphas`: $\alpha_t = 1 - \beta_t$  
        - `alphas_bar`: $\bar{\alpha} _ {t} = \Pi_{s = 1}^t\alpha_{s}$  
        - `alphas_bar_prev`: $\bar{\alpha}_{t - 1}$  
        - `sigma_squares`: $\sigma^2 = \beta$ 
        - `sqrt_alphas_bar`: $\sqrt{\bar{\alpha}_{t}}$  
        - `sqrt_one_minus_alphas_bar`: $\sqrt{1 - \bar{\alpha}_{t}}$  
        - `log_one_minus_alphas_bar`: $\log(1 - \bar{\alpha}_{t})$  
        - `sqrt_reciprocal_alphas_bar`: $\sqrt{\frac{1}{\bar{\alpha}_{t}}}$  
        - `sqrt_reciprocal_of_alphas_bar_minus_1`: $\sqrt{\frac{1}{\bar{\alpha}_{t}} - 1}$  
        - `forward_loss_weights`: $\frac{1}{2\sigma_{t}^2}$

        ``` python
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
        register_buffer('forward_loss_weights', 0.5 / sigma_squares)
        ```
- Implementation  
    ``` python
    def __init__(self, model: nn.Module, image_size, timesteps=1000, 
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
    ```
## 4 Sampling from the Priorior and Posterior Distributions, Loss Calculation and the Forward Function  
### 4.1 Sampling from $q(x_t|x_{0})$
Sampling from the distribution $q(x_t|x_{0})$, to get the image with noise added.   
- This function `q_sample` gets a sample from the distribution $q(x_t|x_{0}) = N(x_t; \sqrt{\bar{\alpha} _ {t}}x_{0}, (1 - \bar{\alpha}_{t})I)$.  
    - Get the noise from standard Gaussian distribution.  
        ``` python
        noise = default(value=noise, default_value=lambda: torch.randn_like(x0))
        ```
    - Coefficients needed in obtaining the distribution at the timestep $t$.  
        - `alphas_bar`: $\bar{\alpha}_{t}$  
        - `standard_deviations`: $\sigma_{t}$  
        ``` python
        alphas_bar = self.sqrt_alphas_bar
        standard_deviations = self.sqrt_one_minus_alphas_bar
        ```  
    - The mean value and standard deviation of distribution $q(\mathbf{x} _ {t}|\mathbf{x}_{0})$.  
        - Mean value: $\sqrt{\bar{\alpha}_{t}}\mathbf{x}_0$.  
        - Standard deviation: $(1 - \bar{\alpha}_{t})$.  
        ``` python
        mean_t = extract(consts=alphas_bar, t=t, x_shape=x0.shape) * x0
        standard_deviation_t = extract(consts=standard_deviations, t=t, x_shape=x0.shape)  
        ```
    - Obtain the sample:  
        ``` python
        return mean_t + standard_deviation_t * noise
        ```
- Implementation  
    ``` python
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
    ```
### 4.2 Loss Calculation in Forward Steps  
- Notes  
    - Obtain the image shape, as well as the standard Gaussian noise $\epsilon$.  
        ``` python
        batch_size, channels, height, width = x0.shape
        noise = default(noise, lambda: torch.randn_like(x0))
        ```
    - Sample from distribution $q(x _ {t} | x _ {0})$, to get images with noise $x _ {t}$.  
        ``` python
        xt = self.q_sample(x0, t, noise)
        ```
    - Obtain the prediction result of the deep model, the real value, and calculate weighted mean loss.  
        ``` python
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
        ```
- Implementation  
``` python
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
```
### 4.3 The `forward` Function  
- This function implements the forward process, i.e. adding noise to the original image.  
    - Obtain the shape, device and image size
        ``` python
        batch_size, channels, height, width = img.shape  
        device = img.device
        img_size = self.image_size
        ```
    - Obtain a sequence of random integers in range $[0, T]$, as timesteps for each image in the batch.  
        ``` python
        t = torch.randint(low=0, high=self.timesteps, size=(batch_size, ), device=device).long()
        ```
    - Normalize the image, then call `forward_p_loss` to calculate $\mathbf{x}_t$, do prediction and calculate the forward loss.  
        ``` python
        # normalize the original image to [-1, 1]
        normalize = transforms.Normalize(mean=(0.5, 0.5, 0.5), 
                                                    std=(0.5, 0.5, 0.5))
        img = transforms.normalize(img)
        return self.forward_p_loss(x0=img, t=t, *args, **kwargs)
        ```
- Implementation  
    ``` python
    def forward(self, img: torch.Tensor, *args, **kwargs):
        # shape of the training image: (N, C, H, W)
        batch_size, channels, height, width = *img.shape,  
        device = img.device
        img_size = self.image_size
        # obtain a sequence of random integers in range [0, T] as timesteps for each image in the batch
        t = torch.randint(low=0, high=self.timesteps, size=(batch_size, ), device=device).long()
        # normalize the original image to [-1, 1]
        normalize = transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        img = transforms.normalize(img)
        return self.forward_p_loss(x0=img, t=t, *args, **kwargs)
    ```
## References  
[1]https://python.readthedocs.io/en/stable/library/typing.html  
[2]https://pytorch.org/docs/stable/generated/torch.cumprod.html  
[3]https://nn.labml.ai/diffusion/ddpm/index.html  
[4][Denoising Diffusion Probabilistic Model](https://arxiv.org/pdf/2006.11239.pdf)  
