# DDPM(A Complete Version)  
## 1 Libraries
``` python 
import math
import torch
import torch.nn as nn
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
    def __init__(self, model, image_size, timesteps=1000, 
                 sampling_timesteps=None, loss_type='l1', 
                 objective='pred_noise', beta_schedule='cosine', 
                 p2_loss_weight_gamma=0, p2_loss_weight_k=1, 
                 ddim_sampling_eta=1):
    ```
- Comments  
    - `model`: The deep model to predict the noise $\epsilon$ or velocity $v$ or original image $x_0$, denoted as $\epsilon_{\theta}(x_{t}, t)$ in the DDPM paper.  
    - `image_size`: The size of input images.  
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
- Class Parameters Configuration

- Comments


### 3.10 `__init__` Function Implementation
``` python

```
## References  
[1]https://python.readthedocs.io/en/stable/library/typing.html  
[2]https://pytorch.org/docs/stable/generated/torch.cumprod.html  
[3]https://nn.labml.ai/diffusion/ddpm/index.html  
[4][Denoising Diffusion Probabilistic Model](https://arxiv.org/pdf/2006.11239.pdf)  
