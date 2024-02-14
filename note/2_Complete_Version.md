# DDPM(A Complete Version)  
## 1 Libraries
``` python 
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
This kind of $\beta$ schedule is proposed in , and it behaves better than linear schedule.  
- Implementation  
    ``` python
    
    ```
## References  
[1]https://python.readthedocs.io/en/stable/library/typing.html  
[2]https://pytorch.org/docs/stable/generated/torch.cumprod.html  
[3]https://nn.labml.ai/diffusion/ddpm/index.html  
[4][Denoising Diffusion Probabilistic Model](https://arxiv.org/pdf/2006.11239.pdf)  