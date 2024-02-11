# DDPM Study Notes 1
## 1 Libraries  
### 1.1 PyTorch import
``` python
import math
import torch
import torch.nn.functional as F
import torch.utils.data
from torch import nn
```  
### 1.2 Other libraries to import and comments
``` python
from labml_nn.diffusion.ddpm.utils import gather
from typing import Tuple, Optional
```  
- typing.Tuple and Type Alias  
    A **type alias** is defined by **assigning the type to the alias**. In this example, `Vector` and `List[float]` will be treated as **interchangeable synonyms**:
    ``` python
    from typing import List
    Vector = List[float]

    def scale(scalar: float, vector: Vector) -> Vector:
        return [scalar * num for num in vector]

    # typechecks; a list of floats qualifies as a Vector.
    new_vector = scale(2.0, [1.0, -4.2, 5.4])
    ```

    **Type aliases** are useful for **simplifying complex type signatures**. For example:  

    ``` python 
    from typing import Dict, Tuple, List

    ConnectionOptions = Dict[str, str]
    Address = Tuple[str, int]
    Server = Tuple[Address, ConnectionOptions]

    def broadcast_message(message: str, servers: List[Server]) -> None:
        ...

    # The static type checker will treat the previous type signature as
    # being exactly equivalent to this one.
    def broadcast_message(
            message: str,
            servers: List[Tuple[Tuple[str, int], Dict[str, str]]]) -> None:
        ...
    ```

- `typing.Optional` and `typing.Union`  

     - `typing.Union`  
    Union type; `Union[X, Y]` means **either `X` or `Y`**.  
        To **define** a union, use e.g. 
        ``` python 
        Union[int, str]
        ```   
        Details:  
        The **arguments must be types** and there must be **at least one**.  
        **Unions of unions** are **flattened**, e.g.:  
        ``` python 
        Union[Union[int, str], float] == Union[int, str, float]
        ```  
        **Unions of a single argument vanish**, e.g.:  
        ``` python 
        Union[int] == int  # The constructor actually returns int
        ```
        **Redundant arguments** are **skipped**, e.g.:
        ``` python 
        Union[int, str, int] == Union[int, str]
        ```
        When **comparing unions**, the **argument order** is **ignored**, e.g.:
        ``` python 
        Union[int, str] == Union[str, int]
        ```
        When a **class and its subclass** are **present**, the **latter** is **skipped**, e.g.:
        ``` python 
        Union[int, object] == object
        ``` 
        You **cannot subclass or instantiate** a union.  
        You **cannot write** `Union[X][Y]`.  
        You can use `Optional[X]` as a **shorthand** for `Union[X, None]`.

    - typing.Optional  
        Optional type.
        `Optional[X]` is **equivalent** to `Union[X, None]`.
        Note that this is **not the same concept as an optional argument**, which is one that **has a default**s. An optional argument **with a default needn’t use the Optional qualifier** on its type annotation (although it is inferred if the default is None). A **mandatory argument** may **still have an Optional type** if an **explicit value of None** is **allowed**.  

## 2 The Overall DDPM Class
``` python 
# The overall class of denoising diffusion
class DenoiseDiffusion:
```
### 2.1 The `__init__` Function  
``` python
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
```  
- Parameters  
    `self.betas`: $\beta$.  
    `self.alphas`: $\alpha = 1 - \beta$.  
    `self.alpha_bars`: $\bar{\alpha}_t = \Pi_{s = 1}^t\alpha_s$    
    `self.sigma_squares`: $\sigma_t^2 = \beta_t$  
    $t$ means the timestep.   

    [Note]  
    There are `2` kinds of variance($\sigma_t^2$) in reverse process $L_{1 : T - 1}$ in the DDPM paper.  
    "Experimentally, both $\sigma_t^2 = \beta_t$" and $\sigma_t^2 = \tilde{\beta}_t = \frac{1 - \tilde{\alpha}_{t - 1}}{1 - \tilde{\alpha}_t}\beta_t$ has similar results."  

- `torch.cumprod`  

    Returns the **cumulative product** of **elements of input** in the **dimension `dim`**.

    For example, if input is a vector of size $N$, the result will also be a vector of size $N$, with elements.

    $y_i = x_1 \times x_2\times x_3\times \dots \times x_i$
    - Parameters  
        `input (Tensor)` – the input tensor.  
        `dim (int)` – the **dimension to do the operation over**.

    - Keyword Arguments  
        `dtype (torch.dtype, optional)` – the **desired data type of returned tensor**. If specified, the input tensor is **casted** to `dtype` **before** the **operation is performed**. This is useful for **preventing data type overflows**. Default: `None`.  
        `out (Tensor, optional)` – the output tensor.  
### 2.2 The `get_q_xt_x0_distribution` Function
This function gets the distribution $q(x_t|x_0) = \mathscr{N}(x_t; \sqrt{\bar{\alpha}_t}x_0, (1 - \bar{\alpha}_t)\mathbf{I})$. Input a tensor $x_0$, it will return the mean value and variance of the distribution $q(x_t|x_0)$.  
``` python
def get_q_xt_x0_distribution(self, x0: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    mean = math.sqrt(gather(consts=self.alpha_bars, t=t)) * x0
    var = 1 - gather(consts=self.alpha_bars, t=t)
    # return mean and varience of the distribution
    return mean, var
```  
- The `gather` function
    ``` python
    def gather(consts: torch.Tensor, t: torch.Tensor):
    """Gather consts for $t$ and reshape to feature map shape"""
    c = consts.gather(-1, t)
    return c.reshape(-1, 1, 1, 1)
    ```
    The `gather` function above gathers constant for timestep $t$, which gets $\bar{\alpha}_t$, and reshape it to the shape of feature map.  
### 2.3 The `sample_from_q_xt_x0` function
This function gets a sample from the distribution $q(x_t|x_0) = \mathscr{N}(x_t; \sqrt{\bar{\alpha}_t}x_0, (1 - \bar{\alpha}_t)\mathbf{I})$.  
``` python
def sample_from_q_xt_x0_distribution(self, x0: torch.Tensor, t: torch.Tensor, 
                        epsilon: Optional[torch.Tensor]=None):
    # if noise is not provided, sample it from normal distribution
    if epsilon is None:
        epsilon = torch.randn_like(x0)
    # get mean and variance of distribution q(x_t|x_0)
    mean, var = self.get_q_xt_x0_distribution(x0, t)
    # apply the noise(epsilon) to get a sample from distribution q(x_t|x_0)
    return mean + math.sqrt(var) * epsilon
```
- `torch.randn_like`  
    Returns a tensor with the **same size as input** that is filled with **random numbers** from a **normal distribution** with mean `0` and variance `1`.  
- Implementation Details  
    - If the noise $\epsilon ∼ \mathscr{N}(0,I)$ is not provided, then we need to sample it from normal distribution. And its shape is the same as $x_0$.  
    - To get the distribution $q(x_t|x_0)$, we can call the function `get_q_xt_x0_distribution` we've just defined.  
    - Sample from $q(x_t|x_0)$ means, apply the noise we've sampled, to get a point of distribution $q(x_t|x_0)$. The scale of $\epsilon$ is determined by the standard deviation(square root of variance, $\sqrt{var}$), and after scaling, it should be shifted by $mean$.  
### 2.4 The `sample_from_p_theta_distribution` Function  

## 3 Loss Function
$L_{simple}(\theta) = \mathbb{E}_{t, \mathbf{x}_0, \epsilon}[||\epsilon - \epsilon_\theta(\sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t}\epsilon, t)||^2]$  
``` python
def simple_loss(self, x0: torch.Tensor, epsilon: Optional[torch.Tensor]=None):
    # get batch size
    batch_size = x0.shape[0]
    # for each sample in the training batch, obtain the timestep t independently and randomly
    t = torch.randint(low=0, high=self.step_num, size=(batch_size, ), device=x0.device, dtype=torch.long) 
    # if noise is not provided, then sample it from normal distribution
    if epsilon is None:
        epsilon = torch.randn_like(x0)
    # obtain xt, by sampling from q distribution
    xt = self.sample_from_q_xt_x0(x0, t, epsilon)
    # obtain the denoising result
    epsilon_theta = self.eps_model(xt, t)
    # return the discrete mean square error
    return F.mse_loss(epsilon, epsilon_theta)
```
- Implement Details  
    - Since the loss function is the expectation of mean square deviation(with discrete form), we can implement it with `F.MSELoss`.  
    - $\epsilon_\theta$ is the denoising result of the model at timestep $t$.  
    - $\epsilon$ is the sampled noise, and should be directly passed into the loss function.  
    - For each sample in the training batch, the timestep $t$ is independently and randomly obtained.  

## References  
[1]https://python.readthedocs.io/en/stable/library/typing.html  
[2]https://pytorch.org/docs/stable/generated/torch.cumprod.html  
[3]https://nn.labml.ai/diffusion/ddpm/index.html  
[4][Denoising Diffusion Probabilistic Model](https://arxiv.org/pdf/2006.11239.pdf)  