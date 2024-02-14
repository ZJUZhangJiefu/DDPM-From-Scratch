# Introduction  
This repository implements the DDPM(Denoising Diffusion Probabilistic Model) in PyTorch with:  
- The **correspondence between mathematical formulae and the code**.  
- **Detailed comments** on the computational steps, and some necessary instructions from official documents to enhance the understanding of the code.  
# Repository Structure  
- A **Simpler Version** of DDPM
    - Code: `model/SimplerModel.py`  
    - Note: `note/1_Simpler_Version.md`  
    
    This section focuses on the implementation of  **sampling from distribution** $p$ and $q$, as well as the **loss function**, which helps you form **a basic understanding** of DDPM. 
- A **Complete Version** of DDPM
    - Code: `model/CompleteModel.py`
    - Note: `note/2_Complete_Version.md`  

    This section implements a complete version of  DDPM model, which provides **clear correspondence between mathmatical formulae in the [paper](https://arxiv.org/pdf/2006.11239.pdf) and the code**.  
# Summary  
To understand the implementation details of DDPM, **you only need a basic grasp of PyTorch grammar**.   
**Hope you can benefit from my notes, and many thanks for your watching and support**.  
# Future Preview  
Notes on mathematical principle of DDPM.  