import numpy as np
import torch 
import random
def set_seed(seed):
    """
    This function fixes all random seeds to
    ensure the reproducibility.
    """
    np.random.seed(seed) 
    torch.manual_seed(seed) 
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)











