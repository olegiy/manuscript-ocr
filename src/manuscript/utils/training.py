import random

try:
    import torch
except ImportError:
    torch = None


def set_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility across random, numpy, and PyTorch.
    """
    if torch is None:
        raise ImportError("PyTorch is required for set_seed. Install with: pip install torch")
    
    random.seed(seed)
    
    torch.manual_seed(seed)
    
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
