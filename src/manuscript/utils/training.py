"""Training utilities for manuscript-ocr."""

import random

try:
    import torch
except ImportError:
    torch = None


def set_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility across random, numpy, and PyTorch.
    
    This function sets seeds for:
    - Python's random module
    - PyTorch (CPU and CUDA)
    - Optimizes PyTorch for performance with cuDNN
    
    Parameters
    ----------
    seed : int, default=42
        Random seed value.
        
    Examples
    --------
    >>> set_seed(42)
    >>> import torch
    >>> torch.rand(1).item()  # Reproducible
    0.8823...
    
    >>> set_seed(42)
    >>> torch.rand(1).item()  # Same value
    0.8823...
    
    Notes
    -----
    - Enables cuDNN benchmark mode for faster training (non-deterministic)
    - Disables deterministic mode for better performance
    - Enables TF32 on Ampere GPUs for faster mixed precision
    - Call this at the start of your training script
    
    Warnings
    --------
    This does NOT guarantee full reproducibility due to:
    - cuDNN benchmark mode (set to True for performance)
    - Non-deterministic CUDA operations
    - Multi-threaded data loading
    """
    if torch is None:
        raise ImportError("PyTorch is required for set_seed. Install with: pip install torch")
    
    # Set Python random seed
    random.seed(seed)
    
    # Set PyTorch seed
    torch.manual_seed(seed)
    
    # Performance optimizations (trade determinism for speed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    
    # Enable TF32 on Ampere GPUs for faster training
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
