"""Reproducibility utilities for GlassBox-LM.

Ensures that pseudorandom number generators across Python, NumPy, and PyTorch (CPU and GPU)
are synchronized to the identical seed for scientific reproducibility.
"""

import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set random seed across Python, NumPy, and PyTorch.

    Args:
        seed: Integer seed value to initialize all generators with.

    Why each call is necessary:
    - PYTHONHASHSEED: Ensures deterministic hash values for str/bytes across runs.
    - random.seed: Seeds Python's built-in pseudorandom generator (used in samplers / shufflers).
    - np.random.seed: Seeds NumPy arrays and legacy NumPy random generation.
    - torch.manual_seed: Seeds PyTorch RNG on CPU.
    - torch.cuda.manual_seed_all: Seeds PyTorch RNG across all accessible CUDA GPUs.
    - torch.backends.cudnn.deterministic = True: Enforces deterministic convolution algorithms.
    - torch.backends.cudnn.benchmark = False: Disables algorithm benchmarking which can pick
      different algorithmic paths between runs.
    """
    if not isinstance(seed, int):
        raise TypeError(f"Seed must be an integer, got: {type(seed).__name__}")

    # Set Python hash seed for string hashing consistency
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Built-in Python random module
    random.seed(seed)

    # NumPy random generator
    np.random.seed(seed)

    # PyTorch CPU generator
    torch.manual_seed(seed)

    # PyTorch GPU generator and cuDNN backend (only if CUDA is available)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
