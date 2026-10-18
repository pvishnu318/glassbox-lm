"""Shared model and tokenizer loading utilities for GlassBox-LM.

Provides centralized logic for Hugging Face authentication, hardware-aware dtype selection
(bfloat16 for compute capability >= 8.0, float16 for older GPUs, and optional float32 override),
and loading the base language model onto the target device.
"""

import os
from typing import Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_hf_token() -> str:
    """Retrieve the Hugging Face access token from the environment.

    Returns:
        The token string (never logged or printed).

    Raises:
        ValueError: If HF_TOKEN is missing or empty.
    """
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise ValueError(
            "HF_TOKEN environment variable is missing or empty.\n"
            "The model 'google/gemma-2-2b' is gated and requires acceptance of terms on Hugging Face.\n"
            "To set it:\n"
            "  - On Kaggle: Add-ons -> Secrets -> Add 'HF_TOKEN' with your read token.\n"
            "  - In Bash/Zsh: export HF_TOKEN=\"hf_...\"\n"
            "  - In PowerShell: $env:HF_TOKEN=\"hf_...\"\n"
            "SECURITY: Never commit or hard-code access tokens into source files."
        )
    return token


def select_dtype(compute_major: int, force_float32: bool = False) -> torch.dtype:
    """Determine the optimal PyTorch floating-point dtype.

    Args:
        compute_major: NVIDIA GPU compute capability major version (e.g. 7 for T4, 8 for A100).
        force_float32: If True, forces float32 regardless of GPU capability (useful if float16 yields NaNs).

    Returns:
        torch.bfloat16 if compute_major >= 8 and not force_float32;
        torch.float16 if compute_major < 8 and not force_float32;
        torch.float32 if force_float32 is True.

    Why hardware capability dictates dtype:
    - Native bfloat16 hardware acceleration begins with NVIDIA Ampere (compute capability >= 8.0).
    - Older GPUs (Turing/T4 at 7.5, Pascal/P100 at 6.0) lack bfloat16 tensor core math and require
      float16 to avoid severe performance degradation or runtime incompatibilities.
    - If float16 causes numerical overflow/underflow (NaNs), force_float32 provides an escape hatch.
    """
    if force_float32:
        return torch.float32
    if compute_major >= 8:
        return torch.bfloat16
    return torch.float16


def load_model_and_tokenizer(
    model_name: str,
    hf_token: str,
    compute_major: int,
    force_float32: bool = False,
    device_map: str = "cuda",
) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    """Load the tokenizer and causal language model with appropriate precision and device placement.

    Args:
        model_name: Hugging Face model repository identifier (e.g. 'google/gemma-2-2b').
        hf_token: Authentication token for gated model access.
        compute_major: CUDA compute capability major version.
        force_float32: Whether to bypass half-precision and force full float32.
        device_map: PyTorch device placement strategy (defaults to 'cuda').

    Returns:
        A tuple of (tokenizer, model).
    """
    dtype = select_dtype(compute_major=compute_major, force_float32=force_float32)

    # Note: Using dtype= instead of deprecated torch_dtype=
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=hf_token,
        dtype=dtype,
        device_map=device_map,
    )
    return tokenizer, model
