"""Environment and sanity check for GlassBox-LM on Kaggle GPU.

This script executes 8 verification steps (a-h) in order to ensure
the runtime environment has adequate GPU resources, required libraries,
valid Hugging Face authentication, and that google/gemma-2-2b can be loaded
and produce expected activation tensors with output_hidden_states=True.

NOTE: This script must ONLY be executed in a GPU-enabled environment (e.g. Kaggle).
Do not run this script on a local CPU-only development machine.
"""

from pathlib import Path
import sys
import time
from typing import Tuple
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

# Ensure src is on sys.path so glassbox modules can be imported
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from glassbox.model_loader import get_hf_token, load_model_and_tokenizer, select_dtype


def step_a() -> bool:
    """Step a: Print Python, torch, and transformers versions."""
    print("=" * 60)
    print("STEP A: Checking core library versions...")
    print(f"  Python version:       {sys.version.split()[0]}")
    print(f"  PyTorch version:      {torch.__version__}")
    print(f"  Transformers version: {transformers.__version__}")
    print("[PASS] Step A: Library versions verified.")
    return True


def step_b() -> Tuple[bool, int, int]:
    """Step b: Check CUDA availability, GPU device, memory, and compute capability.

    Returns:
        Tuple of (success_flag, major_capability, minor_capability)
    """
    print("=" * 60)
    print("STEP B: Checking CUDA GPU availability...")

    if not torch.cuda.is_available():
        print(
            "[FAIL] Step B: CUDA GPU is NOT available.\n"
            "  Warning: GlassBox-LM requires an NVIDIA GPU for Gemma-2-2B.\n"
            "  If running on Kaggle, enable 'GPU T4 x2' or 'GPU P100' in the notebook settings.\n"
            "  Stopping execution now."
        )
        sys.exit(1)

    device_count = torch.cuda.device_count()
    device_name = torch.cuda.get_device_name(0)
    total_memory_bytes = torch.cuda.get_device_properties(0).total_memory
    total_memory_gb = total_memory_bytes / (1024**3)
    major, minor = torch.cuda.get_device_capability(0)

    print(f"  GPU devices found:    {device_count}")
    print(f"  Primary GPU name:     {device_name}")
    print(f"  Total GPU memory:     {total_memory_gb:.2f} GB")
    print(f"  Compute capability:   {major}.{minor}")
    print("[PASS] Step B: CUDA GPU detected and properties verified.")
    return True, major, minor


def step_c() -> str:
    """Step c: Read Hugging Face access token from HF_TOKEN environment variable.

    Returns:
        The token string (never logged or printed).
    """
    print("=" * 60)
    print("STEP C: Verifying Hugging Face authentication token...")

    try:
        hf_token = get_hf_token()
    except ValueError as exc:
        print(f"[FAIL] Step C: {exc}")
        sys.exit(1)

    print("  HF_TOKEN found in environment (value masked for security).")
    print("[PASS] Step C: HF_TOKEN detected.")
    return hf_token


def step_d(hf_token: str, compute_major: int) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    """Step d: Load tokenizer and model using bfloat16 or float16 depending on capability."""
    print("=" * 60)
    print("STEP D: Loading model and tokenizer 'google/gemma-2-2b'...")

    model_name = "google/gemma-2-2b"
    selected_dtype = select_dtype(compute_major=compute_major)
    print(f"  Selected dtype: {selected_dtype} (based on compute capability major {compute_major})")

    try:
        tokenizer, model = load_model_and_tokenizer(
            model_name=model_name,
            hf_token=hf_token,
            compute_major=compute_major,
            device_map="cuda",
        )
    except Exception as exc:
        print(f"[FAIL] Step D: Failed to load model or tokenizer: {exc}")
        raise

    print(f"  Successfully loaded {model_name} onto {model.device}.")
    print("[PASS] Step D: Model and tokenizer loaded successfully.")
    return tokenizer, model


def step_e(tokenizer: AutoTokenizer, model: AutoModelForCausalLM) -> Tuple[torch.Tensor, ...]:
    """Step e: Run ONE forward pass on a short sentence with output_hidden_states=True."""
    print("=" * 60)
    print("STEP E: Executing forward pass with output_hidden_states=True...")

    # Note: Short sentences (~10-15 tokens) rarely trigger float16 numerical instability (underflow/overflow).
    # Float16 issues typically emerge on longer contexts. While this quick check tests a multi-sentence prompt,
    # Stage 1 will systematically evaluate full-length dataset contexts.
    test_sentence = (
        "The internal activations of a deep language model can reveal whether generated "
        "statements are grounded in factual knowledge or represent hallucinations. "
        "In this research, GlassBox-LM analyzes hidden states across intermediate transformer layers "
        "to evaluate confidence, consistency, and truthfulness across diverse question answering benchmarks."
    )
    inputs = tokenizer(test_sentence, return_tensors="pt").to("cuda")

    # inference_mode disables autograd graph building and view tracking, saving memory and compute
    with torch.inference_mode():
        outputs = model(**inputs, output_hidden_states=True)

    if outputs.hidden_states is None:
        print("[FAIL] Step E: outputs.hidden_states is None.")
        sys.exit(1)

    print("  Forward pass succeeded with hidden states captured.")
    print("[PASS] Step E: Forward pass completed.")
    return outputs.hidden_states


def step_f(hidden_states: Tuple[torch.Tensor, ...], model: AutoModelForCausalLM) -> Tuple[int, int]:
    """Step f: Print hidden-state count, tensor shape, and config-based layer dimensions."""
    print("=" * 60)
    print("STEP F: Reading hidden-state dimensions and model config...")

    num_hidden_tensors = len(hidden_states)
    sample_tensor_shape = hidden_states[0].shape

    # Read dimensions directly from the model configuration without hard-coding
    num_hidden_layers = getattr(model.config, "num_hidden_layers", None)
    hidden_size = getattr(model.config, "hidden_size", None)

    if num_hidden_layers is None or hidden_size is None:
        print("[FAIL] Step F: Could not read num_hidden_layers or hidden_size from model.config.")
        sys.exit(1)

    print(f"  Hidden-state tensors returned: {num_hidden_tensors}")
    print(f"  Sample tensor shape (layer 0): {tuple(sample_tensor_shape)}")
    print(f"  model.config.num_hidden_layers: {num_hidden_layers}")
    print(f"  model.config.hidden_size:       {hidden_size}")

    print("[PASS] Step F: Hidden-state dimensions inspected.")
    return num_hidden_layers, hidden_size


def step_g(hidden_states: Tuple[torch.Tensor, ...], num_hidden_layers: int) -> bool:
    """Step g: Check hidden-state count equals num_hidden_layers + 1 and no NaN/inf exists."""
    print("=" * 60)
    print("STEP G: Verifying hidden-state integrity...")

    # Hugging Face causal models return initial embedding layer + each decoder layer output
    expected_tensor_count = num_hidden_layers + 1
    actual_tensor_count = len(hidden_states)

    if actual_tensor_count != expected_tensor_count:
        print(
            f"[FAIL] Step G: Tensor count mismatch. Expected {expected_tensor_count} "
            f"(num_hidden_layers + 1), but got {actual_tensor_count}."
        )
        sys.exit(1)

    # Check every hidden-state tensor for numerical anomalies
    for layer_idx, tensor in enumerate(hidden_states):
        if torch.isnan(tensor).any():
            print(f"[FAIL] Step G: NaN detected in hidden state tensor at index {layer_idx}.")
            sys.exit(1)
        if torch.isinf(tensor).any():
            print(f"[FAIL] Step G: Inf detected in hidden state tensor at index {layer_idx}.")
            sys.exit(1)

    print(f"  Verified {actual_tensor_count} hidden-state tensors match num_hidden_layers + 1.")
    print("  Verified no NaN or Inf values are present across all activation tensors.")
    print("  Note: Smoke test verified on multi-sentence text. Stage 1 will test longer prompts to monitor float16 stability.")
    print("[PASS] Step G: Hidden-state integrity confirmed.")
    return True


def step_h(start_time: float) -> bool:
    """Step h: Print peak GPU memory and total script execution time."""
    print("=" * 60)
    print("STEP H: Reporting peak GPU memory and execution time...")

    peak_memory_mb = torch.cuda.max_memory_allocated(0) / (1024**2)
    peak_memory_gb = peak_memory_mb / 1024
    elapsed_seconds = time.perf_counter() - start_time

    print(f"  Peak GPU memory allocated: {peak_memory_mb:.2f} MB ({peak_memory_gb:.2f} GB)")
    print(f"  Total elapsed time:        {elapsed_seconds:.2f} seconds")
    print("[PASS] Step H: Resource consumption and timing reported.")
    print("=" * 60)
    print("ALL ENVIRONMENT AND SANITY CHECKS PASSED.")
    return True


def main() -> None:
    """Run all verification steps sequentially."""
    start_time = time.perf_counter()

    step_a()
    _, major, _ = step_b()
    hf_token = step_c()
    tokenizer, model = step_d(hf_token, major)
    hidden_states = step_e(tokenizer, model)
    num_hidden_layers, _ = step_f(hidden_states, model)
    step_g(hidden_states, num_hidden_layers)
    step_h(start_time)


if __name__ == "__main__":
    main()
