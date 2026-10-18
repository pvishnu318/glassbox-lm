"""Stage 1: Pilot extraction script for GlassBox-LM on Kaggle GPU.

Loads question records, runs greedy decoding, extracts internal activations
and output confidence metrics, stores results in results/<run_name>/, and executes
five rigorous sanity checks (S1 through S5).

NOTE: This script is intended for execution on GPU (e.g. Kaggle GPU notebook).
"""

import argparse
from datetime import datetime, timezone
import math
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Union
import numpy as np
import torch
import transformers

# Ensure src/ is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from glassbox.config import Config, load_config
from glassbox.data import QuestionExample, load_and_validate_questions
from glassbox.features import ExtractedFeatures, extract_features
from glassbox.generation import generate_answer
from glassbox.model_loader import get_hf_token, load_model_and_tokenizer
from glassbox.prompting import format_prompt, tokenize_prompt
from glassbox.seed import set_seed
from glassbox.storage import get_git_commit_hash, save_run_results


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="GlassBox-LM Pilot Feature Extraction (Stage 1)")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the questions JSONL file (e.g. datasets_src/pilot_questions.jsonl)",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        required=True,
        help="Unique identifier for this run (creates results/<run_name>/)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML configuration file (defaults to configs/default.yaml)",
    )
    return parser.parse_args()


def run_sanity_checks(
    model: torch.nn.Module,
    tokenizer: Any,
    saved_features: Dict[str, np.ndarray],
    extracted_features_list: List[ExtractedFeatures],
    first_example_prompt_ids: Optional[torch.Tensor],
    first_example_answer_ids: Optional[torch.Tensor],
    layers: Sequence[int],
    total_time: float,
    num_examples: int,
) -> None:
    """Run and report the 5 sanity checks (S1-S5)."""
    print("\n" + "=" * 60)
    print("RUNNING SANITY CHECKS (S1 - S5)")
    print("=" * 60)

    # --- S1: Check for NaN or Inf in saved feature arrays ---
    s1_arrays = ["last_prompt", "last_generated", "mean_generated"]
    s1_has_nan_or_inf = False
    for arr_name in s1_arrays:
        arr = saved_features[arr_name]
        if np.isnan(arr).any() or np.isinf(arr).any():
            s1_has_nan_or_inf = True
            print(f"  [FAIL] S1: Found NaN/Inf in array '{arr_name}'.")

    if not s1_has_nan_or_inf:
        print("  [PASS] S1: No NaN or Inf found in any saved activation array.")

    # --- S2: Report greedy-consistency fraction ---
    if extracted_features_list:
        consistencies = [f.greedy_consistency for f in extracted_features_list]
        mean_consistency = float(np.mean(consistencies))
        min_consistency = float(np.min(consistencies))
        print(
            f"  [PASS] S2: Greedy consistency across examples: mean = {mean_consistency:.4f}, "
            f"min = {min_consistency:.4f} (fraction of tokens matching greedy argmax)."
        )
    else:
        print("  [SKIP] S2: No non-empty answers to evaluate greedy consistency.")

    # --- S3: Determinism check (re-extract first example) ---
    if (
        first_example_prompt_ids is not None
        and first_example_answer_ids is not None
        and first_example_answer_ids.shape[1] > 0
        and extracted_features_list
    ):
        f_re = extract_features(model, first_example_prompt_ids, first_example_answer_ids, layers=layers)
        f_orig = extracted_features_list[0]

        diff_lp = (f_orig.last_prompt - f_re.last_prompt).abs().max().item()
        diff_lg = (f_orig.last_generated - f_re.last_generated).abs().max().item()
        diff_mg = (f_orig.mean_generated - f_re.mean_generated).abs().max().item()
        max_abs_diff = max(diff_lp, diff_lg, diff_mg)

        if max_abs_diff <= 1e-5:
            print(f"  [PASS] S3: Determinism verified. Max absolute difference across runs = {max_abs_diff:.6e}")
        else:
            print(f"  [FAIL] S3: Non-deterministic extraction! Max diff = {max_abs_diff:.6e}")
    else:
        print("  [SKIP] S3: First example had an empty answer or no features extracted.")

    # --- S4: Long-prompt stress test (~500 tokens) ---
    neutral_sentence = "In deep language models, activation patterns indicate latent certainty and representation structure. "
    stress_text = neutral_sentence * 35  # ~500 tokens
    device = next(model.parameters()).device
    stress_tokens = tokenize_prompt(stress_text, tokenizer, device=device)
    actual_stress_len = stress_tokens.shape[1]

    with torch.no_grad():
        stress_outputs = model(stress_tokens, output_hidden_states=True)

    stress_nan_or_inf = False
    for i, layer_hs in enumerate(stress_outputs.hidden_states):
        if torch.isnan(layer_hs).any() or torch.isinf(layer_hs).any():
            stress_nan_or_inf = True
            print(f"  [FAIL] S4: NaN or Inf found in stress test layer {i}.")
            break

    if not stress_nan_or_inf:
        print(f"  [PASS] S4: Long-prompt stress test passed ({actual_stress_len} tokens). No NaN or Inf detected.")

    # --- S5: Performance and GPU memory ---
    avg_seconds = total_time / max(num_examples, 1)
    if torch.cuda.is_available():
        peak_memory_mb = torch.cuda.max_memory_allocated(0) / (1024**2)
        print(f"  [PASS] S5: Average time per example: {avg_seconds:.2f}s | Peak GPU memory: {peak_memory_mb:.2f} MB")
    else:
        print(f"  [PASS] S5: Average time per example: {avg_seconds:.2f}s | (Skipping peak GPU memory on CPU)")

    print("=" * 60)


def run_pilot(
    model: torch.nn.Module,
    tokenizer: Any,
    config: Config,
    questions: Sequence[QuestionExample],
    run_name: str,
    output_root: Union[str, Path] = "results",
) -> Path:
    """Execute the pilot evaluation pipeline for a list of questions.

    Args:
        model: Causal language model.
        tokenizer: Tokenizer corresponding to model.
        config: Project configuration dataclass.
        questions: Validated question records.
        run_name: Run directory name.
        output_root: Base results directory.

    Returns:
        Path to the saved run folder.
    """
    device = next(model.parameters()).device
    is_cuda = next(model.parameters()).is_cuda
    gpu_name = torch.cuda.get_device_name(0) if is_cuda else "CPU"

    vocab_size = getattr(model.config, "vocab_size", len(tokenizer))
    max_theoretical_entropy = math.log(vocab_size)
    print(f"Model vocab size: {vocab_size} | Theoretical maximum entropy: {max_theoretical_entropy:.4f} nats\n")

    prompt_ids_list: List[str] = []
    last_prompt_list: List[np.ndarray] = []
    last_gen_list: List[np.ndarray] = []
    mean_gen_list: List[np.ndarray] = []
    extracted_features_list: List[ExtractedFeatures] = []
    records: List[Dict[str, Any]] = []

    first_example_prompt_ids: Optional[torch.Tensor] = None
    first_example_answer_ids: Optional[torch.Tensor] = None

    start_time = time.perf_counter()

    for idx, q in enumerate(questions):
        prompt_text = format_prompt(q.question, config.prompt.few_shot_examples)
        prompt_ids = tokenize_prompt(prompt_text, tokenizer, device=device)

        gen_res = generate_answer(
            model=model,
            tokenizer=tokenizer,
            prompt_ids=prompt_ids,
            max_new_tokens=config.generation.max_new_tokens,
        )

        if idx == 0:
            first_example_prompt_ids = prompt_ids
            first_example_answer_ids = gen_res.answer_ids

        rec: Dict[str, Any] = {
            "prompt_id": q.prompt_id,
            "question": q.question,
            "reference_answer": q.reference_answer,
            "source": q.source,
            "generated_text": gen_res.generated_text,
            "answer_token_ids": gen_res.answer_token_ids,
            "prompt_len": gen_res.prompt_len,
            "answer_len": gen_res.answer_len,
            "empty_answer": gen_res.empty_answer,
        }

        if not gen_res.empty_answer:
            feats = extract_features(
                model=model,
                prompt_ids=prompt_ids,
                answer_ids=gen_res.answer_ids,
                layers=config.features.layers,
            )
            extracted_features_list.append(feats)

            prompt_ids_list.append(q.prompt_id)
            last_prompt_list.append(feats.last_prompt.numpy())
            last_gen_list.append(feats.last_generated.numpy())
            mean_gen_list.append(feats.mean_generated.numpy())

            rec.update(
                {
                    "mean_logprob": feats.mean_logprob,
                    "min_logprob": feats.min_logprob,
                    "first_token_logprob": feats.first_token_logprob,
                    "mean_entropy": feats.mean_entropy,
                    "max_entropy": feats.max_entropy,
                    "first_token_entropy": feats.first_token_entropy,
                    "perplexity": feats.perplexity,
                    "greedy_consistency": feats.greedy_consistency,
                    "token_logprobs": feats.token_logprobs,
                    "token_entropies": feats.token_entropies,
                }
            )
            entropy_str = f"{feats.mean_entropy:.4f}"
        else:
            entropy_str = "N/A (empty)"
            rec.update(
                {
                    "mean_logprob": None,
                    "min_logprob": None,
                    "first_token_logprob": None,
                    "mean_entropy": None,
                    "max_entropy": None,
                    "first_token_entropy": None,
                    "perplexity": None,
                    "greedy_consistency": None,
                    "token_logprobs": [],
                    "token_entropies": [],
                }
            )

        records.append(rec)
        print(
            f"[{q.prompt_id}] Q: {q.question[:35]}... | Gen: {gen_res.generated_text[:30]} | "
            f"Len: {gen_res.answer_len} | Entropy: {entropy_str}"
        )

    total_time = time.perf_counter() - start_time

    # Save results
    num_layers = len(config.features.layers)
    hidden_size = getattr(model.config, "hidden_size", 2304)

    if last_prompt_list:
        saved_features = {
            "prompt_ids": np.array(prompt_ids_list, dtype=str),
            "layers": np.array(list(config.features.layers), dtype=int),
            "last_prompt": np.stack(last_prompt_list, axis=0).astype(np.float32),
            "last_generated": np.stack(last_gen_list, axis=0).astype(np.float32),
            "mean_generated": np.stack(mean_gen_list, axis=0).astype(np.float32),
        }
    else:
        saved_features = {
            "prompt_ids": np.array([], dtype=str),
            "layers": np.array(list(config.features.layers), dtype=int),
            "last_prompt": np.empty((0, num_layers, hidden_size), dtype=np.float32),
            "last_generated": np.empty((0, num_layers, hidden_size), dtype=np.float32),
            "mean_generated": np.empty((0, num_layers, hidden_size), dtype=np.float32),
        }

    run_meta = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "device": str(device),
        "gpu_name": gpu_name,
        "dtype": str(next(model.parameters()).dtype),
        "seed": config.seed,
        "layers": list(config.features.layers),
        "git_commit": get_git_commit_hash(),
        "config": {
            "model": {"name": config.model.name, "force_float32": config.model.force_float32},
            "generation": {"max_new_tokens": config.generation.max_new_tokens},
            "features": {"layers": list(config.features.layers)},
        },
    }

    run_dir = Path(output_root) / run_name
    saved_path = save_run_results(
        run_dir=run_dir,
        features=saved_features,
        records=records,
        run_meta=run_meta,
    )
    print(f"\nSaved results to {saved_path.resolve()}")

    # Execute sanity checks S1 - S5
    run_sanity_checks(
        model=model,
        tokenizer=tokenizer,
        saved_features=saved_features,
        extracted_features_list=extracted_features_list,
        first_example_prompt_ids=first_example_prompt_ids,
        first_example_answer_ids=first_example_answer_ids,
        layers=config.features.layers,
        total_time=total_time,
        num_examples=len(questions),
    )

    return saved_path


def main() -> None:
    """Execute pilot feature extraction and validation pipeline."""
    args = parse_args()

    # 1. Load config and set random seed
    cfg: Config = load_config(args.config)
    set_seed(cfg.seed)

    # 2. Check GPU environment and compute capability
    if torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability(0)
        device = "cuda"
    else:
        print("Warning: Running on CPU. Gemma-2-2B inference is intended for GPU.")
        major = 0
        device = "cpu"

    hf_token = get_hf_token()

    # 3. Load model and tokenizer
    print(f"Loading model '{cfg.model.name}' on {device}...")
    tokenizer, model = load_model_and_tokenizer(
        model_name=cfg.model.name,
        hf_token=hf_token,
        compute_major=major,
        force_float32=cfg.model.force_float32,
        device_map=device,
    )

    # 4. Load and validate input questions
    questions = load_and_validate_questions(args.input)
    print(f"Loaded {len(questions)} validated questions from {args.input}\n")

    # 5. Execute pilot pipeline
    run_pilot(
        model=model,
        tokenizer=tokenizer,
        config=cfg,
        questions=questions,
        run_name=args.run_name,
        output_root=cfg.paths.results_dir,
    )


if __name__ == "__main__":
    main()
