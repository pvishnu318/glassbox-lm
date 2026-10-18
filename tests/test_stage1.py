"""Unit tests for Stage 1 of GlassBox-LM.

Runs strictly on CPU without downloading models, accessing network resources,
or requiring a tokenizer. A tiny random Gemma2 model is instantiated with synthetic weights.
"""

from pathlib import Path
import math
import sys
import numpy as np
import pytest
import torch
from transformers import Gemma2Config, Gemma2ForCausalLM

# Ensure src/ is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from glassbox.data import load_and_validate_questions
from glassbox.features import compute_entropy_and_logprobs, extract_features
from glassbox.generation import truncate_answer
from glassbox.seed import set_seed
from glassbox.storage import load_run_results, save_run_results


@pytest.fixture(scope="module")
def tiny_gemma2_model() -> Gemma2ForCausalLM:
    """Fixture providing a tiny randomly initialized Gemma2 model on CPU."""
    set_seed(42)
    cfg = Gemma2Config(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        query_pre_attn_scalar=16,
        final_logit_softcapping=30.0,
    )
    model = Gemma2ForCausalLM(cfg)
    model.eval()
    return model


def test_t1_output_shapes(tiny_gemma2_model: Gemma2ForCausalLM) -> None:
    """T1: Verify output shapes are (num_layers_selected, hidden_size) for all 3 pooling modes."""
    layers = [1, 2, 3]
    prompt_ids = torch.randint(1, 100, (1, 6), dtype=torch.long)
    answer_ids = torch.randint(1, 100, (1, 4), dtype=torch.long)

    feats = extract_features(tiny_gemma2_model, prompt_ids, answer_ids, layers=layers)

    expected_shape = (len(layers), 32)
    assert feats.last_prompt.shape == expected_shape, f"last_prompt shape {feats.last_prompt.shape} != {expected_shape}"
    assert feats.last_generated.shape == expected_shape, f"last_generated shape {feats.last_generated.shape} != {expected_shape}"
    assert feats.mean_generated.shape == expected_shape, f"mean_generated shape {feats.mean_generated.shape} != {expected_shape}"
    assert feats.last_prompt.dtype == torch.float32
    assert feats.last_generated.dtype == torch.float32
    assert feats.mean_generated.dtype == torch.float32


def test_t2_position_correctness(tiny_gemma2_model: Gemma2ForCausalLM) -> None:
    """T2: Verify each pooled vector matches manual indexing into hidden_states for exact positions."""
    layers = [1, 2]
    prompt_ids = torch.randint(1, 100, (1, 5), dtype=torch.long)
    answer_ids = torch.randint(1, 100, (1, 3), dtype=torch.long)
    p_len = prompt_ids.shape[1]
    a_len = answer_ids.shape[1]

    feats = extract_features(tiny_gemma2_model, prompt_ids, answer_ids, layers=layers)

    # Re-run forward pass manually to verify exact slice match
    full_ids = torch.cat([prompt_ids, answer_ids], dim=1)
    with torch.no_grad():
        out = tiny_gemma2_model(full_ids, output_hidden_states=True)

    for idx, layer_idx in enumerate(layers):
        hs = out.hidden_states[layer_idx][0]

        expected_lp = hs[p_len - 1].to(torch.float32)
        expected_lg = hs[p_len + a_len - 1].to(torch.float32)
        expected_mg = hs[p_len : p_len + a_len].mean(dim=0).to(torch.float32)

        assert torch.allclose(feats.last_prompt[idx], expected_lp, atol=1e-6)
        assert torch.allclose(feats.last_generated[idx], expected_lg, atol=1e-6)
        assert torch.allclose(feats.mean_generated[idx], expected_mg, atol=1e-6)


def test_t3_causality_masking(tiny_gemma2_model: Gemma2ForCausalLM) -> None:
    """T3: Verify causal masking: altering answer tokens does NOT alter last_prompt_token vector.

    Altering prompt tokens MUST alter last_prompt_token to ensure test validity.
    """
    layers = [1, 2]
    prompt_ids_orig = torch.tensor([[10, 20, 30, 40]], dtype=torch.long)
    answer_ids_v1 = torch.tensor([[50, 60, 70]], dtype=torch.long)
    answer_ids_v2 = torch.tensor([[80, 90, 100]], dtype=torch.long)  # Completely different answer tokens

    feats_v1 = extract_features(tiny_gemma2_model, prompt_ids_orig, answer_ids_v1, layers=layers)
    feats_v2 = extract_features(tiny_gemma2_model, prompt_ids_orig, answer_ids_v2, layers=layers)

    # Due to causal attention masking, future answer tokens cannot influence prompt token representations
    assert torch.allclose(
        feats_v1.last_prompt, feats_v2.last_prompt, atol=1e-6
    ), "Causality violation: changing answer tokens modified last_prompt activations!"

    # Verify that changing prompt tokens DOES change the representation (test is not vacuous)
    prompt_ids_diff = torch.tensor([[11, 22, 33, 44]], dtype=torch.long)
    feats_v3 = extract_features(tiny_gemma2_model, prompt_ids_diff, answer_ids_v1, layers=layers)

    assert not torch.allclose(
        feats_v1.last_prompt, feats_v3.last_prompt, atol=1e-3
    ), "Test is vacuous: different prompt tokens produced identical representations!"


def test_t4_entropy_and_logprobs_math() -> None:
    """T4: Verify entropy math on uniform, one-hot, and target token distributions."""
    vocab_size = 16
    seq_len = 2

    # Case A: Uniform logits -> p = 1/V -> entropy = log(V)
    uniform_logits = torch.zeros((seq_len, vocab_size), dtype=torch.float32)
    targets = torch.tensor([0, 1], dtype=torch.long)
    token_logprobs, token_entropies, consistency = compute_entropy_and_logprobs(uniform_logits, targets)

    expected_entropy = math.log(vocab_size)
    expected_logprob = -math.log(vocab_size)

    for h in token_entropies:
        assert math.isclose(h, expected_entropy, rel_tol=1e-5), f"Uniform entropy {h} != {expected_entropy}"
    for lp in token_logprobs:
        assert math.isclose(lp, expected_logprob, rel_tol=1e-5), f"Uniform logprob {lp} != {expected_logprob}"

    # Case B: One-hot (near deterministic) logits -> entropy ~ 0, logprob ~ 0
    one_hot_logits = torch.full((seq_len, vocab_size), -100.0, dtype=torch.float32)
    one_hot_logits[0, targets[0]] = 100.0
    one_hot_logits[1, targets[1]] = 100.0

    _, det_entropies, det_consistency = compute_entropy_and_logprobs(one_hot_logits, targets)
    for h in det_entropies:
        assert math.isclose(h, 0.0, abs_tol=1e-4), f"Deterministic entropy {h} is not close to 0"
    assert math.isclose(det_consistency, 1.0), "Deterministic greedy consistency should be 1.0"


def test_t5_truncate_answer() -> None:
    """T5: Verify truncate_answer cuts at newline and EOS, and handles empty outputs."""
    eos_id = 99
    newline_id = 45

    def is_newline(tid: int) -> bool:
        return tid == newline_id

    # 1. Truncate at EOS
    tokens_with_eos = [1, 2, 3, eos_id, 4, 5]
    res_eos = truncate_answer(tokens_with_eos, eos_id=eos_id, is_newline_token=is_newline)
    assert res_eos == [1, 2, 3], f"Expected [1, 2, 3], got {res_eos}"

    # 2. Truncate at newline
    tokens_with_newline = [10, 20, newline_id, 30]
    res_nl = truncate_answer(tokens_with_newline, eos_id=eos_id, is_newline_token=is_newline)
    assert res_nl == [10, 20], f"Expected [10, 20], got {res_nl}"

    # 3. Truncate at whichever occurs first
    tokens_mixed = [7, 8, newline_id, 9, eos_id]
    res_mixed = truncate_answer(tokens_mixed, eos_id=eos_id, is_newline_token=is_newline)
    assert res_mixed == [7, 8]

    # 4. Immediate boundary token -> empty result
    res_immediate_eos = truncate_answer([eos_id, 1, 2], eos_id=eos_id, is_newline_token=is_newline)
    assert res_immediate_eos == []

    res_empty_input = truncate_answer([], eos_id=eos_id, is_newline_token=is_newline)
    assert res_empty_input == []


def test_t6_pilot_loader_validation(tmp_path: Path) -> None:
    """T6: Verify load_and_validate_questions raises on missing fields, empty strings, and duplicates."""
    # 1. Valid file loads cleanly
    valid_file = tmp_path / "valid.jsonl"
    valid_file.write_text(
        '{"prompt_id": "q1", "question": "What is A?", "reference_answer": "A."}\n'
        '{"prompt_id": "q2", "question": "What is B?", "reference_answer": "B.", "source": "test"}\n',
        encoding="utf-8",
    )
    loaded = load_and_validate_questions(valid_file)
    assert len(loaded) == 2
    assert loaded[0].prompt_id == "q1"
    assert loaded[1].source == "test"

    # 2. Missing field raises ValueError
    missing_file = tmp_path / "missing.jsonl"
    missing_file.write_text('{"prompt_id": "q1", "question": "No answer"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="Missing required field"):
        load_and_validate_questions(missing_file)

    # 3. Empty string raises ValueError
    empty_str_file = tmp_path / "empty_str.jsonl"
    empty_str_file.write_text('{"prompt_id": "q1", "question": "   ", "reference_answer": "Valid"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="must be a non-empty string"):
        load_and_validate_questions(empty_str_file)

    # 4. Duplicate prompt_id raises ValueError
    duplicate_file = tmp_path / "duplicate.jsonl"
    duplicate_file.write_text(
        '{"prompt_id": "dup", "question": "Q1", "reference_answer": "A1"}\n'
        '{"prompt_id": "dup", "question": "Q2", "reference_answer": "A2"}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Duplicate prompt_id"):
        load_and_validate_questions(duplicate_file)


def test_t7_storage_roundtrip_and_no_overwrite(tmp_path: Path) -> None:
    """T7: Verify save/load round-trip preserves row order and overwriting raises FileExistsError."""
    run_dir = tmp_path / "test_run"

    features_data = {
        "prompt_ids": np.array(["p1", "p2"], dtype=str),
        "layers": np.array([1, 2], dtype=int),
        "last_prompt": np.random.randn(2, 2, 8).astype(np.float32),
        "last_generated": np.random.randn(2, 2, 8).astype(np.float32),
        "mean_generated": np.random.randn(2, 2, 8).astype(np.float32),
    }

    records_data = [
        {
            "prompt_id": "p1",
            "question": "Q1",
            "reference_answer": "A1",
            "generated_text": "G1",
            "empty_answer": False,
            "mean_entropy": 1.23,
        },
        {
            "prompt_id": "p2",
            "question": "Q2",
            "reference_answer": "A2",
            "generated_text": "G2",
            "empty_answer": False,
            "mean_entropy": 2.34,
        },
    ]

    meta_data = {"test_run": True, "seed": 42}

    # First save succeeds
    saved_path = save_run_results(run_dir, features_data, records_data, meta_data)
    assert saved_path.is_dir()

    # Second save to the same folder must fail loudly
    with pytest.raises(FileExistsError, match="already exists"):
        save_run_results(run_dir, features_data, records_data, meta_data)

    # Round trip load verifies alignment and content
    loaded_feats, loaded_records, loaded_meta = load_run_results(run_dir)
    assert loaded_feats["prompt_ids"].tolist() == ["p1", "p2"]
    assert np.allclose(loaded_feats["last_prompt"], features_data["last_prompt"])
    assert len(loaded_records) == 2
    assert loaded_records[0]["prompt_id"] == "p1"
    assert loaded_records[1]["prompt_id"] == "p2"
    assert loaded_meta["test_run"] is True


def test_scripts_importability() -> None:
    """Verify that all scripts can be imported without NameErrors or missing type annotations."""
    import importlib.util

    script_files = ["scripts/00_env_check.py", "scripts/01_extract_pilot.py"]
    for script_rel in script_files:
        script_path = PROJECT_ROOT / script_rel
        spec = importlib.util.spec_from_file_location(script_path.stem, script_path)
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        # Executing module verifies all top-level statements, type hints, and imports
        spec.loader.exec_module(mod)

