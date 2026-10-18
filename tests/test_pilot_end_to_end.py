"""End-to-end CPU integration test for the pilot extraction pipeline.

Verifies that run_pilot executes end-to-end on CPU using a tiny random Gemma2 model
and a minimal mock tokenizer:
- Processes questions without GPU dependencies or network downloads
- Extracts internal activation features and computes output confidence metrics
- Saves features.npz, records.jsonl, and run_meta.json
- Validates metadata (python, torch, transformers versions non-empty)
- Runs sanity checks S1 through S4 (S5 GPU memory check is skipped on CPU)
"""

import importlib.util
import json
from pathlib import Path
from typing import Any, Dict, Sequence, Union
import torch
from transformers import Gemma2Config, Gemma2ForCausalLM

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load run_pilot dynamically from scripts/01_extract_pilot.py
spec = importlib.util.spec_from_file_location("extract_pilot", PROJECT_ROOT / "scripts" / "01_extract_pilot.py")
assert spec is not None and spec.loader is not None
extract_pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(extract_pilot)
run_pilot = extract_pilot.run_pilot

from glassbox.config import Config, FeaturesConfig, GenerationConfig, ModelConfig, PathsConfig, PromptConfig
from glassbox.data import QuestionExample
from glassbox.seed import set_seed
from glassbox.storage import load_run_results


class MinimalFakeTokenizer:
    """Minimal mock tokenizer for end-to-end CPU testing."""

    def __init__(self, vocab_size: int = 128, eos_token_id: int = 1, pad_token_id: int = 0) -> None:
        self.vocab_size = vocab_size
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id

    def __len__(self) -> int:
        return self.vocab_size

    def decode(self, token_ids: Union[Sequence[int], torch.Tensor], skip_special_tokens: bool = False) -> str:
        """Decode token IDs into mock text, preserving newline tokens."""
        if isinstance(token_ids, torch.Tensor):
            ids = token_ids.flatten().tolist()
        else:
            ids = list(token_ids)

        if not ids:
            return ""
        # If token 99 is present, represent as newline for testing newline boundary detection
        if any(t == 99 for t in ids):
            return "\n"
        return "Fake generated answer."

    def __call__(
        self,
        text: str,
        return_tensors: str = "pt",
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        """Convert input prompt into a 2D LongTensor of token IDs."""
        words = text.split()
        seq_len = max(5, min(len(words), 30))
        # Ensure tokens are within vocab range [2, vocab_size - 1]
        token_list = [2] + [((hash(w) % (self.vocab_size - 5)) + 4) for w in words[:seq_len]]
        input_ids = torch.tensor([token_list], dtype=torch.long)
        return {"input_ids": input_ids}


def test_pilot_end_to_end_cpu(tmp_path: Path) -> None:
    """Verify run_pilot executes end-to-end on CPU with tiny model and mock tokenizer."""
    set_seed(42)

    # 1. Instantiate tiny random Gemma2 model
    model_cfg = Gemma2Config(
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
    model = Gemma2ForCausalLM(model_cfg)
    model.eval()

    # 2. Instantiate minimal fake tokenizer
    tokenizer = MinimalFakeTokenizer(vocab_size=128, eos_token_id=1, pad_token_id=0)

    # 3. Create project configuration
    cfg = Config(
        model=ModelConfig(name="tiny-gemma2", force_float32=True),
        seed=42,
        paths=PathsConfig(
            data_dir=str(tmp_path / "data"),
            results_dir=str(tmp_path / "results"),
            logs_dir=str(tmp_path / "logs"),
            configs_dir=str(tmp_path / "configs"),
        ),
        prompt=PromptConfig(),
        generation=GenerationConfig(max_new_tokens=6),
        features=FeaturesConfig(layers=(1, 2)),
    )

    # 4. Prepare 3 fake question records
    questions = [
        QuestionExample(prompt_id="test_q1", question="What is artificial intelligence?", reference_answer="AI."),
        QuestionExample(prompt_id="test_q2", question="What is deep learning?", reference_answer="DL."),
        QuestionExample(prompt_id="test_q3", question="What is a neural activation?", reference_answer="Activation."),
    ]

    run_name = "pilot_e2e_test"
    output_root = tmp_path / "results"

    # 5. Execute run_pilot
    saved_run_path = run_pilot(
        model=model,
        tokenizer=tokenizer,
        config=cfg,
        questions=questions,
        run_name=run_name,
        output_root=output_root,
    )

    # 6. Verify directory and output files exist
    assert saved_run_path.is_dir(), f"Run directory not created at: {saved_run_path}"
    features_file = saved_run_path / "features.npz"
    records_file = saved_run_path / "records.jsonl"
    meta_file = saved_run_path / "run_meta.json"

    assert features_file.is_file(), "features.npz is missing from run directory."
    assert records_file.is_file(), "records.jsonl is missing from run directory."
    assert meta_file.is_file(), "run_meta.json is missing from run directory."

    # 7. Verify run_meta.json has non-empty python, torch, and transformers versions
    with open(meta_file, "r", encoding="utf-8") as f:
        meta_data = json.load(f)

    assert meta_data.get("python_version"), "run_meta.json python_version is missing or empty."
    assert meta_data.get("torch_version"), "run_meta.json torch_version is missing or empty."
    assert meta_data.get("transformers_version"), "run_meta.json transformers_version is missing or empty."

    # 8. Verify storage loader round-trip and row alignment
    loaded_feats, loaded_records, loaded_meta = load_run_results(saved_run_path)
    assert len(loaded_records) == 3
    assert len(loaded_feats["prompt_ids"]) == 3
    assert loaded_feats["last_prompt"].shape == (3, 2, 32)
    assert loaded_feats["last_generated"].shape == (3, 2, 32)
    assert loaded_feats["mean_generated"].shape == (3, 2, 32)
