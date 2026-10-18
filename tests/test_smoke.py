"""Smoke tests for GlassBox-LM.

These tests run on CPU without downloading models or accessing the internet.
They verify:
1. set_seed produces strictly reproducible random values across Python, NumPy, and PyTorch.
2. configs/default.yaml loads properly into typed dataclasses with all required fields.
"""

from pathlib import Path
import random
import sys
import numpy as np
import pytest
import torch

# Ensure src/ is on sys.path so tests can import the glassbox package without installation
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from glassbox.config import Config, load_config
from glassbox.seed import set_seed


def test_set_seed_reproducibility() -> None:
    """Verify that set_seed ensures reproducible sequences across random, numpy, and torch."""
    seed = 12345

    # Run 1 with seed 12345
    set_seed(seed)
    py_sample_1 = [random.random() for _ in range(5)]
    np_sample_1 = np.random.rand(5).tolist()
    torch_sample_1 = torch.rand(5).tolist()

    # Run 2 with the same seed 12345
    set_seed(seed)
    py_sample_2 = [random.random() for _ in range(5)]
    np_sample_2 = np.random.rand(5).tolist()
    torch_sample_2 = torch.rand(5).tolist()

    assert py_sample_1 == py_sample_2, "Python random generator is not reproducible."
    assert np_sample_1 == np_sample_2, "NumPy random generator is not reproducible."
    assert torch_sample_1 == torch_sample_2, "PyTorch random generator is not reproducible."

    # Run 3 with a different seed to verify that seeds actually change the generated output
    set_seed(99999)
    py_sample_diff = [random.random() for _ in range(5)]
    np_sample_diff = np.random.rand(5).tolist()
    torch_sample_diff = torch.rand(5).tolist()

    assert py_sample_1 != py_sample_diff, "Python random produced identical values with different seeds."
    assert np_sample_1 != np_sample_diff, "NumPy random produced identical values with different seeds."
    assert torch_sample_1 != torch_sample_diff, "PyTorch random produced identical values with different seeds."


def test_load_config_default() -> None:
    """Verify that configs/default.yaml loads into Config dataclass with expected values."""
    default_config_path = PROJECT_ROOT / "configs" / "default.yaml"
    cfg = load_config(default_config_path)

    assert isinstance(cfg, Config), f"Expected Config instance, got {type(cfg)}"
    assert cfg.model.name == "google/gemma-2-2b"
    assert cfg.seed == 42
    assert cfg.paths.data_dir == "data"
    assert cfg.paths.results_dir == "results"
    assert cfg.paths.logs_dir == "logs"
    assert cfg.paths.configs_dir == "configs"


def test_load_config_missing_file() -> None:
    """Verify that load_config raises FileNotFoundError when the file does not exist."""
    non_existent = PROJECT_ROOT / "configs" / "does_not_exist.yaml"
    with pytest.raises(FileNotFoundError):
        load_config(non_existent)


def test_load_config_invalid_format(tmp_path: Path) -> None:
    """Verify that load_config raises ValueError if required sections are missing."""
    malformed_file = tmp_path / "bad_config.yaml"
    malformed_file.write_text("model:\n  name: test\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Missing required configuration key"):
        load_config(malformed_file)
