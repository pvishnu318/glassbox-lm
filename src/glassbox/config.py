"""Configuration loader for GlassBox-LM.

Provides strongly typed dataclasses for project settings to ensure early validation
and IDE autocompletion rather than relying on unstructured dictionaries.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union
import yaml


@dataclass(frozen=True)
class ModelConfig:
    """Settings specifying the target language model."""

    name: str


@dataclass(frozen=True)
class PathsConfig:
    """Directory paths for datasets, results, logs, and configs."""

    data_dir: str
    results_dir: str
    logs_dir: str
    configs_dir: str


@dataclass(frozen=True)
class Config:
    """Root configuration holding all sub-configurations and experiment parameters."""

    model: ModelConfig
    seed: int
    paths: PathsConfig


def load_config(config_path: Optional[Union[str, Path]] = None) -> Config:
    """Load configuration from a YAML file into typed dataclasses.

    Args:
        config_path: Path to the YAML configuration file. Defaults to 'configs/default.yaml'.

    Returns:
        A validated Config dataclass instance.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If required sections or fields are missing.
    """
    # If no path is provided, default to configs/default.yaml in the project root.
    if config_path is None:
        target_path = Path("configs/default.yaml")
    else:
        target_path = Path(config_path)

    if not target_path.is_file():
        raise FileNotFoundError(f"Configuration file not found at: {target_path.resolve()}")

    # Safe loading prevents execution of arbitrary Python objects inside YAML files.
    with open(target_path, "r", encoding="utf-8") as f:
        raw_data: Optional[Dict[str, Any]] = yaml.safe_load(f)

    if not isinstance(raw_data, dict):
        raise ValueError(f"YAML configuration at {target_path} is empty or not a mapping.")

    # Validate mandatory top-level keys before accessing them to fail loudly on misconfiguration.
    required_keys = {"model", "seed", "paths"}
    missing_keys = required_keys - set(raw_data.keys())
    if missing_keys:
        raise ValueError(f"Missing required configuration key(s): {', '.join(sorted(missing_keys))}")

    # Validate model subsection
    model_data = raw_data.get("model")
    if not isinstance(model_data, dict) or "name" not in model_data:
        raise ValueError("Configuration section 'model' must be a mapping containing 'name'.")

    # Validate paths subsection
    paths_data = raw_data.get("paths")
    expected_paths = {"data_dir", "results_dir", "logs_dir", "configs_dir"}
    if not isinstance(paths_data, dict) or not expected_paths.issubset(paths_data.keys()):
        missing_paths = expected_paths - (set(paths_data.keys()) if isinstance(paths_data, dict) else set())
        raise ValueError(f"Configuration section 'paths' missing required key(s): {', '.join(sorted(missing_paths))}")

    # Seed validation
    seed_val = raw_data.get("seed")
    if not isinstance(seed_val, int):
        raise ValueError(f"'seed' must be an integer, got: {type(seed_val).__name__}")

    return Config(
        model=ModelConfig(name=str(model_data["name"])),
        seed=seed_val,
        paths=PathsConfig(
            data_dir=str(paths_data["data_dir"]),
            results_dir=str(paths_data["results_dir"]),
            logs_dir=str(paths_data["logs_dir"]),
            configs_dir=str(paths_data["configs_dir"]),
        ),
    )
