"""Configuration loader for GlassBox-LM.

Provides strongly typed dataclasses for project settings to ensure early validation
and IDE autocompletion rather than relying on unstructured dictionaries.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import yaml


@dataclass(frozen=True)
class ModelConfig:
    """Settings specifying the target language model."""

    name: str
    force_float32: bool = False


@dataclass(frozen=True)
class PathsConfig:
    """Directory paths for datasets, results, logs, and configs."""

    data_dir: str
    results_dir: str
    logs_dir: str
    configs_dir: str


@dataclass(frozen=True)
class FewShotExample:
    """A single few-shot question-answer demonstration pair."""

    question: str
    answer: str


@dataclass(frozen=True)
class PromptConfig:
    """Prompt template settings, including few-shot demonstrations."""

    few_shot_examples: Tuple[FewShotExample, ...] = field(
        default_factory=lambda: (
            FewShotExample("What is the capital of France?", "Paris."),
            FewShotExample("What is the boiling point of water in Celsius?", "100 degrees Celsius."),
            FewShotExample("How many continents are there on Earth?", "Seven continents."),
        )
    )


@dataclass(frozen=True)
class GenerationConfig:
    """Greedy generation parameters."""

    max_new_tokens: int = 24


@dataclass(frozen=True)
class FeaturesConfig:
    """Feature extraction parameters specifying layer selection."""

    layers: Tuple[int, ...] = (6, 8, 10, 12, 14, 16, 18, 20, 22, 24)


@dataclass(frozen=True)
class Config:
    """Root configuration holding all sub-configurations and experiment parameters."""

    model: ModelConfig
    seed: int
    paths: PathsConfig
    prompt: PromptConfig = field(default_factory=PromptConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    features: FeaturesConfig = field(default_factory=FeaturesConfig)


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
    if config_path is None:
        target_path = Path("configs/default.yaml")
    else:
        target_path = Path(config_path)

    if not target_path.is_file():
        raise FileNotFoundError(f"Configuration file not found at: {target_path.resolve()}")

    with open(target_path, "r", encoding="utf-8") as f:
        raw_data: Optional[Dict[str, Any]] = yaml.safe_load(f)

    if not isinstance(raw_data, dict):
        raise ValueError(f"YAML configuration at {target_path} is empty or not a mapping.")

    # Validate mandatory top-level keys
    required_keys = {"model", "seed", "paths"}
    missing_keys = required_keys - set(raw_data.keys())
    if missing_keys:
        raise ValueError(f"Missing required configuration key(s): {', '.join(sorted(missing_keys))}")

    # Validate model subsection
    model_data = raw_data.get("model")
    if not isinstance(model_data, dict) or "name" not in model_data:
        raise ValueError("Configuration section 'model' must be a mapping containing 'name'.")
    force_float32 = bool(model_data.get("force_float32", False))

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

    # Parse optional prompt config
    prompt_data = raw_data.get("prompt", {})
    if isinstance(prompt_data, dict) and "few_shot_examples" in prompt_data:
        raw_examples = prompt_data["few_shot_examples"]
        examples_list: List[FewShotExample] = []
        for ex in raw_examples:
            if isinstance(ex, dict) and "question" in ex and "answer" in ex:
                examples_list.append(FewShotExample(question=str(ex["question"]), answer=str(ex["answer"])))
        prompt_cfg = PromptConfig(few_shot_examples=tuple(examples_list))
    else:
        prompt_cfg = PromptConfig()

    # Parse optional generation config
    gen_data = raw_data.get("generation", {})
    if isinstance(gen_data, dict) and "max_new_tokens" in gen_data:
        gen_cfg = GenerationConfig(max_new_tokens=int(gen_data["max_new_tokens"]))
    else:
        gen_cfg = GenerationConfig()

    # Parse optional features config
    feat_data = raw_data.get("features", {})
    if isinstance(feat_data, dict) and "layers" in feat_data:
        feat_cfg = FeaturesConfig(layers=tuple(int(layer_idx) for layer_idx in feat_data["layers"]))
    else:
        feat_cfg = FeaturesConfig()

    return Config(
        model=ModelConfig(name=str(model_data["name"]), force_float32=force_float32),
        seed=seed_val,
        paths=PathsConfig(
            data_dir=str(paths_data["data_dir"]),
            results_dir=str(paths_data["results_dir"]),
            logs_dir=str(paths_data["logs_dir"]),
            configs_dir=str(paths_data["configs_dir"]),
        ),
        prompt=prompt_cfg,
        generation=gen_cfg,
        features=feat_cfg,
    )
