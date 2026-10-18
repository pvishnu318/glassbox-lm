"""Storage and retrieval of GlassBox-LM representations and run metadata.

Manages saving and loading of:
- features.npz: compressed numpy archive containing activation arrays for non-empty answers
- records.jsonl: per-question records with generation details and output-confidence metrics
- run_meta.json: environment metadata, git commit hash, configuration, and execution properties

Strict immutability:
Overwriting existing run folders is prohibited to prevent accidental data corruption.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import transformers


def get_git_commit_hash() -> Optional[str]:
    """Retrieve the current git commit hash if available."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def save_run_results(
    run_dir: Union[str, Path],
    features: Dict[str, np.ndarray],
    records: List[Dict[str, Any]],
    run_meta: Dict[str, Any],
) -> Path:
    """Save features, records, and run metadata into an immutable run directory.

    Args:
        run_dir: Target directory path (e.g. results/<run_name>).
        features: Dictionary containing prompt_ids, layers, last_prompt, last_generated, mean_generated.
        records: List of dictionaries corresponding to each evaluated question.
        run_meta: Dictionary containing execution environment metadata.

    Returns:
        Path to the newly created run directory.

    Raises:
        FileExistsError: If the run directory already exists.
    """
    target_path = Path(run_dir)
    if target_path.exists():
        raise FileExistsError(
            f"Run directory already exists and overwriting is strictly prohibited: {target_path.resolve()}"
        )

    target_path.mkdir(parents=True, exist_ok=False)

    # 1. Save features.npz (float32 activation tensors)
    npz_path = target_path / "features.npz"
    np.savez_compressed(
        npz_path,
        prompt_ids=np.asarray(features["prompt_ids"], dtype=str),
        layers=np.asarray(features["layers"], dtype=int),
        last_prompt=np.asarray(features["last_prompt"], dtype=np.float32),
        last_generated=np.asarray(features["last_generated"], dtype=np.float32),
        mean_generated=np.asarray(features["mean_generated"], dtype=np.float32),
    )

    # 2. Save records.jsonl
    records_path = target_path / "records.jsonl"
    with open(records_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # 3. Save run_meta.json
    meta_path = target_path / "run_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2, ensure_ascii=False)

    return target_path


def load_run_results(
    run_dir: Union[str, Path],
) -> Tuple[Dict[str, np.ndarray], List[Dict[str, Any]], Dict[str, Any]]:
    """Load and verify alignment between features.npz and records.jsonl.

    Args:
        run_dir: Path to the run directory.

    Returns:
        Tuple of (features_dict, records_list, run_meta_dict).

    Raises:
        FileNotFoundError: If the run directory or expected files are missing.
        ValueError: If features.npz rows do not line up with non-empty records in records.jsonl.
    """
    target_path = Path(run_dir)
    if not target_path.is_dir():
        raise FileNotFoundError(f"Run directory not found: {target_path.resolve()}")

    npz_path = target_path / "features.npz"
    records_path = target_path / "records.jsonl"
    meta_path = target_path / "run_meta.json"

    if not npz_path.is_file():
        raise FileNotFoundError(f"Missing features.npz in {target_path}")
    if not records_path.is_file():
        raise FileNotFoundError(f"Missing records.jsonl in {target_path}")
    if not meta_path.is_file():
        raise FileNotFoundError(f"Missing run_meta.json in {target_path}")

    # Load archive and copy arrays into memory
    with np.load(npz_path) as loaded:
        features = {k: loaded[k] for k in loaded.files}

    # Load records
    records: List[Dict[str, Any]] = []
    with open(records_path, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if line_str:
                records.append(json.loads(line_str))

    # Load metadata
    with open(meta_path, "r", encoding="utf-8") as f:
        run_meta = json.load(f)

    # Verify alignment between features.npz rows and non-empty records in records.jsonl
    non_empty_records = [r for r in records if not r.get("empty_answer", False)]
    npz_prompt_ids = features["prompt_ids"].tolist()

    if len(npz_prompt_ids) != len(non_empty_records):
        raise ValueError(
            f"Row count mismatch: features.npz contains {len(npz_prompt_ids)} rows, "
            f"but records.jsonl contains {len(non_empty_records)} non-empty records."
        )

    for i, (npz_id, rec) in enumerate(zip(npz_prompt_ids, non_empty_records)):
        if str(npz_id) != str(rec["prompt_id"]):
            raise ValueError(
                f"Row {i} prompt_id mismatch: features.npz has '{npz_id}', "
                f"while records.jsonl has '{rec['prompt_id']}'."
            )

    return features, records, run_meta
