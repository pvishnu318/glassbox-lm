"""Dataset loading and schema validation for GlassBox-LM questions.

Validates that JSONL records conform to the required schema:
- Required non-empty string fields: prompt_id, question, reference_answer
- Optional string field: source
- Unique prompt_id enforcement across all records
"""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import List, Optional, Set, Union


@dataclass(frozen=True)
class QuestionExample:
    """A validated question record."""

    prompt_id: str
    question: str
    reference_answer: str
    source: Optional[str] = None


def load_and_validate_questions(file_path: Union[str, Path]) -> List[QuestionExample]:
    """Load and validate JSONL question records.

    Args:
        file_path: Path to the .jsonl file.

    Returns:
        List of validated QuestionExample instances.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If required fields are missing, empty, or prompt_ids are duplicated.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Questions file not found: {path.resolve()}")

    records: List[QuestionExample] = []
    seen_ids: Set[str] = set()

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue

            try:
                data = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSON on line {line_no} in {path.name}: {exc}") from exc

            if not isinstance(data, dict):
                raise ValueError(f"Line {line_no} in {path.name} is not a JSON object.")

            required_fields = ("prompt_id", "question", "reference_answer")
            for field_name in required_fields:
                if field_name not in data:
                    raise ValueError(f"Missing required field '{field_name}' on line {line_no} in {path.name}")
                val = data[field_name]
                if not isinstance(val, str) or not val.strip():
                    raise ValueError(f"Field '{field_name}' must be a non-empty string on line {line_no} in {path.name}")

            prompt_id = data["prompt_id"].strip()
            if prompt_id in seen_ids:
                raise ValueError(f"Duplicate prompt_id '{prompt_id}' found on line {line_no} in {path.name}")
            seen_ids.add(prompt_id)

            source = data.get("source")
            if source is not None and not isinstance(source, str):
                raise ValueError(f"Optional field 'source' must be a string if provided on line {line_no} in {path.name}")

            records.append(
                QuestionExample(
                    prompt_id=prompt_id,
                    question=data["question"].strip(),
                    reference_answer=data["reference_answer"].strip(),
                    source=source.strip() if source is not None else None,
                )
            )

    if not records:
        raise ValueError(f"No valid records found in {path.resolve()}. File is empty.")

    return records
