"""Prompt template and formatting utilities for GlassBox-LM.

Because base models (like google/gemma-2-2b) are not instruction-tuned,
they require few-shot demonstration patterns to generate concise, direct answers
without drifting into unconstrained continuation or dialogue.
"""

from typing import Any, Dict, Sequence, Union
import torch
from glassbox.config import FewShotExample


def format_prompt(
    question: str,
    few_shot_examples: Sequence[Union[FewShotExample, Dict[str, str], Sequence[str]]],
) -> str:
    """Format a target question using few-shot question-answer demonstrations.

    Format:
      Question: <example question>
      Answer: <example answer>

      ... three examples ...

      Question: <the real question>
      Answer:

    Args:
        question: The real question to be answered.
        few_shot_examples: Sequence of 3 few-shot demonstration pairs.

    Returns:
        Formatted prompt string ending with 'Answer:'.
    """
    blocks = []
    for ex in few_shot_examples:
        if isinstance(ex, FewShotExample):
            q, a = ex.question, ex.answer
        elif isinstance(ex, dict):
            q, a = ex["question"], ex["answer"]
        else:
            q, a = ex[0], ex[1]
        blocks.append(f"Question: {q.strip()}\nAnswer: {a.strip()}")

    target_block = f"Question: {question.strip()}\nAnswer:"
    blocks.append(target_block)

    # Separate each question-answer block with a blank line (double newline)
    return "\n\n".join(blocks)


def tokenize_prompt(
    prompt_text: str,
    tokenizer: Any,
    device: Union[str, torch.device] = "cpu",
) -> torch.Tensor:
    """Tokenize the full prompt with tokenizer's default settings (including BOS).

    Processes exactly ONE prompt at a time (batch size 1, no padding) to completely
    eliminate padding-position misalignment bugs.

    Args:
        prompt_text: Text produced by format_prompt.
        tokenizer: Pretrained Hugging Face tokenizer.
        device: Target torch device for the output tensor.

    Returns:
        A 2D LongTensor of shape (1, prompt_len).
    """
    encoded = tokenizer(prompt_text, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    return input_ids
