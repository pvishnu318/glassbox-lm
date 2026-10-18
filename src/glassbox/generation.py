"""Greedy answer generation and token-level truncation for GlassBox-LM.

Ensures strict token-level alignment:
- Answers are generated greedily (do_sample=False).
- Generated tokens are truncated at the first EOS or newline boundary.
- The full sequence is constructed via tensor concatenation of prompt_ids + answer_ids,
  NEVER by re-tokenizing decoded text (which could alter BPE merges and misalign positions).
"""

from dataclasses import dataclass
from typing import Callable, List, Sequence, Union
import torch


@dataclass(frozen=True)
class GenerationResult:
    """Container holding generation outputs and token alignment data."""

    prompt_ids: torch.Tensor
    answer_ids: torch.Tensor
    answer_token_ids: List[int]
    generated_text: str
    empty_answer: bool
    prompt_len: int
    answer_len: int


def truncate_answer(
    token_ids: Union[Sequence[int], torch.Tensor],
    eos_id: int,
    is_newline_token: Callable[[int], bool],
) -> List[int]:
    """Truncate generated token IDs at the first EOS token or newline boundary.

    Both the EOS token and the newline token itself are excluded from the truncated result.
    This pure function does not require a tokenizer instance, facilitating isolated unit testing.

    Args:
        token_ids: Sequence or 1D tensor of newly generated token IDs.
        eos_id: Integer ID of the end-of-sequence token.
        is_newline_token: Callable predicate returning True if a token contains a newline.

    Returns:
        List of integer token IDs truncated before the first boundary token.
    """
    if isinstance(token_ids, torch.Tensor):
        items = token_ids.detach().cpu().flatten().tolist()
    else:
        items = list(token_ids)

    truncated: List[int] = []
    for token in items:
        # Cut at EOS token (EOS excluded)
        if token == eos_id:
            break
        # Cut at first token whose text contains a newline (newline token excluded)
        if is_newline_token(token):
            break
        truncated.append(int(token))

    return truncated


def generate_answer(
    model: torch.nn.Module,
    tokenizer: object,
    prompt_ids: torch.Tensor,
    max_new_tokens: int = 24,
) -> GenerationResult:
    """Generate an answer using greedy decoding and token-level truncation.

    Args:
        model: Language model with generate() method.
        tokenizer: Tokenizer corresponding to the model.
        prompt_ids: 2D LongTensor of shape (1, prompt_len).
        max_new_tokens: Maximum number of tokens to generate.

    Returns:
        GenerationResult containing tensors, token IDs, decoded string, and empty flag.
    """
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is None:
        raise ValueError("Tokenizer must define an eos_token_id.")

    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = eos_token_id

    # Greedy decoding up to max_new_tokens
    with torch.no_grad():
        full_output = model.generate(
            prompt_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
        )

    # Slice newly generated tokens (excluding prompt)
    prompt_len = prompt_ids.shape[1]
    raw_new_ids = full_output[0, prompt_len:].detach()

    # Predicate to detect if token decoded string contains a newline
    def is_newline_token(tid: int) -> bool:
        decoded_piece = tokenizer.decode([tid], skip_special_tokens=False)
        return "\n" in decoded_piece or "\r" in decoded_piece

    truncated_ids = truncate_answer(
        token_ids=raw_new_ids,
        eos_id=eos_token_id,
        is_newline_token=is_newline_token,
    )

    answer_len = len(truncated_ids)
    empty_answer = (answer_len == 0)

    if empty_answer:
        answer_ids = torch.empty((1, 0), dtype=torch.long, device=prompt_ids.device)
        generated_text = ""
    else:
        answer_ids = torch.tensor([truncated_ids], dtype=torch.long, device=prompt_ids.device)
        generated_text = tokenizer.decode(truncated_ids, skip_special_tokens=True).strip()

    return GenerationResult(
        prompt_ids=prompt_ids,
        answer_ids=answer_ids,
        answer_token_ids=truncated_ids,
        generated_text=generated_text,
        empty_answer=empty_answer,
        prompt_len=prompt_len,
        answer_len=answer_len,
    )
