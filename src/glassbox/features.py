"""Activation feature extraction and output confidence computation for GlassBox-LM.

Performs a single forward pass over the full concatenated sequence (prompt + answer)
in eval mode with torch.no_grad().

Why a second pass and not forward hooks during generate():
During greedy decoding with KV-cache enabled, hooks inside generate() only observe
the newly generated single token at each step. Position slicing and attention
pooling across the full sequence are much cleaner, reproducible, and cannot leak hooks
when performed in a dedicated teacher-forced forward pass over concatenated tokens.
"""

from dataclasses import dataclass
import math
from typing import List, Sequence
import numpy as np
import torch


@dataclass(frozen=True)
class ExtractedFeatures:
    """Features and output-confidence metrics extracted from a forward pass."""

    last_prompt: torch.Tensor  # Shape: (num_layers, hidden_size), float32
    last_generated: torch.Tensor  # Shape: (num_layers, hidden_size), float32
    mean_generated: torch.Tensor  # Shape: (num_layers, hidden_size), float32
    layers: List[int]
    mean_logprob: float
    min_logprob: float
    first_token_logprob: float
    mean_entropy: float
    max_entropy: float
    first_token_entropy: float
    perplexity: float
    token_logprobs: List[float]
    token_entropies: List[float]
    greedy_consistency: float


def compute_entropy_and_logprobs(
    logits: torch.Tensor,
    target_token_ids: torch.Tensor,
) -> tuple[List[float], List[float], float]:
    """Compute per-token log probabilities, Shannon entropies, and greedy consistency.

    Entropy math:
      H = - sum_v p(v) * log p(v)
    Computed stably via log_softmax and softmax in float32.

    Args:
        logits: Float32 tensor of shape (seq_len, vocab_size).
        target_token_ids: LongTensor of shape (seq_len,) containing actual target tokens.

    Returns:
        Tuple of (token_logprobs, token_entropies, greedy_consistency_fraction).
    """
    logits_f32 = logits.to(torch.float32)
    log_probs = torch.log_softmax(logits_f32, dim=-1)
    probs = torch.softmax(logits_f32, dim=-1)

    # Per-token entropy: H = - sum(p * log_p)
    # Using torch.where or clamp to ensure 0 * log(0) = 0 numerically
    safe_p_log_p = torch.where(probs > 0, probs * log_probs, torch.zeros_like(probs))
    token_entropies = (-safe_p_log_p.sum(dim=-1)).cpu().tolist()

    # Per-token log probabilities of target tokens
    seq_len = target_token_ids.shape[0]
    token_logprobs = [
        float(log_probs[t, target_token_ids[t]].item())
        for t in range(seq_len)
    ]

    # Greedy consistency: fraction of target tokens matching the argmax of predicting logits
    argmax_predictions = torch.argmax(logits_f32, dim=-1)
    greedy_consistency = float((argmax_predictions == target_token_ids).to(torch.float32).mean().item())

    return token_logprobs, token_entropies, greedy_consistency


def extract_features(
    model: torch.nn.Module,
    prompt_ids: torch.Tensor,
    answer_ids: torch.Tensor,
    layers: Sequence[int] = (6, 8, 10, 12, 14, 16, 18, 20, 22, 24),
) -> ExtractedFeatures:
    """Extract activation vectors and output-confidence metrics in ONE forward pass.

    Args:
        model: Language model with forward(..., output_hidden_states=True).
        prompt_ids: 2D LongTensor of shape (1, prompt_len).
        answer_ids: 2D LongTensor of shape (1, answer_len).
        layers: Indices of transformer layers to extract hidden states from.

    Returns:
        ExtractedFeatures container with pooled activation tensors and confidence metrics.

    Raises:
        ValueError: If answer_ids has length 0 (empty answers must be skipped).
    """
    prompt_len = prompt_ids.shape[1]
    answer_len = answer_ids.shape[1]

    if answer_len == 0:
        raise ValueError("Cannot extract features for an empty answer (answer_len == 0).")

    # Construct the full sequence strictly by concatenating token IDs (NEVER re-tokenizing)
    full_ids = torch.cat([prompt_ids, answer_ids], dim=1)
    attention_mask = torch.ones_like(full_ids)

    model.eval()
    with torch.no_grad():
        outputs = model(
            input_ids=full_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

    hidden_states = outputs.hidden_states
    if hidden_states is None:
        raise RuntimeError("Model forward pass did not return hidden_states.")

    # Slicing positions:
    # prompt tokens occupy positions 0 .. P-1
    # answer tokens occupy positions P .. P+A-1
    # last_prompt_token: index P - 1
    # last_generated_token: index P + A - 1
    # mean_generated_tokens: mean over positions P .. P + A - 1
    last_prompt_list = []
    last_gen_list = []
    mean_gen_list = []

    for layer_idx in layers:
        layer_hs = hidden_states[layer_idx][0]  # Shape: (total_len, hidden_size)

        lp = layer_hs[prompt_len - 1].to(torch.float32)
        lg = layer_hs[prompt_len + answer_len - 1].to(torch.float32)
        mg = layer_hs[prompt_len : prompt_len + answer_len].mean(dim=0).to(torch.float32)

        last_prompt_list.append(lp)
        last_gen_list.append(lg)
        mean_gen_list.append(mg)

    # Stacks have shape: (num_layers_selected, hidden_size)
    last_prompt_tensor = torch.stack(last_prompt_list, dim=0).cpu()
    last_gen_tensor = torch.stack(last_gen_list, dim=0).cpu()
    mean_gen_tensor = torch.stack(mean_gen_list, dim=0).cpu()

    # Logits analysis:
    # Logits at position t predict the token at position t + 1.
    # Therefore, logits at positions (P - 1) .. (P + A - 2) predict answer tokens 0 .. A - 1.
    raw_logits = outputs.logits[0]  # Shape: (total_len, vocab_size)
    predicting_logits = raw_logits[prompt_len - 1 : prompt_len + answer_len - 1]
    target_answer_tokens = answer_ids[0].to(raw_logits.device)

    token_logprobs, token_entropies, greedy_consistency = compute_entropy_and_logprobs(
        logits=predicting_logits,
        target_token_ids=target_answer_tokens,
    )

    mean_logprob = float(np.mean(token_logprobs))
    min_logprob = float(np.min(token_logprobs))
    first_token_logprob = float(token_logprobs[0])

    mean_entropy = float(np.mean(token_entropies))
    max_entropy = float(np.max(token_entropies))
    first_token_entropy = float(token_entropies[0])

    perplexity = float(math.exp(-mean_logprob))

    return ExtractedFeatures(
        last_prompt=last_prompt_tensor,
        last_generated=last_gen_tensor,
        mean_generated=mean_gen_tensor,
        layers=list(layers),
        mean_logprob=mean_logprob,
        min_logprob=min_logprob,
        first_token_logprob=first_token_logprob,
        mean_entropy=mean_entropy,
        max_entropy=max_entropy,
        first_token_entropy=first_token_entropy,
        perplexity=perplexity,
        token_logprobs=token_logprobs,
        token_entropies=token_entropies,
        greedy_consistency=greedy_consistency,
    )
