"""GlassBox-LM package.

Investigating internal activations for hallucination detection.
"""

from glassbox.config import Config, load_config
from glassbox.data import QuestionExample, load_and_validate_questions
from glassbox.features import ExtractedFeatures, extract_features
from glassbox.generation import GenerationResult, generate_answer, truncate_answer
from glassbox.model_loader import get_hf_token, load_model_and_tokenizer, select_dtype
from glassbox.prompting import format_prompt, tokenize_prompt
from glassbox.seed import set_seed
from glassbox.storage import load_run_results, save_run_results

__all__ = [
    "Config",
    "load_config",
    "set_seed",
    "get_hf_token",
    "select_dtype",
    "load_model_and_tokenizer",
    "format_prompt",
    "tokenize_prompt",
    "truncate_answer",
    "generate_answer",
    "GenerationResult",
    "extract_features",
    "ExtractedFeatures",
    "load_and_validate_questions",
    "QuestionExample",
    "save_run_results",
    "load_run_results",
]
