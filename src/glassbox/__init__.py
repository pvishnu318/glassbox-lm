"""GlassBox-LM package.

Investigating internal activations for hallucination detection.
"""

from glassbox.config import Config, load_config
from glassbox.seed import set_seed

__all__ = ["Config", "load_config", "set_seed"]
