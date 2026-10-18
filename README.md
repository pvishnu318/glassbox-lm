# GlassBox-LM

> **Work in progress (Stage 0)**

## Research Question
"Can a small probe that reads a language model's internal activations (hidden states) detect hallucinated answers better than methods that only look at the model's output, including on datasets the probe was never trained on?"

## Overview
GlassBox-LM investigates whether internal representations (hidden activations) of large language models carry distinguishable signals for factual versus hallucinated generations across varying tasks and distributions.

- **Target Model**: `google/gemma-2-2b` (base model)
- **Primary Execution Environment**: Kaggle GPU Notebook (NVIDIA GPU with compute capability >= 7.5 / 8.0)
- **Status**: Stage 0 - Environment verification, reproducible configuration, and scaffolding.
