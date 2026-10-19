\# GlassBox-LM — Concept \& Codebase Understanding Log



\## 1. Stage 0 Critical Thinking Questions (Part C)



\*\*Q1: Why does the model return `layers + 1` (27) hidden states when `num\_hidden\_layers` is 26?\*\*

When we pass `output\_hidden\_states=True` to Hugging Face's `Gemma2Model`, tuple index `0` (`hidden\_states\[0]`) is the output of the input token embedding layer before the tokens pass through any transformer block. Indices `1` through `26` are the outputs after each of the 26 transformer layers, which gives $26 + 1 = 27$ tensors in total.



\*\*Q2: Why do we fix the random seed?\*\*

We fix the random seed across Python, NumPy, and PyTorch so that every part of our workflow that involves randomness—such as sampling `TRAIN`, `VAL`, and `TEST\_ID` splits from TriviaQA, shuffling labels for our sanity-check control, initializing logistic regression probes, and running bootstrap confidence intervals—is 100% deterministic and reproducible across runs.



\*\*Q3: Why must the `HF\_TOKEN` never appear in code or on GitHub?\*\*

Our GitHub repository (`github.com/pvishnu318/glassbox-lm`) is public, so if my Hugging Face token is ever hardcoded into a script, config file, or notebook output and pushed to git, bots and scrapers will immediately find it in the commit history. Anyone with that token could impersonate my Hugging Face account, access gated models under my name, or abuse my account limits, so we keep it strictly inside Kaggle Secrets and load it at runtime via `os.environ\["HF\_TOKEN"]`.



\---



\## 2. Pre-Stage 1 Core Concepts



\*\*Q1: What is a hidden state?\*\*

In our project, a hidden state is the 2,304-number vector that `google/gemma-2-2b` produces for each token at each of its 26 layers as text passes through the network. Instead of just seeing the final words the model prints out, this vector captures the model's internal representation at that exact layer and token position (such as the last prompt token before answering, or the generated answer tokens).



\*\*Q2: What is a probe?\*\*

A probe is a small, separate classifier—in our case, a regularized logistic regression model—that takes those 2,304-number hidden-state vectors as input and learns to predict whether Gemma's answer is correct (0) or incorrect (1). It does not change how Gemma generates text; it only reads the frozen internal numbers to see if a truth/error signal is already present inside that layer.



\*\*Q3: What is AUROC?\*\*

AUROC measures how well a continuous score (either from our linear probe or from an output baseline) separates the incorrect answers from the correct ones across all possible cutoff thresholds, without forcing us to pick just one threshold first. An AUROC of 0.5 means the score is no better than flipping a coin (which is what our shuffled-label control should get), while 1.0 means it ranks every single incorrect answer higher than every correct answer.



\*\*Q4: What is a baseline?\*\*

A baseline is the simpler, cheaper method we have to compare our probe against on the exact same test questions so we know if reading hidden states is actually worth the extra effort. In GlassBox-LM, our main baselines are output-confidence numbers we already get from the final logits—like softmax entropy, token log-probability, and perplexity—because if a hidden-state probe cannot beat simple output entropy, then the internal probe isn't adding real value.



\---



\## 3. Stage 1 Critical Thinking Questions



\*\*Q1: Why can't changing the generated answer change the `last\_prompt\_token` vector?\*\*

Gemma-2 uses a causal attention mask during the forward pass, meaning any token at position $t$ can only attend to positions $0 \\dots t$ while all later tokens ($> t$) are masked out. Since `last\_prompt\_token` sits at index $P - 1$ and the generated answer tokens start at index $P$, no layer at position $P - 1$ can see any token from index $P$ onward, so the activation vector at $P - 1$ stays identical regardless of what answer follows.



\*\*Q2: Why is an off-by-one token index error dangerous even when the code runs without crashing?\*\*

Because the prompt and generated answer are concatenated into a single sequence of length $P + A$, indexing $P$ instead of $P - 1$ is still a valid in-bounds index whenever at least one answer token is generated, so PyTorch will not throw an `IndexError`. However, reading index $P$ leaks the first generated answer token into the `last\_prompt\_token` feature, and an off-by-one shift in logits pairs `logits\[t]` with the wrong target token since causal language models use `logits\[t]` to predict `input\_ids\[t + 1]`.



\*\*Q3: Why is the last layer (`hidden\_states\[26]`) excluded by default?\*\*

In Hugging Face's `Gemma2Model`, `hidden\_states\[1]` through `hidden\_states\[25]` are the raw outputs of each transformer layer, whereas `hidden\_states\[26]` has already been passed through `self.norm` (the final `RMSNorm` layer) right before the vocabulary projection head (`lm\_head`). Because it is already normalized for next-token logit projection, its scale and geometry differ from the intermediate residual stream layers (`\[6, 8, 10, 12, 14, 16, 18, 20, 22, 24]`).

**Q4: Why do we extract activations from a second forward pass over `prompt + answer` instead of grabbing them inside `generate()`?**
When `model.generate()` runs with KV caching enabled, only the first forward pass processes the full prompt of length $P$ (giving hidden states of shape `(1, P, 2304)`). On every subsequent token step, the earlier tokens' keys and values are cached, so the model only feeds the single newest token into the network and each layer's hidden state tensor has shape `(1, 1, 2304)`. Slicing across prompt and answer positions inside `generate()` would either fail or require manually stitching per-step tensors before answer truncation happens. Running one clean second forward pass over the truncated `prompt_ids + answer_ids` sequence with `output_hidden_states=True` gives the full `(1, P + A, 2304)` tensor at every layer at once, and because of causal attention (earlier tokens cannot attend to later tokens), every token's activation vector matches what was computed during greedy generation—verified by our greedy-consistency check (`S2 = 1.0000`).

\---



\## 4. One-Sentence Explanation of Every File in the Repository



\* `configs/default.yaml`: Stores the locked configuration settings for our project, including the `google/gemma-2-2b` model name, random seed, the 10 middle layer indices, the 3 token-pooling modes, and the 3-shot prompt examples.

\* `src/glassbox/config.py`: Reads and validates `configs/default.yaml` so the rest of the codebase gets typed, checked settings instead of raw dictionary lookups.

\* `src/glassbox/seed.py`: Fixes the random seed across Python's `random`, NumPy, and PyTorch (CPU and CUDA) so our data splits, model checks, and probe runs give the exact same results every time.

\* `src/glassbox/model\_loader.py`: Loads the `google/gemma-2-2b` model and tokenizer using our hardware dtype rule (`bfloat16` if GPU compute capability major $\\ge 8$, otherwise `float16`, with a `float32` fallback option).

\* `src/glassbox/prompting.py`: Formats each input question with our fixed 3-shot Q/A prefix into a single prompt string (`batch\_size=1`, no padding).

\* `src/glassbox/generation.py`: Runs greedy decoding (`do\_sample=False`, `max\_new\_tokens=24`) and truncates the generated answer token IDs at the first EOS token or newline without re-tokenizing text.

\* `src/glassbox/features.py`: Runs a second forward pass over `prompt\_ids + answer\_ids` (`output\_hidden\_states=True`, `eval()`, `torch.no\_grad()`), extracts the float32 hidden states for our 3 token-pooling positions, calculates output entropy/logprob baselines, and checks greedy consistency.

\* `src/glassbox/data.py`: Loads question `.jsonl` files and checks that every row has the required fields (`id`, `question`, accepted answers, and source metadata).

\* `src/glassbox/storage.py`: Saves extracted features (`features.npz`), per-example metadata (`records.jsonl`), and run details (`run\_meta.json`) into `results/<run\_name>/` while refusing to overwrite an existing run folder.

\* `scripts/00\_env\_check.py`: Runs the Stage 0 environment check on Kaggle (Steps A through H) to verify library versions, the GPU, `HF\_TOKEN`, model loading, hidden-state tensor shapes, NaN/Inf absence, and peak GPU memory.

\* `scripts/01\_extract\_pilot.py`: Runs the Stage 1 12-question pilot extraction and executes sanity checks S1 through S5.

\* `tests/test\_smoke.py`: Runs basic CPU unit tests on seed setting, config loading, and environment helpers without downloading the model.

\* `tests/test\_stage1.py`: Tests prompt formatting, token-ID answer truncation, pooling index math, entropy/logprob math, and storage no-overwrite protection.

\* `tests/test\_static\_checks.py`: Runs `pyflakes` across `src/`, `scripts/`, and `tests/` inside `pytest` so missing imports or undefined variable names are caught locally before running on Kaggle.

\* `tests/test\_pilot\_end\_to\_end.py`: Runs the full pilot extraction loop on CPU using a tiny randomly initialized `Gemma2Config` model so runtime bugs inside functions are caught locally.

**Q4: In the hidden-state shape `(1, 12, 2304)` (or `(1, 53, 2304)` on the longer smoke prompt), what does each number mean?**
* **`1` (Batch size):** We process one prompt at a time (`batch_size = 1`) with no padding tokens.
* **`12` (or `53`) (Sequence length in tokens):** The number of token IDs produced by the tokenizer for that input text; the model holds one vector per token position.
* **`2304` (Hidden size):** The length of the internal activation vector (`model.config.hidden_size = 2304`) that `google/gemma-2-2b` maintains for each token at each layer.

**Q5: Why did the script choose `float16` on Kaggle, what will it choose on my RTX 4060, and why does `bfloat16` vs `float16` matter?**
* **The selection rule:** In `src/glassbox/model_loader.py` (and printed directly in the Stage 0 output as `Selected dtype: torch.float16 (based on compute capability major 7)`), the code checks the GPU's compute capability major version: if `major >= 8` it selects `torch.bfloat16`, otherwise it falls back to `torch.float16`.
* **Kaggle T4 vs. RTX 4060:** The Kaggle Tesla T4 has compute capability `7.5` (`major = 7`), which lacks native hardware support for `bfloat16`, so it gets the `float16` fallback. My RTX 4060 has compute capability `8.9` (`major = 8 >= 8`), so it gets `bfloat16`.
* **Why it matters numerically:** `bfloat16` uses 8 exponent bits—the exact same dynamic range as `float32`—with fewer mantissa bits, whereas `float16` only has 5 exponent bits (max value 65,504). Because of that wider range, `bfloat16` overflows into `Inf`/`NaN` far less often than `float16` on deeper or longer Gemma-2 forward passes.
* **Lesson learned:** Never guess what a script will do when the exact rule is already printed in the console output or written in the source file.

