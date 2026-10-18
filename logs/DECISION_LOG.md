1\. What question am I trying to answer?

"When a small language model (Gemma-2-2B) is about to hallucinate a false fact or fake citation, can we catch it more accurately by reading its internal neural activations (middle-layer residual streams) than by just checking its output word probabilities (softmax entropy)—especially when tested on a brand-new topic it wasn't trained on (Indian legal and regulatory facts) and under 4-bit quantization?"



In simple words: Does looking inside the AI's brain catch lies better than just looking at how confident its final words sound?



2\. Why did I pick this project?

"Every company deploying LLMs struggles with hallucinations, and standard fixes like checking output token confidence fail when a model is 'confidently wrong.' Meanwhile, 99% of student projects treat LLMs as black-box APIs. I picked GlassBox-LM because it proves I can work inside the Transformer architecture itself—using PyTorch forward hooks, hidden-state linear probes, and rigorous out-of-distribution (OOD) evaluation—while packaging the findings into an interactive visual scanner that anyone can test in a browser."



Why this matters for you: It costs $0 to run on free GPUs, it cannot be dismissed as a "ChatGPT wrapper," and it directly proves both Deep Learning/NLP math and Full-Stack AI engineering.



3\. What result would make me happy, and what result would I still accept?

The Result That Makes Me Thrilled (Positive Result):

The internal linear probe (around Layer 14–18 of Gemma-2-2B) beats output softmax entropy by 5% to 12% AUROC on both the standard test set and the Indian legal/regulatory OOD test set, and retains most of its accuracy even when quantized to 4-bit (INT4).



The Result I Will Still Proudly Accept (Nuanced or Negative Result):

The internal probe beats output entropy on in-distribution data (e.g., 0.81 vs. 0.72 AUROC), but drops when tested on out-of-distribution Indian legal prompts or when compressed to 4-bit quantization.



Why a negative/mixed result is still a 10/10 resume win: Real ML engineers know that honest ablation studies are gold. Publishing a clean ROC curve and writing in your README: "Internal probes beat softmax entropy by +9% in-distribution, but degrade by 11% under domain shift unless recalibrated with 50 domain examples" makes you sound like a Staff Research Engineer, not a student chasing fake 99% accuracy numbers.



4\. What would make me stop or change direction? (Your Hard Kill-Switches)

To make sure you never get stuck in an endless 3-month loop, set these three strict boundaries right now:



The Day-3 Hardware/Extraction Kill-Switch: If extracting activations from Gemma-2-2B on 300 prompts takes more than 2 hours on a free Colab/Kaggle T4 GPU (or crashes memory repeatedly), we do not quit the project—we immediately downshift the base model to Qwen2.5-1.5B-Instruct or Llama-3.2-1B-Instruct so extraction finishes in 12 minutes.



The Day-7 Probe Sanity Kill-Switch: If the linear probe cannot even beat random guessing (AUROC < 0.58) on the same dataset it was trained on (in-distribution), it means there is a bug in how we sliced the token positions or labeled the hallucinations. We stop and fix the data/token index before touching any OOD experiments or UI.



The 3-Week Time Cap: No matter what happens, at the end of Day 21, whatever numbers and ROC curves we have go into the README and the interactive web viewer, and the project ships so you can start applying to jobs. We do not add extra features before shipping v1.0.

