\# GlassBox-LM — Decision Log



\## 2026-10-18 — Entry 1: Core Research Question, Project Choice, and Stop Rules



\*\*1. What question am I trying to answer?\*\*

I am trying to find out whether linear probes trained on the middle-layer hidden states of `google/gemma-2-2b` can detect when the model gives an incorrect factual answer better than standard output-confidence scores like softmax entropy and log-probability, both on general English trivia and on an out-of-distribution test set of Indian law and finance questions.



\*\*2. Why did I pick this project over the others?\*\*

I chose GlassBox-LM over the other project ideas because it runs cleanly on free Kaggle T4 GPUs, tests a focused research question with clear controls, and won't get blocked by external live-data limits or browser hardware bugs. I also kept the repo name `glassbox-lm` because I like it, while using a descriptive README title so the research question is clear right away.



\*\*3. What result would make me happy, and what result would I still accept?\*\*

I would be happy if a middle-layer probe beats softmax entropy on held-out TriviaQA and still holds up on my Indian legal and financial question set (my ambition is a 5 to 12 AUROC point gain, though I know that is only a goal and not guaranteed). At the same time, I will still accept a negative or mixed result—where the probe ties or loses to output entropy, or fails under domain shift—as long as my code, labels, and controls are bug-free and I report the exact measured numbers honestly.



\*\*4. What would make me stop or change direction?\*\*

\* \*\*Validation-only rule:\*\* All modeling choices (layer index, token pooling mode, regularization strength, and threshold) will be selected strictly on `VAL`, never on `TEST\_ID` or `OOD\_INDIA`.

\* \*\*Class-balance rule:\*\* If `google/gemma-2-2b` gets almost everything right or almost everything wrong so that either class (`correct` or `incorrect`) has fewer than about 50 examples, I will stop and adjust the question difficulty mix before training any probes, rather than switching the base model.

\* \*\*Probe sanity stop:\*\* If the probe cannot beat \~0.58 AUROC on in-distribution validation data, I will first check for bugs (token positions, label alignment, leakage) and verify the shuffled-label control (\~0.50 AUROC). If the code is clean, I will stop tweaking and report the negative result honestly. If the whole project stalls completely, my fallback project is the contract / offer-letter trap detector.



\---

## 2026-10-18 — Validation-Only Selection and Class-Balance Rules

1. **Validation-only selection rule:** All layer, pooling position, threshold, and hyperparameter choices will be made on the validation split (`VAL`) only, never on the held-out test set (`TEST_ID`) or the Indian domain set (`OOD_INDIA`), so my reported results are never quietly tuned to the test data.
2. **Class-balance rule:** If fewer than about 50 examples end up in either class (correct or incorrect), I will stop and adjust the question difficulty mix before training any probes.



\## 2026-10-19 — Entry 2: Definition of an "Incorrect Answer" and OOD Scope



\*\*What is my definition of an "incorrect answer"?\*\*

I use the label "incorrect answer" instead of "hallucination" because a wrong factual answer can also come from outdated pretraining data or misreading a prompt. After greedy decoding and cutting at the first EOS or newline, an answer is labeled \*\*correct\*\* if its normalized text (lowercased, punctuation and articles removed, and standard number or currency forms matched like `"6"` vs `"six"` or `"Rs. 5 lakh"` vs `"5,00,000 rupees"`) matches an entry in the verified accepted-answer list. It is labeled \*\*incorrect\*\* if it gives a wrong fact, number, year, section, or entity (like giving the old IPC Section 302 when asked for the BNS murder section), or fails to answer the question. Any answer that is partially right or phrased in an unexpected way not covered by the list is marked \*\*ambiguous / needs review\*\* for hand audit rather than guessed by string matching.



\*\*What does the OOD\_INDIA set actually test?\*\*

`OOD\_INDIA` tests domain shift for the linear probe (which is trained only on TriviaQA), not whether the base `google/gemma-2-2b` model ever saw Indian law or finance text during pretraining.

## 2026-10-19 — Source Link Verification for Pilot Questions 10 and 12

I audited all 12 URLs in `datasets_src/pilot_questions.jsonl` without changing any question text or reference answers:
* **`pilot_10`:** Replaced the generic IT Act PDF link with the full text of the Supreme Court of India judgment in *Shreya Singhal v. Union of India* (decided March 24, 2015), which directly strikes down Section 66A of the Information Technology Act, 2000.
* **`pilot_12`:** Confirmed that Paragraph 1 of SEBI Circular `SEBI/HO/MRD/MRD-PoD-3/P/CIR/2024/20` on `sebi.gov.in` explicitly states on page 1: *"All stock exchanges, clearing corporations and depositories ... jointly decided to shift to T+1 settlement cycle in a phased manner, which was fully implemented w.e.f. January 27, 2023."*

