1\. What question am I trying to answer?

I am trying to find out whether linear probes trained on the middle-layer hidden states of google/gemma-2-2b can detect when the model gives an incorrect factual answer better than standard output-confidence scores like softmax entropy and log-probability, both on general English trivia and on an out-of-distribution test set of Indian law and finance questions.



2\. Why did I pick this project over the others?

I chose GlassBox-LM over the other project ideas because it runs cleanly on free Kaggle T4 GPUs, tests a focused research question with clear controls, and won't get blocked by external live-data limits or browser hardware bugs.



3\. What result would make me happy, and what result would I still accept?

I would be happy if a middle-layer probe beats softmax entropy on held-out TriviaQA and still holds up on my Indian legal and financial question set (my ambition is a 5 to 12 AUROC point gain, though I know that is only a goal and not guaranteed). At the same time, I will still accept a negative or mixed result—where the probe ties or loses to output entropy, or fails under domain shift—as long as my code, labels, and controls are bug-free and I report the exact measured numbers honestly.



4\. What would make me stop or change direction?

First, all choices like layer, token position, regularization strength, and threshold will be picked strictly on VAL and never tuned on TEST\_ID or OOD\_INDIA. Second, if Gemma-2-2B gets almost everything right or wrong so that either the correct or incorrect class has fewer than about 50 examples, I will stop and adjust the question difficulty before training any probe rather than changing the model. Third, if the probe cannot beat \~0.58 AUROC on in-distribution data after I check for label-alignment and token-position bugs and run the shuffled-label control, I will stop trying to improve it and report that result honestly.

