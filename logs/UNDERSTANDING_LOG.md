1. What is a hidden state
In our project, a hidden state is the 2,304-number vector that googlegemma-2-2b produces for each token at each of its 26 layers as text passes through the network. Instead of just seeing the final words the model prints out, this vector captures the model's internal representation at that exact layer and token position (such as the last prompt token before answering, or the generated answer tokens).

2. What is a probe
A probe is a small, separate classifier—in our case, a regularized logistic regression model—that takes those 2,304-number hidden-state vectors as input and learns to predict whether Gemma's answer is correct (0) or incorrect (1). It does not change how Gemma generates text; it only reads the frozen internal numbers to see if a trutherror signal is already present inside that layer.

3. What is AUROC
AUROC measures how well a continuous score (either from our linear probe or from an output baseline) separates the incorrect answers from the correct ones across all possible cutoff thresholds, without forcing us to pick just one threshold first. An AUROC of 0.5 means the score is no better than flipping a coin (which is what our shuffled-label control should get), while 1.0 means it ranks every single incorrect answer higher than every correct answer.

4. What is a baseline
A baseline is the simpler, cheaper method we have to compare our probe against on the exact same test questions so we know if reading hidden states is actually worth the extra effort. In GlassBox-LM, our main baselines are output-confidence numbers we already get from the final logits—like softmax entropy, token log-probability, and perplexity—because if a hidden-state probe cannot beat simple output entropy, then the internal probe isn't adding real value.