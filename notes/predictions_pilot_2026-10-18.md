# Pilot Run Predictions (`pilot_questions.jsonl`)

* **Date Recorded:** 2026-10-18
* **Timing Note (Flag A disclosure):** The first version of this prediction list was written right after seeing the console printout of the first run attempt. Starting in Stage 2, all prediction files in `notes/` will be written, dated, and committed to git before running any extraction script.

## Predictions vs. Actual (`pilot_002`)

| ID | Question Topic | Reference Answer | My Prediction (Will Gemma Get It Right?) | Why I Predicted That | Actual Gemma Output (`pilot_002`) | Actual Verdict | Prediction Matched? |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `pilot_01` | Mountain farthest from Earth's center | `Mount Chimborazo` | **Wrong** | Strong pretraining prior for "highest mountain = Mount Everest". | `Mount Everest` | Wrong | Matched |
| `pilot_02` | Largest desert on Earth by area | `Antarctica` | **Right** | Thought the model would know the standard trivia fact about Antarctica. | `The Sahara Desert` | Wrong (definition-dependent) | Missed |
| `pilot_03` | Longest river in the US | `Missouri River` | **Wrong** | Models usually default to the more famous Mississippi River. | `The Mississippi River` | Wrong (measurement-dependent) | Matched |
| `pilot_04` | Great Wall visible from the Moon | `No` | **Right** | Common myth that is widely debunked in training text. | `No` | Right | Matched |
| `pilot_05` | Are bats blind? | `No` | **Right** | Well-known science FAQ ("No, bats can see"). | `No, bats have excellent eyesight` | Right | Matched |
| `pilot_06` | Can lightning strike same place twice? | `Yes` | **Right** | Common weather FAQ with clear consensus. | `Yes` | Right | Matched |
| `pilot_07` | BNS 2023 section for murder | `Section 103` | **Wrong** | Expected stale knowledge from the old IPC (`Section 302`). | `Section 302` | Wrong | Matched |
| `pilot_08` | Year SEBI Act was enacted | `1992` | **Right** | "SEBI Act, 1992" appears as a fixed phrase in Indian finance text. | `1992` | Right | Matched |
| `pilot_09` | Number of RBI MPC members | `6` | **Right** | Fixed 6-member structure since 2016 is widely mentioned. | `Six members` | Right | Matched |
| `pilot_10` | Year Section 66A IT Act struck down | `2015` | **Right** | Landmark *Shreya Singhal (2015)* Supreme Court ruling. | `2015` | Right | Matched |
| `pilot_11` | DICGC bank deposit insurance limit | `Rs. 5 lakh` | **Wrong** | Expected confusion with the old pre-2020 `Rs. 1 lakh` limit. | `Rs. 5 lakh` | Right | Missed |
| `pilot_12` | Year full T+1 equity settlement completed | `2023` | **Wrong** | Phased rollout (2021–2023) makes the final completion year easy for a 2B model to miss. | `2017` | Wrong | Matched |