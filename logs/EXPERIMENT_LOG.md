# Experiment Log

| Date | Tried | Expected | Happened | Changed |
| --- | --- | --- | --- | --- |
| 2026-10-18 | Initial project setup (Stage 0 scaffolding) | Establish reproducible config, seed utility, tests, and env check script | All local files created and unit tests passed | Created repository structure |
| 2026-10-18 | Rename `torch_dtype` to `dtype` and expand sanity check prompt in `scripts/00_env_check.py` | Eliminate deprecation warning in transformers; document float16 stability on longer texts for Stage 1 | Script updated, py_compile and pytest passed cleanly | Renamed parameter to `dtype`, expanded `test_sentence` in `00_env_check.py` |
| 2026-10-18 | Stage 1 implementation: generation, feature extraction, storage, and pilot pipeline | Modular architecture for answer generation, teacher-forced feature extraction, confidence metrics, and storage | All 11 unit tests (T1-T7 and smoke tests) passed on CPU with tiny Gemma2 model | Created model_loader, prompting, generation, features, data, storage, 01_extract_pilot, test_stage1 |


