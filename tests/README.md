# `tests` — pytest suite

**Flow.** Run `pytest` from the repo root. The **numpy-only** tests run anywhere; the
graph/model tests use `pytest.importorskip` and **skip automatically** unless `torch`,
`torch_geometric`, and `mne` are installed — so the suite is green on a lightweight
checkout and exercises the models on a full install.

| Script | What it checks | Needs torch/PyG? |
|---|---|---|
| `test_config.py` | `TrainConfig` defaults reproduce the thesis's primary setup; run-id bookkeeping works. | no |
| `test_features.py` | DE feature engineering — rolling-variance shape, differential entropy. | no |
| `test_normalization.py` | Per-`(subject, session)` group-wise z-scoring preserves shape and normalises correctly. | no |
| `test_graph.py` | k-NN channel-graph construction from the 10–20 montage. | yes |
| `test_models.py` | The model registry and each architecture's forward pass. | yes |
| `test_smoke_e2e.py` | End-to-end: the real `run_session_holdout` / `run_single_subject_fold` entry points on synthetic data (both protocols, all 3 models), with stdout forced through a strict `cp1250` console to guard against non-ASCII print regressions. | yes |
