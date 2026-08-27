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
| `test_smoke_e2e.py` | End-to-end: the real `run_session_holdout` / `run_single_subject_fold` entry points on synthetic data (both protocols, all 3 models), with stdout forced through a strict `cp1250` console to guard against non-ASCII print regressions. Also carries the **evaluation-integrity regression guards** below. | yes |

**Evaluation-integrity guards** (in `test_smoke_e2e.py`) — these exist because the suite
once passed while the headline protocol was broken:

- `test_loso_fold_writes_the_file_the_aggregator_reads` — a LOSO fold must write
  `final_test_preds_sub<k>.npy`, the exact path `_aggregate_loso` reads. When the engine
  did not write it, every fold counted as missing and the global summary reported
  **0.00%** — while the old “some `.npy` exists” assertion still passed.
- `test_best_epoch_is_not_selected_on_the_test_split` — the saved fold must record
  `leaky_selection == False`, i.e. the checkpoint was chosen on the validation split.
- `test_loso_aggregation_produces_a_real_score` — a full sweep aggregates every fold into
  a summary with a non-zero mean and no missing/leaky warnings.
