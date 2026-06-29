# `eeg_gnn.data` — data layer

**Flow.** `seed.py` loads SEED's pre-computed DE `.mat` files → optionally augments each
sample with the rolling-variance channel from `features.py` (5 → 10 features) →
`normalization.py` z-scores within each `(subject, session)` group → returns a flat
`SeedBundle` that `train.py` consumes. `feature_engineering.py` is a separate, heavier
preprocessing path used only by the exploratory GraphSAGE experiments.

| Script | What it does |
|---|---|
| `seed.py` | Loads the SEED `de_LDS*` feature files and assembles an aligned `SeedBundle` (`X`, `y`, `sessions`, `subjects`, `trials`). The main entry point of the data layer. |
| `features.py` | DE feature engineering: the canonical frequency bands, `compute_rolling_variance` (the 5 → 10 channel augmentation), and a dataset-agnostic raw→DE extractor. |
| `normalization.py` | `groupwise_zscore`: independent z-scoring within each `(subject, session)` group — the key cross-session stabiliser. |
| `feature_engineering.py` | `SmartPreprocessor` (RobustScaler + MNE bad-channel interpolation) and channel-name helpers; used by the exploratory preprocessing pipeline, not the canonical loader. |
| `__init__.py` | Public API; exposes the SEED loader **lazily** so numpy-only consumers (and unit tests) don't import `scipy` until they load data. |
