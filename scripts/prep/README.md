# `scripts/prep` — exploratory raw→feature engineering

> ⚠️ **Largely archival.** The canonical, reproducible pipeline consumes SEED's
> **pre-computed** DE features directly (see [README §7.1](../../README.md#71-get-the-data-seed-is-not-freely-downloadable)).
> These scripts document the raw-signal feature-engineering *exploration*: several are
> fully commented out, and the live ones target intermediate `Data/` directories that are
> **not** part of the reproducible path. Kept as a record of the work, not a turnkey build.

**Flow (conceptual).** raw EEG → ICA cleaning → segmentation into windows → band-pass +
differential entropy + smoothing → normalization/feature study. No single command chains
these into the loader's `ExtractedFeatures_*` inputs.

| Script | What it does | Status |
|---|---|---|
| `ICA_for_SEED.py` | MNE-ICA artifact removal over SEED raw EEG. | live |
| `segmentation.py` | Slices raw EEG into 2 s band-decomposed windows (`fs=200`) to a memmap. | live |
| `build_custom_dataset.py` | Builds a custom 2 s / 25%-overlap DE set (Butterworth + Kalman smoothing, RobustScaler) to a memmap. | live |
| `load_custom.py` | Loads that memmapped custom dataset back (shape pickle + `np.memmap`). | live |
| `normalization_pipeline.py` | Runs `SmartPreprocessor` over `ExtractedFeatures` and caches normalized tensors (for the GraphSAGE experiments). | live |
| `calculate_band_weights.py` | Per-subject Fisher / ANOVA (`f_classif`) discriminative weights per frequency band → JSON. | live |
| `feature_inspection.py` | Visual inspection of per-subject DE feature distributions. | live |
| `performance_analysis.py` | Scans `Errors/` across runs to produce the `Subject_Performance_Report*.csv` summaries (now in `docs/results/`). | live |
| `preprocessing.py` | Dataset-agnostic raw→DE extraction functions (128 Hz / 14-channel, DREAMER-era). | live |
| `print_matlab_data.py` | Small utility that recursively prints the structure of a `.mat` file. | utility |
| `advanced_feature_analysis.py` | Per-band feature-distribution analysis. | archival (commented out) |
| `build_extracted_features.py` | Earlier DE-extraction attempt; does **not** regenerate the loader's inputs. | archival (commented out) |
| `build_preprocessed_raw.py` | ICA + ICLabel preprocessing to 2 s windows. | archival (commented out) |
| `inductive_graph.py` | Inductive-graph (`NeighborLoader`) construction for GraphSAGE. | archival (commented out) |
