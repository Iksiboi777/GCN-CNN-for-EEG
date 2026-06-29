# `scripts/analysis` — post-training analysis & visualization

**Flow.** After training writes per-run outputs to `Results/` and `Errors/`, these scripts
**aggregate** the raw predictions into headline numbers and **visualize / diagnose** them.
The aggregation scripts are how the LOSO mean±σ and confusion matrices in the README were
produced; the diagnostic scripts produced the figures behind the research findings.
(Each script hard-codes a `MODEL_NAME` / `ATTEMPT_ID` at the top — edit those to point at
your run.)

| Script | What it does |
|---|---|
| `aggregate_loso_results.py` | Collects the 15 per-subject LOSO prediction files for one run into a single global accuracy, classification report, and confusion matrix. The main aggregator. |
| `aggregate_results.py` | Lighter parser that scrapes accuracy numbers out of saved `classification_report.txt` files for a run directory. |
| `analyze_de_error.py` | Rebuilds subject/session metadata and breaks down *where* a model fails (confusion matrices, per-class reports). |
| `analyze_history.py` | Plots per-epoch learning curves, confusion matrices, and t-SNE of the learned embeddings from a run's saved history. |
| `analyze_session_drift.py` | Quantifies the statistical distance between sessions (PCA / t-SNE on DE features) — the diagnostic behind the "session drift" finding. |
| `visualize_loso_results.py` | Renders per-subject LOSO results, including MNE scalp topomaps of channel importance. |
| `analyze_per_trial.py` | *(archival — fully commented out)* Per-trial accuracy heatmaps used during the Subject-Identity-Trap investigation. |
| `visualize_results.py` | *(archival — fully commented out)* Side-by-side session confusion-matrix comparison. |
