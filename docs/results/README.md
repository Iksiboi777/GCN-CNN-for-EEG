# `docs/results/` — curated per-subject summaries

One file lives here. Read this page before comparing anything in it to the
headline table in the [README](../../README.md#4-headline-results); the two
measure **different things**.

## `Subject_Performance_Report_1s.csv`

| | |
|---|---|
| **Protocol** | **Session-Holdout** (train sessions 1+2, test session 3) — *not* LOSO |
| **Window** | 1 s DE features |
| **Produced by** | [`scripts/prep/performance_analysis.py`](../../scripts/prep/performance_analysis.py) |
| **How** | Scans `Errors/**/predictions.npy`, slices each run's session-3 predictions by subject using the per-subject sample counts read from `Data/ExtractedFeatures_1s`, and averages per subject across every run that matched |
| **`Num_Runs`** | 7 — the number of *different attempts* pooled, spanning several configurations |
| **Mean over the 15 subjects** | **67.6 %** (σ 14.1, range 36.4 – 94.0) |

### Why this is not the headline number

The README reports **82.3 % LOSO** for `GCN_DE` at its best configuration. This
CSV reports **67.6 %**. Both are real; they differ because:

1. **Different protocol.** This is Session-Holdout, not Leave-One-Subject-Out.
2. **Pooled across attempts.** It averages 7 runs of *varying* configurations —
   including early, weak ones — rather than reporting one model at its best
   setting. It is a *diagnostic* view of which subjects are consistently hard,
   which is what it was built for (see
   [`../reports/HARD_SUBJECTS_ANALYSIS.md`](../reports/HARD_SUBJECTS_ANALYSIS.md)).
3. **Different selection rule.** These runs predate the validation-split fix
   described in the README's *Evaluation integrity* note.

Use it to see the **subject difficulty ordering** (15, 6, 14 easy; 2, 12, 13
hard) — a pattern that is stable across protocols. Do not use it as a headline
accuracy for any architecture.

### `Classification` column

Thresholds applied by the generating script: `Hard` < 0.60, `Medium` 0.60–0.80,
`Easy` ≥ 0.80.
