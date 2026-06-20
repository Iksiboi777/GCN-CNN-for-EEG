"""eeg_gnn — Graph Neural Networks for EEG emotion recognition on SEED.

A clean, reproducible implementation comparing three graph-neural-network
architectures for 3-class emotion classification from differential-entropy (DE)
EEG features:

* :class:`~eeg_gnn.models.gcn_de.GCN_DE_Model` — spectral convolution over a
  *static* physical electrode graph (strongest structural prior).
* :class:`~eeg_gnn.models.adaptive_dgcnn.Adaptive_DGCNN` — *dynamic* topology
  learned via query/key attention, blended with the static prior.
* :class:`~eeg_gnn.models.graphsage.GraphSAGE_EEG_Model` — inductive local
  neighbourhood aggregation (weakest structural prior).

Each is evaluated under two protocols: Session-Holdout (within-subject, across
sessions) and Leave-One-Subject-Out (cross-subject generalization).
"""
from __future__ import annotations

__version__ = "1.0.0"
__all__ = ["__version__"]
