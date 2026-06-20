#!/usr/bin/env python
"""Thin launcher so the pipeline runs without an editable install.

Prefer the installed console script after ``pip install -e .``::

    eeg-gnn-train --model_type GCN --window_size 1s --mode sub_indep

This wrapper exists for ``python scripts/train.py ...`` from a fresh clone.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eeg_gnn.train import main

if __name__ == "__main__":
    main()
