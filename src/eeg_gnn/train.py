"""Training orchestration for the EEG-GNN emotion-recognition pipeline.

Two evaluation protocols are supported via ``--mode``:

* ``sub_indep`` — Leave-One-Subject-Out (LOSO), run as parallel folds.
* ``sub_dep``   — Session-Holdout (train sessions 1+2, test session 3).

Run via the console script ``eeg-gnn-train`` or ``python scripts/train.py``.
"""
from __future__ import annotations

import argparse
import logging
import os

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset

from eeg_gnn.config import DATA_DIR, LOCS_FILE, TrainConfig, get_device, next_run_id
from eeg_gnn.data import load_seed_de
from eeg_gnn.graph import get_knn_adjacency_matrix
from eeg_gnn.models import build_model
from eeg_gnn.training import FocalLoss, evaluate, train_model_with_interrupt

logger = logging.getLogger(__name__)

CLASS_NAMES = ["Negative", "Neutral", "Positive"]


# --- CLI ---------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> TrainConfig:
    """Parse command-line flags into a :class:`TrainConfig`."""
    p = argparse.ArgumentParser(description="Train an EEG-GNN for emotion recognition.")
    p.add_argument("--mode", choices=["sub_dep", "sub_indep"], default="sub_indep",
                   help="sub_indep = LOSO, sub_dep = Session-Holdout")
    p.add_argument("--window_size", choices=["1s", "2s", "4s"], default="1s")
    p.add_argument("--model_type", choices=["GCN", "ADAPTIVE_DGCNN", "GraphSAGE"], default="GCN")
    p.add_argument("--in_features", type=int, choices=[5, 10], default=10)
    p.add_argument("--use_se", action="store_true", default=True)
    p.add_argument("--no_se", dest="use_se", action="store_false")
    p.add_argument("--use_doubling", action="store_true", default=False)
    p.add_argument("--max_parallel", type=int, default=3, help="LOSO folds run concurrently")
    p.add_argument("--val_subjects", type=int, default=2,
                   help="LOSO: training subjects held out to select the best epoch")
    p.add_argument("--val_fraction", type=float, default=0.15,
                   help="Session-Holdout: fraction of training trials held out for selection")
    p.add_argument("--seed", type=int, default=42, help="RNG seed for the run")
    a = p.parse_args(argv)
    return TrainConfig(
        mode=a.mode, window_size=a.window_size, model_type=a.model_type,
        in_features=a.in_features, use_se=a.use_se, use_doubling=a.use_doubling,
        max_parallel=a.max_parallel, val_subjects=a.val_subjects,
        val_fraction=a.val_fraction, seed=a.seed,
    )


# --- Shared training helpers -------------------------------------------------
def build_optimizer(model: nn.Module, cfg: TrainConfig) -> optim.Optimizer:
    """Adam with a stronger L2 penalty on the AGLI gain (``static_norm.gamma``).

    The heavier weight decay on the input-gain parameter drives unreliable
    channels toward silence (gamma -> 0), as described in the thesis.
    """
    gamma_params = [p for n, p in model.named_parameters() if "static_norm.gamma" in n]
    other_params = [p for n, p in model.named_parameters() if "static_norm.gamma" not in n]
    return optim.Adam(
        [
            {"params": other_params, "weight_decay": cfg.weight_decay},
            {"params": gamma_params, "weight_decay": 1e-2},
        ],
        lr=cfg.learning_rate,
    )


def seed_everything(seed: int) -> None:
    """Seed python, numpy and torch RNGs so a run is repeatable."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loaders(X, y, sub, train_mask, test_mask, batch_size, device=None, val_mask=None):
    """Build train/val/test ``DataLoader``s from boolean masks over the sample axis.

    ``val_mask`` is optional; when omitted the returned validation loader is
    ``None``. The three masks must be mutually exclusive -- the validation split
    exists so that best-epoch selection never sees the test split.
    """
    def _to(t):
        return t.to(device) if device is not None else t

    pin = device is None

    def _loader(mask, shuffle):
        ds = TensorDataset(_to(X[mask]), _to(y[mask]), _to(sub[mask]))
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, pin_memory=pin)

    train_loader = _loader(train_mask, True)
    test_loader = _loader(test_mask, False)
    val_loader = _loader(val_mask, False) if val_mask is not None else None
    return train_loader, val_loader, test_loader


# --- LOSO (sub_indep) --------------------------------------------------------
def run_single_subject_fold(subject_id, cfg, X_full, y_full, sub_full,
                            base_edge_index, run_id, model_name):
    """Train and evaluate one LOSO fold (held-out ``subject_id``) in its own process."""
    torch.set_num_threads(2)
    num_gpus = torch.cuda.device_count()
    device = torch.device(f"cuda:{subject_id % num_gpus}" if num_gpus else "cpu")
    logger.info("[fold] subject %d -> %s", subject_id, device)

    seed_everything(cfg.seed + subject_id)

    # Test = the held-out subject. Validation = a disjoint slice of the REMAINING
    # subjects, used only to pick the best epoch. Holding out whole subjects (not
    # random samples) makes the selection signal match the cross-subject shift the
    # fold is actually measuring.
    test_mask = sub_full == subject_id
    train_pool_mask = ~test_mask
    pool_subjects = sorted({int(v) for v in sub_full[train_pool_mask].tolist()})
    n_val = min(cfg.val_subjects, max(0, len(pool_subjects) - 1))

    if n_val > 0:
        rng = np.random.default_rng(cfg.seed + subject_id)
        val_subjects = sorted(int(v) for v in rng.choice(pool_subjects, size=n_val, replace=False))
        val_mask = train_pool_mask & torch.isin(sub_full, torch.tensor(val_subjects))
        train_mask = train_pool_mask & ~val_mask
        logger.info("[fold %d] val subjects %s | train subjects %s",
                    subject_id, val_subjects, [s for s in pool_subjects if s not in val_subjects])
    else:
        val_mask, train_mask = None, train_pool_mask
        logger.warning("[fold %d] too few subjects to carve a validation split; "
                       "best-epoch selection will leak.", subject_id)

    train_loader, val_loader, test_loader = make_loaders(
        X_full, y_full, sub_full, train_mask, test_mask, cfg.batch_size, val_mask=val_mask
    )

    model = build_model(cfg.model_type, cfg.in_features, base_edge_index, device,
                        num_nodes=cfg.num_nodes, num_classes=cfg.num_classes,
                        hidden_dim=cfg.hidden_dim, num_layers=cfg.num_layers,
                        use_se=cfg.use_se, use_doubling=cfg.use_doubling,
                        num_subjects=cfg.num_subjects)
    optimizer = build_optimizer(model, cfg)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=cfg.learning_rate, total_steps=cfg.epochs)
    criterion = FocalLoss(alpha=torch.tensor([1.2, 1.1, 0.9]).to(device), gamma=2.0)

    run_name = f"Attempt_{run_id}_LOSO_Parallel"
    subdir = os.path.join(model_name, run_name, f"Subject_{subject_id}")
    results_dir = os.path.join("Results", subdir)
    train_model_with_interrupt(
        model=model, train_loader=train_loader, test_loader=test_loader,
        val_loader=val_loader,
        optimizer=optimizer, criterion=criterion, scheduler=scheduler,
        epochs=cfg.epochs, device=device,
        results_dir=results_dir,
        params_dir=os.path.join("Params", subdir),
        errors_dir=os.path.join("Errors", subdir),
        subject_tag=f"Subject_{subject_id}",
        base_edge_index=base_edge_index.to(device),
        evaluate_fn=evaluate, hyperparams=cfg.to_dict(), in_features=cfg.in_features,
        final_preds_path=os.path.join(results_dir, f"final_test_preds_sub{subject_id}.npy"),
    )


def run_loso(cfg: TrainConfig, bundle, base_edge_index, run_id, model_name):
    """Run all LOSO folds in parallel chunks, then aggregate from disk."""
    X = torch.tensor(bundle.X, dtype=torch.float32).share_memory_()
    y = torch.tensor(bundle.y, dtype=torch.long).share_memory_()
    sub = torch.tensor(bundle.subjects, dtype=torch.long).share_memory_()
    base_edge_index = base_edge_index.share_memory_()

    subjects = list(range(1, cfg.num_subjects + 1))
    for i in range(0, len(subjects), cfg.max_parallel):
        chunk = subjects[i : i + cfg.max_parallel]
        logger.info("Starting LOSO chunk: %s", chunk)
        procs = [mp.Process(target=run_single_subject_fold,
                            args=(s, cfg, X, y, sub, base_edge_index, run_id, model_name))
                 for s in chunk]
        for p in procs:
            p.start()
        for p in procs:
            p.join()

    _aggregate_loso(cfg, subjects, run_id, model_name)


def _aggregate_loso(cfg, subjects, run_id, model_name):
    """Collect per-subject predictions written to disk into a global report."""
    root = os.path.join("Results", model_name, f"Attempt_{run_id}_LOSO_Parallel")
    all_preds, all_trues, accs = [], [], {}
    missing, leaky_folds = [], []
    for s in subjects:
        res_file = os.path.join(root, f"Subject_{s}", f"final_test_preds_sub{s}.npy")
        if not os.path.exists(res_file):
            logger.warning("Results missing for subject %d (%s)", s, res_file)
            missing.append(s)
            continue
        data = np.load(res_file, allow_pickle=True).item()
        accs[s] = data["acc"]
        if data.get("leaky_selection"):
            leaky_folds.append(s)
        all_preds.extend(data["preds"])
        all_trues.extend(data["true"])

    if not all_preds:
        logger.error("No predictions aggregated; did the folds finish? Expected files "
                     "like Results/<model>/Attempt_<N>_LOSO_Parallel/Subject_<k>/"
                     "final_test_preds_sub<k>.npy")
        return

    if missing:
        logger.error("INCOMPLETE: %d/%d folds missing (%s). The mean below is over "
                     "completed folds only and is NOT a valid LOSO score.",
                     len(missing), len(subjects), missing)

    mean_acc, std_acc = float(np.mean(list(accs.values()))), float(np.std(list(accs.values())))
    logger.info("[LOSO complete] mean acc %.2f%% (+/- %.2f%%) over %d folds",
                mean_acc, std_acc, len(accs))
    report = classification_report(all_trues, all_preds, target_names=CLASS_NAMES)
    print(report)
    print(confusion_matrix(all_trues, all_preds))

    lines = [
        f"Global LOSO Average: {mean_acc:.2f}% (+/- {std_acc:.2f}%)",
        f"Folds aggregated: {len(accs)}/{len(subjects)}",
    ]
    if missing:
        lines.append(f"!! MISSING FOLDS: {missing} -- this is NOT a valid LOSO score.")
    if leaky_folds:
        lines.append(f"!! LEAKY SELECTION in folds {leaky_folds}: the best epoch was "
                     f"chosen on the test split, so the score is optimistically biased.")
    else:
        lines.append("Best epoch selected on a held-out validation split of the training "
                     "subjects; each test fold was evaluated exactly once.")
    lines.append("")
    lines.extend(f"Subject {s}: {acc:.2f}%" for s, acc in accs.items())
    lines += ["", report, "", str(confusion_matrix(all_trues, all_preds))]

    with open(os.path.join(root, "LOSO_Global_Summary.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")


# --- Session-Holdout (sub_dep) ----------------------------------------------
def run_session_holdout(cfg: TrainConfig, bundle, base_edge_index, run_id, model_name):
    """Train on sessions 1+2 and test on session 3 (within-subject, across time)."""
    device = get_device()
    X = torch.tensor(bundle.X, dtype=torch.float32).to(device)
    y = torch.tensor(bundle.y, dtype=torch.long).to(device)
    sub = torch.tensor(bundle.subjects, dtype=torch.long).to(device)
    sessions = bundle.sessions

    seed_everything(cfg.seed)

    # Test = session 3. Validation = a disjoint slice of the session-1+2 TRIALS,
    # used only to pick the best epoch. Splitting by trial (not by sample) keeps
    # windows from the same film clip out of both sides; a sample-wise split would
    # leak through the temporal correlation between adjacent windows.
    train_pool = (sessions == 1) | (sessions == 2)
    test_mask_np = sessions == 3
    trials = bundle.trials
    pool_trials = np.unique(trials[train_pool])
    n_val_trials = min(max(1, int(round(len(pool_trials) * cfg.val_fraction))),
                       max(0, len(pool_trials) - 1))

    if n_val_trials > 0:
        rng = np.random.default_rng(cfg.seed)
        val_trials = rng.choice(pool_trials, size=n_val_trials, replace=False)
        val_mask_np = train_pool & np.isin(trials, val_trials)
        train_mask_np = train_pool & ~val_mask_np
        logger.info("Validation split: %d/%d training trials held out for epoch selection",
                    n_val_trials, len(pool_trials))
    else:
        val_mask_np, train_mask_np = None, train_pool
        logger.warning("Too few trials to carve a validation split; "
                       "best-epoch selection will leak.")

    def _mask(m):
        return None if m is None else torch.from_numpy(np.asarray(m)).to(X.device)

    train_loader, val_loader, test_loader = make_loaders(
        X, y, sub, _mask(train_mask_np), _mask(test_mask_np), cfg.batch_size, device,
        val_mask=_mask(val_mask_np),
    )

    model = build_model(cfg.model_type, cfg.in_features, base_edge_index.to(device), device,
                        num_nodes=cfg.num_nodes, num_classes=cfg.num_classes,
                        hidden_dim=cfg.hidden_dim, num_layers=cfg.num_layers,
                        use_se=cfg.use_se, use_doubling=cfg.use_doubling,
                        num_subjects=cfg.num_subjects)
    optimizer = build_optimizer(model, cfg)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=cfg.learning_rate, total_steps=cfg.epochs)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor([1.2, 0.9, 1.0]).to(device), label_smoothing=0.1
    )

    run_name = f"Attempt_{run_id}_Phase2"
    train_model_with_interrupt(
        model=model, train_loader=train_loader, test_loader=test_loader,
        val_loader=val_loader,
        optimizer=optimizer, criterion=criterion, scheduler=scheduler,
        epochs=cfg.epochs, device=device,
        results_dir=os.path.join("Results", model_name, run_name),
        params_dir=os.path.join("Params", model_name, run_name),
        errors_dir=os.path.join("Errors", model_name, run_name),
        subject_tag="SessionHoldout", base_edge_index=base_edge_index.to(device),
        evaluate_fn=evaluate, hyperparams=cfg.to_dict(), in_features=cfg.in_features,
    )


def main(argv: list[str] | None = None) -> None:
    """CLI entry point: load data, build the graph, and dispatch on ``--mode``."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    cfg = parse_args(argv)
    seed_everything(cfg.seed)
    bundle = load_seed_de(DATA_DIR / f"ExtractedFeatures_{cfg.window_size}",
                          in_features=cfg.in_features, rolling_var_window=cfg.rolling_var_window)
    run_id = next_run_id(cfg.window_size)
    model_name = f"{cfg.model_type}_DE_{cfg.window_size}"
    base_edge_index = get_knn_adjacency_matrix(str(LOCS_FILE), k=cfg.knn_k)

    if cfg.mode == "sub_indep":
        logger.info("Leave-One-Subject-Out (parallel)")
        run_loso(cfg, bundle, base_edge_index, run_id, model_name)
    else:
        logger.info("Session-Holdout (train S1+S2, test S3)")
        run_session_holdout(cfg, bundle, base_edge_index, run_id, model_name)


if __name__ == "__main__":
    main()
