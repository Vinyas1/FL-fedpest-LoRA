"""Evaluation + metrics.  (Phase 7)

The money metric is worst-hospital AUROC. We also report per-disease AUROC/AUPRC,
macro AUROC, and bootstrap confidence intervals.
"""
from __future__ import annotations
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


def per_disease_auroc(y_true: np.ndarray, y_score: np.ndarray, labels: list[str]) -> dict:
    out = {}
    for i, lab in enumerate(labels):
        col = y_true[:, i]
        if col.min() == col.max():          # only one class present -> undefined
            out[lab] = float("nan")
        else:
            out[lab] = roc_auc_score(col, y_score[:, i])
    return out


def macro_auroc(y_true, y_score, labels) -> float:
    vals = [v for v in per_disease_auroc(y_true, y_score, labels).values()
            if not np.isnan(v)]
    return float(np.mean(vals)) if vals else float("nan")


def worst_hospital_auroc(per_client_macro: dict[str, float]) -> tuple[str, float]:
    """Given {client: macro_auroc}, return the weakest client and its score."""
    client, score = min(per_client_macro.items(), key=lambda kv: kv[1])
    return client, score


def bootstrap_ci(y_true, y_score, labels, n: int = 1000, seed: int = 0):
    """Bootstrap CI on macro AUROC. Returns (mean, lo, hi) at 95%."""
    rng = np.random.default_rng(seed)
    N = len(y_true)
    stats = []
    for _ in range(n):
        idx = rng.integers(0, N, N)
        stats.append(macro_auroc(y_true[idx], y_score[idx], labels))
    stats = np.array([s for s in stats if not np.isnan(s)])
    return float(stats.mean()), float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))
