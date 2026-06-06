from __future__ import annotations

import time
from typing import Callable, Dict

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


def eval_binary(y_true: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true).reshape(-1)
    scores = np.asarray(scores).reshape(-1)
    preds = np.where(scores >= 0, 1, -1)
    y_bin = (y_true == 1).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, preds)),
        "precision": float(precision_score(y_true, preds, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_true, preds, pos_label=1, zero_division=0)),
        "f1": float(f1_score(y_true, preds, pos_label=1, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_bin, scores)) if len(np.unique(y_bin)) > 1 else 0.5,
    }


def timed_run(fn: Callable[[], Dict[str, float]]) -> Dict[str, float]:
    t0 = time.perf_counter()
    out = fn()
    out["time_s"] = float(time.perf_counter() - t0)
    return out


def add_bias(X: np.ndarray) -> np.ndarray:
    return np.hstack([X, np.ones((X.shape[0], 1), dtype=X.dtype)])
