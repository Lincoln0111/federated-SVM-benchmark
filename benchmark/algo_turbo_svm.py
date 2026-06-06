from __future__ import annotations

import time
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC, SVC

from common import eval_binary
from sdca_data import iid_partitions


def _train_client_linear(X_local, y_local, C=1.0, random_state=42, w_global=None):
    d = X_local.shape[1]
    X64 = X_local.astype(np.float64, copy=False)
    y_arr = y_local.astype(np.int32, copy=False)
    if w_global is None:
        svc = LinearSVC(C=C, max_iter=2000, dual=True, random_state=random_state)
        svc.fit(X64, y_arr)
        return np.concatenate([svc.coef_.ravel(), svc.intercept_.ravel()]).astype(np.float32)

    # Warm-start path for tabular setting using SGD hinge updates.
    sgd = SGDClassifier(
        loss="hinge",
        alpha=1.0 / max(C * len(y_local), 1.0),
        max_iter=200,
        tol=1e-3,
        random_state=random_state,
        warm_start=True,
    )
    sgd.partial_fit(X64, y_arr, classes=np.array([-1, 1], dtype=np.int32))
    sgd.coef_ = w_global[:d].reshape(1, -1).astype(np.float64)
    sgd.intercept_ = np.array([w_global[d]], dtype=np.float64)
    sgd.t_ = max(getattr(sgd, "t_", 1.0), 1.0)
    sgd.partial_fit(X64, y_arr)
    return np.concatenate([sgd.coef_.ravel(), sgd.intercept_.ravel()]).astype(np.float32)


def turbo_server_aggregate(embeddings, labels_per_client, n_workers, C_svm=1.0):
    E = np.vstack(embeddings).astype(np.float32)

    if E.shape[0] < 3:
        return E.mean(axis=0).astype(np.float32), np.arange(E.shape[0]), True

    svm_labels = np.array([1 if lp > 0.5 else -1 for lp in labels_per_client], dtype=np.int32)
    if len(np.unique(svm_labels)) < 2:
        svm_labels = np.where(np.arange(len(svm_labels)) % 2 == 0, 1, -1)

    try:
        meta_svc = SVC(kernel="linear", C=C_svm, max_iter=3000)
        meta_svc.fit(E, svm_labels)
        sv_idx = meta_svc.support_
        if len(sv_idx) > 0:
            w_selected = E[sv_idx].mean(axis=0)
        else:
            w_selected = E.mean(axis=0)

        svm_w = meta_svc.coef_.ravel().astype(np.float32)
        norm_sq = float(np.dot(svm_w, svm_w))
        if norm_sq > 1e-10:
            proj = float(np.dot(w_selected, svm_w) / norm_sq)
            w_proj = w_selected - proj * svm_w
            return w_proj.astype(np.float32), sv_idx, False
        return w_selected.astype(np.float32), sv_idx, False
    except Exception:
        # FedAvg fallback keeps training stable for tabular inputs.
        return E.mean(axis=0).astype(np.float32), np.arange(E.shape[0]), True


def run_turbo_svm(
    X_train,
    y_train,
    X_test,
    y_test,
    n_workers: int = 3,
    rounds: int = 10,
    C: float = 1.0,
    random_state: int = 42,
    return_history: bool = False,
):
    t0 = time.perf_counter()
    Xtr = X_train.astype(np.float32)
    Xte = X_test.astype(np.float32)
    parts = iid_partitions(Xtr, y_train, n_workers=n_workers, random_state=random_state)

    w_global = None
    history = []

    for t in range(1, rounds + 1):
        embeddings = []
        local_weights = []
        for k, (Xk, yk) in enumerate(parts):
            coef = _train_client_linear(
                Xk,
                yk,
                C=C,
                random_state=random_state + t * 97 + k,
                w_global=w_global,
            )
            local_weights.append(coef)
            embeddings.append(coef)

        dom_labels = [float(np.mean(yk == 1)) for _, yk in parts]
        w_global, sv_idx, used_fallback = turbo_server_aggregate(
            embeddings,
            dom_labels,
            n_workers=len(parts),
            C_svm=C,
        )

        d = Xte.shape[1]
        w_vec = w_global[:d]
        bias = float(w_global[d]) if len(w_global) > d else 0.0
        scores = Xte @ w_vec + bias
        preds = np.where(scores >= 0, 1.0, -1.0)
        acc = float(np.mean(preds == y_test))
        loss = float(np.mean(np.maximum(0.0, 1.0 - y_test * scores)))
        history.append(
            {
                "round": t,
                "accuracy": acc,
                "hinge_loss": loss,
                "fallback": bool(used_fallback),
                "n_sv_clients": int(len(sv_idx)),
            }
        )
        print(f"TurboSVM-FL Round {t}/{rounds}: acc={acc:.4f} loss={loss:.4f}")

    d = Xte.shape[1]
    w_vec = w_global[:d]
    bias = float(w_global[d]) if len(w_global) > d else 0.0
    scores = Xte @ w_vec + bias
    metrics = eval_binary(y_test, scores)
    metrics["time_s"] = float(time.perf_counter() - t0)

    if return_history:
        return metrics, history
    return metrics
