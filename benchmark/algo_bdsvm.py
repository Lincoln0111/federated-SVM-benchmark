from __future__ import annotations

import numpy as np

from common import eval_binary, timed_run
from sdca_data import iid_partitions


def _rbf_kernel(X: np.ndarray, C: np.ndarray, sigma: float) -> np.ndarray:
    x2 = np.sum(X * X, axis=1, keepdims=True)
    c2 = np.sum(C * C, axis=1, keepdims=True).T
    d2 = np.maximum(x2 - 2.0 * (X @ C.T) + c2, 0.0)
    return np.exp(-d2 / (2.0 * sigma * sigma)).astype(np.float32)


def _client_local_sgd(Kb: np.ndarray, y: np.ndarray, w: np.ndarray, C: float, steps: int, lr: float, rng: np.random.Generator) -> np.ndarray:
    w_local = w.copy()
    n = len(y)
    for _ in range(steps):
        i = int(rng.integers(n))
        xi = Kb[i]
        yi = y[i]
        margin = yi * float(xi @ w_local)
        grad = w_local.copy()
        grad[-1] = 0.0
        if margin < 1.0:
            grad -= C * yi * xi
        w_local -= lr * grad
    return w_local


def run_bdsvm(
    X_train,
    y_train,
    X_test,
    y_test,
    n_workers: int = 3,
    budget_size: int = 50,
    sigma: float = 1.0,
    c_value: float = 1.0,
    rounds: int = 10,
    random_state: int = 42,
):
    rng = np.random.default_rng(random_state)

    def _run():
        X_train_n = (X_train / np.sqrt(max(X_train.shape[1], 1))).astype(np.float32)
        X_test_n = (X_test / np.sqrt(max(X_test.shape[1], 1))).astype(np.float32)

        budget_size_eff = min(budget_size, len(X_train_n))
        centroid_idx = rng.choice(len(X_train_n), size=budget_size_eff, replace=False)
        centroids = X_train_n[centroid_idx]

        parts = iid_partitions(X_train_n, y_train, n_workers=n_workers, random_state=random_state)
        parts_k = [(_rbf_kernel(xk, centroids, sigma), yk.astype(np.float32)) for xk, yk in parts]
        parts_k = [(np.hstack([K, np.ones((K.shape[0], 1), dtype=np.float32)]), yk) for K, yk in parts_k]

        w = np.zeros(budget_size_eff + 1, dtype=np.float32)
        for t in range(1, rounds + 1):
            local_ws = []
            for k, (Kk, yk) in enumerate(parts_k):
                lr = 0.05 / np.sqrt(t)
                wk = _client_local_sgd(Kk, yk, w, C=c_value, steps=150, lr=lr, rng=np.random.default_rng(random_state + 1000 * t + k))
                local_ws.append(wk)
            w = np.mean(local_ws, axis=0)
            print(f"[BDSVM] round {t}/{rounds}")

        K_test = _rbf_kernel(X_test_n, centroids, sigma)
        K_test = np.hstack([K_test, np.ones((K_test.shape[0], 1), dtype=np.float32)])
        scores = K_test @ w

        metrics = eval_binary(y_test, scores)
        return metrics

    return timed_run(_run)
