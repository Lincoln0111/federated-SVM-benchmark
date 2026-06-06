from __future__ import annotations

import numpy as np
from sklearn.svm import LinearSVC

from common import eval_binary, timed_run
from sdca_data import iid_partitions


class FedKSVMClient:
    def __init__(self, D=500, sigma=1.0, seed=42):
        self.D = D
        self.sigma = sigma
        self.seed = seed
        self.W = None
        self.b = None

    def fit_random_features(self, X):
        rng = np.random.RandomState(self.seed)
        d = X.shape[1]
        self.W = rng.randn(self.D, d).astype(np.float32) / self.sigma
        self.b = rng.uniform(0, 2 * np.pi, self.D).astype(np.float32)
        return self.transform(X)

    def transform(self, X):
        return np.sqrt(2.0 / self.D) * np.cos(X @ self.W.T + self.b)

    def train(self, X_local, y_local, C=1.0, n_blocks=5):
        phi_X = self.fit_random_features(X_local)
        block_size = self.D // n_blocks
        w = np.zeros(self.D + 1, dtype=np.float32)

        for b in range(n_blocks):
            start = b * block_size
            end = (b + 1) * block_size if b < n_blocks - 1 else self.D
            phi_block = np.hstack([phi_X[:, start:end], np.ones((len(phi_X), 1), dtype=np.float32)])
            svc = LinearSVC(C=C, max_iter=2500, dual=True, random_state=self.seed + b)
            svc.fit(phi_block, y_local)
            coef = svc.coef_.ravel().astype(np.float32)
            w[start:end] = coef[:-1]
            w[-1] += coef[-1] / n_blocks
        return w


def run_fed_ksvm(
    X_train,
    y_train,
    X_test,
    y_test,
    n_workers=3,
    rounds=10,
    D=500,
    sigma=1.0,
    C=1.0,
    n_blocks=5,
    random_state=42,
):
    def _run():
        Xtr = X_train.astype(np.float32)
        Xte = X_test.astype(np.float32)
        parts = iid_partitions(Xtr, y_train, n_workers=n_workers, random_state=random_state)

        shared_client = FedKSVMClient(D=D, sigma=sigma, seed=random_state)
        shared_client.fit_random_features(Xtr[: min(len(Xtr), 4)])
        W, b = shared_client.W.copy(), shared_client.b.copy()

        w_global = np.zeros(D + 1, dtype=np.float32)
        for t in range(1, rounds + 1):
            ws = []
            for k, (xk, yk) in enumerate(parts):
                client = FedKSVMClient(D=D, sigma=sigma, seed=random_state)
                client.W = W
                client.b = b
                client.transform = lambda X, _W=W, _b=b, _D=D: np.sqrt(2.0 / _D) * np.cos(X @ _W.T + _b)
                wk = client.train(xk, yk, C=C, n_blocks=n_blocks)
                ws.append(wk)
            w_global = np.mean(ws, axis=0)
            print(f"[Fed-KSVM] round {t}/{rounds}")

        phi_test = np.sqrt(2.0 / D) * np.cos(Xte @ W.T + b)
        scores = phi_test @ w_global[:-1] + w_global[-1]
        return eval_binary(y_test, scores)

    return timed_run(_run)
