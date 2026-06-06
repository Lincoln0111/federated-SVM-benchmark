from __future__ import annotations

import numpy as np
from sklearn.svm import LinearSVC
from sklearn.exceptions import ConvergenceWarning
import warnings

from common import add_bias, eval_binary, timed_run
from sdca_data import iid_partitions


class FDRSVMClient_SM:
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X
        self.y = y

    def compute_subgradient(self, w, epsilon=0.1, C=1.0):
        margins = self.y * (self.X @ w)
        mask = margins < 1.0
        v = np.zeros_like(w)
        if mask.sum() > 0:
            v -= C * (self.y[mask, None] * self.X[mask]).mean(axis=0)
        norm_w = np.linalg.norm(w)
        if norm_w > 1e-10:
            v += epsilon * w / norm_w
        return v


class FDRSVMClient_ADMM:
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X
        self.y = y

    def local_update(self, w_global, mu, rho=1.0, C=1.0, epsilon=0.1):
        svc = LinearSVC(C=C, max_iter=1200, dual=True, random_state=0)
        svc.fit(self.X[:, :-1], self.y)
        w_svc = np.concatenate([svc.coef_.ravel(), svc.intercept_.ravel()])
        wg = (C * w_svc + rho * (w_global - mu)) / (C + rho + epsilon)
        lambda_g = np.linalg.norm(wg)
        return wg, lambda_g


def run_fdr_svm_sm(
    X_train,
    y_train,
    X_test,
    y_test,
    n_workers: int = 3,
    rounds: int = 100,
    lr0: float = 0.1,
    epsilon: float = 0.1,
    C: float = 1.0,
    random_state: int = 42,
):
    def _run():
        Xb_train = add_bias(X_train.astype(np.float32))
        Xb_test = add_bias(X_test.astype(np.float32))
        parts = iid_partitions(Xb_train, y_train, n_workers=n_workers, random_state=random_state)
        clients = [FDRSVMClient_SM(x, y.astype(np.float32)) for x, y in parts]

        d = Xb_train.shape[1]
        w = np.zeros(d, dtype=np.float32)
        alpha = 1.0 / max(len(clients), 1)

        for t in range(1, rounds + 1):
            gamma_t = lr0 / np.sqrt(t)
            g = np.zeros_like(w)
            for c in clients:
                g += alpha * c.compute_subgradient(w, epsilon=epsilon, C=C)
            w = w - gamma_t * g
            if t % max(1, rounds // 10) == 0 or t == 1:
                print(f"[FDR-SVM-SM] round {t}/{rounds}")

        scores = Xb_test @ w
        return eval_binary(y_test, scores)

    return timed_run(_run)


def run_fdr_svm_admm(
    X_train,
    y_train,
    X_test,
    y_test,
    n_workers: int = 3,
    rounds: int = 20,
    rho: float = 1.0,
    epsilon: float = 0.1,
    C: float = 1.0,
    random_state: int = 42,
):
    def _run():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        Xb_train = add_bias(X_train.astype(np.float32))
        Xb_test = add_bias(X_test.astype(np.float32))

        parts = iid_partitions(Xb_train, y_train, n_workers=n_workers, random_state=random_state)
        clients = [FDRSVMClient_ADMM(x, y.astype(np.int32)) for x, y in parts]

        d = Xb_train.shape[1]
        w = np.zeros(d, dtype=np.float32)
        mus = [np.zeros(d, dtype=np.float32) for _ in clients]
        alpha = 1.0 / max(len(clients), 1)

        for t in range(1, rounds + 1):
            wgs = []
            for i, c in enumerate(clients):
                wg, _ = c.local_update(w_global=w, mu=mus[i], rho=rho, C=C, epsilon=epsilon)
                wgs.append(wg.astype(np.float32))

            w = sum(alpha * (wgs[i] + mus[i]) for i in range(len(clients)))
            for i in range(len(clients)):
                mus[i] = mus[i] + wgs[i] - w

            print(f"[FDR-SVM-ADMM] round {t}/{rounds}")

        scores = Xb_test @ w
        return eval_binary(y_test, scores)

    return timed_run(_run)
