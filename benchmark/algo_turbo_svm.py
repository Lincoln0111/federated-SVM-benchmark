from __future__ import annotations

import numpy as np
from sklearn.svm import LinearSVC, SVC

from common import add_bias, eval_binary, timed_run
from sdca_data import iid_partitions


class TurboSVMClient:
    def __init__(self, C: float = 1.0, random_state: int = 42):
        self.C = C
        self.random_state = random_state

    def train(self, X_local, y_local, w_init=None, C=1.0):
        svc = LinearSVC(C=C, max_iter=4000, dual=True, random_state=self.random_state)
        svc.fit(X_local, y_local)
        emb = svc.coef_.ravel().astype(np.float32)
        if w_init is not None:
            # LinearSVC has no warm_start; blend toward previous global init.
            emb = 0.7 * emb + 0.3 * w_init.astype(np.float32)
        return emb


def turbo_server_aggregate(embeddings, labels_per_client, n_workers, C_svm=1.0):
    E = np.vstack(embeddings).astype(np.float32)

    svm_labels = np.array([1 if lp >= 0.5 else -1 for lp in labels_per_client], dtype=np.int32)
    if len(np.unique(svm_labels)) < 2:
        svm_labels = np.where(np.arange(len(svm_labels)) % 2 == 0, 1, -1)

    svc = SVC(kernel="linear", C=C_svm, max_iter=3000)
    svc.fit(E, svm_labels)

    sv_clients = svc.support_
    if len(sv_clients) == 0:
        sv_clients = np.arange(n_workers)

    w_global = E[sv_clients].mean(axis=0)

    h = svc.coef_.ravel().astype(np.float32)
    h_norm2 = float(np.dot(h, h))
    if h_norm2 > 1e-12:
        w_global = w_global - (float(np.dot(w_global, h)) / h_norm2) * h

    return w_global.astype(np.float32), sv_clients


def run_turbo_svm(
    X_train,
    y_train,
    X_test,
    y_test,
    n_workers: int = 3,
    rounds: int = 10,
    C: float = 1.0,
    random_state: int = 42,
):
    def _run():
        Xtr = X_train.astype(np.float32)
        Xte = X_test.astype(np.float32)

        parts = iid_partitions(Xtr, y_train, n_workers=n_workers, random_state=random_state)
        clients = [TurboSVMClient(C=C, random_state=random_state + i) for i in range(len(parts))]

        w_global = np.zeros(Xtr.shape[1], dtype=np.float32)
        for t in range(1, rounds + 1):
            embs = []
            dom_labels = []
            for i, ((xk, yk), client) in enumerate(zip(parts, clients)):
                emb = client.train(xk, yk, w_init=w_global, C=C)
                embs.append(emb)
                dom_labels.append(float((yk == 1).mean()))

            w_global, sv_clients = turbo_server_aggregate(
                embs, dom_labels, n_workers=len(parts), C_svm=C
            )
            print(f"[TurboSVM-FL] round {t}/{rounds}, sv_clients={len(sv_clients)}")

        scores = Xte @ w_global
        return eval_binary(y_test, scores)

    return timed_run(_run)
