from __future__ import annotations

import numpy as np

from benchmark.algo_turbo_svm import turbo_server_aggregate


class TurboSVMAlgorithm:
    """FL_PyTorch-style adapter for TurboSVM server aggregation.

    This adapter expects each client pseudo-gradient vector to encode a local
    embedding as a flat vector (client LinearSVC coef_). The server decodes
    them, runs turbo_server_aggregate, and returns the aggregated vector as the
    server gradient update.
    """

    def __init__(self, n_workers: int, C_svm: float = 1.0):
        self.n_workers = n_workers
        self.C_svm = C_svm

    def deserialize_embeddings(self, pseudo_gradients):
        embeddings = [np.asarray(g, dtype=np.float32).reshape(-1) for g in pseudo_gradients]
        return embeddings

    def server_gradient(self, pseudo_gradients, labels_per_client):
        embeddings = self.deserialize_embeddings(pseudo_gradients)
        w_global, sv_clients = turbo_server_aggregate(
            embeddings=embeddings,
            labels_per_client=labels_per_client,
            n_workers=self.n_workers,
            C_svm=self.C_svm,
        )
        return {
            "server_gradient": w_global,
            "support_vector_clients": np.asarray(sv_clients, dtype=np.int32),
        }


# Registry hook expected by some FL_PyTorch setups.
ALGORITHM_REGISTRY = {
    "turbo_svm": TurboSVMAlgorithm,
}
