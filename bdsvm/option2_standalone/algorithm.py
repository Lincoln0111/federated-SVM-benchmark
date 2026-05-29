"""Standalone federated BDSVM algorithm following the requested component split."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import numpy as np

from fl_pytorch_bdsvm.dataset import ClientDataset, TestDataset
from fl_pytorch_bdsvm.model import SVMModel


@dataclass
class RoundMetrics:
    """Per-round monitoring data."""

    round_idx: int
    loss: float
    accuracy: float


class BDSVMAlgorithm:
    """Federated BDSVM with local sufficient statistics and server-side solves."""

    def __init__(
        self,
        model: SVMModel,
        regularizer_matrix: np.ndarray,
        c_value: float = 1.0,
        server_lr: float = 0.5,
    ) -> None:
        self.model = model
        self.regularizer_matrix = regularizer_matrix.astype(np.float64, copy=True)
        self.c_value = float(c_value)
        self.server_lr = float(server_lr)

    def serialize_update(self, c_m: np.ndarray, d_m: np.ndarray) -> np.ndarray:
        """Match the requested flattened [C_m, d_m] transport shape."""
        return np.concatenate([c_m.reshape(-1), d_m.reshape(-1)])

    def deserialize_update(self, payload: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Recover local sufficient statistics from a flattened vector."""
        num_params = self.model.num_params
        matrix_size = num_params * num_params
        c_m = payload[:matrix_size].reshape(num_params, num_params)
        d_m = payload[matrix_size:].reshape(num_params)
        return c_m, d_m

    def local_gradient(self, client: ClientDataset) -> np.ndarray:
        """Compute the client-local IRWLS sufficient statistics."""
        predictions = client.k_tilde @ self.model.beta
        e = client.y - predictions
        ey = e * client.y

        safe_ey = np.maximum(ey, 1e-8)
        a = np.where(ey < 0.0, 0.0, 2.0 * self.c_value / safe_ey)

        weighted_features = client.k_tilde * a[:, None]
        c_m = client.k_tilde.T @ weighted_features
        d_m = client.k_tilde.T @ (a * client.y)
        return self.serialize_update(c_m, d_m)

    def server_gradient(self, serialized_updates: Iterable[np.ndarray]) -> np.ndarray:
        """Aggregate client updates and solve for the next beta candidate."""
        sum_c = self.regularizer_matrix.copy()
        sum_d = np.zeros(self.model.num_params, dtype=np.float64)

        for payload in serialized_updates:
            c_m, d_m = self.deserialize_update(payload)
            sum_c += c_m
            sum_d += d_m

        try:
            return np.linalg.solve(sum_c, sum_d)
        except np.linalg.LinAlgError:
            return np.linalg.lstsq(sum_c, sum_d, rcond=None)[0]

    def server_opt(self, beta_candidate: np.ndarray) -> np.ndarray:
        """Apply the requested server update rule with lambda=0.5."""
        self.model.beta = (1.0 - self.server_lr) * self.model.beta + self.server_lr * beta_candidate
        return self.model.beta

    def evaluate(self, dataset: TestDataset) -> Dict[str, float]:
        """Evaluate the current model on a kernelized dataset."""
        scores = self.model.forward(dataset.k_tilde)
        preds = np.where(scores >= 0.0, 1.0, -1.0)
        accuracy = float(np.mean(preds == dataset.y))
        margins = dataset.y * scores
        hinge = np.maximum(0.0, 1.0 - margins)
        loss = 0.5 * float(self.model.beta @ self.model.beta) + self.c_value * float(np.mean(hinge))
        return {'loss': loss, 'accuracy': accuracy}

    def fit(self, clients: List[ClientDataset], eval_dataset: TestDataset, rounds: int = 10) -> List[RoundMetrics]:
        """Run the requested communication-round training loop."""
        history: List[RoundMetrics] = []
        for round_idx in range(1, rounds + 1):
            serialized_updates = [self.local_gradient(client) for client in clients]
            beta_candidate = self.server_gradient(serialized_updates)
            self.server_opt(beta_candidate)
            metrics = self.evaluate(eval_dataset)
            history.append(RoundMetrics(round_idx=round_idx, loss=metrics['loss'], accuracy=metrics['accuracy']))
            print(f"Round {round_idx:02d}/{rounds}: loss={metrics['loss']:.6f}, accuracy={metrics['accuracy']:.4f}")
        return history
