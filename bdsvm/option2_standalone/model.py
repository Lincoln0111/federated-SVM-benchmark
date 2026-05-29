"""Minimal SVM model abstraction for the standalone BDSVM federated loop."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SVMModel:
    """Stores the semiparametric BDSVM coefficients beta."""

    num_params: int
    beta: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.beta = np.zeros(self.num_params, dtype=np.float64)

    def forward(self, k_tilde: np.ndarray) -> np.ndarray:
        return k_tilde @ self.beta

    def decision_function(self, k_tilde: np.ndarray) -> np.ndarray:
        return self.forward(k_tilde)

    def predict(self, k_tilde: np.ndarray) -> np.ndarray:
        scores = self.forward(k_tilde)
        return np.where(scores >= 0.0, 1.0, -1.0)
