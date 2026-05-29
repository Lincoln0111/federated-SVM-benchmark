"""Dataset preparation for the standalone BDSVM federated simulation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import StratifiedKFold, train_test_split


@dataclass
class ClientDataset:
    """Client-local features in budgeted kernel form."""

    client_id: str
    k_tilde: np.ndarray
    y: np.ndarray


@dataclass
class TestDataset:
    """Shared test split for evaluation."""

    k_tilde: np.ndarray
    y: np.ndarray


@dataclass
class BDSVMDataset:
    """All data needed by the standalone federated BDSVM runner."""

    clients: List[ClientDataset]
    test: TestDataset
    centroids: np.ndarray
    sigma: float
    k_tilde_cc: np.ndarray


def rbf_kernel(a: np.ndarray, b: np.ndarray, sigma: float) -> np.ndarray:
    """Compute an RBF kernel matrix."""
    a_sq = np.sum(a * a, axis=1, keepdims=True)
    b_sq = np.sum(b * b, axis=1, keepdims=True).T
    sq_dist = np.maximum(a_sq - 2.0 * (a @ b.T) + b_sq, 0.0)
    return np.exp(-sq_dist / (2.0 * sigma * sigma))


def append_bias_column(kernel_matrix: np.ndarray) -> np.ndarray:
    """Extend kernel features with a bias feature."""
    return np.hstack([kernel_matrix, np.ones((kernel_matrix.shape[0], 1), dtype=kernel_matrix.dtype)])


def _load_mnist_binary(random_state: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load MNIST and convert it to digit-0-vs-rest labels."""
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False, parser='auto')
    X = X.astype(np.float32) / 255.0
    y = y.astype(np.int64)
    y_binary = np.where(y == 0, 1.0, -1.0).astype(np.float64)

    # Keep sigma=1.0 meaningful in 784 dimensions by scaling distances to O(1).
    X = X / np.sqrt(X.shape[1])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_binary,
        test_size=10000,
        random_state=random_state,
        stratify=y_binary,
    )
    return X_train, X_test, y_train, y_test


def load_bdsvm_dataset(
    n_clients: int = 3,
    budget_size: int = 50,
    sigma: float = 1.0,
    random_state: int = 42,
) -> BDSVMDataset:
    """Load MNIST binary, create client splits, and precompute budgeted kernel features."""
    X_train, X_test, y_train, y_test = _load_mnist_binary(random_state)

    rng = np.random.default_rng(random_state)
    centroid_idx = rng.choice(X_train.shape[0], size=budget_size, replace=False)
    centroids = X_train[centroid_idx].astype(np.float64, copy=True)

    train_kernel = append_bias_column(rbf_kernel(X_train, centroids, sigma).astype(np.float64, copy=False))
    test_kernel = append_bias_column(rbf_kernel(X_test, centroids, sigma).astype(np.float64, copy=False))
    centroid_kernel = append_bias_column(rbf_kernel(centroids, centroids, sigma).astype(np.float64, copy=False))
    k_tilde_cc = centroid_kernel.T @ centroid_kernel

    clients: List[ClientDataset] = []
    splitter = StratifiedKFold(n_splits=n_clients, shuffle=True, random_state=random_state)
    for split_index, (_, client_idx) in enumerate(splitter.split(train_kernel, y_train), start=1):
        clients.append(
            ClientDataset(
                client_id=f'client_{split_index}',
                k_tilde=train_kernel[client_idx],
                y=y_train[client_idx],
            )
        )

    return BDSVMDataset(
        clients=clients,
        test=TestDataset(k_tilde=test_kernel, y=y_test),
        centroids=centroids,
        sigma=sigma,
        k_tilde_cc=k_tilde_cc,
    )
