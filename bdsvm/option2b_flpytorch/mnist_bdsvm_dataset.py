#!/usr/bin/env python3
"""
MNIST BDSVM Federated Learning Dataset.

Prepares MNIST as a binary classification task (digit 0 vs rest) using
RBF kernel features for Budget Distributed SVM (BDSVM) federated learning.
Centroids are chosen from the training set with a fixed random seed so that
both the train and test dataset instances use identical feature maps.
"""

import numpy as np
import torch
from torchvision import datasets, transforms

from data_preprocess.fl_dataset import FLDataset


class MNISTBDSVMDataset(FLDataset):
    """MNIST dataset with RBF kernel features for BDSVM federated learning.

    Attributes:
        num_clients (int): Number of federated clients (for train split).
        num_params  (int): Dimension of the kernel feature vector = budget_size + 1.
        k_tilde_cc  (np.ndarray): Centroid Gram matrix K̃_cc^T K̃_cc, shape (P+1, P+1).
    """

    def __init__(self, exec_ctx, args, train: bool = True,
                 download: bool = True, data_path: str = '../data/'):
        # --- Config (read from args with sensible defaults) ---
        self.n_clients = int(getattr(args, 'bdsvm_n_clients', 3))
        self.budget_size = int(getattr(args, 'bdsvm_budget_size', 50))
        self.sigma = float(getattr(args, 'bdsvm_sigma', 1.0))
        self.binary_class = int(getattr(args, 'bdsvm_binary_class', 0))

        self.train = train
        self.data_path = data_path

        # Required by FL framework
        self.num_clients = self.n_clients
        self.num_params = self.budget_size + 1   # P+1 (P RBF features + 1 bias)

        # Populated in load_data()
        self._current_client = None
        self._loaded = False

        self.k_tilde_cc = None          # (P+1, P+1) centroid Gram matrix
        self.centroids = None           # (P,  D_input)
        self.k_tilde_arrays = None      # list[ndarray] – per-client train features
        self.y_arrays = None            # list[ndarray] – per-client train labels
        self.k_tilde_all = None         # (N, P+1) full features (train or test)
        self.y_all = None               # (N,)  labels {+1, -1}

        # Eager loading ensures num_params and k_tilde_cc are ready before
        # initialise_model() tries to read them.
        self.load_data()

    # ------------------------------------------------------------------
    def load_data(self):
        """Compute RBF kernel features and split data across clients."""
        if self._loaded:
            return
        self._loaded = True

        # ---- Always select centroids from training data (fixed seed) ----
        train_raw = datasets.MNIST(
            root=self.data_path, train=True, download=True,
            transform=transforms.ToTensor()
        )
        X_train_raw = (train_raw.data.numpy()
                       .reshape(-1, 784)
                       .astype(np.float64) / 255.0)
        X_train = X_train_raw / np.sqrt(X_train_raw.shape[1])   # L2 norm normalise

        rng = np.random.default_rng(42)
        centroid_idx = rng.choice(len(X_train), self.budget_size, replace=False)
        self.centroids = X_train[centroid_idx]   # (P, D)

        # k_tilde_cc  = K̃_cc^T K̃_cc  where K̃_cc = [K_cc | 1]  (P×(P+1))
        K_cc = self._rbf_kernel(self.centroids, self.centroids)           # (P, P)
        K_tilde_cc = np.hstack([K_cc, np.ones((self.budget_size, 1))])   # (P, P+1)
        self.k_tilde_cc = K_tilde_cc.T @ K_tilde_cc                      # (P+1, P+1)

        # ---- Load the target split (train or test) ----
        if self.train:
            X_raw = X_train_raw
            y_raw = train_raw.targets.numpy()
            X = X_train
        else:
            test_raw = datasets.MNIST(
                root=self.data_path, train=False, download=True,
                transform=transforms.ToTensor()
            )
            X_raw = (test_raw.data.numpy()
                     .reshape(-1, 784)
                     .astype(np.float64) / 255.0)
            X = X_raw / np.sqrt(X_raw.shape[1])
            y_raw = test_raw.targets.numpy()

        # Binary labels: target class -> +1, all others -> -1
        y = np.where(y_raw == self.binary_class, 1.0, -1.0)

        K = self._rbf_kernel(X, self.centroids)                     # (N, P)
        K_tilde = np.hstack([K, np.ones((len(K), 1))])              # (N, P+1)

        if self.train:
            # IID random split across n_clients
            indices = rng.permutation(len(X))
            splits = np.array_split(indices, self.n_clients)
            self.k_tilde_arrays = [K_tilde[s] for s in splits]
            self.y_arrays = [y[s] for s in splits]

        self.k_tilde_all = K_tilde
        self.y_all = y

    # ------------------------------------------------------------------
    def _rbf_kernel(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """RBF kernel K[i,j] = exp(-||A[i]-B[j]||^2 / (2σ^2))."""
        A_sq = np.sum(A ** 2, axis=1, keepdims=True)           # (n, 1)
        B_sq = np.sum(B ** 2, axis=1, keepdims=True).T         # (1, m)
        dist_sq = np.maximum(A_sq + B_sq - 2.0 * (A @ B.T), 0.0)
        return np.exp(-dist_sq / (2.0 * self.sigma ** 2))

    # ------------------------------------------------------------------
    def set_client(self, index=None):
        """Set active client (None = all data combined)."""
        self._current_client = index

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        if self._current_client is None:
            return len(self.k_tilde_all)
        return len(self.k_tilde_arrays[self._current_client])

    def __getitem__(self, idx):
        if self._current_client is None:
            k = self.k_tilde_all[idx]
            y = self.y_all[idx]
        else:
            k = self.k_tilde_arrays[self._current_client][idx]
            y = self.y_arrays[self._current_client][idx]

        # Return label as shape (1,) so batching gives (batch, 1).
        # This keeps output.shape[1] == 1 < 5 inside update_metrics,
        # preventing the top-5-accuracy branch from executing.
        return (
            torch.tensor(k, dtype=torch.float64),
            torch.tensor([y], dtype=torch.float64),
        )
