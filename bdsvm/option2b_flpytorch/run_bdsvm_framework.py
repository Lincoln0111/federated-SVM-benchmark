#!/usr/bin/env python3
"""
run_bdsvm_framework.py
======================
Run Budget Distributed SVM (BDSVM) using the FL_PyTorch framework's native
simulation loop.

Setup:
  - 3 clients, 10 communication rounds, budget P = 50
  - Binary classification: MNIST digit-0 vs rest
  - Global optimiser: SGD(lr=1.0), so  β_new = β_old - grad_server = β_momentum
  - Metric: MSE loss (enables model_last.pth.tar checkpointing)

After training the framework saves model_last.pth.tar.  We load it and compute
the true binary classification accuracy on the MNIST test set.

Usage (from the MMLL project root):
    python fl_pytorch_bdsvm/run_bdsvm_framework.py
"""

import os
import sys

import numpy as np
import torch

# ── Locate the fl_pytorch package and add it to sys.path ─────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_FL_DIR = os.path.normpath(
    os.path.join(_HERE, '..', 'fl_pytorch_repo', 'flpytorch-main', 'fl_pytorch')
)
if _FL_DIR not in sys.path:
    sys.path.insert(0, _FL_DIR)

from opts import parse_args
from run import main
from utils.logger import Logger


# ── Helper: binary accuracy using sign(K̃ β) vs {-1, +1} labels ─────────────
def evaluate_bdsvm_accuracy(beta_vec, data_path, binary_class=0,
                             budget_size=50, sigma=1.0):
    """Load MNIST test set, recompute kernel features, return binary accuracy."""
    from torchvision import datasets, transforms

    test_raw = datasets.MNIST(
        root=data_path, train=False, download=True,
        transform=transforms.ToTensor()
    )
    X_raw = (test_raw.data.numpy()
             .reshape(-1, 784)
             .astype(np.float64) / 255.0)
    X = X_raw / np.sqrt(X_raw.shape[1])
    y = np.where(test_raw.targets.numpy() == binary_class, 1.0, -1.0)

    # Re-select the same centroids (fixed seed = 42, same as MNISTBDSVMDataset)
    train_raw = datasets.MNIST(
        root=data_path, train=True, download=True,
        transform=transforms.ToTensor()
    )
    X_train_raw = (train_raw.data.numpy()
                   .reshape(-1, 784)
                   .astype(np.float64) / 255.0)
    X_train = X_train_raw / np.sqrt(X_train_raw.shape[1])

    rng = np.random.default_rng(42)
    centroid_idx = rng.choice(len(X_train), budget_size, replace=False)
    centroids = X_train[centroid_idx]

    # RBF kernel features  K̃ = [K | 1]
    A_sq = np.sum(X ** 2, axis=1, keepdims=True)
    B_sq = np.sum(centroids ** 2, axis=1, keepdims=True).T
    dist_sq = np.maximum(A_sq + B_sq - 2.0 * (X @ centroids.T), 0.0)
    K = np.exp(-dist_sq / (2.0 * sigma ** 2))
    K_tilde = np.hstack([K, np.ones((len(K), 1))])   # (N, P+1)

    predictions = np.sign(K_tilde @ beta_vec)
    accuracy = np.mean(predictions == y)
    return accuracy, predictions, y


# ── Main training entry point ─────────────────────────────────────────────────
def run():
    data_path = os.path.normpath(os.path.join(_HERE, '..', 'data'))
    checkpoint_dir = os.path.normpath(os.path.join(_HERE, '..', 'checkpoints'))

    # Must be called before main() so that Logger.registered_loggers exists
    Logger.setup_logging(loglevel='INFO')

    raw_cmdline = [
        '--algorithm',               'bdsvm',
        '--dataset',                 'mnist_bdsvm',
        '--model',                   'bdsvm',
        '--rounds',                  '10',
        '--num-clients-per-round',   '3',
        '--loss',                    'mse',
        '--global-lr',               '1.0',
        '--global-optimiser',        'sgd',
        '--local-optimiser',         'sgd',
        '--local-lr',                '0.0',   # local grad is zero; lr has no effect
        '--compute-type',            'fp64',
        '--metric',                  'loss',  # enable checkpointing via MSE loss
        '--data-path',               data_path,
        '--checkpoint-dir',          checkpoint_dir,
        '--run-id',                  'bdsvm_framework_run',
        '--batch-size',              '512',
        '--number-of-local-iters',   '1',
        '--run-local-steps',
        '--global-regulizer',        'none',
        '--client-sampling-type',    'uniform',
        '--initialize-shifts-policy', 'zero',
        '--eval-every',              '1',     # checkpoint every round
        '--loglevel',                'info',
        '--gpu',                     '-1',    # CPU only (no CUDA in this env)
        '--algorithm-options',       'internal_sgd:full-gradient',
    ]

    # parse_args() modifies the list in-place; pass a copy
    args = parse_args(list(raw_cmdline))

    # BDSVM-specific hyper-parameters.
    # MNISTBDSVMDataset and BDSVMAlgorithm read these via getattr(args, 'bdsvm_*').
    args.bdsvm_n_clients = 3
    args.bdsvm_budget_size = 50
    args.bdsvm_sigma = 1.0
    args.bdsvm_binary_class = 0
    args.bdsvm_C = 1.0

    print("=" * 60)
    print("BDSVM via FL_PyTorch framework")
    print(f"  Rounds            : {args.rounds}")
    print(f"  Clients per round : {args.num_clients_per_round}")
    print(f"  Budget (P)        : {args.bdsvm_budget_size}")
    print(f"  sigma             : {args.bdsvm_sigma}")
    print(f"  C                 : {args.bdsvm_C}")
    print(f"  Binary class      : digit {args.bdsvm_binary_class} vs rest")
    print("=" * 60)

    main(args, raw_cmdline, None)

    # ── Post-training evaluation ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Post-training binary accuracy evaluation")
    print("=" * 60)

    from utils.utils import create_model_dir
    model_dir = create_model_dir(args)
    model_path = os.path.join(model_dir, 'model_last.pth.tar')

    if not os.path.exists(model_path):
        print(f"Checkpoint not found at: {model_path}")
        print("Cannot compute final accuracy.")
        return

    # torch.load saves the full nn.Module (BDSVMModule)
    try:
        saved_obj = torch.load(model_path, map_location='cpu', weights_only=False)
    except TypeError:
        saved_obj = torch.load(model_path, map_location='cpu')

    # Extract the beta parameter vector
    if hasattr(saved_obj, 'beta'):
        beta_vec = saved_obj.beta.detach().cpu().numpy().astype(np.float64)
    elif isinstance(saved_obj, torch.Tensor):
        beta_vec = saved_obj.detach().cpu().numpy().astype(np.float64)
    else:
        from models import mutils
        beta_vec = mutils.get_params(saved_obj).detach().cpu().numpy().astype(np.float64)

    accuracy, predictions, y_true = evaluate_bdsvm_accuracy(
        beta_vec,
        data_path=data_path,
        binary_class=args.bdsvm_binary_class,
        budget_size=args.bdsvm_budget_size,
        sigma=args.bdsvm_sigma,
    )

    correct = int(np.sum(predictions == y_true))
    total = len(y_true)
    print(f"\nTest accuracy : {accuracy:.4f}  ({correct} / {total} correct)")
    if accuracy >= 0.90:
        print("Target of >= 90% accuracy achieved.")
    else:
        print("Accuracy below 90% target.")
    print("=" * 60)


if __name__ == '__main__':
    run()
