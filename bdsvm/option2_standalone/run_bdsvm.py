"""Run the standalone fallback BDSVM federated simulation."""

from __future__ import annotations

import os
import sys
import time

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

if __package__ is None or __package__ == '':
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fl_pytorch_bdsvm.algorithm import BDSVMAlgorithm
from fl_pytorch_bdsvm.dataset import load_bdsvm_dataset
from fl_pytorch_bdsvm.model import SVMModel


def main() -> int:
    n_clients = 3
    rounds = 10
    budget_size = 50
    sigma = 1.0
    c_value = 1.0

    print('Loading BDSVM dataset...')
    t0 = time.perf_counter()
    dataset = load_bdsvm_dataset(
        n_clients=n_clients,
        budget_size=budget_size,
        sigma=sigma,
        random_state=42,
    )
    print(f'Dataset ready in {time.perf_counter() - t0:.2f}s')
    print(f'Clients: {len(dataset.clients)}, budget vectors: {budget_size}, sigma: {sigma}')

    model = SVMModel(num_params=budget_size + 1)
    algorithm = BDSVMAlgorithm(
        model=model,
        regularizer_matrix=dataset.k_tilde_cc,
        c_value=c_value,
        server_lr=0.5,
    )

    print(f'\nStarting federated BDSVM for {rounds} rounds...')
    fit_start = time.perf_counter()
    algorithm.fit(dataset.clients, dataset.test, rounds=rounds)
    fit_seconds = time.perf_counter() - fit_start

    scores = model.forward(dataset.test.k_tilde)
    preds = np.where(scores >= 0.0, 1.0, -1.0)
    y_true = dataset.test.y
    y_true_binary = (y_true == 1.0).astype(np.int64)

    accuracy = accuracy_score(y_true, preds)
    precision = precision_score(y_true, preds, pos_label=1.0, zero_division=0)
    recall = recall_score(y_true, preds, pos_label=1.0, zero_division=0)
    f1 = f1_score(y_true, preds, pos_label=1.0, zero_division=0)
    roc_auc = roc_auc_score(y_true_binary, scores)

    print('\nFinal test metrics')
    print('------------------')
    print(f'Accuracy : {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall   : {recall:.4f}')
    print(f'F1-score : {f1:.4f}')
    print(f'ROC-AUC  : {roc_auc:.4f}')
    print(f'Fit time : {fit_seconds:.2f}s')

    if accuracy < 0.90:
        print('\nWARNING: Final accuracy is below the 90% target.')
        return 1

    return 0


if __name__ == '__main__':
    raise SystemExit(main())