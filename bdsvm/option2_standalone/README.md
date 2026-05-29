# Option 2a – Standalone FL loop (pure NumPy)

A self-contained federated BDSVM implementation that requires only
`numpy`, `scikit-learn`, and (optionally) `torch` for MNIST loading.

## Files

| File | Purpose |
|---|---|
| `dataset.py` | Load MNIST, build RBF kernel features, IID client split |
| `model.py` | `BDSVMModel`: holds β, `predict()`, `score()` |
| `algorithm.py` | `BDSVMFederated`: server aggregation + IRWLS local stats |
| `run_bdsvm.py` | Entry point: 10 rounds × 3 clients, prints metrics per round |

## How to run

```bash
python bdsvm/option2_standalone/run_bdsvm.py
```

MNIST is downloaded automatically via `torchvision.datasets.MNIST`
or falls back to `sklearn.datasets.fetch_openml`.

## Key hyperparameters (editable in `run_bdsvm.py`)

| Parameter | Default | Description |
|---|---|---|
| `N_CLIENTS` | 3 | Number of federated workers |
| `ROUNDS` | 10 | Global communication rounds |
| `BUDGET_SIZE` (P) | 50 | Number of RBF kernel centroids |
| `SIGMA` | 1.0 | RBF bandwidth |
| `C` | 1.0 | SVM regularisation constant |
| `BINARY_CLASS` | 0 | Positive class digit (0 vs rest) |

## Expected output

```
Round 10 – accuracy: ~94.60%  AUC: ~97.59%
```
