# federated-SVM-benchmark

Benchmark repository for federated learning implementations of SVM-based models.

## Methods

| Method | Path | Notes |
|---|---|---|
| DSVM (quick test) | `test_dsvm_bdsvm_quick.py` | Baseline centralised / federated DSVM |
| BDSVM FL | [`bdsvm/`](bdsvm/) | Budget Distributed SVM — three FL implementations |

## BDSVM Federated Learning

[Budget Distributed SVM](bdsvm/) implements BDSVM via primal IRWLS in a
federated setting. Workers share only sufficient statistics (no raw data),
and the server solves a linear system each round.

**Dataset:** MNIST binary (digit 0 vs rest) · **Workers:** 3 · **Rounds:** 10  
**Features:** RBF kernel map with P = 50 budget centroids

| Implementation | Accuracy | AUC |
|---|---|---|
| Threading simulation | 95.37% | – |
| Standalone FL loop | 94.60% | 97.59% |
| FL_PyTorch framework | 94.84% | – |

See [`bdsvm/README.md`](bdsvm/README.md) for details and quickstart commands.

## Requirements

```bash
pip install -r requirements.txt
```

For the FL_PyTorch framework variant, see
[`bdsvm/option2b_flpytorch/INSTALL.md`](bdsvm/option2b_flpytorch/INSTALL.md).
