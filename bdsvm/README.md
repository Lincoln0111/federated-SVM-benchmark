# BDSVM Federated Learning

Budget Distributed SVM (BDSVM) — federated binary SVM using primal IRWLS.

Each client computes local sufficient statistics (C_m, d_m) from its data
using Iteratively Re-Weighted Least Squares (IRWLS). The server aggregates
these matrices and solves the global linear system to update the SVM
weight vector β. A momentum blend (λ = 0.5) ensures stable convergence.

**Dataset:** MNIST binary classification — digit 0 vs rest  
**Features:** RBF kernel map using P = 50 centroids (→ 51-dim input with bias)  
**Workers:** 3 clients, IID data split

## Results

| Implementation | Accuracy | Notes |
|---|---|---|
| [Option 1 – Threading](option1_threading/) | 95.37% | DSVM.py unchanged; 3 workers via Python Queue |
| [Option 2a – Standalone](option2_standalone/) | 94.60% | AUC 97.59%; 10 rounds; pure NumPy |
| [Option 2b – FL_PyTorch](option2b_flpytorch/) | 94.84% | Loss 0.337→0.150; framework simulation loop |

## Algorithm sketch

```
FOR each round t = 1..T:
  FOR each client m:
    β_old ← current global SVM weights
    FOR each batch (K̃, y) in local data:
      e   = y − K̃ β_old
      a_i = 2C / max(e_i y_i, ε)   if e_i y_i ≥ 0 else 0
      C_m += K̃ᵀ diag(a) K̃
      d_m += K̃ᵀ diag(a) y
    SEND (C_m, d_m) to server

  SERVER:
    (Σ C_m + K̃_cc) β_new = Σ d_m    (least-squares solve)
    β ← (1−λ) β_old + λ β_new        (momentum blend, λ=0.5)
```

`K̃_cc = K̃_P^T K̃_P` is the regulariser Gram matrix evaluated at the P
centroids — a fixed matrix computed once before training.

## Quick start

```bash
# Option 1 – needs MMLL package installed
python bdsvm/option1_threading/test_bdsvm_fl_threading.py

# Option 2a – standalone, no extra dependencies
python bdsvm/option2_standalone/run_bdsvm.py

# Option 2b – needs FL_PyTorch installed (see option2b_flpytorch/INSTALL.md)
python bdsvm/option2b_flpytorch/run_bdsvm_framework.py
```
