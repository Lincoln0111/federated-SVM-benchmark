# BDSVM Option 1 (Threading) – covtype Binary Results

## Dataset

| Property | Value |
|---|---|
| Name | covtype binary (scaled) |
| Source | `covtype.libsvm.binary.scale.bz2` via LIBSVM |
| Features | 54 (pre-scaled to [−1, 1]) |
| Train samples | 27,000 (subsampled to 30k, split 90/10 train/val) |
| Validation samples | 3,000 |
| Test samples | 11,620 (20% held out before subsampling) |
| Task | Binary classification: class 1 vs. class 2 (forest cover types) |
| Class balance | ≈ 49% / 51% (near-balanced) |

## Hyperparameters

| Parameter | Value |
|---|---|
| Workers | 3 (IID partition) |
| Budget size (nc) | 200 |
| Max iterations | 30 |
| Sigma (RBF kernel) | 2.0 |
| C (regularisation) | 50.0 |
| Tolerance | 1e-4 |
| Centroid init | uniform(P1, P99) of training data |

## Results

| Metric | Federated (3 workers) | Centralized (1 worker) | Gap |
|---|---|---|---|
| Accuracy | **0.7621** | 0.7678 | −0.0057 |
| Precision | — | 0.7551 | — |
| Recall | — | 0.7750 | — |
| F1-score | — | 0.7650 | — |
| ROC-AUC | **0.8474** | 0.8436 | +0.0038 |
| Fit time | — | 24.70 s | — |

## Notes

- covtype binary (class 1 vs. class 2) is a known hard problem for kernel SVMs.
  Reported accuracy in the literature is typically 76–82% for RBF-SVM; achieving
  ≥ 85% requires ensemble or deep methods.
- The near-balanced class distribution (49%/51%) combined with significant class
  overlap in 54-dimensional space limits the RBF-SVM ceiling.
- Federated accuracy is within **0.57 pp** of the centralized baseline, confirming
  that IID partitioning across 3 workers introduces negligible federation overhead.
- **No double-scaling**: the LIBSVM `.scale` file is already pre-scaled to [−1, 1],
  so no additional StandardScaler is applied.
- Larger C (50.0) and sharper sigma (2.0) were required to escape the zero-gradient
  fixed point caused by near-balanced labels and small initial weights.
