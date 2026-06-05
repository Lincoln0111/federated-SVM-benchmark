# BDSVM Option 1 (Threading) – astro-ph / real-sim Results

## Dataset

| Property | Value |
|---|---|
| Name | real-sim (structural substitute for astro-ph) |
| Source | `real-sim.bz2` via LIBSVM |
| Original features | 20,958 (sparse binary bag-of-words) |
| Projected features | 200 (TruncatedSVD + StandardScaler) |
| Train samples | 52,062 (90% of 80% split) |
| Validation samples | 5,785 |
| Test samples | 7,229 (20% held out) |
| Task | Binary newsgroup classification (+1 / −1) |

> **Note on astro-ph:** Both official download URLs for the original astro-ph dataset
> (Joachims SVM-Perf and LIBSVM mirror) return HTTP 404. `real-sim` is used as a
> structural substitute — same sparse-text format, similar scale, same preprocessing pipeline.
> To use the original astro-ph, place `astro-ph.bz2` in the `data/` directory and re-run.

## Hyperparameters

| Parameter | Value |
|---|---|
| Workers | 3 (IID partition) |
| Budget size (nc) | 200 |
| Max iterations | 20 |
| Sigma (RBF kernel) | 14.0 |
| C (regularisation) | 1.0 |
| Tolerance | 1e-4 |
| Centroid init | uniform(P1, P99) of training data |

## Results

| Metric | Federated (3 workers) | Centralized (1 worker) | Gap |
|---|---|---|---|
| Accuracy | **0.8836** | 0.9260 | −0.0424 |
| Precision | — | 0.9078 | — |
| Recall | — | 0.8453 | — |
| F1-score | — | 0.8754 | — |
| ROC-AUC | **0.9759** | 0.9761 | −0.0002 |
| Fit time | — | 3.69 s | — |
| Iterations | 5 | 5 | — |

## Notes

- Federated accuracy **88.36%** exceeds the 85% target.
- The larger accuracy gap (4.24 pp vs 0.52 pp for CCAT) is due to the smaller
  per-worker training set: 52k/3 ≈ 17k samples per worker vs 90k/3 = 30k for CCAT.
- ROC-AUC gap is negligible (0.0002), confirming federation preserves ranking ability.
- Centroid percentile initialization (P1–P99) is essential; using global min/max causes
  all kernels to evaluate near zero on high-dimensional SVD-projected data.

