# BDSVM Option 1 (Threading) – CCAT / RCV1 Binary Results

## Dataset

| Property | Value |
|---|---|
| Name | CCAT (RCV1 binary, LIBSVM mirror) |
| Source | `rcv1_train.binary.bz2` + `rcv1_test.binary.bz2` via LIBSVM |
| Original features | 47,236 (sparse TF-IDF) |
| Projected features | 200 (TruncatedSVD + StandardScaler) |
| Train samples | 90,000 (subsampled from 781,265, split 90/10 train/val) |
| Validation samples | 10,000 |
| Test samples | 20,000 (subsampled from 677,399) |
| Task | Binary classification: CCAT topic (+1) vs. rest (−1) |
| Class balance | ≈ 53% positive / 47% negative |

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
| Accuracy | **0.9275** | 0.9327 | −0.0052 |
| Precision | 0.9326 | 0.9314 | — |
| Recall | 0.9306 | 0.9424 | — |
| F1-score | 0.9316 | 0.9369 | — |
| ROC-AUC | **0.9791** | 0.9794 | −0.0003 |
| Fit time | — | 0.74 s | — |
| Iterations | 2 | 2 | — |

## Notes

- Percentile-based centroid initialization (`P1`–`P99` of training data) is essential.
  Using global `min/max` places all centroids in the far tails of the SVD distribution,
  giving near-zero kernel values and preventing weight updates.
- The IRWLS linesearch converges in 2 iterations once centroids are well-initialized.
- Federated accuracy is within **0.52 pp** of the centralized baseline, demonstrating
  that IID data partitioning across 3 workers causes negligible performance loss.
- astro-ph results are not included in this benchmark run; both official download URLs
  (Joachims and LIBSVM) return HTTP 404. See `results_astro_ph.md`.
