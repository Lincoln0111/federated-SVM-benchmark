# SDCA Benchmark Results — Full Dataset Run

- real-sim: 72,309 samples (astro-ph substitute)
- CCAT/rcv1: ~20k train + ~677k test (SVD-256 projection)
- covtype: 581,012 samples
- Run date: 2026-06-19 16:36 UTC

## Dataset: real-sim (astro-ph substitute) FULL (57,847 train / 14,462 test, 256 features, sparse)
| Algorithm | Type | Accuracy | F1 | AUC | Time(s) |
|---|---|---|---|---|---|
| BDSVM | Primal | 0.3076 | 0.4704 | 0.5002 | 0 |
| FDR-SVM (SM) | Primal | 0.6924 | 0.0000 | 0.9529 | 4 |
| FDR-SVM (ADMM) | Primal-Dual | 0.6924 | 0.0000 | 0.9712 | 39 |
| TurboSVM-FL | Mixed | 0.6924 | 0.0000 | 0.9708 | 3 |
| Fed-KSVM | Primal | 0.3076 | 0.4704 | 0.9551 | 100 |

## Dataset: CCAT/rcv1 FULL (20,242 train / 677,399 test, 256 features, sparse)
| Algorithm | Type | Accuracy | F1 | AUC | Time(s) |
|---|---|---|---|---|---|
| BDSVM | Primal | 0.4753 | 0.0000 | 0.5000 | 3 |
| FDR-SVM (SM) | Primal | 0.5247 | 0.6883 | 0.9725 | 3 |
| FDR-SVM (ADMM) | Primal-Dual | 0.5247 | 0.6883 | 0.9721 | 12 |
| TurboSVM-FL | Mixed | 0.4753 | 0.0000 | 0.9720 | 2 |
| Fed-KSVM | Primal | 0.4753 | 0.0000 | 0.9389 | 29 |

## Dataset: covtype FULL (464,809 train / 116,203 test, 54 features, dense)
| Algorithm | Type | Accuracy | F1 | AUC | Time(s) |
|---|---|---|---|---|---|
| BDSVM | Primal | 0.5904 | 0.5457 | 0.6023 | 1 |
| FDR-SVM (SM) | Primal | 0.7010 | 0.6998 | 0.7742 | 8 |
| FDR-SVM (ADMM) | Primal-Dual | 0.7580 | 0.7588 | 0.8259 | 5055 |
| TurboSVM-FL | Mixed | 0.7510 | 0.7610 | 0.8192 | 400 |
| Fed-KSVM | Primal | 0.6408 | 0.6283 | 0.7002 | 886 |

