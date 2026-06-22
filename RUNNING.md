# Running the Federated SVM Benchmark

## Environment

| Item | Value |
|---|---|
| Python | 3.10 or later |
| OS tested | Windows 10/11 (PowerShell), Linux (bash) |
| Key packages | numpy ≥ 1.24, scipy ≥ 1.10, scikit-learn ≥ 1.3 |

## Installation

```bash
# Clone or check out your branch
git checkout -b jiaye-config-refactor

# Install dependencies
pip install -r requirements.txt
```

## Dataset Setup

Datasets are downloaded from [LIBSVM](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/).
They are stored in the `data/` directory (excluded from git via `.gitignore`).

```bash
# Download all datasets (rcv1 train + test, covtype)
python benchmark.py --download

# Verify datasets and run a quick partition demo
python benchmark.py --demo
```

| Dataset | Size | Source URL |
|---|---|---|
| rcv1 train | 20,242 × 47,236 (sparse) | LIBSVM `rcv1_train.binary.bz2` |
| rcv1 test | 677,399 × 47,236 (sparse) | LIBSVM `rcv1_test.binary.bz2` |
| covtype | 581,012 × 54 (dense) | LIBSVM `covtype.libsvm.binary.scale.bz2` |

## Key Hyperparameters (from `config.py`)

| Parameter | Value | Description |
|---|---|---|
| `NUM_WORKERS` | 10 | FL clients |
| `ROUNDS` | 100 | Communication rounds |
| `SEED` | 42 | Global random seed |
| `BDSVM_P` | 100 | Budget size (BDSVM pre-image vectors) |
| `BDSVM_C` | 1.0 | SVM cost C |
| `MOMENTUM_ALPHA` | 0.5 | BDSVM momentum blend λ |
| `IRWLS_ETA` | 5e-3 | BDSVM convergence threshold |
| `FEDAVG_LAMBDA` | 0.01 | Pegasos regularisation (FedAvg-SVM) |
| `FDR_LAMBDA` / `FDR_RHO` | 0.01 / 1.0 | FDR-SVM ADMM parameters |
| `FEDSSL_D_ENC` | 64 | FedSSL-AMC encoder dimension |

All parameters can be changed in `config.py`; no source files need editing.

## Data Preparation

Datasets are downloaded from [LIBSVM](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/).
They are stored in `data/` (flat layout — no `data/raw/` or `data/processed/` subdirectories).
The `data/` directory is excluded from git via `.gitignore`.

```bash
# Recommended: use the standalone download script
python scripts/download_data.py

# Alternative: trigger download through the benchmark runner
python run_benchmark.py --download
```

Expected files after download:

| File | Size | Dataset |
|---|---|---|
| `data/rcv1_train.binary.bz2` | ~13 MB | rcv1 training set (20,242 samples) |
| `data/rcv1_test.binary.bz2` | ~435 MB | rcv1 test set (677,399 samples) |
| `data/covtype.libsvm.binary.scale.bz2` | ~8 MB | covtype (581,012 samples) |

If automatic download fails (e.g. network issues), download the files manually from:
`https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/`
and place them in `data/`.

---

## Running the Full Benchmark

```bash
# Full benchmark (rcv1 + covtype, 10 workers, 100 rounds)
python run_benchmark.py

# With dataset download
python run_benchmark.py --download

# Specific datasets and methods
python run_benchmark.py --datasets rcv1 --methods bdsvm fedavg_svm

# Custom output file
python run_benchmark.py --out my_results.csv

# Custom seed
python run_benchmark.py --seed 123
```

## Running Individual Models

Each method can be run independently using `--methods`:

```bash
python run_benchmark.py --methods centralized
python run_benchmark.py --methods fedavg_svm
python run_benchmark.py --methods fdr_svm
python run_benchmark.py --methods bdsvm
python run_benchmark.py --methods fedssl_amc
python run_benchmark.py --methods all
```

`--methods all` is equivalent to running without `--methods` (all registered methods).

## Fast / Smoke-Test Mode

```bash
# Uses FAST_ROUNDS=10 and caps train/test to 5k/2k samples
python run_benchmark.py --fast
```

## Output

Results are written to `results.csv` (default) with columns:

```
dataset, scheme, num_clients, method, test_acc, elapsed_s, seed
```

A pivot summary is printed to stdout after the run.

## Expected Results (approximate, 100 rounds, 10 workers)

| Method | rcv1 (IID) acc | covtype (IID) acc |
|---|---|---|
| Centralized | ~0.95 | ~0.75 |
| FedAvg-SVM | ~0.90 | ~0.72 |
| BDSVM | ~0.88 | ~0.70 |
| FDR-SVM | ~0.91 | ~0.73 |
| FedSSL-AMC | ~0.85 | ~0.68 |

Numbers are indicative; actual values depend on partitioning scheme and seed.

## Partitioning Schemes Swept

| Scheme | Description |
|---|---|
| `iid` | Uniform random split |
| `dirichlet α=1.0` | Mildly non-IID |
| `dirichlet α=0.3` | Moderately non-IID (standard benchmark) |
| `dirichlet α=0.1` | Highly non-IID |
| `label_skew` | Each client sees only one class (extreme non-IID) |

## Important Configuration Notes

- **`NUM_WORKERS = 10`** was explicitly aligned with sreekarsWindows (original value was a `[10, 50]` sweep).
- **`ROUNDS = 100`** was explicitly aligned with sreekarsWindows (original value was `50`).
- **All other algorithmic parameters** (`FEDAVG_LAMBDA`, `FDR_LAMBDA`, `BDSVM_C`, `BDSVM_P`, `MOMENTUM_ALPHA`, `IRWLS_ETA`, `FDR_RHO`, etc.) were preserved from the original implementation without change.
- **`LAMBDA = 1e-4`** from the sreekarsWindows reference was intentionally **not adopted**. The original code uses `lambda_reg=0.01` (Pegasos) and `C=1.0` (primal SVM) — kept as separate CONFIG keys.
- **p2p / gossip keys** (`TOPOLOGY`, `EPOCHS`, `BASE_PORT`, `GOSSIP_K`) are structural placeholders only. They are not read by `run_benchmark.py` or any algorithm.
- **Dataset path**: data files are in `data/` (flat layout). Do not rename or move them without also updating `DATA_DIR` in `config.py`.

---

## Current Distributed Setup

The current benchmark implements federated / distributed SVM training as a **Python local simulation**.

The configured number of workers represents simulated clients inside the Python benchmark runner. These workers are **not** currently separate physical machines, independent network processes, or CCR/HPC jobs.

Current team-aligned configuration:

| Setting | Value | Notes |
|---|---:|---|
| Workers / clients | 10 | Simulated Python workers |
| Communication rounds | 100 | Simulated training rounds |
| Communication backend | `local_simulation` | No TCP/gRPC/socket communication |
| Execution mode | `python_local_simulation` | Single-machine Python benchmark runner |
| Network topology | placeholder | No active p2p networking in the current version |

The current implementation does **not** use TCP, gRPC, socket communication, or real inter-node networking.

---

## Future CCR / HPC Plan

This implementation is intended to be adapted later for CCR / HPC execution.

In a future CCR/HPC version, simulated workers may be mapped to separate processes, jobs, or compute nodes. Additional work will be needed for job scheduling, process management, inter-process communication, and possibly real networking or message passing.

At the current stage, fields such as `BASE_PORT`, `TOPOLOGY`, and `GOSSIP_K` are kept mainly for structural alignment with the reference implementation and future distributed execution planning. They should **not** be interpreted as active TCP/gRPC/socket networking parameters in the current Python simulation.

---

## Known Limitations

- **No p2p networking**: The benchmark simulates federation in a single process. `TOPOLOGY`, `BASE_PORT`, `GOSSIP_K`, `EPOCHS` are reserved stubs.
- **LAMBDA reconciliation pending**: `lambda_reg=0.01` (Pegasos) vs `LAMBDA=1e-4` (SDCA reference). See `docs/hyperparameters.md` TODO #1.
- **rcv1 test is large**: The test set has 677k rows. `MAX_TEST=20_000` caps evaluation by default.
- **Function-signature defaults**: Method files still carry `n_rounds=50` as fallback API defaults. The runner always overrides via `METHOD_KWARGS`.

## Unresolved TODOs

See `docs/hyperparameters.md` for the full list. Critical items:

1. Agree on a single SVM regularisation parameterisation (`lambda_reg=0.01` vs `LAMBDA=1e-4`).
2. Clarify `T0_FRACTION` usage with Sreekar.
3. Wire p2p / HPC layer before merging into main repo.

## Branching Workflow

```bash
# Once invited to the main Gadget SVM repository:
git remote add upstream <gadget-svm-repo-url>
git fetch upstream
git checkout -b jiaye-config-refactor upstream/main
git cherry-pick <your-commits>   # or rebase
git push origin jiaye-config-refactor
# Open a PR: jiaye-config-refactor → main
```
