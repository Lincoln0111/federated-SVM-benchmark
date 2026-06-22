# Federated SVM Benchmark

Benchmarks four federated / distributed SVM methods against centralized and FedAvg baselines
on standard SDCA datasets (rcv1, covtype).

**Current execution mode:** Python local simulation — workers are simulated in-process clients.
No TCP, gRPC, socket communication, or real multi-node execution is currently used.
Future adaptation for CCR / HPC is planned.

## Methods

| Method | Paper | Key idea |
|---|---|---|
| BDSVM | ACM TIST 2022, [DOI 10.1145/3539734](https://dl.acm.org/doi/10.1145/3539734) | Budget pre-image kernel, IRWLS aggregation |
| FDR-SVM | [arXiv 2410.03877](https://arxiv.org/abs/2410.03877) | Distributionally robust SVM + ADMM |
| FedAvg-SVM | [arXiv 1812.11750](https://arxiv.org/abs/1812.11750) | FedAvg over Pegasos updates |
| FedSSL-AMC | [arXiv 2510.04927](https://arxiv.org/abs/2510.04927) | Federated encoder + local SVM |

## Quick Start

```bash
git clone https://github.com/Lincoln0111/federated-SVM-benchmark.git
cd federated-SVM-benchmark

python -m venv .venv
.venv\Scripts\activate      # Windows
source .venv/bin/activate   # macOS / Linux

pip install -r requirements.txt
python scripts/download_data.py
python smoke_test.py
python run_benchmark.py --methods all
```

## Per-Model Commands

```bash
python run_benchmark.py --methods centralized
python run_benchmark.py --methods fedavg_svm
python run_benchmark.py --methods fdr_svm
python run_benchmark.py --methods bdsvm
python run_benchmark.py --methods fedssl_amc
python run_benchmark.py --methods all
```

## Fast Validation

```bash
python run_benchmark.py --methods all --fast
```

`--fast` uses 10 rounds and 5,000 training samples. It is **not** the full benchmark setting.
Full benchmark: `NUM_WORKERS = 10`, `ROUNDS = 100` (from `config.py`).

## Configuration Summary

All experiment parameters live in [`config.py`](config.py). No source files need editing.

| Setting | Value | Notes |
|---|---|---|
| Workers / clients | 10 | Simulated — team-aligned with sreekarsWindows |
| Communication rounds | 100 | Simulated — team-aligned |
| Communication backend | `local_simulation` | No network I/O |
| Execution mode | `python_local_simulation` | Single-machine |
| Future target | CCR / HPC | Planned adaptation |

## Documentation

- [`RUNNING.md`](RUNNING.md) — detailed setup, per-model commands, data preparation, CCR/HPC plan
- [`docs/hyperparameters.md`](docs/hyperparameters.md) — full hyperparameter table and distributed settings
- [`config.py`](config.py) — all configuration in one place
