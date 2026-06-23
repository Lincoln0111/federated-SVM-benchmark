# Sreekar Branch Comparison

## Scope

Reference branch: `https://github.com/Haimonti/GadgetSVM/tree/sreekar`

This repository is intentionally not a direct copy of Sreekar's branch. The reference branch is a real p2p-SDCA implementation built around p2pfl nodes, local addresses, topology wiring, and learning rounds started through `set_start_learning(rounds=..., epochs=...)`. This benchmark repository remains a Python local simulation for comparing multiple SVM methods under shared dataset and partitioning conditions.

## Structural Comparison

| Area | Sreekar `sreekar` branch | Current benchmark repository |
| --- | --- | --- |
| Layout | `src/` layout with `src/config.py`, `src/data_loader.py`, `src/main.py`, `src/model.py`, `src/network_topology.py` | Flat benchmark layout rooted at `benchmark.py`, `run_benchmark.py`, and `methods/*.py` |
| Config structure | Central `CONFIG` dict for p2p node setup and SDCA hyperparameters | Passive `config.py` alignment surface plus method-specific defaults kept in code paths such as `run_benchmark.py` |
| Dataset path handling | `CODE_DIR`, `RAW_DIR`, `DATA_DIR`; train data loaded from extracted files under `src/data` | `DATA_DIR` points to repository `data/`; `benchmark.py` loads cached `.npz` or LIBSVM-style files from the local benchmark data directory |
| Topology / network fields | Active `TOPOLOGY`, `BASE_PORT`, node addresses like `127.0.0.1:<port>`, topology helpers for `ring`, `full`, `star`, `mesh` | `TOPOLOGY`, `BASE_PORT`, and `GOSSIP_K` are placeholders only; no active topology wiring or node addressing exists in the local simulation |
| Execution mode | Real p2pfl node startup, topology connection, and round-based learning | Local in-process Python simulation driven by `run_benchmark.py` |
| Communication backend | p2pfl over local node addresses | `COMMUNICATION_BACKEND = "local_simulation"` and `EXECUTION_MODE = "python_local_simulation"` |
| Number of workers | Public branch currently shows `NUM_WORKERS = 5` | Team-aligned config surface keeps `NUM_WORKERS = 10`; benchmark sweeps may still evaluate other client counts separately |
| Rounds | Public branch currently shows `ROUNDS = 10` | Team-aligned config surface keeps `ROUNDS = 100`; benchmark fast mode remains a separate smoke-test path |
| Lambda / regularization | Active SDCA fields include `LAMBDA = 1e-4` and `T0_FRACTION = 0.5` | These remain absent from active config; this repo uses method-specific parameters such as `FEDAVG_LAMBDA`, `FDR_LAMBDA`, `BDSVM_C`, `FDR_RHO`, and `FEDSSL_SVM_C` |
| Model organization | Single p2p-SDCA model stack centered on p2pfl and topology-aware execution | Separate runnable method modules in `methods/centralized.py`, `methods/fedavg_svm.py`, `methods/fdr_svm.py`, `methods/bdsvm.py`, and `methods/fedssl_amc.py` |
| Data loader organization | `data_loader.py` prepares worker shards and p2pfl/HF dataset wrappers | `benchmark.py` handles downloads, cached loading, and local client partitioners for IID, label-skew, and Dirichlet splits |
| Documentation / reproducibility | Reproducibility is tied to p2pfl node setup and topology-specific execution | Reproducibility is tied to deterministic local seeds, cached dataset loading, and per-method benchmark commands |

## Relation to Sreekar's p2p-SDCA Branch

Sreekar's reference branch implements a p2pfl-based p2p SDCA workflow with node setup, topology wiring, local addresses, and p2p learning rounds.

This benchmark repository currently remains a Python local simulation. It does not yet use p2pfl, TCP/gRPC/socket communication, or real multi-node execution.

The current benchmark borrows the centralized configuration style and field names where appropriate, but it does not directly copy Sreekar's SDCA-specific hyperparameters such as `LAMBDA=1e-4` and `T0_FRACTION=0.5`.

In practice, the branch roles differ:

- Sreekar branch: p2p SDCA implementation built around topology-aware nodes.
- This branch: benchmark harness comparing multiple SVM methods under a shared local simulation workflow.
- Future work: possible CCR/HPC adaptation, potentially with p2pfl or a different distributed backend.

## Config Alignment Notes

The current `config.py` keeps the following Sreekar-style field names for structural alignment and future compatibility work:

```python
CONFIG = {
    "NUM_WORKERS": 10,
    "TOPOLOGY": "star",
    "ROUNDS": 100,
    "EPOCHS": 1,
    "BASE_PORT": 6000,
    "GOSSIP_K": 1,
    "SEED": 42,
    "COMMUNICATION_BACKEND": "local_simulation",
    "EXECUTION_MODE": "python_local_simulation",
    "FUTURE_TARGET_ENV": "CCR_HPC",
}
```

`TOPOLOGY`, `EPOCHS`, `BASE_PORT`, and `GOSSIP_K` are not active networking controls in the current benchmark. They are retained only for structural alignment and future CCR/HPC or p2p adaptation. Current communication is still local Python simulation.

Sreekar p2p-SDCA branch values that are intentionally not active here:

- `LAMBDA = 1e-4`
- `T0_FRACTION = 0.5`

This repository instead keeps method-specific parameters close to the corresponding method implementation or benchmark runner configuration.

## Optional Future p2p/CCR Dependencies

These are not active benchmark requirements today and should not be added to active requirements until real distributed code is implemented:

- `p2pfl`
- `lightning`
- `torch`
- `datasets`

## Future p2p / CCR Adaptation

To adapt this benchmark toward Sreekar's p2pfl branch or CCR/HPC execution, future work should:

1. Decide whether to use p2pfl, multiprocessing, MPI, Ray, or another backend.
2. Map simulated workers to real processes/jobs/nodes.
3. Wire `TOPOLOGY`, `BASE_PORT`, and node addresses into an actual communication backend.
4. Decide whether Sreekar's SDCA-specific `LAMBDA` and `T0_FRACTION` apply only to p2p-SDCA or should be added as a separate method.
5. Keep the benchmark method API separate so each model remains independently runnable.