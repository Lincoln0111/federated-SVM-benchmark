# Hyperparameter Inventory — Federated SVM Benchmark

All parameters are centralised in [`config.py`](../config.py).
See [`RUNNING.md`](../RUNNING.md) for setup and execution instructions.

---

## SVM Implementation Details

All values below were **preserved from the original implementation**. None were changed during the config refactor (only `NUM_WORKERS` and `ROUNDS` were intentionally modified for team alignment — those are in the next section).

| Parameter | Value | CONFIG key | Used by | Purpose | Source | Changed from original? | Notes |
|---|---|---|---|---|---|---|---|
| `FEDAVG_LAMBDA` | `0.01` | `CONFIG["FEDAVG_LAMBDA"]` | FedAvg-SVM | Pegasos regularisation λ in `min λ‖w‖² + loss` | original `lambda_reg=0.01` in `fedavg_svm.py` | No | preserved |
| `FDR_LAMBDA` | `0.01` | `CONFIG["FDR_LAMBDA"]` | FDR-SVM | Base regularisation λ for DRO objective | original `lambda_reg=0.01` in `fdr_svm.py` | No | preserved; same value as `FEDAVG_LAMBDA` but kept separate for independent tuning |
| `BDSVM_C` | `1.0` | `CONFIG["BDSVM_C"]` | BDSVM | Primal SVM cost C in `min ½‖β‖² + C·Σ hinge` | original `C=1.0` in `bdsvm.py` | No | preserved |
| `CENTRALIZED_C` | `1.0` | `CONFIG["CENTRALIZED_C"]` | Centralized SVM | LinearSVC cost C | original `C=1.0` in `centralized.py` | No | preserved |
| `BDSVM_P` | `100` | `CONFIG["BDSVM_P"]` | BDSVM | Budget size — number of random pre-image vectors P; kernel matrix is N×P, communication per round is O(P²) | original `P=100` in `bdsvm.py` | No | preserves communication efficiency guarantee from ACM TIST 2022 paper |
| `MOMENTUM_ALPHA` | `0.5` | `CONFIG["MOMENTUM_ALPHA"]` | BDSVM | Momentum blend λ in Algorithm 2 step 8: `β̃^(n+1) = λ·β̃^(n) + (1−λ)·β̃_new`; 0 = no momentum | original `lam=0.5` in `bdsvm.py` | No | preserved |
| `IRWLS_ETA` | `5e-3` | `CONFIG["IRWLS_ETA"]` | BDSVM (IRWLS convergence) | Convergence threshold η: stop when `‖β_new−β‖/‖β‖ < η`; IRWLS weights are `a_i = 2C/max(e_i y_i, ε)` clipped at 1e6 | original `eta=5e-3` in `bdsvm.py` | No | preserved |
| `FDR_RHO` | `1.0` | `CONFIG["FDR_RHO"]` | FDR-SVM | ADMM penalty parameter ρ in augmented Lagrangian | original `rho=1.0` in `fdr_svm.py` | No | preserved |
| `FDR_EPS_SCALE` | `1.0` | `CONFIG["FDR_EPS_SCALE"]` | FDR-SVM | Wasserstein ball radius multiplier: `ε_k = FDR_EPS_SCALE / sqrt(n_k)` | original `eps_scale=1.0` in `fdr_svm.py` | No | preserved; larger values → more distributionally robust |
| `FEDSSL_D_ENC` | `64` | `CONFIG["FEDSSL_D_ENC"]` | FedSSL-AMC | Federated encoder output dimension d_enc (clamped to `min(d_enc, d−1)`) | original `d_enc=64` in `fedssl_amc.py` | No | preserved |
| `FEDSSL_SVM_C` | `1.0` | `CONFIG["FEDSSL_SVM_C"]` | FedSSL-AMC | Phase-2 local LinearSVC cost C | original `svm_C=1.0` in `fedssl_amc.py` | No | preserved |
| `CENTRALIZED_MAX_ITER` | `5000` | `CONFIG["CENTRALIZED_MAX_ITER"]` | Centralized SVM | liblinear solver max iterations | original `max_iter=5000` in `centralized.py` | No | preserved |
| `CENTRALIZED_SUBSAMPLE` | `50_000` | `CONFIG["CENTRALIZED_SUBSAMPLE"]` | Centralized SVM | Max training samples for the centralized baseline | original `subsample=50_000` in `centralized.py` | No | preserved |
| `FEDAVG_LOCAL_STEPS` | `100` | `CONFIG["FEDAVG_LOCAL_STEPS"]` | FedAvg-SVM | Local Pegasos SGD steps per round | original `n_local_steps=100` in `fedavg_svm.py` | No | preserved |
| `FDR_LOCAL_STEPS` | `100` | `CONFIG["FDR_LOCAL_STEPS"]` | FDR-SVM | Local Pegasos steps per ADMM round | original `n_local_steps=100` in `fdr_svm.py` | No | preserved |

### Reference values intentionally absent

The following values appear in the sreekarsWindows reference `config.py` but are **not present** in this benchmark's `config.py`:

| Reference key | Reference value | Why absent |
|---|---|---|
| `LAMBDA` | `1e-4` | SDCA formulation; conflicts with `FEDAVG_LAMBDA=0.01` and `FDR_LAMBDA=0.01` (different parameterisations). Importing it would silently change algorithmic behaviour without team agreement. |
| `T0_FRACTION` | `0.5` | SDCA step-size schedule. No counterpart exists in the current Pegasos/ADMM/IRWLS implementations. |

---

## Distributed / Federated Settings

| Setting | Value | Active? | CONFIG key | Notes |
|---|---|---|---|---|
| Workers / clients | `10` | **Yes** | `CONFIG["NUM_WORKERS"]` | **Team-aligned** (was `[10, 50]` sweep); simulated Python clients — not separate network nodes |
| Communication rounds | `100` | **Yes** | `CONFIG["ROUNDS"]` | **Team-aligned** (was `50`); simulated rounds — no actual network I/O |
| Communication backend | `local_simulation` | **Yes** | `CONFIG["COMMUNICATION_BACKEND"]` | No TCP, gRPC, socket, or MPI communication in current runner |
| Execution mode | `python_local_simulation` | **Yes** | `CONFIG["EXECUTION_MODE"]` | Single-machine Python benchmark; workers are in-process function calls |
| Future target environment | `CCR_HPC` | Planned | `CONFIG["FUTURE_TARGET_ENV"]` | Future adaptation to HPC cluster; requires job scheduling and IPC work |
| Network topology | `"star"` | **No — placeholder** | `CONFIG["TOPOLOGY"]` | Not an active real-network topology; kept for schema alignment with sreekarsWindows |
| Base TCP port | `6000` | **No — placeholder** | `CONFIG["BASE_PORT"]` | Reserved for future networking; port is not currently bound or used |
| Gossip fan-out | `1` | **No — placeholder** | `CONFIG["GOSSIP_K"]` | Only meaningful once a gossip/p2p layer is implemented |
| Local epochs per round | `1` | **No — placeholder** | `CONFIG["EPOCHS"]` | Meaning TBD (per-round local epochs vs global epochs); not currently read |

### Two intentionally changed settings

| Setting | Original value | New value | Reason |
|---|---|---|---|
| `NUM_WORKERS` | `[10, 50]` (sweep) | `10` | Team alignment with sreekarsWindows |
| `ROUNDS` | `50` | `100` | Team alignment with sreekarsWindows |

---

## Shared / Infrastructure Settings

| Parameter | Value | CONFIG key | Category |
|---|---|---|---|
| Random seed | `42` | `CONFIG["SEED"]` | Active |
| Datasets | `["rcv1", "covtype"]` | `CONFIG["DATASETS"]` | Active |
| Train file | `data/rcv1_train.binary.bz2` | `CONFIG["TRAIN_PATH"]` | Active — flat `data/` layout |
| Test file | `data/rcv1_test.binary.bz2` | `CONFIG["TEST_PATH"]` | Active |
| Train cap | `{"rcv1": None, "covtype": 50_000}` | `CONFIG["MAX_TRAIN"]` | Active |
| Test cap | `20_000` | `CONFIG["MAX_TEST"]` | Active — rcv1 test set has 677k rows |
| Dirichlet α sweep | `[1.0, 0.3, 0.1]` | `CONFIG["DIRICHLET_ALPHAS"]` | Active |
| Fast rounds | `10` | `CONFIG["FAST_ROUNDS"]` | Active (`--fast` flag) |

**Path note:** `DATA_DIR = CODE_DIR / "data"` (flat layout). The reference sreekarsWindows config uses `data/processed/`. Do not change the path to match the reference without first moving the data files.

---

## Unresolved TODOs

| # | Item | Action required |
|---|---|---|
| 1 | `LAMBDA` unification | Agree with Sreekar: `lambda_reg=0.01` (Pegasos) vs `LAMBDA=1e-4` (SDCA). Until resolved, each method uses its own key. |
| 2 | `T0_FRACTION` | Decide whether to implement SDCA step-size schedule. Currently absent. |
| 3 | `TOPOLOGY` | Confirm value and wire into p2p layer before CCR/HPC merge. |
| 4 | `EPOCHS` | Clarify: per-round local epochs or global dataset passes? |
| 5 | `BASE_PORT` | Confirm port range with Sreekar for future networking. |
| 6 | `GOSSIP_K` | Confirm gossip fan-out once gossip/p2p is implemented. |
| 7 | `n_rounds=50` function defaults | Method files still carry `n_rounds=50` as fallback API defaults. Runner always overrides via `METHOD_KWARGS`. Optionally change signatures to `n_rounds=None` to make the override explicit. |

- **Active** — read by `run_benchmark.py` or a method file at runtime
- **Placeholder** — in `config.py` for structural alignment only; not read by any current code
- **Team-alignment** — value was deliberately changed to match sreekarsWindows
- **TODO** — value or usage needs team confirmation before the branch can be merged

---

## Shared Training Parameters

| Parameter | CONFIG key | Value | Original value | Changed? | Must match Sreekar? | Category |
|---|---|---|---|---|---|---|
| Workers | `NUM_WORKERS` | `10` | `[10, 50]` sweep | **Yes — team alignment** | **Yes** | Team-alignment |
| Rounds | `ROUNDS` | `100` | `50` | **Yes — team alignment** | **Yes** | Team-alignment |
| Seed | `SEED` | `42` | `42` (argparse default) | No | Yes | Active |

---

## Dataset / Path Parameters

| Parameter | CONFIG key | Value | Notes | Category |
|---|---|---|---|---|
| Dataset list | `DATASETS` | `["rcv1", "covtype"]` | Unchanged from original | Active |
| Train file | `TRAIN_PATH` | `data/rcv1_train.binary.bz2` | Flat `data/` layout (no `data/processed/` subdirectory exists) | Active |
| Test file | `TEST_PATH` | `data/rcv1_test.binary.bz2` | Same | Active |
| covtype file | `COVTYPE_PATH` | `data/covtype.libsvm.binary.scale.bz2` | Same | Active |
| Train cap | `MAX_TRAIN` | `{"rcv1": None, "covtype": 50_000}` | Unchanged | Active |
| Test cap | `MAX_TEST` | `20_000` | rcv1 test is 677k rows; cap limits eval cost | Active |

**Path note:** The reference sreekarsWindows config uses `DATA_DIR = CODE_DIR / "data" / "processed"`.
This benchmark uses `DATA_DIR = CODE_DIR / "data"` (flat layout). Files are downloaded directly into `data/` by `benchmark.py --download`. Do not change the path to match the reference without first moving the data files.

---

## p2p / Gossip Parameters — PLACEHOLDERS

**These keys are currently NOT read by any code in this benchmark runner.**
They exist solely for structural alignment with the sreekarsWindows config schema.
They must be wired up before merging into the main p2p-capable Gadget SVM repo.

| Parameter | CONFIG key | Value | Category |
|---|---|---|---|
| Topology | `TOPOLOGY` | `"star"` | Placeholder — TODO: confirm with Sreekar |
| Local epochs | `EPOCHS` | `1` | Placeholder — TODO: clarify meaning (per-round local epochs?) |
| Base TCP port | `BASE_PORT` | `6000` | Placeholder — TODO: confirm with Sreekar |
| Gossip fan-out | `GOSSIP_K` | `1` | Placeholder — TODO: confirm with Sreekar |

---

## SDCA / SVM Regularisation

Two distinct regularisation parameterisations are used across methods. They are **not collapsed** into a single ambiguous `LAMBDA`.

| Concept | CONFIG key | Value | Original source | Used by | Category |
|---|---|---|---|---|---|
| Pegasos λ | `FEDAVG_LAMBDA` | `0.01` | `lambda_reg=0.01` in `fedavg_svm.py` | FedAvg-SVM | Active |
| DRO base λ | `FDR_LAMBDA` | `0.01` | `lambda_reg=0.01` in `fdr_svm.py` | FDR-SVM | Active |
| Primal SVM cost C | `BDSVM_C` | `1.0` | `C=1.0` in `bdsvm.py` | BDSVM | Active |
| Centralized cost C | `CENTRALIZED_C` | `1.0` | `C=1.0` in `centralized.py` | Centralized | Active |
| Phase-2 SVM cost C | `FEDSSL_SVM_C` | `1.0` | `svm_C=1.0` in `fedssl_amc.py` | FedSSL-AMC | Active |

**sreekarsWindows reference values intentionally absent:**
- `LAMBDA = 1e-4` — would silently override `FEDAVG_LAMBDA=0.01` and `FDR_LAMBDA=0.01`
- `T0_FRACTION = 0.5` — SDCA step-size schedule; has no counterpart in this implementation

Both require reconciliation with Sreekar before the branch is merged.

---

## BDSVM Parameters (ACM TIST 2022)

| Parameter | CONFIG key | Value | Original source | Changed? | Category |
|---|---|---|---|---|---|
| Budget size | `BDSVM_P` | `100` | `P=100` in `bdsvm.py` signature | No | Active |
| SVM cost C | `BDSVM_C` | `1.0` | `C=1.0` in `bdsvm.py` signature | No | Active |
| Momentum blend λ | `MOMENTUM_ALPHA` | `0.5` | `lam=0.5` in `bdsvm.py` signature | No | Active |
| IRWLS convergence η | `IRWLS_ETA` | `5e-3` | `eta=5e-3` in `bdsvm.py` signature | No | Active |

IRWLS weight formula (Algorithm 3 step 6): `a_i = 2C / max(e_i y_i, ε)`, clipped at 1e6.
Stop condition (Algorithm 2 step 9): `‖β_new − β‖ / ‖β‖ < IRWLS_ETA`.

---

## FedAvg-SVM Parameters (arXiv 1812.11750)

| Parameter | CONFIG key | Value | Original source | Changed? | Category |
|---|---|---|---|---|---|
| Regularisation λ | `FEDAVG_LAMBDA` | `0.01` | `lambda_reg=0.01` in signature | No | Active |
| Local SGD steps | `FEDAVG_LOCAL_STEPS` | `100` | `n_local_steps=100` in signature | No | Active |

---

## FDR-SVM Parameters (arXiv 2410.03877)

| Parameter | CONFIG key | Value | Original source | Changed? | Category |
|---|---|---|---|---|---|
| Base regularisation λ | `FDR_LAMBDA` | `0.01` | `lambda_reg=0.01` in signature | No | Active |
| ADMM penalty ρ | `FDR_RHO` | `1.0` | `rho=1.0` in signature | No | Active |
| Wasserstein radius scale | `FDR_EPS_SCALE` | `1.0` | `eps_scale=1.0` in signature | No | Active |
| Local Pegasos steps | `FDR_LOCAL_STEPS` | `100` | `n_local_steps=100` in signature | No | Active |

Radius formula: `ε_k = FDR_EPS_SCALE / sqrt(n_k)`.

---

## FedSSL-AMC Parameters (arXiv 2510.04927)

| Parameter | CONFIG key | Value | Original source | Changed? | Category |
|---|---|---|---|---|---|
| Encoder dimension | `FEDSSL_D_ENC` | `64` | `d_enc=64` in signature | No | Active |
| Phase-2 SVM cost C | `FEDSSL_SVM_C` | `1.0` | `svm_C=1.0` in signature | No | Active |

---

## Centralized Baseline

| Parameter | CONFIG key | Value | Original source | Changed? | Category |
|---|---|---|---|---|---|
| SVM cost C | `CENTRALIZED_C` | `1.0` | `C=1.0` in signature | No | Active |
| Max solver iterations | `CENTRALIZED_MAX_ITER` | `5000` | `max_iter=5000` in signature | No | Active |
| Training subsample cap | `CENTRALIZED_SUBSAMPLE` | `50_000` | `subsample=50_000` in signature | No | Active |

Note: the previous `METHOD_KWARGS["centralized"]` was `{}` (relying on function defaults).
It now passes these values explicitly from CONFIG. Behaviour is identical since values are unchanged.

---

## Fast / Smoke-Test Parameters

| Parameter | CONFIG key | Value | Category |
|---|---|---|---|
| Fast datasets | `FAST_DATASETS` | `["rcv1"]` | Active (`--fast` flag) |
| Fast rounds | `FAST_ROUNDS` | `10` | Active |
| Fast train cap | `FAST_MAX_TRAIN` | `5_000` | Active |
| Fast test cap | `FAST_MAX_TEST` | `2_000` | Active |

---

## Partitioning

| Parameter | CONFIG key | Value | Category |
|---|---|---|---|
| Dirichlet α sweep | `DIRICHLET_ALPHAS` | `[1.0, 0.3, 0.1]` | Active |

---

## Unresolved TODOs

| # | Item | Action required |
|---|---|---|
| 1 | `LAMBDA` / regularisation unification | Agree with Sreekar on one formulation (`lambda_reg=0.01` vs `LAMBDA=1e-4`). Until then, each method keeps its own key. |
| 2 | `T0_FRACTION` | Decide if SDCA step-size schedule will be implemented. Currently absent from all algorithm code. |
| 3 | `TOPOLOGY` | Confirm topology value when implementing p2p layer. |
| 4 | `EPOCHS` | Clarify: per-round local epochs, or global epochs over the dataset? |
| 5 | `BASE_PORT` | Confirm port range with Sreekar. |
| 6 | `GOSSIP_K` | Confirm gossip fan-out with Sreekar. |
| 7 | `n_rounds=50` function defaults | Method files still carry their original `n_rounds=50` fallback defaults. The runner always overrides via `METHOD_KWARGS`, so runtime is correct. Optionally update the signatures to `n_rounds=None` to make the override explicit. |

## Parameter Table

| Parameter | Value | Used in | Purpose | Must match Sreekar? | Notes |
|---|---|---|---|---|---|
| `TOPOLOGY` | `"star"` | _reserved_ | p2p network topology | Yes | **TODO: NEEDS_CONFIRMATION** — no p2p layer in current centralized-sim code |
| `EPOCHS` | `1` | _reserved_ | local training epochs per round | Yes | **TODO: NEEDS_CONFIRMATION** — not used in current code |
| `BASE_PORT` | `6000` | _reserved_ | base TCP port for worker sockets | Yes | **TODO: NEEDS_CONFIRMATION** — no socket layer in current code |
| `GOSSIP_K` | `1` | _reserved_ | gossip fan-out per round | Yes | **TODO: NEEDS_CONFIRMATION** — no gossip layer in current code |
| `NUM_WORKERS` | `10` | `run_benchmark.py` | number of FL clients / workers | **Yes** | Aligned with sreekarsWindows. Previous value in this repo: `[10, 50]` sweep |
| `ROUNDS` | `100` | `run_benchmark.py` | FL communication rounds | **Yes** | Aligned with sreekarsWindows. Previous value in this repo: `50` |
| `SEED` | `42` | `run_benchmark.py` | global random seed | Yes | Consistent across implementations |
| `LAMBDA` | `1e-4` | _reference only_ | SDCA regularisation λ | Yes | **TODO: NEEDS_CONFIRMATION** — reference value from sreekarsWindows SDCA formulation; current code uses `lambda_reg=0.01` (Pegasos) and `C=1.0` (primal SVM). These are different parameterisations and must be reconciled with the team |
| `T0_FRACTION` | `0.5` | _reference only_ | SDCA step-size schedule fraction | Yes | **TODO: NEEDS_CONFIRMATION** — appears in sreekarsWindows SDCA schedule; has no counterpart in current implementation |
| `TRAIN_PATH` | `data/rcv1_train.binary.bz2` | `benchmark.py`, `config.py` | rcv1 training set path | Partial | Absolute path constructed from `DATA_DIR`; filename matches LIBSVM convention |
| `TEST_PATH` | `data/rcv1_test.binary.bz2` | `benchmark.py`, `config.py` | rcv1 test set path | Partial | Same as above |
| `DIRICHLET_ALPHAS` | `[1.0, 0.3, 0.1]` | `run_benchmark.py` | heterogeneity sweep values for Dirichlet partition | No | Standard FL benchmark sweep (α→0 = highly non-IID) |
| `MAX_TRAIN` | `{"rcv1": None, "covtype": 50_000}` | `run_benchmark.py` | per-dataset training cap | No | covtype capped at 50k; rcv1 uses full set |
| `MAX_TEST` | `20_000` | `run_benchmark.py` | test evaluation cap (rcv1 test = 677k) | No | Caps evaluation cost |
| **BDSVM** | | | | | |
| `BDSVM_P` | `100` | `methods/bdsvm.py` | budget size (random pre-image vectors P) | No | Paper: "P determines communication overhead O(P²) per round" |
| `BDSVM_C` | `1.0` | `methods/bdsvm.py` | SVM hinge-loss cost C (primal form) | No | Inverse of λ in dual; differs from `FEDAVG_LAMBDA` |
| `MOMENTUM_ALPHA` | `0.5` | `methods/bdsvm.py` | momentum blend λ in Algorithm 2 step 8 | No | `0` = no momentum, `1` = frozen weights |
| `IRWLS_ETA` | `5e-3` | `methods/bdsvm.py` | IRWLS convergence threshold η | No | Stop when ‖β_new−β‖/‖β‖ < η |
| **FedAvg-SVM** | | | | | |
| `FEDAVG_LAMBDA` | `0.01` | `methods/fedavg_svm.py` | Pegasos regularisation λ | No | |
| `FEDAVG_LOCAL_STEPS` | `100` | `methods/fedavg_svm.py` | local SGD steps per round | No | |
| **FDR-SVM** | | | | | |
| `FDR_LAMBDA` | `0.01` | `methods/fdr_svm.py` | base regularisation λ | No | Same value as `FEDAVG_LAMBDA`; kept separate for independent tuning |
| `FDR_RHO` | `1.0` | `methods/fdr_svm.py` | ADMM penalty parameter ρ | No | |
| `FDR_EPS_SCALE` | `1.0` | `methods/fdr_svm.py` | Wasserstein ball radius multiplier | No | ε_k = FDR_EPS_SCALE / √n_k |
| `FDR_LOCAL_STEPS` | `100` | `methods/fdr_svm.py` | local Pegasos steps per ADMM round | No | |
| **FedSSL-AMC** | | | | | |
| `FEDSSL_D_ENC` | `64` | `methods/fedssl_amc.py` | federated encoder output dimension | No | Clamped to min(d_enc, d−1) inside the method |
| `FEDSSL_SVM_C` | `1.0` | `methods/fedssl_amc.py` | phase-2 local LinearSVC cost C | No | |
| **Centralized baseline** | | | | | |
| `CENTRALIZED_C` | `1.0` | `methods/centralized.py` | LinearSVC cost C | No | |
| `CENTRALIZED_MAX_ITER` | `5000` | `methods/centralized.py` | liblinear solver max iterations | No | |
| `CENTRALIZED_SUBSAMPLE` | `50_000` | `methods/centralized.py` | max training samples for baseline | No | Applied after any `MAX_TRAIN` cap in the runner |
| **Fast/smoke-test** | | | | | |
| `FAST_ROUNDS` | `10` | `run_benchmark.py` | rounds in `--fast` mode | No | |
| `FAST_MAX_TRAIN` | `5_000` | `run_benchmark.py` | training cap in `--fast` mode | No | |
| `FAST_MAX_TEST` | `2_000` | `run_benchmark.py` | test cap in `--fast` mode | No | |

---

## BDSVM Implementation Note

The current `methods/bdsvm.py` uses the **IRWLS / pre-image kernel-matrix** variant
(Algorithms 2 & 3 from the ACM TIST 2022 paper). This differs from an earlier committed
ADMM/Pegasos-style BDSVM that used kwargs `budget`, `n_local_steps`, `lambda_reg`, `rho`.

The current implementation passes these kwargs from `CONFIG`:

| CONFIG key | Value | kwarg name | Purpose |
|---|---|---|---|
| `BDSVM_P` | `100` | `P` | Number of random pre-image vectors |
| `BDSVM_C` | `1.0` | `C` | SVM hinge-loss cost |
| `MOMENTUM_ALPHA` | `0.5` | `lam` | Momentum blend λ (Algorithm 2, step 8) |
| `IRWLS_ETA` | `5e-3` | `eta` | IRWLS convergence threshold η |

To recover the earlier ADMM/Pegasos variant, check the git history of `methods/bdsvm.py`.

---

## Unresolved TODOs

| # | Parameter | Conflict / Question |
|---|---|---|
| 1 | `LAMBDA` | sreekarsWindows uses `1e-4` (SDCA formulation). Current code uses `lambda_reg=0.01` (Pegasos, FedAvg/FDR) and `C=1.0` (primal SVM, BDSVM). Must decide on a single canonical formulation before merging. |
| 2 | `T0_FRACTION` | Appears in sreekarsWindows SDCA schedule. Not implemented here. Decide whether to add SDCA-specific schedule or leave as unused stub. |
| 3 | `TOPOLOGY` | Current code runs a centralized simulation loop — no network topology. Required when merging into p2p-capable main repo. |
| 4 | `BASE_PORT` | Required for p2p layer. Currently unused. |
| 5 | `GOSSIP_K` | Required for gossip protocol. Currently unused. |
| 6 | `EPOCHS` | Unclear whether this means local epochs per round or global epochs. Needs clarification with Sreekar. |
| 7 | `ROUNDS` | Previous value was 50 in this repo. Changed to 100 per team spec. Method function-signature defaults (`n_rounds=50`) are overridden via `METHOD_KWARGS`; runtime behaviour is correct. |
| 8 | `NUM_WORKERS` | Previous value was a sweep `[10, 50]`. Changed to single value `10` per team spec. |
