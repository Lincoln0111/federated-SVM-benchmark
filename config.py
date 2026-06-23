"""Centralised experiment configuration.

Follows the same structure as the sreekarsWindows reference config so this
repository can be merged cleanly into the main Gadget SVM repo under a
personal branch (e.g. ``git checkout -b jiaye-config-refactor``).

Usage::

    from config import CONFIG, DATA_DIR

    num_workers = CONFIG["NUM_WORKERS"]
    rounds      = CONFIG["ROUNDS"]
    train_path  = CONFIG["TRAIN_PATH"]
"""
from pathlib import Path

CODE_DIR = Path(__file__).parent
RAW_DIR  = CODE_DIR / "data" / "raw"      # reserved — no raw/ subdirectory exists yet
DATA_DIR = CODE_DIR / "data"              # actual location: data/*.bz2 and data/*.npz
                                           # (not data/processed/ — flat layout used here)

CONFIG = {
    # ── Execution mode ───────────────────────────────────────────────────────
    # This benchmark currently runs as a Python local simulation.
    # No TCP, gRPC, socket communication, or real multi-node execution is used.
    "COMMUNICATION_BACKEND": "local_simulation",
    "EXECUTION_MODE":        "python_local_simulation",
    # Future target: simulated workers may be mapped to separate CCR/HPC jobs.
    "FUTURE_TARGET_ENV":     "CCR_HPC",

    # ── p2p / gossip placeholders ──────────────────────────────────────────
    # Currently NOT used by this Python local simulation benchmark runner.
    # Included only for structural alignment with sreekarsWindows and future
    # CCR/HPC or networking-based work. These are NOT active real-network
    # parameters — "star" is not a live topology; BASE_PORT is not bound.
    "TOPOLOGY":   "star",
    "EPOCHS":     1,
    "BASE_PORT":  6000,
    "GOSSIP_K":   1,

    # ── Shared training (aligned with sreekarsWindows) ─────────────────────
    # Previous values in run_benchmark.py:
    #   FULL_CLIENTS = [10, 50]  (sweep — reduced to single value per team spec)
    #   FULL_ROUNDS  = 50        (increased to 100 per team spec)
    "NUM_WORKERS": 10,
    "ROUNDS":      100,
    "SEED":        42,

    # ── Dataset paths ──────────────────────────────────────────────────────
    "DATASETS":     ["rcv1", "covtype"],
    "TRAIN_PATH":   DATA_DIR / "rcv1_train.binary.bz2",
    "TEST_PATH":    DATA_DIR / "rcv1_test.binary.bz2",
    "COVTYPE_PATH": DATA_DIR / "covtype.libsvm.binary.scale.bz2",
    "MAX_TRAIN":    {"rcv1": None, "covtype": 50_000},
    "MAX_TEST":     20_000,

    # ── SVM regularisation ──────────────────────────────────────────────────
    # Two independent regularisation parameters are used, matching the original code:
    #   LAMBDA_REG = 0.01  — Pegasos/ADMM formulation (FedAvg-SVM, FDR-SVM)
    #   C          = 1.0   — primal SVM cost parameter (BDSVM, Centralized, FedSSL)
    # These are NOT collapsed into a single LAMBDA; they are different
    # parameterisations and each method section below references its own key.
    # NOTE: the sreekarsWindows reference config carries LAMBDA=1e-4 (SDCA
    # formulation). That value is NOT imported here because it would silently
    # change algorithmic behaviour. Reconciliation with Sreekar is required
    # before adopting a unified regularisation constant.

    # ── BDSVM (ACM TIST 2022, DOI 10.1145/3539734) ─────────────────────────
    "BDSVM_P":    100,        # budget size: number of random pre-image vectors P
    "BDSVM_C":    1.0,        # SVM hinge-loss cost C (primal formulation)

    # Momentum blend used in Algorithm 2 step 8:
    #   β̃^(n+1) = MOMENTUM_ALPHA · β̃^(n) + (1 − MOMENTUM_ALPHA) · β̃_new
    # Corresponds to parameter lam (λ) in the paper. 0 = no momentum.
    "MOMENTUM_ALPHA": 0.5,

    # IRWLS convergence — Algorithm 2 step 9 / Algorithm 3 step 6:
    #   weights  a_i = 2C / max(e_i y_i, ε)  (clipped at 1e6 for stability)
    #   stop when  ‖β_new − β‖ / ‖β‖ < IRWLS_ETA
    "IRWLS_ETA":  5e-3,

    # ── FedAvg-SVM (arXiv 1812.11750) ─────────────────────────────────────
    "FEDAVG_LAMBDA":      0.01,   # Pegasos regularisation λ
    "FEDAVG_LOCAL_STEPS": 100,    # local SGD steps per round

    # ── FDR-SVM (arXiv 2410.03877) ────────────────────────────────────────
    "FDR_LAMBDA":      0.01,      # base regularisation λ
    "FDR_RHO":         1.0,       # ADMM penalty parameter ρ
    "FDR_EPS_SCALE":   1.0,       # Wasserstein ball radius multiplier
                                  #   ε_k = FDR_EPS_SCALE / sqrt(n_k)
    "FDR_LOCAL_STEPS": 100,       # local Pegasos steps per ADMM round

    # ── FedSSL-AMC (arXiv 2510.04927) ─────────────────────────────────────
    "FEDSSL_D_ENC":  64,          # encoder output dimension d_enc
    "FEDSSL_SVM_C":  1.0,         # phase-2 local SVM cost C

    # ── Centralized baseline ───────────────────────────────────────────────
    "CENTRALIZED_C":         1.0,
    "CENTRALIZED_MAX_ITER":  5000,
    "CENTRALIZED_SUBSAMPLE": 50_000,  # max training samples for the baseline

    # ── Fast / smoke-test overrides ────────────────────────────────────────
    "FAST_DATASETS":   ["rcv1"],
    "FAST_ROUNDS":     10,
    "FAST_MAX_TRAIN":  5_000,
    "FAST_MAX_TEST":   2_000,

    # ── Partitioning scheme ────────────────────────────────────────────────
    "DIRICHLET_ALPHAS": [1.0, 0.3, 0.1],  # α values for non-IID heterogeneity sweep

    # ── Output ─────────────────────────────────────────────────────────────
    "RESULTS_PATH":  CODE_DIR / "results.csv",
}

INACTIVE_ALIGNMENT_FIELDS = {
    "TOPOLOGY",
    "EPOCHS",
    "BASE_PORT",
    "GOSSIP_K",
}

CONFIG_ALIGNMENT_NOTE = (
    "TOPOLOGY, EPOCHS, BASE_PORT, and GOSSIP_K are retained only for structural "
    "alignment with Sreekar's p2p-SDCA branch and future CCR/HPC adaptation. "
    "The current benchmark remains a Python local simulation with no active "
    "TCP, gRPC, socket, or p2pfl backend."
)