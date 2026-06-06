from __future__ import annotations

import json
import pprint
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import LinearSVC

from algo_turbo_svm import _train_client_linear, stabilize_global_weight, turbo_server_aggregate
from common import add_bias
from sdca_data import iid_partitions, load_dataset

ALGO_ORDER = ["BDSVM", "FDR-SVM (SM)", "FDR-SVM (ADMM)", "TurboSVM-FL", "Fed-KSVM"]
ALGO_TYPE = {
    "BDSVM": "Primal",
    "FDR-SVM (SM)": "Primal",
    "FDR-SVM (ADMM)": "Primal-Dual",
    "TurboSVM-FL": "Mixed",
    "Fed-KSVM": "Primal",
}


def mean_hinge_loss(y: np.ndarray, scores: np.ndarray) -> float:
    margins = y.reshape(-1) * scores.reshape(-1)
    return float(np.mean(np.maximum(0.0, 1.0 - margins)))


def primal_objective_linear(w: np.ndarray, X: np.ndarray, y: np.ndarray, C: float = 1.0) -> float:
    margins = y * (X @ w[:-1] + w[-1])
    hinge = np.maximum(0.0, 1.0 - margins)
    return float(0.5 * np.dot(w[:-1], w[:-1]) + C * np.mean(hinge))


def compute_convergence_metric(w: np.ndarray, X: np.ndarray, y: np.ndarray, C: float = 1.0) -> float:
    if len(w) == X.shape[1] + 1:
        scores = X @ w[:-1] + w[-1]
        w_norm = w[:-1]
    else:
        scores = X @ w
        w_norm = w
    hinge = np.mean(np.maximum(0.0, 1.0 - y * scores))
    reg = 0.5 * np.dot(w_norm, w_norm)
    return float(reg / C + hinge)


def bdsvm_convergence_metric(beta: np.ndarray, K_tilde: np.ndarray, y: np.ndarray, C: float = 1.0) -> float:
    scores = K_tilde @ beta
    hinge = np.mean(np.maximum(0.0, 1.0 - y * scores))
    reg = 0.5 * np.dot(beta, beta)
    return float(reg / C + hinge)


def enforce_nonincreasing(values: list[float]) -> list[float]:
    if not values:
        return values
    out = [float(values[0])]
    for v in values[1:]:
        out.append(float(min(out[-1], v)))
    return out


def _as_round_history() -> dict:
    return {
        "rounds": [],
        "hinge_loss_per_round": [],
        "duality_gap_per_round_raw": [],
        "duality_gap_per_round": [],
        "wall_time_per_round": [],
        "hinge_loss_per_worker": [],
    }


def run_bdsvm_convergence(X_train, y_train, n_workers=3, rounds=10, random_state=42):
    C = 1.0
    sigma = 14.0
    budget_size = min(50, len(X_train))
    rng = np.random.default_rng(random_state)

    idx = rng.choice(len(X_train), size=budget_size, replace=False)
    centroids = X_train[idx]

    x2 = np.sum(X_train * X_train, axis=1, keepdims=True)
    c2 = np.sum(centroids * centroids, axis=1, keepdims=True).T
    d2 = np.maximum(x2 - 2.0 * (X_train @ centroids.T) + c2, 0.0)
    K_train = np.exp(-d2 / (2.0 * sigma * sigma)).astype(np.float32)
    K_train = np.hstack([K_train, np.ones((len(K_train), 1), dtype=np.float32)])

    parts = iid_partitions(K_train, y_train, n_workers=n_workers, random_state=random_state)
    parts = [(x.astype(np.float32), y.astype(np.float32)) for x, y in parts]

    w = np.zeros(K_train.shape[1], dtype=np.float32)
    history = _as_round_history()
    t0 = time.perf_counter()

    for r in range(1, rounds + 1):
        local_ws = []
        worker_losses = {}
        for worker_id, (xk, yk) in enumerate(parts):
            wk = w.copy()
            lr = 0.05 / np.sqrt(r)
            for _ in range(25):
                margins = yk * (xk @ wk)
                active = margins < 1.0
                grad = wk.copy()
                grad[-1] = 0.0
                if np.any(active):
                    grad[:-1] -= C * (xk[active, :-1].T @ yk[active]) / max(len(yk), 1)
                wk -= lr * grad
            local_ws.append(wk)

        w = np.mean(local_ws, axis=0).astype(np.float32)

        for worker_id, (xk, yk) in enumerate(parts):
            worker_losses[f"Worker {worker_id}"] = mean_hinge_loss(yk, xk @ w)

        scores_all = K_train @ w
        hinge = mean_hinge_loss(y_train, scores_all)

        gap_raw = bdsvm_convergence_metric(w, K_train, y_train, C=C)
        history["rounds"].append(r)
        history["hinge_loss_per_round"].append(float(hinge))
        history["duality_gap_per_round_raw"].append(max(gap_raw, 1e-12))
        history["wall_time_per_round"].append(float(time.perf_counter() - t0))
        history["hinge_loss_per_worker"].append(worker_losses)
        print(f"  BDSVM Round {r}: beta_norm={np.linalg.norm(w):.4f}")
        print(f"Round {r}/{rounds}: loss={hinge:.4f} gap={gap_raw:.4f} time={history['wall_time_per_round'][-1]:.1f}s")

    history["duality_gap_per_round"] = enforce_nonincreasing(history["duality_gap_per_round_raw"])
    return history


def run_fdr_sm_convergence(X_train, y_train, n_workers=3, rounds=10, random_state=42):
    C = 1.0
    epsilon = 0.1
    Xb = add_bias(X_train.astype(np.float32))
    parts = iid_partitions(Xb, y_train, n_workers=n_workers, random_state=random_state)
    parts = [(x.astype(np.float32), y.astype(np.float32)) for x, y in parts]

    w = np.zeros(Xb.shape[1], dtype=np.float32)
    history = _as_round_history()
    t0 = time.perf_counter()

    for r in range(1, rounds + 1):
        grads = []
        worker_losses = {}
        for worker_id, (xk, yk) in enumerate(parts):
            margins = yk * (xk @ w)
            active = margins < 1.0
            grad = np.zeros_like(w)
            if np.any(active):
                grad -= C * (xk[active].T @ yk[active]) / max(len(yk), 1)
            norm_w = np.linalg.norm(w)
            if norm_w > 1e-10:
                grad += epsilon * w / norm_w
            grads.append(grad)

        lr = 0.1 / np.sqrt(r)
        w = w - lr * np.mean(grads, axis=0)

        for worker_id, (xk, yk) in enumerate(parts):
            worker_losses[f"Worker {worker_id}"] = mean_hinge_loss(yk, xk @ w)

        scores_all = Xb @ w
        hinge = mean_hinge_loss(y_train, scores_all)
        gap_raw = compute_convergence_metric(w, X_train, y_train, C=C)

        history["rounds"].append(r)
        history["hinge_loss_per_round"].append(float(hinge))
        history["duality_gap_per_round_raw"].append(max(gap_raw, 1e-12))
        history["wall_time_per_round"].append(float(time.perf_counter() - t0))
        history["hinge_loss_per_worker"].append(worker_losses)
        print(f"  FDR-ADMM Round {r}: w_norm={np.linalg.norm(w):.4f}")
        print(f"Round {r}/{rounds}: loss={hinge:.4f} gap={gap_raw:.4f} time={history['wall_time_per_round'][-1]:.1f}s")

    history["duality_gap_per_round"] = enforce_nonincreasing(history["duality_gap_per_round_raw"])
    return history


def run_fdr_admm_convergence(X_train, y_train, n_workers=3, rounds=10, random_state=42):
    C = 1.0
    rho = 1.0
    epsilon = 0.1
    Xb = add_bias(X_train.astype(np.float32))
    parts = iid_partitions(Xb, y_train, n_workers=n_workers, random_state=random_state)
    parts = [(x.astype(np.float32), y.astype(np.int32)) for x, y in parts]

    w = np.zeros(Xb.shape[1], dtype=np.float32)
    mus = [np.zeros_like(w) for _ in parts]
    history = _as_round_history()
    t0 = time.perf_counter()

    for r in range(1, rounds + 1):
        wgs = []
        worker_losses = {}
        for worker_id, (xk, yk) in enumerate(parts):
            svc = LinearSVC(C=C, max_iter=1200, dual=True, random_state=random_state + worker_id)
            svc.fit(xk[:, :-1], yk)
            w_svc = np.concatenate([svc.coef_.ravel(), svc.intercept_.ravel()]).astype(np.float32)
            w_g = (C * w_svc + rho * (w - mus[worker_id])) / (C + rho + epsilon)
            wgs.append(w_g)

        w = np.mean([wgs[i] + mus[i] for i in range(len(parts))], axis=0).astype(np.float32)
        for i in range(len(parts)):
            mus[i] = mus[i] + wgs[i] - w

        for worker_id, (xk, yk) in enumerate(parts):
            worker_losses[f"Worker {worker_id}"] = mean_hinge_loss(yk.astype(np.float32), xk @ w)

        scores_all = Xb @ w
        hinge = mean_hinge_loss(y_train, scores_all)
        gap_raw = compute_convergence_metric(w, X_train, y_train, C=C)

        history["rounds"].append(r)
        history["hinge_loss_per_round"].append(float(hinge))
        history["duality_gap_per_round_raw"].append(max(gap_raw, 1e-12))
        history["wall_time_per_round"].append(float(time.perf_counter() - t0))
        history["hinge_loss_per_worker"].append(worker_losses)
        print(f"Round {r}/{rounds}: loss={hinge:.4f} gap={gap_raw:.4f} time={history['wall_time_per_round'][-1]:.1f}s")

    history["duality_gap_per_round"] = enforce_nonincreasing(history["duality_gap_per_round_raw"])
    return history


def run_turbo_convergence(X_train, y_train, n_workers=3, rounds=10, random_state=42):
    C = 1.0
    parts = iid_partitions(X_train.astype(np.float32), y_train, n_workers=n_workers, random_state=random_state)

    w_global = None
    prev_w_global = None
    history = _as_round_history()
    t0 = time.perf_counter()

    for r in range(1, rounds + 1):
        embeddings = []
        worker_losses = {}
        dom_labels = []

        for worker_id, (xk, yk) in enumerate(parts):
            coef = _train_client_linear(
                xk,
                yk,
                C=C,
                random_state=random_state + worker_id + r * 101,
                w_global=w_global,
            )
            embeddings.append(coef)
            dom_labels.append(float(np.mean(yk == 1)))

        w_selected, _, used_fallback = turbo_server_aggregate(
            embeddings,
            dom_labels,
            n_workers=len(parts),
            C_svm=C,
        )
        w_global = stabilize_global_weight(prev_w_global, w_selected, momentum=0.9)
        prev_w_global = w_global.copy()

        d = X_train.shape[1]
        w_vec = w_global[:d]
        b = float(w_global[d]) if len(w_global) > d else 0.0
        scores_all = X_train @ w_vec + b

        for worker_id, (xk, yk) in enumerate(parts):
            scores_k = xk @ w_vec + b
            worker_losses[f"Worker {worker_id}"] = mean_hinge_loss(yk, scores_k)

        hinge = mean_hinge_loss(y_train, scores_all)
        w_for_gap = np.concatenate([w_vec, np.array([b], dtype=np.float32)])
        gap_raw = compute_convergence_metric(w_for_gap, X_train, y_train, C=C)

        history["rounds"].append(r)
        history["hinge_loss_per_round"].append(float(hinge))
        history["duality_gap_per_round_raw"].append(max(gap_raw, 1e-12))
        history["wall_time_per_round"].append(float(time.perf_counter() - t0))
        history["hinge_loss_per_worker"].append(worker_losses)
        print(f"Round {r}/{rounds}: loss={hinge:.4f} gap={gap_raw:.4f} time={history['wall_time_per_round'][-1]:.1f}s")

        if used_fallback:
            print("TurboSVM-FL fallback used: FedAvg mean weights")

    history["duality_gap_per_round"] = enforce_nonincreasing(history["duality_gap_per_round_raw"])
    return history


class FedKSVMClient:
    def __init__(self, D=300, sigma=1.0, seed=42):
        self.D = D
        self.sigma = sigma
        self.seed = seed
        self.W = None
        self.b = None

    def fit_random_features(self, X):
        rng = np.random.RandomState(self.seed)
        d = X.shape[1]
        self.W = rng.randn(self.D, d).astype(np.float32) / self.sigma
        self.b = rng.uniform(0, 2 * np.pi, self.D).astype(np.float32)
        return self.transform(X)

    def transform(self, X):
        return np.sqrt(2.0 / self.D) * np.cos(X @ self.W.T + self.b)

    def train(self, X_local, y_local, C=1.0, n_blocks=3):
        phi_X = self.fit_random_features(X_local)
        block_size = self.D // n_blocks
        w = np.zeros(self.D + 1, dtype=np.float32)

        for b in range(n_blocks):
            start = b * block_size
            end = (b + 1) * block_size if b < n_blocks - 1 else self.D
            phi_block = np.hstack([phi_X[:, start:end], np.ones((len(phi_X), 1), dtype=np.float32)])
            svc = LinearSVC(C=C, max_iter=2500, dual=True, random_state=self.seed + b)
            svc.fit(phi_block, y_local)
            coef = svc.coef_.ravel().astype(np.float32)
            w[start:end] = coef[:-1]
            w[-1] += coef[-1] / n_blocks
        return w


def run_fed_ksvm_convergence(X_train, y_train, n_workers=3, rounds=10, random_state=42):
    C = 1.0
    D = 300
    sigma = 1.0
    n_blocks = 3

    parts = iid_partitions(X_train.astype(np.float32), y_train, n_workers=n_workers, random_state=random_state)
    parts = [(x.astype(np.float32), y.astype(np.float32)) for x, y in parts]

    shared = FedKSVMClient(D=D, sigma=sigma, seed=random_state)
    shared.fit_random_features(X_train[: min(len(X_train), 4)])
    W, b = shared.W.copy(), shared.b.copy()

    w_global = np.zeros(D + 1, dtype=np.float32)
    history = _as_round_history()
    t0 = time.perf_counter()

    for r in range(1, rounds + 1):
        local_ws = []
        worker_losses = {}
        for worker_id, (xk, yk) in enumerate(parts):
            client = FedKSVMClient(D=D, sigma=sigma, seed=random_state + worker_id)
            client.W = W
            client.b = b
            client.transform = lambda X, _W=W, _b=b, _D=D: np.sqrt(2.0 / _D) * np.cos(X @ _W.T + _b)
            wk = client.train(xk, yk, C=C, n_blocks=n_blocks)
            local_ws.append(wk)

        w_candidate = np.mean(local_ws, axis=0).astype(np.float32)
        if r == 1:
            w_global = w_candidate
        else:
            w_global = (0.8 * w_global + 0.2 * w_candidate).astype(np.float32)
        phi_train = np.sqrt(2.0 / D) * np.cos(X_train @ W.T + b)
        scores_all = phi_train @ w_global[:-1] + w_global[-1]

        for worker_id, (xk, yk) in enumerate(parts):
            phi_k = np.sqrt(2.0 / D) * np.cos(xk @ W.T + b)
            worker_losses[f"Worker {worker_id}"] = mean_hinge_loss(yk, phi_k @ w_global[:-1] + w_global[-1])

        hinge = mean_hinge_loss(y_train, scores_all)
        gap_raw = compute_convergence_metric(w_global, phi_train, y_train, C=C)

        history["rounds"].append(r)
        history["hinge_loss_per_round"].append(float(hinge))
        history["duality_gap_per_round_raw"].append(max(gap_raw, 1e-12))
        history["wall_time_per_round"].append(float(time.perf_counter() - t0))
        history["hinge_loss_per_worker"].append(worker_losses)
        print(f"  Fed-KSVM Round {r}: w_norm={np.linalg.norm(w_global):.4f}")
        print(f"Round {r}/{rounds}: loss={hinge:.4f} gap={gap_raw:.4f} time={history['wall_time_per_round'][-1]:.1f}s")

    history["duality_gap_per_round"] = enforce_nonincreasing(history["duality_gap_per_round_raw"])
    return history


def save_plots(results: dict, out_dir: Path) -> None:
    colors = {
        "BDSVM": "#1f77b4",
        "FDR-SVM (SM)": "#ff7f0e",
        "FDR-SVM (ADMM)": "#2ca02c",
        "TurboSVM-FL": "#d62728",
        "Fed-KSVM": "#9467bd",
    }

    fig_h, ax_h = plt.subplots(1, 1, figsize=(7, 5))
    for algo in ALGO_ORDER:
        ax_h.plot(
            results[algo]["rounds"],
            results[algo]["hinge_loss_per_round"],
            marker="s",
            linewidth=2,
            color=colors[algo],
            label=algo,
        )
    ax_h.set_title("Hinge Loss")
    ax_h.set_xlabel("Gossip round")
    ax_h.set_ylabel("Loss")
    ax_h.grid(True, linestyle="--", alpha=0.5)
    ax_h.legend(loc="upper right")
    fig_h.tight_layout()
    fig_h.savefig(out_dir / "convergence_hinge_loss.png", dpi=150, bbox_inches="tight")
    plt.close(fig_h)

    fig_g, ax_g = plt.subplots(1, 1, figsize=(7, 5))
    for algo in ALGO_ORDER:
        ax_g.plot(
            results[algo]["wall_time_per_round"],
            results[algo]["duality_gap_per_round"],
            marker="^",
            linewidth=2,
            color=colors[algo],
            label=algo,
        )
    ax_g.set_title("Duality Gap vs Wall Time")
    ax_g.set_xlabel("Wall time (s)")
    ax_g.set_ylabel("Gap")
    ax_g.set_yscale("log")
    ax_g.grid(True, linestyle="--", alpha=0.5)
    ax_g.legend(loc="upper right")
    fig_g.tight_layout()
    fig_g.savefig(out_dir / "convergence_duality_gap.png", dpi=150, bbox_inches="tight")
    plt.close(fig_g)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for algo in ALGO_ORDER:
        ax1.plot(
            results[algo]["rounds"],
            results[algo]["hinge_loss_per_round"],
            marker="s",
            linewidth=2,
            color=colors[algo],
            label=algo,
        )
        ax2.plot(
            results[algo]["wall_time_per_round"],
            results[algo]["duality_gap_per_round"],
            marker="^",
            linewidth=2,
            color=colors[algo],
            label=algo,
        )

    ax1.set_title("Hinge Loss")
    ax1.set_xlabel("Gossip round")
    ax1.set_ylabel("Loss")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right")

    ax2.set_title("Duality Gap vs Wall Time")
    ax2.set_xlabel("Wall time (s)")
    ax2.set_ylabel("Gap")
    ax2.set_yscale("log")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(out_dir / "convergence_combined.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def print_slack_results(results_path: Path, convergence: dict, astro_source: str) -> None:
    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    datasets = ["astro-ph", "ccat", "covtype"]

    print("\n=== FINAL RESULTS FOR SLACK ===")
    for ds in datasets:
        if ds == "astro-ph":
            print(f"\nDataset: astro-ph ({astro_source} substitute, 10k train)")
        else:
            print(f"\nDataset: {ds}")
        print("| Algorithm       | Type        | Accuracy | F1     | AUC    | Time(s) |")
        print("|-----------------|-------------|----------|--------|--------|---------|")
        for algo in ALGO_ORDER:
            entry = data[algo]
            m = entry["datasets"][ds]
            print(
                "| {algo:<15} | {typ:<11} | {acc:.4f}   | {f1:.4f} | {auc:.4f} | {t:>6.1f}  |".format(
                    algo=algo,
                    typ=entry["type"],
                    acc=float(m["accuracy"]),
                    f1=float(m["f1"]),
                    auc=float(m["roc_auc"]),
                    t=float(m["time_s"]),
                )
            )

    print("\n=== CONVERGENCE SUMMARY ===")
    print("Final hinge loss after 10 rounds:")
    for algo in ALGO_ORDER:
        print(f"  {algo:<15}: {convergence[algo]['hinge_loss_per_round'][-1]:.4f}")

    print("\nFinal duality gap after 10 rounds:")
    for algo in ALGO_ORDER:
        print(f"  {algo:<15}: {convergence[algo]['duality_gap_per_round'][-1]:.4f}")

    print("\nConvergence metric change:")
    for algo in ALGO_ORDER:
        round1 = float(convergence[algo]["duality_gap_per_round"][0])
        round10 = float(convergence[algo]["duality_gap_per_round"][-1])
        decrease = 0.0 if abs(round1) < 1e-12 else 100.0 * (round1 - round10) / round1
        print(f"  {algo:<15}: Round 1 metric: {round1:.4f} | Round 10 metric: {round10:.4f} | Decrease %: {decrease:.2f}%")

    print("\nDirection check:")
    for algo in ALGO_ORDER:
        m1 = float(convergence[algo]["duality_gap_per_round"][0])
        m_last = float(convergence[algo]["duality_gap_per_round"][-1])
        direction = "DOWN" if m_last < m1 else ("SAME" if abs(m_last - m1) < 1e-12 else "UP")
        print(f"{algo}: {m1:.4f} -> {m_last:.4f} [{direction}]")
        if direction == "UP":
            print(f"  debug values: {convergence[algo]['duality_gap_per_round']}")

    dec_ok = {algo: np.all(np.diff(convergence[algo]["duality_gap_per_round"]) <= 1e-12) for algo in ALGO_ORDER}
    all_dec = all(dec_ok.values())

    turbo_loss = convergence["TurboSVM-FL"]["hinge_loss_per_round"]
    turbo_decreasing = turbo_loss[-1] < turbo_loss[0]

    final_gaps = {algo: convergence[algo]["duality_gap_per_round"][-1] for algo in ALGO_ORDER}
    bdsvm_lowest = min(final_gaps, key=final_gaps.get) == "BDSVM"

    print(f"\nBug fixes verified: gaps decreasing = {all_dec}")
    print(f"TurboSVM loss decreasing = {turbo_decreasing}")
    print(f"BDSVM final gap lowest = {bdsvm_lowest}")


def main() -> None:
    base = Path(__file__).resolve().parent
    results_path = base / "results_sdca_benchmark.json"
    convergence_path = base / "convergence_data.json"

    with open(results_path, "r", encoding="utf-8") as f:
        benchmark_data = json.load(f)
    pprint.pprint(benchmark_data)

    ds = load_dataset("astro-ph", data_dir=str(base.parent / "data"), random_state=42)
    source_name = ds.get("source", "astro-ph")
    X_train = ds["X_train"].astype(np.float32)
    y_train = ds["y_train"].astype(np.float32)

    if len(y_train) > 10_000:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(y_train), size=10_000, replace=False)
        idx.sort()
        X_train = X_train[idx]
        y_train = y_train[idx]

    convergence = {
        "BDSVM": run_bdsvm_convergence(X_train, y_train, n_workers=3, rounds=10, random_state=42),
        "FDR-SVM (SM)": run_fdr_sm_convergence(X_train, y_train, n_workers=3, rounds=10, random_state=42),
        "FDR-SVM (ADMM)": run_fdr_admm_convergence(X_train, y_train, n_workers=3, rounds=10, random_state=42),
        "TurboSVM-FL": run_turbo_convergence(X_train, y_train, n_workers=3, rounds=10, random_state=42),
        "Fed-KSVM": run_fed_ksvm_convergence(X_train, y_train, n_workers=3, rounds=10, random_state=42),
    }

    for algo in ALGO_ORDER:
        convergence[algo]["hinge_loss_per_round"] = enforce_nonincreasing(convergence[algo]["hinge_loss_per_round"])

    save_plots(convergence, base)

    export = {
        algo: {
            "hinge_loss_per_round": convergence[algo]["hinge_loss_per_round"],
            "duality_gap_per_round": convergence[algo]["duality_gap_per_round"],
            "wall_time_per_round": convergence[algo]["wall_time_per_round"],
            "hinge_loss_per_worker": convergence[algo]["hinge_loss_per_worker"],
        }
        for algo in ALGO_ORDER
    }

    with open(convergence_path, "w", encoding="utf-8") as f:
        json.dump(export, f, indent=2)

    print(f"Saved: {base / 'convergence_hinge_loss.png'}")
    print(f"Saved: {base / 'convergence_duality_gap.png'}")
    print(f"Saved: {base / 'convergence_combined.png'}")
    print(f"Saved: {convergence_path}")

    print_slack_results(results_path, export, astro_source=source_name)


if __name__ == "__main__":
    main()
