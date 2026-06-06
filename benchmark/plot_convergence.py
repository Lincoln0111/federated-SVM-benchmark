from __future__ import annotations

import json
import pprint
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import LinearSVC, SVC

from common import add_bias
from sdca_data import load_dataset, iid_partitions

plt.style.use("default")


ALGORITHMS = ["BDSVM", "FDR-SVM (SM)", "FDR-SVM (ADMM)", "TurboSVM-FL", "Fed-KSVM"]


def print_full_json_results(results_path: Path) -> None:
    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    pprint.pprint(data)


def print_final_metrics_table(results_path: Path) -> None:
    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    dataset_names: list[str] = []
    for algo_data in data.values():
        if isinstance(algo_data, dict) and "datasets" in algo_data:
            dataset_names = list(algo_data["datasets"].keys())
            break

    print("\n=== FINAL BENCHMARK RESULTS ===")
    for dataset in dataset_names:
        print(f"Dataset: {dataset}")
        print("| Algorithm | Type | Accuracy | F1 | ROC-AUC | Time(s) |")
        for algo_name, algo_data in data.items():
            if not isinstance(algo_data, dict):
                continue
            datasets = algo_data.get("datasets", {})
            if dataset not in datasets:
                continue
            metrics = datasets[dataset]
            print(
                "| {algo} | {type} | {acc:.4f} | {f1:.4f} | {auc:.4f} | {time:.1f} |".format(
                    algo=algo_name,
                    type=algo_data.get("type", ""),
                    acc=float(metrics.get("accuracy", float("nan"))),
                    f1=float(metrics.get("f1", float("nan"))),
                    auc=float(metrics.get("roc_auc", float("nan"))),
                    time=float(metrics.get("time_s", 0.0)),
                )
            )


def train_test_subsample(X: np.ndarray, y: np.ndarray, max_train_samples: int, random_state: int) -> tuple[np.ndarray, np.ndarray]:
    if len(y) <= max_train_samples:
        return X.astype(np.float32), y.astype(np.float32)
    rng = np.random.default_rng(random_state)
    idx = rng.choice(len(y), size=max_train_samples, replace=False)
    idx.sort()
    return X[idx].astype(np.float32), y[idx].astype(np.float32)


def rbf_kernel(X: np.ndarray, centroids: np.ndarray, sigma: float) -> np.ndarray:
    x2 = np.sum(X * X, axis=1, keepdims=True)
    c2 = np.sum(centroids * centroids, axis=1, keepdims=True).T
    d2 = np.maximum(x2 - 2.0 * (X @ centroids.T) + c2, 0.0)
    return np.exp(-d2 / (2.0 * sigma * sigma)).astype(np.float32)


def mean_hinge_loss(y: np.ndarray, scores: np.ndarray) -> float:
    margins = y.reshape(-1) * scores.reshape(-1)
    return float(np.mean(np.maximum(0.0, 1.0 - margins)))


def primal_objective(y: np.ndarray, scores: np.ndarray, w: np.ndarray, c_value: float) -> float:
    hinge = mean_hinge_loss(y, scores)
    return 0.5 * float(np.dot(w.reshape(-1), w.reshape(-1))) + c_value * hinge


def aggregate_weighted(values: list[float], sizes: list[int]) -> float:
    sizes_arr = np.asarray(sizes, dtype=np.float64)
    values_arr = np.asarray(values, dtype=np.float64)
    denom = float(np.sum(sizes_arr))
    if denom <= 0:
        return float(np.mean(values_arr))
    return float(np.sum(values_arr * sizes_arr) / denom)


def plot_metric_rounds(data: dict, out_path: Path, title: str, ylabel: str, marker: str) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    for idx, algo in enumerate(ALGORITHMS):
        ax.plot(
            data[algo]["rounds"],
            data[algo]["hinge_loss_per_round"] if marker == "s" else data[algo]["duality_gap_per_round"],
            marker=marker,
            linewidth=2,
            markersize=5,
            color=colors[idx % len(colors)],
            label=algo,
        )
    ax.set_title(title)
    ax.set_xlabel("Gossip round")
    ax.set_ylabel(ylabel)
    if marker == "^":
        ax.set_yscale("log")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_metric_wall_time(data: dict, out_path: Path) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    for idx, algo in enumerate(ALGORITHMS):
        ax.plot(
            data[algo]["wall_time_per_round"],
            data[algo]["duality_gap_per_round"],
            marker="^",
            linewidth=2,
            markersize=5,
            color=colors[idx % len(colors)],
            label=algo,
        )
    ax.set_title("Duality Gap vs Wall Time")
    ax.set_xlabel("Wall time (s)")
    ax.set_ylabel("Gap")
    ax.set_yscale("log")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_combined(data: dict, out_path: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    for idx, algo in enumerate(ALGORITHMS):
        color = colors[idx % len(colors)]
        ax1.plot(
            data[algo]["rounds"],
            data[algo]["hinge_loss_per_round"],
            marker="s",
            linewidth=2,
            markersize=5,
            color=color,
            label=algo,
        )
        ax2.plot(
            data[algo]["wall_time_per_round"],
            data[algo]["duality_gap_per_round"],
            marker="^",
            linewidth=2,
            markersize=5,
            color=color,
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
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_bdsvm(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, n_workers: int, rounds: int, random_state: int) -> dict:
    c_value = 1.0
    sigma = 14.0
    budget_size = min(50, len(X_train))
    rng = np.random.default_rng(random_state)
    centroid_idx = rng.choice(len(X_train), size=budget_size, replace=False)
    centroids = X_train[centroid_idx]

    phi_train = rbf_kernel(X_train, centroids, sigma)
    phi_test = rbf_kernel(X_test, centroids, sigma)
    phi_train = np.hstack([phi_train, np.ones((len(phi_train), 1), dtype=np.float32)])
    phi_test = np.hstack([phi_test, np.ones((len(phi_test), 1), dtype=np.float32)])

    partitions = iid_partitions(phi_train, y_train, n_workers=n_workers, random_state=random_state)
    local_parts = [(x.astype(np.float32), y.astype(np.float32)) for x, y in partitions]
    w_global = np.zeros(phi_train.shape[1], dtype=np.float32)

    history = {
        "rounds": [],
        "hinge_loss_per_round": [],
        "duality_gap_per_round": [],
        "wall_time_per_round": [],
        "hinge_loss_per_worker": [],
    }

    start = time.perf_counter()
    for round_idx in range(1, rounds + 1):
        local_ws = []
        worker_losses = {}
        for worker_idx, (xk, yk) in enumerate(local_parts):
            w_local = w_global.copy()
            lr = 0.05 / np.sqrt(round_idx)
            for _ in range(25):
                margins = yk * (xk @ w_local)
                active = margins < 1.0
                grad = w_local.copy()
                grad[-1] = 0.0
                if np.any(active):
                    grad[:-1] -= c_value * (xk[active, :-1].T @ yk[active]) / max(len(yk), 1)
                w_local -= lr * grad
            local_ws.append(w_local)

        w_global = np.mean(local_ws, axis=0).astype(np.float32)
        for worker_idx, (xk, yk) in enumerate(local_parts):
            worker_losses[f"Worker {worker_idx}"] = mean_hinge_loss(yk, xk @ w_global)
        train_scores = phi_train @ w_global
        hinge = mean_hinge_loss(y_train, train_scores)
        gap = primal_objective(y_train, train_scores, w_global, c_value)
        elapsed = time.perf_counter() - start

        history["rounds"].append(round_idx)
        history["hinge_loss_per_round"].append(float(hinge))
        history["duality_gap_per_round"].append(float(max(gap, 1e-12)))
        history["wall_time_per_round"].append(float(elapsed))
        history["hinge_loss_per_worker"].append(worker_losses)
        print(f"[BDSVM] Round {round_idx}/{rounds}: loss={hinge:.4f} gap={gap:.4f} time={elapsed:.1f}s")

    scores = phi_test @ w_global
    return {"history": history, "accuracy": float(np.mean(np.where(scores >= 0, 1, -1) == y_test))}


def run_fdr_svm_sm(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, n_workers: int, rounds: int, random_state: int) -> dict:
    c_value = 1.0
    epsilon = 0.1
    Xb_train = add_bias(X_train)
    Xb_test = add_bias(X_test)
    partitions = iid_partitions(Xb_train, y_train, n_workers=n_workers, random_state=random_state)
    local_parts = [(x.astype(np.float32), y.astype(np.float32)) for x, y in partitions]
    w = np.zeros(Xb_train.shape[1], dtype=np.float32)

    history = {"rounds": [], "hinge_loss_per_round": [], "duality_gap_per_round": [], "wall_time_per_round": [], "hinge_loss_per_worker": []}
    start = time.perf_counter()
    for round_idx in range(1, rounds + 1):
        grads = []
        worker_losses = {}
        for worker_idx, (xk, yk) in enumerate(local_parts):
            margins = yk * (xk @ w)
            active = margins < 1.0
            grad = np.zeros_like(w)
            if np.any(active):
                grad -= c_value * (xk[active].T @ yk[active]) / max(len(yk), 1)
            norm_w = np.linalg.norm(w)
            if norm_w > 1e-10:
                grad += epsilon * w / norm_w
            grads.append(grad)
        lr = 0.1 / np.sqrt(round_idx)
        w = w - lr * np.mean(grads, axis=0)
        for worker_idx, (xk, yk) in enumerate(local_parts):
            worker_losses[f"Worker {worker_idx}"] = mean_hinge_loss(yk, xk @ w)
        scores = Xb_train @ w
        hinge = mean_hinge_loss(y_train, scores)
        gap = primal_objective(y_train, scores, w, c_value)
        elapsed = time.perf_counter() - start
        history["rounds"].append(round_idx)
        history["hinge_loss_per_round"].append(float(hinge))
        history["duality_gap_per_round"].append(float(max(gap, 1e-12)))
        history["wall_time_per_round"].append(float(elapsed))
        history["hinge_loss_per_worker"].append(worker_losses)
        print(f"[FDR-SVM (SM)] Round {round_idx}/{rounds}: loss={hinge:.4f} gap={gap:.4f} time={elapsed:.1f}s")

    scores_test = Xb_test @ w
    return {"history": history, "accuracy": float(np.mean(np.where(scores_test >= 0, 1, -1) == y_test))}


def run_fdr_svm_admm(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, n_workers: int, rounds: int, random_state: int) -> dict:
    c_value = 1.0
    rho = 1.0
    Xb_train = add_bias(X_train)
    Xb_test = add_bias(X_test)
    partitions = iid_partitions(Xb_train, y_train, n_workers=n_workers, random_state=random_state)
    local_parts = [(x.astype(np.float32), y.astype(np.int32)) for x, y in partitions]
    w = np.zeros(Xb_train.shape[1], dtype=np.float32)
    mus = [np.zeros_like(w) for _ in range(len(local_parts))]

    history = {"rounds": [], "hinge_loss_per_round": [], "duality_gap_per_round": [], "wall_time_per_round": [], "hinge_loss_per_worker": []}
    start = time.perf_counter()
    for round_idx in range(1, rounds + 1):
        wgs = []
        worker_losses = {}
        for worker_idx, (xk, yk) in enumerate(local_parts):
            svc = LinearSVC(C=c_value, max_iter=1200, dual=True, random_state=random_state + worker_idx)
            svc.fit(xk[:, :-1], yk)
            w_svc = np.concatenate([svc.coef_.ravel(), svc.intercept_.ravel()]).astype(np.float32)
            w_g = (c_value * w_svc + rho * (w - mus[worker_idx])) / (c_value + rho + 0.1)
            wgs.append(w_g)
        w = np.mean([wgs[i] + mus[i] for i in range(len(wgs))], axis=0).astype(np.float32)
        for i in range(len(wgs)):
            mus[i] = mus[i] + wgs[i] - w
        for worker_idx, (xk, yk) in enumerate(local_parts):
            worker_losses[f"Worker {worker_idx}"] = mean_hinge_loss(yk, xk @ w)
        scores = Xb_train @ w
        hinge = mean_hinge_loss(y_train, scores)
        gap = primal_objective(y_train, scores, w, c_value)
        elapsed = time.perf_counter() - start
        history["rounds"].append(round_idx)
        history["hinge_loss_per_round"].append(float(hinge))
        history["duality_gap_per_round"].append(float(max(gap, 1e-12)))
        history["wall_time_per_round"].append(float(elapsed))
        history["hinge_loss_per_worker"].append(worker_losses)
        print(f"[FDR-SVM (ADMM)] Round {round_idx}/{rounds}: loss={hinge:.4f} gap={gap:.4f} time={elapsed:.1f}s")

    scores_test = Xb_test @ w
    return {"history": history, "accuracy": float(np.mean(np.where(scores_test >= 0, 1, -1) == y_test))}


class FedKSVMClient:
    def __init__(self, D: int = 300, sigma: float = 1.0, seed: int = 42):
        self.D = D
        self.sigma = sigma
        self.seed = seed
        self.W = None
        self.b = None

    def fit_random_features(self, X: np.ndarray) -> np.ndarray:
        rng = np.random.RandomState(self.seed)
        d = X.shape[1]
        self.W = rng.randn(self.D, d).astype(np.float32) / self.sigma
        self.b = rng.uniform(0, 2 * np.pi, self.D).astype(np.float32)
        return self.transform(X)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return np.sqrt(2.0 / self.D) * np.cos(X @ self.W.T + self.b)

    def train(self, X_local: np.ndarray, y_local: np.ndarray, C: float = 1.0, n_blocks: int = 3) -> np.ndarray:
        phi_X = self.fit_random_features(X_local)
        block_size = self.D // n_blocks
        w = np.zeros(self.D + 1, dtype=np.float32)
        for b in range(n_blocks):
            start = b * block_size
            end = (b + 1) * block_size if b < n_blocks - 1 else self.D
            phi_block = np.hstack([phi_X[:, start:end], np.ones((len(phi_X), 1), dtype=np.float32)])
            svc = LinearSVC(C=C, max_iter=1200, dual=True, random_state=self.seed + b)
            svc.fit(phi_block, y_local)
            coef = svc.coef_.ravel().astype(np.float32)
            w[start:end] = coef[:-1]
            w[-1] += coef[-1] / n_blocks
        return w


class TurboSVMClient:
    def __init__(self, C: float = 1.0, random_state: int = 42):
        self.C = C
        self.random_state = random_state

    def train(self, X_local: np.ndarray, y_local: np.ndarray, w_init: np.ndarray | None = None, C: float = 1.0) -> np.ndarray:
        svc = LinearSVC(C=C, max_iter=4000, dual=True, random_state=self.random_state)
        svc.fit(X_local, y_local)
        emb = svc.coef_.ravel().astype(np.float32)
        if w_init is not None:
            emb = 0.7 * emb + 0.3 * w_init.astype(np.float32)
        return emb


def run_fed_ksvm(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, n_workers: int, rounds: int, random_state: int) -> dict:
    c_value = 1.0
    D = 300
    sigma = 1.0
    n_blocks = 3
    partitions = iid_partitions(X_train, y_train, n_workers=n_workers, random_state=random_state)
    local_parts = [(x.astype(np.float32), y.astype(np.float32)) for x, y in partitions]

    shared_client = FedKSVMClient(D=D, sigma=sigma, seed=random_state)
    shared_client.fit_random_features(X_train[: min(len(X_train), 4)])
    W, b = shared_client.W.copy(), shared_client.b.copy()

    w_global = np.zeros(D + 1, dtype=np.float32)
    history = {"rounds": [], "hinge_loss_per_round": [], "duality_gap_per_round": [], "wall_time_per_round": [], "hinge_loss_per_worker": []}
    start = time.perf_counter()
    for round_idx in range(1, rounds + 1):
        local_ws = []
        worker_losses = {}
        for worker_idx, (xk, yk) in enumerate(local_parts):
            client = FedKSVMClient(D=D, sigma=sigma, seed=random_state)
            client.W = W
            client.b = b
            client.transform = lambda X, _W=W, _b=b, _D=D: np.sqrt(2.0 / _D) * np.cos(X @ _W.T + _b)
            wk = client.train(xk, yk, C=c_value, n_blocks=n_blocks)
            local_ws.append(wk)
            phi_local = np.sqrt(2.0 / D) * np.cos(xk @ W.T + b)
            worker_losses[f"Worker {worker_idx}"] = mean_hinge_loss(yk, phi_local @ w_global[:-1] + w_global[-1])
        w_global = np.mean(local_ws, axis=0).astype(np.float32)
        phi_train = np.sqrt(2.0 / D) * np.cos(X_train @ W.T + b)
        scores = phi_train @ w_global[:-1] + w_global[-1]
        hinge = mean_hinge_loss(y_train, scores)
        gap = primal_objective(y_train, scores, w_global, c_value)
        elapsed = time.perf_counter() - start
        history["rounds"].append(round_idx)
        history["hinge_loss_per_round"].append(float(hinge))
        history["duality_gap_per_round"].append(float(max(gap, 1e-12)))
        history["wall_time_per_round"].append(float(elapsed))
        history["hinge_loss_per_worker"].append(worker_losses)
        print(f"[Fed-KSVM] Round {round_idx}/{rounds}: loss={hinge:.4f} gap={gap:.4f} time={elapsed:.1f}s")

    phi_test = np.sqrt(2.0 / D) * np.cos(X_test @ W.T + b)
    scores_test = phi_test @ w_global[:-1] + w_global[-1]
    return {"history": history, "accuracy": float(np.mean(np.where(scores_test >= 0, 1, -1) == y_test))}


def main() -> None:
    base = Path(__file__).resolve().parent
    benchmark_json = base / "results_sdca_benchmark.json"
    convergence_json = base / "convergence_data.json"
    out_hinge = base / "convergence_hinge_loss.png"
    out_gap = base / "convergence_duality_gap.png"
    out_combined = base / "convergence_combined.png"

    print_full_json_results(benchmark_json)

    random_state = 42
    n_workers = 3
    rounds = 10

    dataset = load_dataset("astro-ph", data_dir=str(base.parent / "data"), random_state=random_state)
    X_train = dataset["X_train"].astype(np.float32)
    y_train = dataset["y_train"].astype(np.float32)
    X_test = dataset["X_test"].astype(np.float32)
    y_test = dataset["y_test"].astype(np.float32)

    X_train, y_train = train_test_subsample(X_train, y_train, 10_000, random_state)

    all_results = {}
    all_results["BDSVM"] = run_bdsvm(X_train, y_train, X_test, y_test, n_workers=n_workers, rounds=rounds, random_state=random_state)
    all_results["FDR-SVM (SM)"] = run_fdr_svm_sm(X_train, y_train, X_test, y_test, n_workers=n_workers, rounds=rounds, random_state=random_state)
    all_results["FDR-SVM (ADMM)"] = run_fdr_svm_admm(X_train, y_train, X_test, y_test, n_workers=n_workers, rounds=rounds, random_state=random_state)
    all_results["TurboSVM-FL"] = run_turbo_svm_wrapper(X_train, y_train, X_test, y_test, n_workers=n_workers, rounds=rounds, random_state=random_state)
    all_results["Fed-KSVM"] = run_fed_ksvm(X_train, y_train, X_test, y_test, n_workers=n_workers, rounds=rounds, random_state=random_state)

    convergence_data = {
        algo: {
            "hinge_loss_per_round": res["history"]["hinge_loss_per_round"],
            "duality_gap_per_round": res["history"]["duality_gap_per_round"],
            "wall_time_per_round": res["history"]["wall_time_per_round"],
            "hinge_loss_per_worker": res["history"]["hinge_loss_per_worker"],
        }
        for algo, res in all_results.items()
    }

    with open(convergence_json, "w", encoding="utf-8") as f:
        json.dump(convergence_data, f, indent=2)

    plot_metric_rounds({k: {"rounds": list(range(1, rounds + 1)), **v} for k, v in convergence_data.items()}, out_hinge, "Hinge Loss", "Loss", "s")
    plot_metric_wall_time(convergence_data, out_gap)
    plot_combined({k: {"rounds": list(range(1, rounds + 1)), **v} for k, v in convergence_data.items()}, out_combined)

    print(f"Saved: {out_hinge}")
    print(f"Saved: {out_gap}")
    print(f"Saved: {out_combined}")
    print(f"Saved: {convergence_json}")

    print_final_metrics_table(benchmark_json)


def run_turbo_svm_wrapper(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, n_workers: int, rounds: int, random_state: int) -> dict:
    c_value = 1.0
    Xtr = X_train.astype(np.float32)
    Xte = X_test.astype(np.float32)
    parts = iid_partitions(Xtr, y_train, n_workers=n_workers, random_state=random_state)
    clients = [TurboSVMClient(C=c_value, random_state=random_state + i) for i in range(len(parts))]
    w_global = np.zeros(Xtr.shape[1], dtype=np.float32)

    history = {"rounds": [], "hinge_loss_per_round": [], "duality_gap_per_round": [], "wall_time_per_round": [], "hinge_loss_per_worker": []}
    start = time.perf_counter()
    for round_idx in range(1, rounds + 1):
        embs = []
        dom_labels = []
        worker_losses = {}
        for worker_idx, ((xk, yk), client) in enumerate(zip(parts, clients)):
            emb = client.train(xk, yk, w_init=w_global, C=c_value)
            embs.append(emb)
            dom_labels.append(float((yk == 1).mean()))
        w_global, _ = turbo_server_aggregate(embs, dom_labels, n_workers=len(parts), C_svm=c_value)
        for worker_idx, (xk, yk) in enumerate(parts):
            worker_losses[f"Worker {worker_idx}"] = mean_hinge_loss(yk, xk @ w_global)
        scores = Xtr @ w_global
        hinge = mean_hinge_loss(y_train, scores)
        gap = primal_objective(y_train, scores, w_global, c_value)
        elapsed = time.perf_counter() - start
        history["rounds"].append(round_idx)
        history["hinge_loss_per_round"].append(float(hinge))
        history["duality_gap_per_round"].append(float(max(gap, 1e-12)))
        history["wall_time_per_round"].append(float(elapsed))
        history["hinge_loss_per_worker"].append(worker_losses)
        print(f"[TurboSVM-FL] Round {round_idx}/{rounds}: loss={hinge:.4f} gap={gap:.4f} time={elapsed:.1f}s")

    scores_test = Xte @ w_global
    return {"history": history, "accuracy": float(np.mean(np.where(scores_test >= 0, 1, -1) == y_test))}


def turbo_server_aggregate(embeddings: list[np.ndarray], labels_per_client: list[float], n_workers: int, C_svm: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    E = np.vstack(embeddings).astype(np.float32)
    svm_labels = np.array([1 if lp >= 0.5 else -1 for lp in labels_per_client], dtype=np.int32)
    if len(np.unique(svm_labels)) < 2:
        svm_labels = np.where(np.arange(len(svm_labels)) % 2 == 0, 1, -1)
    svc = SVC(kernel="linear", C=C_svm, max_iter=3000)
    svc.fit(E, svm_labels)
    sv_clients = svc.support_
    if len(sv_clients) == 0:
        sv_clients = np.arange(n_workers)
    w_global = E[sv_clients].mean(axis=0)
    h = svc.coef_.ravel().astype(np.float32)
    h_norm2 = float(np.dot(h, h))
    if h_norm2 > 1e-12:
        w_global = w_global - (float(np.dot(w_global, h)) / h_norm2) * h
    return w_global.astype(np.float32), sv_clients


class FedKSVMClient:
    def __init__(self, D: int = 300, sigma: float = 1.0, seed: int = 42):
        self.D = D
        self.sigma = sigma
        self.seed = seed
        self.W = None
        self.b = None

    def fit_random_features(self, X: np.ndarray) -> np.ndarray:
        rng = np.random.RandomState(self.seed)
        d = X.shape[1]
        self.W = rng.randn(self.D, d).astype(np.float32) / self.sigma
        self.b = rng.uniform(0, 2 * np.pi, self.D).astype(np.float32)
        return self.transform(X)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return np.sqrt(2.0 / self.D) * np.cos(X @ self.W.T + self.b)

    def train(self, X_local: np.ndarray, y_local: np.ndarray, C: float = 1.0, n_blocks: int = 3) -> np.ndarray:
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


def run_fed_ksvm_wrapper(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, n_workers: int, rounds: int, random_state: int) -> dict:
    c_value = 1.0
    D = 300
    sigma = 1.0
    n_blocks = 3
    parts = iid_partitions(X_train.astype(np.float32), y_train, n_workers=n_workers, random_state=random_state)
    local_parts = [(x.astype(np.float32), y.astype(np.float32)) for x, y in parts]

    shared_client = FedKSVMClient(D=D, sigma=sigma, seed=random_state)
    shared_client.fit_random_features(X_train[: min(len(X_train), 4)])
    W, b = shared_client.W.copy(), shared_client.b.copy()

    w_global = np.zeros(D + 1, dtype=np.float32)
    history = {"rounds": [], "hinge_loss_per_round": [], "duality_gap_per_round": [], "wall_time_per_round": [], "hinge_loss_per_worker": []}
    start = time.perf_counter()
    for round_idx in range(1, rounds + 1):
        local_ws = []
        worker_losses = {}
        for worker_idx, (xk, yk) in enumerate(local_parts):
            client = FedKSVMClient(D=D, sigma=sigma, seed=random_state)
            client.W = W
            client.b = b
            client.transform = lambda X, _W=W, _b=b, _D=D: np.sqrt(2.0 / _D) * np.cos(X @ _W.T + _b)
            wk = client.train(xk, yk, C=c_value, n_blocks=n_blocks)
            local_ws.append(wk)
        w_global = np.mean(local_ws, axis=0).astype(np.float32)
        for worker_idx, (xk, yk) in enumerate(local_parts):
            phi_local = np.sqrt(2.0 / D) * np.cos(xk @ W.T + b)
            worker_losses[f"Worker {worker_idx}"] = mean_hinge_loss(yk, phi_local @ w_global[:-1] + w_global[-1])
        phi_train = np.sqrt(2.0 / D) * np.cos(X_train @ W.T + b)
        scores = phi_train @ w_global[:-1] + w_global[-1]
        hinge = mean_hinge_loss(y_train, scores)
        gap = primal_objective(y_train, scores, w_global, c_value)
        elapsed = time.perf_counter() - start
        history["rounds"].append(round_idx)
        history["hinge_loss_per_round"].append(float(hinge))
        history["duality_gap_per_round"].append(float(max(gap, 1e-12)))
        history["wall_time_per_round"].append(float(elapsed))
        history["hinge_loss_per_worker"].append(worker_losses)
        print(f"[Fed-KSVM] Round {round_idx}/{rounds}: loss={hinge:.4f} gap={gap:.4f} time={elapsed:.1f}s")

    phi_test = np.sqrt(2.0 / D) * np.cos(X_test @ W.T + b)
    scores_test = phi_test @ w_global[:-1] + w_global[-1]
    return {"history": history, "accuracy": float(np.mean(np.where(scores_test >= 0, 1, -1) == y_test))}


if __name__ == "__main__":
    main()
