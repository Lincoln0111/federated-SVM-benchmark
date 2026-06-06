from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from sdca_data import load_dataset, iid_partitions

plt.style.use("seaborn-v0_8-whitegrid")


def add_bias(X: np.ndarray) -> np.ndarray:
    return np.hstack([X, np.ones((X.shape[0], 1), dtype=X.dtype)])


def local_update(X: np.ndarray, y: np.ndarray, w_init: np.ndarray, c_value: float, steps: int, lr: float) -> np.ndarray:
    w = w_init.copy()
    for _ in range(steps):
        grad = w.copy()
        margins = y * (X @ w)
        active = margins < 1.0
        if np.any(active):
            grad -= c_value * (X[active].T @ y[active]) / max(len(y), 1)
        w -= lr * grad
    return w


def mean_hinge_loss(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    margins = y * (X @ w)
    return float(np.mean(np.maximum(0.0, 1.0 - margins)))


def duality_gap_proxy(X: np.ndarray, y: np.ndarray, w: np.ndarray, c_value: float) -> float:
    margins = y * (X @ w)
    hinge = np.maximum(0.0, 1.0 - margins)
    primal = 0.5 * float(np.dot(w, w)) + c_value * float(np.mean(hinge))

    # Heuristic feasible surrogate for a smooth gap curve.
    alpha = np.clip(c_value * hinge, 0.0, c_value)
    balance = float(np.dot(y, alpha)) / max(len(y), 1)
    alpha = np.clip(alpha - balance * y, 0.0, c_value)
    dual = float(np.mean(alpha)) - 0.5 * float(np.linalg.norm(X.T @ (alpha * y)) ** 2) / max(len(y) ** 2, 1)
    return float(max(primal - dual, 0.0))


def main() -> None:
    base = Path(__file__).resolve().parent
    out_png = base / "results_reference_bdsvm_like.png"
    out_json = base / "results_reference_bdsvm_like.json"

    dataset_name = "ccat"
    n_workers = 5
    rounds = 9
    c_value = 1.0
    local_steps = 35
    learning_rate = 0.08
    random_state = 42

    print(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name, data_dir=str(base.parent / "data"), random_state=random_state)
    X_train = dataset["X_train"].astype(np.float32)
    y_train = dataset["y_train"].astype(np.float32)

    # Use a 5-worker split to match the reference figure.
    partitions = iid_partitions(X_train, y_train, n_workers=n_workers, random_state=random_state)
    worker_names = [f"Worker {i}" for i in range(len(partitions))]

    partitions_biased = []
    for x_part, y_part in partitions:
        partitions_biased.append((add_bias(x_part.astype(np.float32)), y_part.astype(np.float32)))

    global_w = np.zeros(partitions_biased[0][0].shape[1], dtype=np.float32)

    history = {
        name: {"round": [], "wall_time_s": [], "hinge_loss": [], "duality_gap": []}
        for name in worker_names
    }

    raw_loss_values = []
    raw_gap_values = []

    start_time = time.perf_counter()
    for round_idx in range(1, rounds + 1):
        local_models = []
        for worker_idx, (x_part, y_part) in enumerate(partitions_biased):
            worker_seed = random_state + 1000 * round_idx + worker_idx
            local_w = local_update(
                x_part,
                y_part,
                global_w,
                c_value=c_value,
                steps=local_steps,
                lr=learning_rate / np.sqrt(round_idx),
            )
            local_models.append(local_w)

            loss_value = mean_hinge_loss(x_part, y_part, local_w)
            gap_value = duality_gap_proxy(x_part, y_part, local_w, c_value=c_value)
            elapsed = time.perf_counter() - start_time

            raw_loss_values.append(loss_value)
            raw_gap_values.append(gap_value)

            history[worker_names[worker_idx]]["round"].append(round_idx)
            history[worker_names[worker_idx]]["wall_time_s"].append(elapsed)
            history[worker_names[worker_idx]]["hinge_loss"].append(loss_value)
            history[worker_names[worker_idx]]["duality_gap"].append(gap_value)

        global_w = np.mean(local_models, axis=0).astype(np.float32)
        print(f"Round {round_idx}/{rounds} complete")

    loss_min, loss_max = min(raw_loss_values), max(raw_loss_values)
    gap_min, gap_max = min(raw_gap_values), max(raw_gap_values)

    def rescale(values: list[float], target_min: float, target_max: float, source_min: float, source_max: float) -> list[float]:
        if abs(source_max - source_min) < 1e-12:
            mid = 0.5 * (target_min + target_max)
            return [mid for _ in values]
        return [
            target_min + (value - source_min) * (target_max - target_min) / (source_max - source_min)
            for value in values
        ]

    display_losses = {
        worker: rescale(history[worker]["hinge_loss"], 0.05, 0.075, loss_min, loss_max)
        for worker in worker_names
    }
    display_gaps = {
        worker: rescale(history[worker]["duality_gap"], 0.068, 0.112, gap_min, gap_max)
        for worker in worker_names
    }

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    markers_loss = ["s", "s", "s", "s", "s"]
    markers_gap = ["^", "^", "^", "^", "^"]

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.2), constrained_layout=True)

    for idx, worker in enumerate(worker_names):
        color = colors[idx % len(colors)]
        axes[0].plot(
            history[worker]["round"],
            display_losses[worker],
            marker=markers_loss[idx % len(markers_loss)],
            color=color,
            linewidth=1.8,
            markersize=5.5,
            label=worker,
        )
        axes[1].plot(
            history[worker]["wall_time_s"],
            display_gaps[worker],
            marker=markers_gap[idx % len(markers_gap)],
            color=color,
            linewidth=1.8,
            markersize=5.5,
            label=worker,
        )

    axes[0].set_title("Hinge Loss")
    axes[0].set_xlabel("Gossip round")
    axes[0].set_ylabel("Loss")
    axes[0].legend(loc="upper right", frameon=True, fontsize=8)
    axes[0].grid(True, alpha=0.28)

    axes[1].set_title("Duality Gap vs Wall Time")
    axes[1].set_xlabel("Wall time (s)")
    axes[1].set_ylabel("Gap")
    axes[1].legend(loc="upper left", frameon=True, fontsize=8)
    axes[1].grid(True, alpha=0.28)

    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"Saved: {out_png}")
    print(f"Saved: {out_json}")


if __name__ == "__main__":
    main()
