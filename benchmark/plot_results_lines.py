from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


plt.style.use("seaborn-v0_8-whitegrid")


def main() -> None:
    base = Path(__file__).resolve().parent
    src = base / "results_sdca_benchmark.json"
    out = base / "results_sdca_benchmark_lines.png"

    data = json.loads(src.read_text(encoding="utf-8"))

    datasets = ["astro-ph", "ccat", "covtype"]
    algorithms = list(data.keys())
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    markers = ["s", "o", "^", "D", "v"]

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), constrained_layout=True)
    metrics = [
        ("accuracy", "Accuracy", axes[0]),
        ("roc_auc", "ROC-AUC", axes[1]),
    ]

    for metric_key, title, ax in metrics:
        for idx, algo in enumerate(algorithms):
            y = [float(data[algo]["datasets"][ds][metric_key]) for ds in datasets]
            ax.plot(
                datasets,
                y,
                marker=markers[idx % len(markers)],
                color=colors[idx % len(colors)],
                linewidth=2,
                markersize=6,
                label=algo,
            )

        ax.set_title(title)
        ax.set_xlabel("Dataset")
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="x", rotation=0)

    axes[0].set_ylabel("Score")
    axes[1].set_ylabel("Score")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 1.06),
        frameon=False,
    )
    fig.suptitle("FL SVM Benchmark Comparison", y=1.12, fontsize=13)

    fig.savefig(out, dpi=220, bbox_inches="tight")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
