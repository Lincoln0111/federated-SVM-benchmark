from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    base = Path(__file__).resolve().parent
    src = base / "results_sdca_benchmark.json"
    out = base / "results_sdca_benchmark_lines.png"

    data = json.loads(src.read_text(encoding="utf-8"))

    datasets = ["astro-ph", "ccat", "covtype"]
    algorithms = list(data.keys())

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    metrics = [
        ("accuracy", "Accuracy", axes[0]),
        ("roc_auc", "ROC-AUC", axes[1]),
        ("time_s", "Time (s)", axes[2]),
    ]

    for metric_key, title, ax in metrics:
        for algo in algorithms:
            y = [float(data[algo]["datasets"][ds][metric_key]) for ds in datasets]
            ax.plot(datasets, y, marker="o", linewidth=2, label=algo)

        ax.set_title(title)
        ax.set_xlabel("Dataset")
        ax.grid(alpha=0.25)

    axes[0].set_ylabel("Score")
    axes[1].set_ylabel("Score")
    axes[2].set_ylabel("Seconds")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.08), frameon=False)
    fig.suptitle("FL SVM Benchmark: Line Comparison Across Datasets", y=1.14, fontsize=12)

    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
