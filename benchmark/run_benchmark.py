from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from algo_bdsvm import run_bdsvm
from algo_fdr_svm import run_fdr_svm_admm, run_fdr_svm_sm
from algo_fed_ksvm import run_fed_ksvm
from algo_turbo_svm import run_turbo_svm
from sdca_data import load_dataset

DATASETS = ["real-sim", "ccat", "covtype"]

DATASET_DISPLAY = {
    "real-sim": "real-sim (astro-ph substitute)",
    "ccat":     "CCAT/rcv1",
    "covtype":  "covtype",
}

ALGORITHMS = {
    "BDSVM":          (run_bdsvm,        {"budget_size": 50, "sigma": 1.0, "rounds": 10}),
    "FDR-SVM (SM)":   (run_fdr_svm_sm,   {"rounds": 50, "lr0": 0.1, "epsilon": 0.1}),
    "FDR-SVM (ADMM)": (run_fdr_svm_admm, {"rounds": 20, "rho": 1.0, "epsilon": 0.1}),
    "TurboSVM-FL":    (run_turbo_svm,    {"rounds": 10}),
    "Fed-KSVM":       (run_fed_ksvm,     {"D": 500, "sigma": 1.0, "rounds": 10}),
}
ALGO_TYPE = {
    "BDSVM":          "Primal",
    "FDR-SVM (SM)":   "Primal",
    "FDR-SVM (ADMM)": "Primal-Dual",
    "TurboSVM-FL":    "Mixed",
    "Fed-KSVM":       "Primal",
}


def _to_float(v):
    try:
        return float(v)
    except Exception:
        return float("nan")


def _dataset_header(ds: str, d: dict) -> str:
    X_tr, X_te = d["X_train"], d["X_test"]
    n_tr, n_te = X_tr.shape[0], X_te.shape[0]
    n_feat = X_tr.shape[1]
    density = "sparse" if ds in ("real-sim", "ccat") else "dense"
    display = DATASET_DISPLAY[ds]
    return f"Dataset: {display} FULL ({n_tr:,} train / {n_te:,} test, {n_feat:,} features, {density})"


def _slack_table(ds: str, algo_results: dict) -> list[str]:
    rows = []
    rows.append("| Algorithm | Type | Accuracy | F1 | AUC | Time(s) |")
    rows.append("|---|---|---|---|---|---|")
    for algo_name in ALGORITHMS:
        r = algo_results.get(algo_name, {}).get(ds)
        if r is None:
            rows.append(f"| {algo_name} | {ALGO_TYPE[algo_name]} | N/A | N/A | N/A | N/A |")
        else:
            rows.append(
                "| {} | {} | {:.4f} | {:.4f} | {:.4f} | {:.0f} |".format(
                    algo_name,
                    ALGO_TYPE[algo_name],
                    _to_float(r.get("accuracy", float("nan"))),
                    _to_float(r.get("f1", float("nan"))),
                    _to_float(r.get("roc_auc", float("nan"))),
                    _to_float(r.get("time_s", float("nan"))),
                )
            )
    return rows


def main():
    out_dir = Path(__file__).resolve().parent
    md_path = out_dir / "results_sdca_benchmark.md"
    json_path = out_dir / "results_sdca_benchmark.json"

    random_state = 42
    n_workers = 3
    c_value = 1.0

    dataset_cache = {}
    for ds in DATASETS:
        print(f"\n=== Loading dataset: {ds} ===")
        dataset_cache[ds] = load_dataset(ds, data_dir=str(out_dir.parent / "data"), random_state=random_state)
        d = dataset_cache[ds]
        print(f"[{ds}] train={d['X_train'].shape}, test={d['X_test'].shape}, source={d.get('source', ds)}")

    # algo_results[algo_name][ds] = metrics dict
    algo_results: dict = {name: {} for name in ALGORITHMS}

    for ds in DATASETS:
        d = dataset_cache[ds]
        n_tr, n_feat = d["X_train"].shape
        for algo_name, (algo_fn, algo_cfg) in ALGORITHMS.items():
            print(f"\nRunning {algo_name} on {DATASET_DISPLAY[ds]} (train_size={n_tr:,}, features={n_feat:,})...")
            kwargs = dict(algo_cfg)
            kwargs.update({"n_workers": n_workers, "random_state": random_state})
            if "C" in algo_fn.__code__.co_varnames:
                kwargs["C"] = c_value
            if "c_value" in algo_fn.__code__.co_varnames:
                kwargs["c_value"] = c_value

            t_wall = time.perf_counter()
            try:
                res = algo_fn(
                    d["X_train"],
                    d["y_train"],
                    d["X_test"],
                    d["y_test"],
                    **kwargs,
                )
                algo_results[algo_name][ds] = {k: _to_float(v) for k, v in res.items()}
                elapsed = time.perf_counter() - t_wall
                print(
                    f"{algo_name} on {ds}: acc={res['accuracy']:.4f}  "
                    f"f1={res['f1']:.4f}  auc={res['roc_auc']:.4f}  time={elapsed:.0f}s"
                )
            except Exception as exc:
                elapsed = time.perf_counter() - t_wall
                algo_results[algo_name][ds] = {}
                print(f"[ERROR] {algo_name} on {ds} failed after {elapsed:.0f}s: {exc}")

    # Build JSON output (keyed by algo → ds)
    json_out = {}
    for algo_name in ALGORITHMS:
        json_out[algo_name] = {"type": ALGO_TYPE[algo_name], "datasets": algo_results[algo_name]}
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_out, f, indent=2)

    # Build Slack-ready console + MD output
    run_date = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    console_lines = ["\n=== FULL DATASET BENCHMARK RESULTS ===\n"]
    md_lines = [
        "# SDCA Benchmark Results — Full Dataset Run",
        "",
        f"- real-sim: 72,309 samples (astro-ph substitute)",
        f"- CCAT/rcv1: ~20k train + ~677k test (SVD-256 projection)",
        f"- covtype: 581,012 samples",
        f"- Run date: {run_date}",
        "",
    ]

    for ds in DATASETS:
        header = _dataset_header(ds, dataset_cache[ds])
        table_rows = _slack_table(ds, algo_results)

        console_lines.append(header)
        console_lines.extend(table_rows)
        console_lines.append("")

        md_lines.append(f"## {header}")
        md_lines.extend(table_rows)
        md_lines.append("")

    slack_block = "\n".join(console_lines)
    print(slack_block)

    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"Saved: {md_path}")
    print(f"Saved: {json_path}")


if __name__ == "__main__":
    main()
