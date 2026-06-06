from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from algo_bdsvm import run_bdsvm
from algo_fdr_svm import run_fdr_svm_admm, run_fdr_svm_sm
from algo_fed_ksvm import run_fed_ksvm
from algo_turbo_svm import run_turbo_svm
from sdca_data import load_dataset

DATASETS = ["astro-ph", "ccat", "covtype"]
ALGORITHMS = {
    "BDSVM": (run_bdsvm, {"budget_size": 50, "sigma": 1.0, "rounds": 6}),
    "FDR-SVM (SM)": (run_fdr_svm_sm, {"rounds": 20, "lr0": 0.1, "epsilon": 0.1}),
    "FDR-SVM (ADMM)": (run_fdr_svm_admm, {"rounds": 8, "rho": 1.0, "epsilon": 0.1}),
    "TurboSVM-FL": (run_turbo_svm, {"rounds": 6}),
    "Fed-KSVM": (run_fed_ksvm, {"D": 300, "sigma": 1.0, "rounds": 6, "n_blocks": 3}),
}
ALGO_TYPE = {
    "BDSVM": "Primal",
    "FDR-SVM (SM)": "Primal",
    "FDR-SVM (ADMM)": "Primal-Dual",
    "TurboSVM-FL": "Mixed",
    "Fed-KSVM": "Primal",
}


def _to_float(v):
    try:
        return float(v)
    except Exception:
        return float("nan")


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

    raw = {}
    for algo_name, (algo_fn, algo_cfg) in ALGORITHMS.items():
        print(f"\n############################")
        print(f"Running algorithm: {algo_name}")
        print(f"############################")
        raw[algo_name] = {"type": ALGO_TYPE[algo_name], "datasets": {}}

        for ds in DATASETS:
            d = dataset_cache[ds]
            kwargs = dict(algo_cfg)
            kwargs.update({"n_workers": n_workers, "random_state": random_state})
            if "C" in algo_fn.__code__.co_varnames:
                kwargs["C"] = c_value
            if "c_value" in algo_fn.__code__.co_varnames:
                kwargs["c_value"] = c_value

            print(f"\n--> {algo_name} on {ds}")
            res = algo_fn(
                d["X_train"],
                d["y_train"],
                d["X_test"],
                d["y_test"],
                **kwargs,
            )
            raw[algo_name]["datasets"][ds] = {k: _to_float(v) for k, v in res.items()}
            print(
                f"{ds}: acc={res['accuracy']:.4f}, auc={res['roc_auc']:.4f}, "
                f"f1={res['f1']:.4f}, t={res['time_s']:.2f}s"
            )

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(raw, f, indent=2)

    lines = []
    lines.append("| Algorithm | Type | astro-ph Acc | astro-ph AUC | ccat Acc | ccat AUC | covtype Acc | covtype AUC | Avg Time(s) |")
    lines.append("|-----------|------|-------------|--------------|----------|----------|-------------|-------------|-------------|")

    for algo_name in ALGORITHMS:
        r = raw[algo_name]
        a = r["datasets"].get("astro-ph", {})
        c = r["datasets"].get("ccat", {})
        v = r["datasets"].get("covtype", {})
        avg_t = np.nanmean([
            _to_float(a.get("time_s", np.nan)),
            _to_float(c.get("time_s", np.nan)),
            _to_float(v.get("time_s", np.nan)),
        ])
        lines.append(
            "| {} | {} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.2f} |".format(
                algo_name,
                r["type"],
                _to_float(a.get("accuracy", np.nan)),
                _to_float(a.get("roc_auc", np.nan)),
                _to_float(c.get("accuracy", np.nan)),
                _to_float(c.get("roc_auc", np.nan)),
                _to_float(v.get("accuracy", np.nan)),
                _to_float(v.get("roc_auc", np.nan)),
                float(avg_t),
            )
        )

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nSaved: {md_path}")
    print(f"Saved: {json_path}")


if __name__ == "__main__":
    main()
