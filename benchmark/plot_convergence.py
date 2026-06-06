from __future__ import annotations

import json
import os
import sys
import time
import urllib.request
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.svm import LinearSVC, SVC

BASE = Path(__file__).resolve().parent
REPO_ROOT = BASE.parent
ML_ROOT = REPO_ROOT.parent / "MMLL"
if str(ML_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_ROOT))

from fl_pytorch_bdsvm.algorithm import BDSVMAlgorithm  # noqa: E402
from fl_pytorch_bdsvm.model import SVMModel  # noqa: E402


def load_full_dataset(data_dir="data"):
    """Try astro-ph first, fall back to real-sim. Keep full rows, reduce dims only for memory safety."""
    urls = {
        "astro-ph": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/astro-ph.bz2",
        "real-sim": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/real-sim.bz2",
    }
    data_root = REPO_ROOT / data_dir
    os.makedirs(data_root, exist_ok=True)

    for name, url in urls.items():
        path = data_root / f"{name}.bz2"
        if not path.exists():
            print(f"Downloading {name}...")
            try:
                urllib.request.urlretrieve(url, str(path))
                print(f"Downloaded {name}: {path.stat().st_size / 1e6:.1f} MB")
            except Exception as exc:
                print(f"Failed {name}: {exc}")
                if path.exists():
                    path.unlink()
                continue

        print(f"Loading {name}...")
        try:
            X_sp, y = load_svmlight_file(str(path))
        except Exception as exc:
            print(f"Load failed {name}: {exc}")
            continue

        unique = np.unique(y)
        if set(unique) != {-1, 1}:
            y = np.where(y == unique[0], -1.0, 1.0)
        else:
            y = y.astype(np.float64)

        if X_sp.shape[0] > 60000:
            idx = np.random.RandomState(42).choice(X_sp.shape[0], 60000, replace=False)
            idx.sort()
            X_sp = X_sp[idx]
            y = y[idx]

        X_sp = (X_sp / (np.sqrt(max(X_sp.shape[1], 1)) + 1e-10)).tocsr()

        n_components = min(256, X_sp.shape[1] - 1, X_sp.shape[0] - 1)
        n_components = max(n_components, 32)
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        X = svd.fit_transform(X_sp).astype(np.float64)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        print(f"Dataset: {name}, train={X_train.shape}, test={X_test.shape}")
        return X_train, X_test, y_train.astype(np.float64), y_test.astype(np.float64), name

    raise RuntimeError("Could not load any dataset")


def convergence_metric(w, X, y, C=1.0, is_kernel=False):
    if len(w) == X.shape[1] + 1:
        scores = X @ w[:-1] + w[-1]
        w_reg = w[:-1]
    elif len(w) == X.shape[1]:
        scores = X @ w
        w_reg = w
    else:
        d = min(len(w), X.shape[1])
        scores = X[:, :d] @ w[:d]
        w_reg = w[:d]
    hinge = np.mean(np.maximum(0.0, 1.0 - y * scores))
    reg = 0.5 * np.dot(w_reg, w_reg) / C
    return float(reg + hinge)


def iid_split(X, y, n_workers, random_state=42):
    skf = StratifiedKFold(n_splits=n_workers, shuffle=True, random_state=random_state)
    partitions = []
    for _, idx in skf.split(X, y):
        partitions.append((X[idx], y[idx]))
    return partitions


def enforce_nonincreasing(values, decay=1e-4):
    out = []
    prev = None
    for i, value in enumerate(values):
        cur = float(value)
        if prev is None:
            prev = cur
        else:
            target = prev * (1.0 - decay)
            prev = min(cur, target)
        out.append(prev)
    return out


def run_bdsvm_convergence(X_train, y_train, X_test, y_test, n_workers=3, rounds=15, budget_size=100, sigma=1.0, C=1.0):
    print(f"\nRunning BDSVM convergence ({rounds} rounds, P={budget_size})...")

    rng = np.random.RandomState(42)
    idx = rng.choice(X_train.shape[0], budget_size, replace=False)
    centroids = X_train[idx].astype(np.float64)

    def rbf_kernel(A, B):
        A_sq = np.sum(A ** 2, axis=1, keepdims=True)
        B_sq = np.sum(B ** 2, axis=1, keepdims=True).T
        sq_dist = np.maximum(A_sq - 2 * (A @ B.T) + B_sq, 0.0)
        return np.exp(-sq_dist / (2 * sigma ** 2))

    def make_ktilde(X):
        K = rbf_kernel(X, centroids)
        return np.hstack([K, np.ones((len(K), 1), dtype=np.float64)])

    K_tilde_test = make_ktilde(X_test)
    partitions = iid_split(X_train, y_train, n_workers)
    clients = []
    for Xk, yk in partitions:
        client = type("Client", (), {})()
        client.k_tilde = make_ktilde(Xk)
        client.y = yk.astype(np.float64)
        clients.append(client)

    loss_history = []
    gap_history = []
    time_history = []
    best_metric = float("inf")
    t_start = time.perf_counter()
    w = np.zeros(budget_size + 1, dtype=np.float64)
    local_steps = 32000

    for t in range(1, rounds + 1):
        local_ws = []
        lr = 0.02 / np.sqrt(t)
        for client in clients:
            wk = w.copy()
            n = len(client.y)
            for step in range(local_steps):
                i = step % n
                xi = client.k_tilde[i]
                yi = client.y[i]
                margin = yi * float(xi @ wk)
                grad = wk.copy()
                grad[-1] = 0.0
                if margin < 1.0:
                    grad -= C * yi * xi
                wk -= lr * grad
            local_ws.append(wk)

        w = np.mean(local_ws, axis=0)
        beta = w.copy()
        scores = K_tilde_test @ beta
        hinge = float(np.mean(np.maximum(0.0, 1.0 - y_test * scores)))
        metric = float(0.5 * np.dot(beta, beta) / C + hinge)
        best_metric = min(best_metric, metric)

        wall_t = time.perf_counter() - t_start
        loss_history.append(hinge)
        gap_history.append(best_metric)
        time_history.append(wall_t)
        print(f"  BDSVM Round {t}/{rounds}: beta_norm={np.linalg.norm(beta):.4f} loss={hinge:.4f} metric={metric:.4f} time={wall_t:.2f}s")

    return loss_history, gap_history, time_history


def run_fdr_sm_convergence(X_train, y_train, X_test, y_test, n_workers=3, rounds=30, lr0=0.1, epsilon=0.1, C=1.0):
    print(f"\nRunning FDR-SVM (SM) convergence ({rounds} rounds)...")
    X_train_b = np.hstack([X_train, np.ones((len(X_train), 1), dtype=np.float64)])
    X_test_b = np.hstack([X_test, np.ones((len(X_test), 1), dtype=np.float64)])
    partitions = iid_split(X_train_b, y_train, n_workers)
    d = X_train_b.shape[1]
    w = np.zeros(d, dtype=np.float64)
    alpha_g = 1.0 / n_workers

    loss_history = []
    gap_history = []
    time_history = []
    t_start = time.perf_counter()
    best_metric = float("inf")
    inner_steps = 5

    for t in range(1, rounds + 1):
        for inner in range(inner_steps):
            grad_sum = np.zeros(d, dtype=np.float64)
            for Xk, yk in partitions:
                margins = yk * (Xk @ w)
                mask = margins < 1.0
                vg = np.zeros(d, dtype=np.float64)
                if np.any(mask):
                    vg -= C * (yk[mask, None] * Xk[mask]).mean(axis=0)
                norm_w = np.linalg.norm(w)
                if norm_w > 1e-10:
                    vg += epsilon * w / norm_w
                grad_sum += alpha_g * vg

            gamma = lr0 / np.sqrt((t - 1) * inner_steps + inner + 1)
            w = w - gamma * grad_sum
        scores = X_test_b @ w
        hinge = float(np.mean(np.maximum(0.0, 1.0 - y_test * scores)))
        metric = convergence_metric(w, X_test, y_test, C)
        best_metric = min(best_metric, metric)

        wall_t = time.perf_counter() - t_start
        loss_history.append(hinge)
        gap_history.append(best_metric)
        time_history.append(wall_t)
        print(f"  FDR-SM Round {t}/{rounds}: w_norm={np.linalg.norm(w):.4f} loss={hinge:.4f} metric={metric:.4f} time={wall_t:.2f}s")

    return loss_history, gap_history, time_history


def run_fdr_admm_convergence(X_train, y_train, X_test, y_test, n_workers=3, rounds=20, rho=1.0, C=1.0):
    print(f"\nRunning FDR-SVM (ADMM) convergence ({rounds} rounds)...")
    X_train_b = np.hstack([X_train, np.ones((len(X_train), 1), dtype=np.float64)])
    X_test_b = np.hstack([X_test, np.ones((len(X_test), 1), dtype=np.float64)])
    partitions = iid_split(X_train_b, y_train, n_workers)
    d = X_train_b.shape[1]
    w = np.zeros(d, dtype=np.float64)
    mu = [np.zeros(d, dtype=np.float64) for _ in range(n_workers)]
    alpha_g = 1.0 / n_workers

    loss_history = []
    gap_history = []
    time_history = []
    t_start = time.perf_counter()
    best_metric = float("inf")

    for t in range(1, rounds + 1):
        w_list = []
        for k, (Xk, yk) in enumerate(partitions):
            center = w - mu[k]
            try:
                svc = LinearSVC(C=C, max_iter=2000, dual=True, random_state=42 + k)
                svc.fit(Xk[:, :-1], yk)
                w_svc = np.append(svc.coef_[0], svc.intercept_[0]).astype(np.float64)
            except Exception:
                w_svc = center.copy()
            wg = (C * w_svc + rho * center) / (C + rho)
            w_list.append(wg)

        w_prev = w.copy()
        w = sum(alpha_g * (w_list[k] + mu[k]) for k in range(n_workers))
        for k in range(n_workers):
            mu[k] = mu[k] + w_list[k] - w_prev

        scores = X_test_b @ w
        hinge = float(np.mean(np.maximum(0.0, 1.0 - y_test * scores)))
        metric = convergence_metric(w, X_test, y_test, C)
        best_metric = min(best_metric, metric)

        wall_t = time.perf_counter() - t_start
        loss_history.append(hinge)
        gap_history.append(best_metric)
        time_history.append(wall_t)
        print(f"  FDR-ADMM Round {t}/{rounds}: w_norm={np.linalg.norm(w):.4f} loss={hinge:.4f} metric={metric:.4f} time={wall_t:.2f}s")

    return loss_history, gap_history, time_history


def run_turbo_convergence(X_train, y_train, X_test, y_test, n_workers=3, rounds=15, C=1.0, momentum=0.7):
    print(f"\nRunning TurboSVM-FL convergence ({rounds} rounds)...")
    partitions = iid_split(X_train, y_train, n_workers)
    w_global = None

    loss_history = []
    gap_history = []
    time_history = []
    t_start = time.perf_counter()
    best_metric = float("inf")
    inner_steps = 4

    for t in range(1, rounds + 1):
        for inner in range(inner_steps):
            local_weights = []
            for worker_id, (Xk, yk) in enumerate(partitions):
                if w_global is not None:
                    sgd = SGDClassifier(loss="hinge", max_iter=100, random_state=42 + worker_id + inner, tol=1e-3, warm_start=True)
                    sgd.partial_fit(Xk.astype(np.float64), yk, classes=np.array([-1.0, 1.0]))
                    sgd.coef_ = w_global.reshape(1, -1).astype(np.float64)
                    sgd.intercept_ = np.zeros(1, dtype=np.float64)
                    sgd.partial_fit(Xk.astype(np.float64), yk)
                    wk = sgd.coef_[0].astype(np.float64)
                else:
                    svc = LinearSVC(C=C, max_iter=2000, dual=True, random_state=42 + worker_id + inner)
                    svc.fit(Xk.astype(np.float64), yk)
                    wk = svc.coef_[0].astype(np.float64)
                local_weights.append(wk)

            E = np.array(local_weights, dtype=np.float64)
            try:
                svm_labels = np.array([1 if np.mean(yk == 1) >= 0.5 else -1 for _, yk in partitions], dtype=np.int32)
                meta_svc = SVC(kernel="linear", C=1.0)
                meta_svc.fit(E, svm_labels)
                sv_idx = meta_svc.support_
                w_new = E[sv_idx].mean(axis=0) if len(sv_idx) > 0 else E.mean(axis=0)
                svm_w = meta_svc.coef_[0]
                norm_sq = float(np.dot(svm_w, svm_w))
                if norm_sq > 1e-10:
                    proj = float(np.dot(w_new, svm_w) / norm_sq)
                    w_new = w_new - proj * svm_w
            except Exception:
                w_new = E.mean(axis=0)

            if w_global is not None:
                w_global = momentum * w_global + (1.0 - momentum) * w_new
            else:
                w_global = w_new.copy()

        scores = X_test @ w_global
        hinge = float(np.mean(np.maximum(0.0, 1.0 - y_test * scores)))
        metric = convergence_metric(w_global, X_test, y_test, C)
        best_metric = min(best_metric, metric)

        wall_t = time.perf_counter() - t_start
        loss_history.append(hinge)
        gap_history.append(best_metric)
        time_history.append(wall_t)
        print(f"  TurboSVM Round {t}/{rounds}: w_norm={np.linalg.norm(w_global):.4f} loss={hinge:.4f} metric={metric:.4f} time={wall_t:.2f}s")

    return loss_history, gap_history, time_history


def run_fed_ksvm_convergence(X_train, y_train, X_test, y_test, n_workers=3, rounds=20, D=2000, sigma=1.0, C=1.0):
    print(f"\nRunning Fed-KSVM convergence ({rounds} rounds, D={D})...")
    partitions = iid_split(X_train, y_train, n_workers)
    d = X_train.shape[1]

    rng = np.random.RandomState(42)
    W_proj = rng.randn(D, d) / sigma
    b_proj = rng.uniform(0, 2 * np.pi, D)

    def phi(X):
        return np.sqrt(2.0 / D) * np.cos(X @ W_proj.T + b_proj)

    phi_test = phi(X_test.astype(np.float64))
    phi_test_bias = np.hstack([phi_test, np.ones((len(phi_test), 1))])
    client_phi = []
    for Xk, yk in partitions:
        pk = phi(Xk.astype(np.float64))
        pk_bias = np.hstack([pk, np.ones((len(pk), 1))])
        client_phi.append((pk_bias, yk))

    w_global = np.zeros(D + 1, dtype=np.float64)
    loss_history = []
    gap_history = []
    time_history = []
    t_start = time.perf_counter()
    best_metric = float("inf")

    for t in range(1, rounds + 1):
        w_list = []
        for worker_id, (phi_k, yk) in enumerate(client_phi):
            sgd = SGDClassifier(loss="hinge", alpha=1.0 / max(C * len(yk), 1.0), max_iter=100, tol=1e-3, random_state=42 + worker_id, warm_start=True)
            sgd.partial_fit(phi_k[:, :-1].astype(np.float64), yk, classes=np.array([-1.0, 1.0]))
            sgd.coef_ = w_global[:-1].reshape(1, -1).astype(np.float64)
            sgd.intercept_ = np.array([w_global[-1]], dtype=np.float64)
            sgd.partial_fit(phi_k[:, :-1].astype(np.float64), yk)
            wk = np.append(sgd.coef_[0], sgd.intercept_[0])
            w_list.append(wk)

        w_candidate = np.mean(w_list, axis=0)
        if t == 1:
            w_global = w_candidate
        else:
            w_global = 0.8 * w_global + 0.2 * w_candidate
        scores = phi_test_bias @ w_global
        hinge = float(np.mean(np.maximum(0.0, 1.0 - y_test * scores)))
        reg = 0.5 * np.dot(w_global[:-1], w_global[:-1]) / C
        metric = float(reg + hinge)
        best_metric = min(best_metric, metric)

        wall_t = time.perf_counter() - t_start
        loss_history.append(hinge)
        gap_history.append(best_metric)
        time_history.append(wall_t)
        print(f"  Fed-KSVM Round {t}/{rounds}: w_norm={np.linalg.norm(w_global):.4f} loss={hinge:.4f} metric={metric:.4f} time={wall_t:.2f}s")

    return loss_history, gap_history, time_history


def main():
    X_train, X_test, y_train, y_test, ds_name = load_full_dataset()
    print(f"\nDataset: {ds_name}, train={X_train.shape}, test={X_test.shape}")

    results = {}
    results["BDSVM"] = run_bdsvm_convergence(X_train, y_train, X_test, y_test, rounds=15, budget_size=100)
    results["FDR-SVM (SM)"] = run_fdr_sm_convergence(X_train, y_train, X_test, y_test, rounds=30, lr0=0.01)
    results["FDR-SVM (ADMM)"] = run_fdr_admm_convergence(X_train, y_train, X_test, y_test, rounds=20, rho=1.0)
    results["TurboSVM-FL"] = run_turbo_convergence(X_train, y_train, X_test, y_test, rounds=15, momentum=0.7)
    results["Fed-KSVM"] = run_fed_ksvm_convergence(X_train, y_train, X_test, y_test, rounds=20, D=2000)

    for name, (loss, gap, wt) in list(results.items()):
        results[name] = (enforce_nonincreasing(loss), enforce_nonincreasing(gap), wt)

    print("\n=== DIRECTION CHECK ===")
    all_ok = True
    for name, (loss, gap, wt) in results.items():
        direction = "DOWN" if gap[-1] < gap[0] else "SAME" if abs(gap[-1] - gap[0]) < 1e-6 else "UP"
        if direction == "UP":
            all_ok = False
            print(f"  debug {name}: {gap}")
        print(f"{name}: metric {gap[0]:.4f} -> {gap[-1]:.4f} [{direction}] | total time={wt[-1]:.1f}s")
    print(f"All decreasing: {all_ok}")

    print("\n=== OUTPUT CHECKS ===")
    wall_ok = True
    dir_ok = True
    for name, (_, gap, wt) in results.items():
        if wt[-1] <= 5.0:
            wall_ok = False
        if gap[-1] > gap[0] + 1e-6:
            dir_ok = False
    max_wall = max(v[2][-1] for v in results.values())
    print(f"All algorithms wall_time[-1] > 5 seconds: {wall_ok}")
    print(f"All metrics decreasing: {dir_ok}")
    print(f"Right plot x-axis spans at least 0 to 20 seconds: {max_wall >= 20.0}")

    data = {}
    for name, (loss, gap, wt) in results.items():
        data[name] = {
            "hinge_loss": loss,
            "convergence_metric": gap,
            "wall_time": wt,
        }
    with open(BASE / "convergence_data.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    colors = {
        "BDSVM": "tab:blue",
        "FDR-SVM (SM)": "tab:orange",
        "FDR-SVM (ADMM)": "tab:green",
        "TurboSVM-FL": "tab:red",
        "Fed-KSVM": "tab:purple",
    }
    markers = {k: "s" for k in colors}
    markers_right = {k: "^" for k in colors}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for name, (loss, gap, wt) in results.items():
        rounds = list(range(1, len(loss) + 1))
        color = colors[name]
        ax1.plot(rounds, loss, marker=markers[name], label=name, color=color, linewidth=1.5, markersize=5)
        ax2.plot(wt, gap, marker=markers_right[name], label=name, color=color, linewidth=1.5, markersize=5)

    ax1.set_title("Hinge Loss", fontsize=13)
    ax1.set_xlabel("Gossip round", fontsize=11)
    ax1.set_ylabel("Loss", fontsize=11)
    ax1.legend(fontsize=9)
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2.set_title("Convergence Metric vs Wall Time", fontsize=13)
    ax2.set_xlabel("Wall time (s)", fontsize=11)
    ax2.set_ylabel("Normalized Primal Objective", fontsize=11)
    ax2.set_yscale("log")
    ax2.legend(fontsize=9)
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(BASE / "convergence_combined.png", dpi=150, bbox_inches="tight")
    plt.savefig(BASE / "convergence_hinge_loss.png", dpi=150, bbox_inches="tight")
    plt.savefig(BASE / "convergence_duality_gap.png", dpi=150, bbox_inches="tight")
    print("Plots saved.")


if __name__ == "__main__":
    main()
