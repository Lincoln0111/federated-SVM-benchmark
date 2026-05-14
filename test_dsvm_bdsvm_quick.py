# -*- coding: utf-8 -*-
"""
Quick standalone test of DSVM (= BDSVM) on MNIST binary classification.
Digit 0 vs Others | IRWLS optimization + Gaussian (RBF) kernel
Reference: ACM TIST 2022
"""

import time
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.cluster import MiniBatchKMeans
from scipy.spatial.distance import cdist

C = 1.0
sigma = 1.0
NC = 50
Nmaxiter = 5
tolerance = 1e-4
N_TRAIN = 10000
N_TEST = 2000


def gaussian_kernel_matrix(A, B, sigma):
    dists2 = cdist(A, B, 'sqeuclidean')
    return np.exp(-dists2 / (2.0 * sigma ** 2))


def irwls_dsvm(Kmn, y, C, Nmaxiter, tolerance):
    N, M = Kmn.shape
    w = np.zeros((M, 1))
    Y = y.reshape(-1, 1)
    history = []

    for it in range(Nmaxiter):
        margin = Kmn.dot(w) * Y
        hinge = 1.0 - margin
        hinge_pos = np.maximum(hinge, 0)
        cost = 0.5 * float(w.T.dot(w).item()) + C * hinge_pos.sum()
        history.append(cost)

        sv_mask = (hinge > 0).flatten()
        if sv_mask.sum() == 0:
            print(f"   Iter {it+1}: No support vectors, cost={cost:.4f} 鈥?converged early")
            break

        clYC = sv_mask.astype(float).reshape(-1, 1) * C * (-Y)
        grad = Kmn.T.dot(clYC) + w

        W_sv = sv_mask.astype(float).reshape(-1, 1)
        KtWK = (Kmn * W_sv).T.dot(Kmn)
        H = KtWK * C + np.eye(M)

        try:
            delta = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            delta = np.linalg.lstsq(H, grad, rcond=None)[0]

        w_new = w - delta
        diff = np.linalg.norm(w_new - w)
        w = w_new
        print(f"   Iter {it+1}/{Nmaxiter}: cost={cost:.4f}, ||螖w||={diff:.6f}, SVs={sv_mask.sum()}")

        if diff < tolerance:
            print(f"   Converged at iteration {it+1}")
            break

    return w, history


print("=" * 60)
print("  DSVM (BDSVM) Quick Test on MNIST Binary Classification")
print("  Digit 0 vs Others | IRWLS + Gaussian Kernel")
print("=" * 60)

print("\n[Step 1] Loading MNIST...")
t0 = time.time()
mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
X_all = mnist.data.astype(np.float32) / 255.0
y_raw = mnist.target.astype(int)
print(f"  鉁?MNIST loaded: {X_all.shape[0]} samples, {X_all.shape[1]} features ({time.time()-t0:.1f}s)")

print("\n[Step 2] Preprocessing (Digit 0 vs Others)...")
y_all = np.where(y_raw == 0, 1, -1)
X_train_full, y_train_full = X_all[:60000], y_all[:60000]
X_test_full, y_test_full = X_all[60000:], y_all[60000:]

rng = np.random.default_rng(42)
idx = rng.choice(len(X_train_full), N_TRAIN, replace=False)
idx2 = rng.choice(len(X_test_full), N_TEST, replace=False)
X_train, y_train = X_train_full[idx], y_train_full[idx]
X_test, y_test = X_test_full[idx2], y_test_full[idx2]

pos_tr = (y_train == 1).sum()
print(f"  鉁?Train: {len(X_train)} samples ({pos_tr} positives, {len(X_train)-pos_tr} negatives)")
print(f"  鉁?Test : {len(X_test)} samples")

print(f"\n[Step 3] Computing {NC} budget vectors via Mini-Batch K-Means...")
t1 = time.time()
km = MiniBatchKMeans(n_clusters=NC, random_state=42, max_iter=100, n_init=3)
km.fit(X_train)
centroids = km.cluster_centers_.astype(np.float32)
print(f"  鉁?Centroids shape: {centroids.shape}  ({time.time()-t1:.1f}s)")

print(f"\n[Step 4] Computing Gram (kernel) matrices 鈥?sigma={sigma}...")
t2 = time.time()
Kmn_train = gaussian_kernel_matrix(X_train, centroids, sigma)
Kmn_test = gaussian_kernel_matrix(X_test, centroids, sigma)
print(f"  鉁?K_train shape: {Kmn_train.shape}")
print(f"  鉁?K_test  shape: {Kmn_test.shape}  ({time.time()-t2:.1f}s)")

print(f"\n[Step 5] IRWLS optimisation (C={C}, Nmaxiter={Nmaxiter})...")
t3 = time.time()
w, cost_history = irwls_dsvm(Kmn_train, y_train, C, Nmaxiter, tolerance)
print(f"  鉁?Training done in {time.time()-t3:.1f}s")

print("\n[Step 6] Evaluating model...")

def predict(Kmn, w):
    scores = Kmn.dot(w).flatten()
    labels = np.where(scores >= 0, 1, -1)
    return labels, scores

y_pred_tr, scores_tr = predict(Kmn_train, w)
y_pred_te, scores_te = predict(Kmn_test, w)

acc_tr = accuracy_score(y_train, y_pred_tr)
prec_tr = precision_score(y_train, y_pred_tr, pos_label=1, zero_division=0)
rec_tr = recall_score(y_train, y_pred_tr, pos_label=1, zero_division=0)
f1_tr = f1_score(y_train, y_pred_tr, pos_label=1, zero_division=0)

acc_te = accuracy_score(y_test, y_pred_te)
prec_te = precision_score(y_test, y_pred_te, pos_label=1, zero_division=0)
rec_te = recall_score(y_test, y_pred_te, pos_label=1, zero_division=0)
f1_te = f1_score(y_test, y_pred_te, pos_label=1, zero_division=0)
auc_te = roc_auc_score(y_test, scores_te)

total_time = time.time() - t0

print("\n" + "=" * 60)
print("  RESULTS")
print("=" * 60)
print(f"  {'Metric':<15} {'Train':>10} {'Test':>10}")
print(f"  {'-'*35}")
print(f"  {'Accuracy':<15} {acc_tr:>10.4f} {acc_te:>10.4f}")
print(f"  {'Precision':<15} {prec_tr:>10.4f} {prec_te:>10.4f}")
print(f"  {'Recall':<15} {rec_tr:>10.4f} {rec_te:>10.4f}")
print(f"  {'F1-Score':<15} {f1_tr:>10.4f} {f1_te:>10.4f}")
print(f"  {'ROC-AUC':<15} {'N/A':>10} {auc_te:>10.4f}")
print(f"  {'-'*35}")
print(f"  Total wall-clock time: {total_time:.1f}s")
print(f"\n  Parameters: C={C}, sigma={sigma}, NC={NC}, Nmaxiter={Nmaxiter}")
print(f"  Train size={len(X_train)}, Test size={len(X_test)}")
print("=" * 60)
print("  鉁?DSVM (BDSVM) test completed successfully!")
print("=" * 60)
