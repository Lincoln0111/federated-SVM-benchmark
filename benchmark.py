"""
SDCA benchmark datasets (astro-ph, CCAT/rcv1, cov1/covtype) loader
+ Federated Learning partitioners (IID / label-skew / Dirichlet)

For benchmarking the 4 federated SVM papers:
  1. BDSVM (ACM TIST 2022)
  2. FDR-SVM (arXiv 2410.03877)
  3. Over-the-Air FL (arXiv 1812.11750)
  4. FedSSL-AMC (arXiv 2510.04927)

Usage:
    from sdca_fl_data import download_all, load_dataset, partition

    download_all(data_dir="./data")
    X, y = load_dataset("rcv1", split="train", data_dir="./data")
    client_idx = partition(y, num_clients=10, scheme="dirichlet", alpha=0.3)
    # client_idx is a list of np.array(indices) per client
"""
import os
import sys
import bz2
import ssl
import urllib.request
from pathlib import Path

_SSL_CTX = ssl.create_default_context()
_SSL_CTX.check_hostname = False
_SSL_CTX.verify_mode = ssl.CERT_NONE

import numpy as np
from sklearn.datasets import load_svmlight_file


def _normalize_labels(y):
    """Map raw labels to {-1, +1}. Works for {-1,1}, {0,1}, {1,2}, etc."""
    classes = np.unique(y)
    if len(classes) != 2:
        raise ValueError(f"Expected binary labels, got {classes}")
    lo, hi = classes[0], classes[1]
    if lo == -1 and hi == 1:
        return y.astype(np.float64)
    mid = (lo + hi) / 2.0
    return np.where(y > mid, 1.0, -1.0).astype(np.float64)


LIBSVM_BASE = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary"

# (filename_on_libsvm, local_filename) — astro-ph not on LIBSVM, see note below
DATASETS = {
    "rcv1_train": (f"{LIBSVM_BASE}/rcv1_train.binary.bz2", "rcv1_train.binary.bz2"),
    "rcv1_test":  (f"{LIBSVM_BASE}/rcv1_test.binary.bz2",  "rcv1_test.binary.bz2"),
    "covtype":    (f"{LIBSVM_BASE}/covtype.libsvm.binary.scale.bz2",
                   "covtype.libsvm.binary.scale.bz2"),
    # astro-ph: Joachims SVMperf example data. If the URL below 404s,
    # fall back to mirror or OpenML. In SDCA paper this is "Physics ArXiv",
    # ~62k samples, ~99k features.
    "astro_ph":   ("http://download.joachims.org/svm_perf/examples/example3.tar.gz",
                   "svmperf_example3.tar.gz"),
}


def _download(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"[skip] {dest.name} already exists ({dest.stat().st_size/1e6:.1f} MB)")
        return
    print(f"[download] {url} -> {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, context=_SSL_CTX) as resp, open(dest, "wb") as f:
        f.write(resp.read())
    print(f"[done] {dest.stat().st_size/1e6:.1f} MB")


def download_all(data_dir: str = "./data", include_astro_ph: bool = True) -> None:
    """Download all SDCA benchmark datasets to data_dir."""
    data_dir = Path(data_dir)
    for name, (url, fname) in DATASETS.items():
        if name == "astro_ph" and not include_astro_ph:
            continue
        try:
            _download(url, data_dir / fname)
        except Exception as e:
            print(f"[warn] failed to download {name}: {e}")
            if name == "astro_ph":
                print("       astro-ph is the trickiest one — Joachims's URL may have moved.")
                print("       Try OpenML (search 'arxiv physics') or contact the authors.")


def load_dataset(name: str, split: str = "train", data_dir: str = "./data"):
    """
    Load a dataset as (X, y) where X is scipy.sparse.csr_matrix and y is np.ndarray of {-1, +1}.

    name in {"rcv1", "covtype", "astro_ph"}
    split in {"train", "test"} — only matters for rcv1
    """
    data_dir = Path(data_dir)
    if name == "rcv1":
        fname = f"rcv1_{split}.binary.bz2"
    elif name == "covtype":
        fname = "covtype.libsvm.binary.scale.bz2"
    elif name == "astro_ph":
        # after extracting svmperf_example3.tar.gz you get train.dat / test.dat
        sub = "train.dat" if split == "train" else "test.dat"
        fname = f"svmperf_example3/example3/{sub}"
    else:
        raise ValueError(f"unknown dataset: {name}")

    path = data_dir / fname
    if not path.exists():
        raise FileNotFoundError(f"{path} — run download_all() first")

    cache_key = f"{name}_{split}"
    cache = data_dir / f"{cache_key}.npz"

    # covtype has no official train/test split — split 80/20 by index
    if name == "covtype" and not cache.exists():
        full_cache = data_dir / "covtype_full.npz"
        if not full_cache.exists():
            print(f"[parse] {path.name} (first time — will cache as covtype_full.npz)", flush=True)
            X_full, y_full = load_svmlight_file(str(path))
            y_full = _normalize_labels(y_full)
            X_full = X_full.tocsr()
            np.savez(full_cache, data=X_full.data, indices=X_full.indices,
                     indptr=X_full.indptr, shape=X_full.shape, y=y_full)
            print(f"[cache] saved {full_cache.name}", flush=True)
        z = np.load(full_cache, allow_pickle=True)
        import scipy.sparse
        X_full = scipy.sparse.csr_matrix(
            (z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"])
        )
        y_full = z["y"]
        n = X_full.shape[0]
        cut = int(n * 0.8)
        if split == "train":
            X, y = X_full[:cut], y_full[:cut]
        else:
            X, y = X_full[cut:], y_full[cut:]
        np.savez(cache, data=X.data, indices=X.indices, indptr=X.indptr,
                 shape=X.shape, y=y)
        print(f"[cache] saved {cache.name}", flush=True)
        return X, y
    if cache.exists():
        print(f"[cache] loading {cache.name}", flush=True)
        z = np.load(cache, allow_pickle=True)
        import scipy.sparse
        X = scipy.sparse.csr_matrix(
            (z["data"], z["indices"], z["indptr"]), shape=z["shape"]
        )
        y = z["y"]
        return X, y

    print(f"[parse] {path.name} (first time — will cache as {cache.name})", flush=True)
    X, y = load_svmlight_file(str(path))
    y = _normalize_labels(y)
    X = X.tocsr()
    np.savez(cache, data=X.data, indices=X.indices, indptr=X.indptr,
             shape=X.shape, y=y)
    print(f"[cache] saved {cache.name}", flush=True)
    return X, y


# ======================================================================
# Federated partitioners
# ======================================================================

def partition_iid(y: np.ndarray, num_clients: int, seed: int = 0):
    """Random uniform partition. Returns list of np.array(indices)."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    return np.array_split(idx, num_clients)


def partition_label_skew(y: np.ndarray, num_clients: int,
                          classes_per_client: int = 1, seed: int = 0):
    """
    Each client only sees `classes_per_client` of the 2 labels.
    For binary: classes_per_client=1 means each client sees only +1 OR only -1
    (extreme non-IID, used in BDSVM evals).
    """
    rng = np.random.default_rng(seed)
    if classes_per_client == 2:
        return partition_iid(y, num_clients, seed)

    pos_idx = np.where(y > 0)[0]
    neg_idx = np.where(y < 0)[0]
    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)

    half = num_clients // 2
    pos_chunks = np.array_split(pos_idx, half)
    neg_chunks = np.array_split(neg_idx, num_clients - half)
    return list(pos_chunks) + list(neg_chunks)


def partition_dirichlet(y: np.ndarray, num_clients: int,
                         alpha: float = 0.3, seed: int = 0):
    """
    Dirichlet(alpha) partition — the standard non-IID benchmark used in
    FedAvg / FedProx / SCAFFOLD and most modern FL papers (incl. FDR-SVM).
    Smaller alpha => more heterogeneous. alpha=0.3 is "moderately non-IID",
    alpha=0.05 is "highly non-IID", alpha->inf approaches IID.
    """
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    client_idx = [[] for _ in range(num_clients)]

    for c in classes:
        idx_c = np.where(y == c)[0]
        rng.shuffle(idx_c)
        # Dirichlet draws — proportions for this class across clients
        proportions = rng.dirichlet([alpha] * num_clients)
        # cumulative split points
        cuts = (np.cumsum(proportions) * len(idx_c)).astype(int)[:-1]
        splits = np.split(idx_c, cuts)
        for i, s in enumerate(splits):
            client_idx[i].extend(s.tolist())

    return [np.array(ci, dtype=np.int64) for ci in client_idx]


def partition(y, num_clients, scheme="iid", **kwargs):
    """Dispatcher. scheme in {'iid', 'label_skew', 'dirichlet'}."""
    if scheme == "iid":
        return partition_iid(y, num_clients, **kwargs)
    elif scheme == "label_skew":
        return partition_label_skew(y, num_clients, **kwargs)
    elif scheme == "dirichlet":
        return partition_dirichlet(y, num_clients, **kwargs)
    else:
        raise ValueError(f"unknown scheme: {scheme}")


def summarize_partition(y, client_idx):
    """Print a small report — useful for verifying non-IID-ness."""
    print(f"{'client':>8} {'n':>10} {'n_pos':>10} {'n_neg':>10} {'pos_frac':>10}")
    for i, ci in enumerate(client_idx):
        yi = y[ci]
        n_pos = int((yi > 0).sum())
        n_neg = int((yi < 0).sum())
        n = len(ci)
        frac = n_pos / max(n, 1)
        print(f"{i:>8} {n:>10} {n_pos:>10} {n_neg:>10} {frac:>10.3f}")


# ======================================================================
# Quick sanity check
# ======================================================================
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--demo", action="store_true",
                    help="Load covtype (smallest to load fast) and run demo partitions")
    args = ap.parse_args()

    if args.download:
        download_all(args.data_dir)

    if args.demo:
        print("\n=== Loading covtype ===")
        X, y = load_dataset("covtype", data_dir=args.data_dir)
        print(f"X: {X.shape} ({type(X).__name__}, nnz={X.nnz})")
        print(f"y: {y.shape}, +1: {(y>0).sum()}, -1: {(y<0).sum()}")

        print("\n=== IID partition (10 clients) ===")
        ci = partition(y, 10, "iid")
        summarize_partition(y, ci)

        print("\n=== Dirichlet alpha=0.3 (10 clients) ===")
        ci = partition(y, 10, "dirichlet", alpha=0.3)
        summarize_partition(y, ci)

        print("\n=== Label skew (10 clients, 1 class each) ===")
        ci = partition(y, 10, "label_skew", classes_per_client=1)
        summarize_partition(y, ci)