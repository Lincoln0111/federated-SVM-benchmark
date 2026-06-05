"""Shared SDCA benchmark dataset loaders for federated SVM experiments.

Provides:
    load_astro_ph(data_dir, n_svd_components, random_state)
    load_ccat(data_dir, n_svd_components, max_train_samples, random_state)
    load_covtype(data_dir, test_size, random_state)

All return (X_train, y_train, X_val, y_val, X_test, y_test) as dense float32
numpy arrays with labels in {-1.0, +1.0}.

High-dimensional sparse datasets (astro-ph: 99k dims, CCAT/rcv1: 47k dims) are
projected to n_svd_components dimensions via TruncatedSVD before returning dense
arrays, making them compatible with RBF-kernel methods that expect dense inputs.

URLs:
  astro-ph  : http://download.joachims.org/svm_perf/examples/example3.tar.gz
  rcv1/CCAT : https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/
  covtype   : https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/
              covtype.libsvm.binary.scale.bz2
"""
from __future__ import annotations

import ssl
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ── SSL context (bypasses corporate cert chains) ─────────────────────────────
_SSL_CTX = ssl.create_default_context()
_SSL_CTX.check_hostname = False
_SSL_CTX.verify_mode = ssl.CERT_NONE

# ── Download URLs ─────────────────────────────────────────────────────────────
_LIBSVM_BASE = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary"
# astro-ph: original Joachims URL is dead; try LIBSVM mirror, then fail gracefully
_ASTRO_PH_URLS = [
    f"{_LIBSVM_BASE}/astro-ph.bz2",
    "http://download.joachims.org/svm_perf/examples/example3.tar.gz",
]
_RCV1_TRAIN_URL = f"{_LIBSVM_BASE}/rcv1_train.binary.bz2"
_RCV1_TEST_URL = f"{_LIBSVM_BASE}/rcv1_test.binary.bz2"
_COVTYPE_URL = f"{_LIBSVM_BASE}/covtype.libsvm.binary.scale.bz2"
# real-sim: 72,309 samples, 20,958 sparse features — used as astro-ph fallback
_REAL_SIM_URL = f"{_LIBSVM_BASE}/real-sim.bz2"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _download(url: str, dest: Path, retries: int = 3) -> None:
    """Download url to dest, skipping if already cached. Retries on error."""
    if dest.exists():
        print(f"[cache] {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, retries + 1):
        tmp = dest.with_suffix(dest.suffix + ".tmp")
        try:
            print(f"[download] {url} (attempt {attempt}/{retries})")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, context=_SSL_CTX) as resp, open(tmp, "wb") as f:
                while True:
                    chunk = resp.read(1 << 20)  # 1 MB chunks
                    if not chunk:
                        break
                    f.write(chunk)
            tmp.rename(dest)
            print(f"[done] {dest.stat().st_size / 1e6:.1f} MB")
            return
        except Exception as e:
            if tmp.exists():
                tmp.unlink()
            if attempt == retries:
                raise
            print(f"[retry] {e}")



def _normalize_labels(y: np.ndarray) -> np.ndarray:
    """Map raw labels to {-1.0, +1.0}."""
    classes = np.unique(y)
    if len(classes) != 2:
        raise ValueError(f"Expected binary labels, got {classes}")
    lo, hi = classes[0], classes[1]
    if lo == -1 and hi == 1:
        return y.astype(np.float64)
    mid = (lo + hi) / 2.0
    return np.where(y > mid, 1.0, -1.0).astype(np.float64)


def _svd_project(X_train_sp, X_val_sp, X_test_sp, n_components: int, random_state: int):
    """Fit TruncatedSVD on train, project all splits. Returns dense float32 arrays."""
    print(f"[svd] TruncatedSVD(n_components={n_components}) on {X_train_sp.shape} ...")
    svd = TruncatedSVD(n_components=n_components, random_state=random_state)
    X_train = svd.fit_transform(X_train_sp).astype(np.float32)
    X_val = svd.transform(X_val_sp).astype(np.float32)
    X_test = svd.transform(X_test_sp).astype(np.float32)
    print(f"[svd] Explained variance: {svd.explained_variance_ratio_.sum():.3f}")
    return X_train, X_val, X_test


def _download_first_working(urls: list, dest: Path) -> str:
    """Try each url in order; return the one that worked, or raise."""
    if dest.exists():
        print(f"[cache] {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")
        return str(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    last_err = None
    for url in urls:
        try:
            print(f"[download] trying {url}")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, context=_SSL_CTX) as resp, open(dest, "wb") as f:
                f.write(resp.read())
            print(f"[done] {dest.stat().st_size / 1e6:.1f} MB")
            return str(dest)
        except Exception as e:
            print(f"[warn] {url} failed: {e}")
            last_err = e
            if dest.exists():
                dest.unlink()   # remove partial download
    raise RuntimeError(
        f"All download URLs failed for {dest.name}. "
        "Place the file manually in the data/ directory.\n"
        f"Last error: {last_err}"
    )


# ── astro-ph ──────────────────────────────────────────────────────────────────

def load_astro_ph(
    data_dir: str = "data",
    n_svd_components: int = 200,
    random_state: int = 42,
):
    """Load astro-ph astrophysics classification dataset.

    Tries LIBSVM (.bz2) first, then Joachims SVMperf (.tar.gz).
    TruncatedSVD projects sparse features to n_svd_components dims.
    Labels: +1/-1.  Split: 80/10/10 (stratified).
    """
    import scipy.sparse

    data_dir = Path(data_dir)
    bz2_path = data_dir / "astro-ph.bz2"
    tgz_path = data_dir / "svmperf_example3.tar.gz"

    # --- try .bz2 (LIBSVM) format first ---
    libsvm_ok = False
    if bz2_path.exists():
        libsvm_ok = True
    else:
        try:
            _download_first_working([_ASTRO_PH_URLS[0]], bz2_path)
            libsvm_ok = True
        except Exception as e:
            print(f"[warn] LIBSVM astro-ph.bz2 not available: {e}")

    if libsvm_ok:
        print("[load] astro-ph from .bz2 ...")
        X_all_sp, y_all = load_svmlight_file(str(bz2_path))
        y_all = _normalize_labels(y_all)
    else:
        # --- fallback: try tar.gz (Joachims) format ---
        extract_dir = data_dir / "svmperf_example3"
        train_dat = extract_dir / "example3" / "train.dat"
        test_dat = extract_dir / "example3" / "test.dat"

        # Joachims tar.gz fallback
        joachims_ok = False
        try:
            _download_first_working([_ASTRO_PH_URLS[1]], tgz_path)

            if not train_dat.exists():
                print("[extract] svmperf_example3.tar.gz ...")
                extract_dir.mkdir(parents=True, exist_ok=True)
                with tarfile.open(tgz_path, "r:gz") as tf:
                    tf.extractall(extract_dir)

            print("[load] astro-ph train.dat + test.dat ...")
            X_tr_sp, y_tr = load_svmlight_file(str(train_dat))
            X_te_sp, y_te = load_svmlight_file(str(test_dat), n_features=X_tr_sp.shape[1])
            y_tr = _normalize_labels(y_tr)
            y_te = _normalize_labels(y_te)
            X_all_sp = scipy.sparse.vstack([X_tr_sp, X_te_sp])
            y_all = np.concatenate([y_tr, y_te])
            joachims_ok = True
        except Exception as e:
            print(f"[warn] Joachims astro-ph not available: {e}")

        if not joachims_ok:
            # Final fallback: real-sim (72k samples, 20k sparse features)
            # Structural substitute for astro-ph — sparse binary text classification.
            print("[fallback] astro-ph unavailable (both URLs 404). Using real-sim dataset.")
            print("[fallback] real-sim: 72,309 samples, 20,958 features (binary newsgroup text)")
            real_sim_path = data_dir / "real-sim.bz2"
            _download(_REAL_SIM_URL, real_sim_path)
            print("[load] real-sim ...")
            X_all_sp, y_all = load_svmlight_file(str(real_sim_path))
            y_all = _normalize_labels(y_all)

    X_tmp, X_test_sp, y_tmp, y_test = train_test_split(
        X_all_sp, y_all, test_size=0.10, random_state=random_state, stratify=y_all
    )
    X_train_sp, X_val_sp, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=0.10 / 0.90, random_state=random_state, stratify=y_tmp
    )

    X_train, X_val, X_test = _svd_project(
        X_train_sp, X_val_sp, X_test_sp, n_svd_components, random_state
    )

    # Center SVD components so centroids initialize symmetrically around data cloud.
    scaler = StandardScaler(with_mean=True, with_std=True)
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    dataset_label = "astro-ph" if (bz2_path.exists() or tgz_path.exists()) else "real-sim"
    print(
        f"[{dataset_label}] Train: {X_train.shape[0]:,}  Val: {X_val.shape[0]:,}  "
        f"Test: {X_test.shape[0]:,}  Dim: {X_train.shape[1]}"
    )
    return X_train, y_train, X_val, y_val, X_test, y_test


# ── CCAT (rcv1 binary) ────────────────────────────────────────────────────────

def load_ccat(
    data_dir: str = "data",
    n_svd_components: int = 200,
    max_train_samples: int = 100_000,
    random_state: int = 42,
):
    """Load CCAT / rcv1 binary classification dataset from LIBSVM.

    Subsamples training to max_train_samples (default 100k) to keep memory
    and runtime manageable. TruncatedSVD reduces 47k-dim sparse features to
    n_svd_components for dense use.
    Labels: +1/-1.  Test uses the official rcv1 test split (capped at 20k).
    """
    data_dir = Path(data_dir)
    train_path = data_dir / "rcv1_train.binary.bz2"
    test_path = data_dir / "rcv1_test.binary.bz2"

    _download(_RCV1_TRAIN_URL, train_path)
    _download(_RCV1_TEST_URL, test_path)

    print("[load] rcv1 train ...")
    X_train_sp, y_train = load_svmlight_file(str(train_path))
    y_train = _normalize_labels(y_train)

    print("[load] rcv1 test ...")
    X_test_sp, y_test = load_svmlight_file(str(test_path), n_features=X_train_sp.shape[1])
    y_test = _normalize_labels(y_test)

    # Subsample training set if too large
    if X_train_sp.shape[0] > max_train_samples:
        print(f"[subsample] {max_train_samples:,} of {X_train_sp.shape[0]:,} train samples")
        rng = np.random.default_rng(random_state)
        idx = np.sort(rng.choice(X_train_sp.shape[0], size=max_train_samples, replace=False))
        X_train_sp = X_train_sp[idx]
        y_train = y_train[idx]

    # Split train → train + val
    X_tr_sp, X_val_sp, y_tr, y_val = train_test_split(
        X_train_sp, y_train, test_size=0.10, random_state=random_state, stratify=y_train
    )

    # Cap test set
    if X_test_sp.shape[0] > 20_000:
        rng = np.random.default_rng(random_state + 1)
        idx = np.sort(rng.choice(X_test_sp.shape[0], size=20_000, replace=False))
        X_test_sp = X_test_sp[idx]
        y_test = y_test[idx]

    X_train, X_val, X_test = _svd_project(
        X_tr_sp, X_val_sp, X_test_sp, n_svd_components, random_state
    )

    # Center each SVD component so data spans [-x, +x] symmetrically.
    # This ensures centroid initialization from uniform(min, max) covers the
    # actual data cloud instead of only the positive half-space.
    scaler = StandardScaler(with_mean=True, with_std=True)
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    print(
        f"[CCAT/rcv1] Train: {X_train.shape[0]:,}  Val: {X_val.shape[0]:,}  "
        f"Test: {X_test.shape[0]:,}  Dim: {X_train.shape[1]}"
    )
    return X_train, y_tr, X_val, y_val, X_test, y_test


# ── covtype binary ────────────────────────────────────────────────────────────

def load_covtype(
    data_dir: str = "data",
    test_size: float = 0.20,
    max_train_samples: int = 30_000,
    random_state: int = 42,
):
    """Load covtype binary (scaled) dataset from LIBSVM. Dense 54-feature data.

    The file is already LIBSVM-scaled to [-1, 1], so no second StandardScaler
    is applied — this avoids double-normalisation artifacts.

    Labels: +1/-1 (class 2 → +1, class 1 → -1).
    Split: ~72/8/20 stratified; training capped at max_train_samples.
    """
    data_dir = Path(data_dir)
    path = data_dir / "covtype.libsvm.binary.scale.bz2"

    _download(_COVTYPE_URL, path)

    print("[load] covtype ...")
    X_sp, y = load_svmlight_file(str(path))
    y = _normalize_labels(y)

    X = X_sp.toarray().astype(np.float32)

    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    X_train_full, X_val, y_train_full, y_val = train_test_split(
        X_tmp, y_tmp, test_size=0.10, random_state=random_state, stratify=y_tmp
    )

    # Subsample training set to keep IRWLS feasible
    if max_train_samples and X_train_full.shape[0] > max_train_samples:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(X_train_full.shape[0], size=max_train_samples, replace=False)
        idx.sort()
        X_train = X_train_full[idx].astype(np.float32)
        y_train = y_train_full[idx]
        print(f"[subsample] covtype train: {max_train_samples:,} of {X_train_full.shape[0]:,}")
    else:
        X_train = X_train_full.astype(np.float32)
        y_train = y_train_full

    # Data is already in [-1,1] — no second scaler needed
    X_val = X_val.astype(np.float32)
    X_test = X_test.astype(np.float32)

    print(
        f"[covtype] Train: {X_train.shape[0]:,}  Val: {X_val.shape[0]:,}  "
        f"Test: {X_test.shape[0]:,}  Dim: {X_train.shape[1]}"
    )
    return X_train, y_train, X_val, y_val, X_test, y_test
