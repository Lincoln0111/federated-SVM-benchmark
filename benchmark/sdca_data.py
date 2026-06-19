from __future__ import annotations

import ssl
import urllib.request
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

_SSL_CTX = ssl.create_default_context()
_SSL_CTX.check_hostname = False
_SSL_CTX.verify_mode = ssl.CERT_NONE

_LIBSVM_BASE = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary"

_URLS = {
    "astro-ph": [
        f"{_LIBSVM_BASE}/astro-ph.bz2",
        "http://download.joachims.org/svm_perf/examples/example3.tar.gz",
    ],
    "real-sim": [f"{_LIBSVM_BASE}/real-sim.bz2"],
    "ccat_train": [f"{_LIBSVM_BASE}/rcv1_train.binary.bz2"],
    "ccat_test": [f"{_LIBSVM_BASE}/rcv1_test.binary.bz2"],
    "covtype": [
        f"{_LIBSVM_BASE}/covtype.libsvm.binary.bz2",
        f"{_LIBSVM_BASE}/covtype.libsvm.binary.scale.bz2",
    ],
}


def _download(urls: list[str], dest: Path, retries: int = 3, timeout_s: int = 20) -> str:
    if dest.exists() and dest.stat().st_size > 0:
        return "cache"

    dest.parent.mkdir(parents=True, exist_ok=True)
    last_err = None
    for url in urls:
        for attempt in range(1, retries + 1):
            tmp = dest.with_suffix(dest.suffix + ".tmp")
            try:
                print(f"[download] {url} (attempt {attempt}/{retries})")
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, context=_SSL_CTX, timeout=timeout_s) as resp, open(tmp, "wb") as f:
                    while True:
                        chunk = resp.read(1 << 20)
                        if not chunk:
                            break
                        f.write(chunk)
                tmp.rename(dest)
                print(f"[done] {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")
                return url
            except Exception as e:
                last_err = e
                if tmp.exists():
                    tmp.unlink()
                if attempt == retries:
                    break
    raise RuntimeError(f"Failed to download {dest.name}: {last_err}")


def _norm_sparse(X_sp):
    scale = np.sqrt(max(X_sp.shape[1], 1))
    return (X_sp / scale).tocsr()


def _svd_project(X_train_sp, X_test_sp, random_state: int, n_components: int = 256):
    max_c = min(X_train_sp.shape[1] - 1, X_train_sp.shape[0] - 1, n_components)
    max_c = max(max_c, 8)
    svd = TruncatedSVD(n_components=max_c, random_state=random_state)
    X_train = svd.fit_transform(X_train_sp).astype(np.float32)
    X_test = svd.transform(X_test_sp).astype(np.float32)
    return X_train, X_test


def _map_binary_labels(y: np.ndarray) -> np.ndarray:
    uniq = np.unique(y)
    if len(uniq) != 2:
        raise ValueError(f"Expected binary labels, got {uniq}")
    if np.array_equal(uniq, np.array([-1, 1])):
        return y.astype(np.int8)
    # covtype mapping requirement: 1 -> +1, 2 -> -1
    if np.array_equal(uniq, np.array([1, 2])):
        return np.where(y == 1, 1, -1).astype(np.int8)
    lo, hi = uniq[0], uniq[1]
    return np.where(y == hi, 1, -1).astype(np.int8)


def _iid_split(X: np.ndarray, y: np.ndarray, n_workers: int, random_state: int) -> list[tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(random_state)
    idx = np.arange(len(y))
    rng.shuffle(idx)
    chunks = np.array_split(idx, n_workers)
    return [(X[c], y[c]) for c in chunks if len(c) > 0]


def _align_dims(X_train, X_test):
    d = max(X_train.shape[1], X_test.shape[1])
    if X_train.shape[1] != d:
        from scipy.sparse import hstack

        X_train = hstack([X_train, np.zeros((X_train.shape[0], d - X_train.shape[1]))]).tocsr()
    if X_test.shape[1] != d:
        from scipy.sparse import hstack

        X_test = hstack([X_test, np.zeros((X_test.shape[0], d - X_test.shape[1]))]).tocsr()
    return X_train, X_test


def _load_svmlight_capped(path: Path, n_features: int, max_rows: int, random_state: int, read_bytes: int = 120_000_000):
    # Fast path: partial read from very large test file. n_features is required when length is used.
    safe_n_features = max(int(n_features), 50_000)
    X_sp, y = load_svmlight_file(str(path), n_features=safe_n_features, offset=0, length=read_bytes)
    # Align to n_features expected by train split.
    X_sp = X_sp[:, :n_features]
    if len(y) > max_rows:
        rng = np.random.default_rng(random_state)
        sel = rng.choice(len(y), size=max_rows, replace=False)
        sel.sort()
        X_sp = X_sp[sel]
        y = y[sel]
    return X_sp, y


def load_astro_ph(data_dir: str = "data", random_state: int = 42) -> Dict[str, np.ndarray]:
    data_path = Path(data_dir)
    astro_file = data_path / "astro-ph.bz2"
    real_file = data_path / "real-sim.bz2"

    source = "astro-ph"
    if astro_file.exists():
        X_sp, y = load_svmlight_file(str(astro_file))
    elif real_file.exists():
        source = "real-sim"
        print("[fallback] using cached real-sim")
        X_sp, y = load_svmlight_file(str(real_file))
    else:
        try:
            _download(_URLS["astro-ph"], astro_file)
            X_sp, y = load_svmlight_file(str(astro_file))
        except Exception:
            # Fallback requested by user instruction when URL fails.
            source = "real-sim"
            print("[fallback] astro-ph unavailable; switching to real-sim")
            _download(_URLS["real-sim"], real_file)
            X_sp, y = load_svmlight_file(str(real_file))

    y = _map_binary_labels(y)

    X_train_sp, X_test_sp, y_train, y_test = train_test_split(
        X_sp, y, test_size=0.2, random_state=random_state, stratify=y
    )
    X_train_sp = _norm_sparse(X_train_sp)
    X_test_sp = _norm_sparse(X_test_sp)
    X_train, X_test = _svd_project(X_train_sp, X_test_sp, random_state=random_state, n_components=256)
    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "source": source,
    }


def load_ccat(
    data_dir: str = "data",
    random_state: int = 42,
) -> Dict[str, np.ndarray]:
    data_path = Path(data_dir)
    tr_file = data_path / "rcv1_train.binary.bz2"
    te_file = data_path / "rcv1_test.binary.bz2"
    _download(_URLS["ccat_train"], tr_file)
    _download(_URLS["ccat_test"], te_file)

    print("[ccat] loading sparse train/test (full)")
    X_train_sp, y_train = load_svmlight_file(str(tr_file))
    X_test_sp, y_test = load_svmlight_file(str(te_file))

    X_train_sp, X_test_sp = _align_dims(X_train_sp, X_test_sp)

    y_train = _map_binary_labels(y_train)
    y_test = _map_binary_labels(y_test)

    # OOM protection: dense conversion of n*d floats > 8 GB → subsample train to 200k
    dense_bytes = X_train_sp.shape[0] * X_train_sp.shape[1] * 8
    if dense_bytes > 8 * (1 << 30):
        max_safe = 200_000
        print(f"[ccat] CCAT too large for dense ({dense_bytes / 1e9:.1f} GB), subsampling train to {max_safe}")
        rng = np.random.default_rng(random_state)
        sel = rng.choice(len(y_train), size=min(max_safe, len(y_train)), replace=False)
        sel.sort()
        X_train_sp = X_train_sp[sel]
        y_train = y_train[sel]

    print("[ccat] sparse normalize + SVD projection")
    X_train_sp = _norm_sparse(X_train_sp)
    X_test_sp = _norm_sparse(X_test_sp)
    X_train, X_test = _svd_project(X_train_sp, X_test_sp, random_state=random_state, n_components=256)

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "source": "ccat",
    }


def load_covtype(
    data_dir: str = "data",
    random_state: int = 42,
) -> Dict[str, np.ndarray]:
    data_path = Path(data_dir)
    cov_file = data_path / "covtype.libsvm.binary.bz2"
    cov_file_scaled = data_path / "covtype.libsvm.binary.scale.bz2"
    if cov_file.exists():
        selected = cov_file
    elif cov_file_scaled.exists():
        selected = cov_file_scaled
    else:
        used = _download(_URLS["covtype"], cov_file)
        if used.endswith("covtype.libsvm.binary.scale.bz2") and not cov_file.exists():
            # Keep filename consistent with downloaded source when mirror serves the scaled variant.
            alt = data_path / "covtype.libsvm.binary.scale.bz2"
            if cov_file.exists():
                cov_file.rename(alt)
            selected = alt
        else:
            selected = cov_file

    X_sp, y = load_svmlight_file(str(selected))
    y = _map_binary_labels(y)

    X_train_sp, X_test_sp, y_train, y_test = train_test_split(
        X_sp, y, test_size=0.2, random_state=random_state, stratify=y
    )

    X_train = X_train_sp.toarray().astype(np.float32)
    X_test = X_test_sp.toarray().astype(np.float32)
    scaler = StandardScaler(with_mean=True, with_std=True)
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)
    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "source": "covtype",
    }


def load_dataset(name: str, data_dir: str = "data", random_state: int = 42) -> Dict[str, np.ndarray]:
    if name in ("astro-ph", "real-sim"):
        return load_astro_ph(data_dir=data_dir, random_state=random_state)
    if name == "ccat":
        return load_ccat(data_dir=data_dir, random_state=random_state)
    if name == "covtype":
        return load_covtype(data_dir=data_dir, random_state=random_state)
    raise ValueError(f"Unknown dataset: {name}")


def iid_partitions(X_train: np.ndarray, y_train: np.ndarray, n_workers: int = 3, random_state: int = 42):
    return _iid_split(X_train, y_train, n_workers=n_workers, random_state=random_state)
