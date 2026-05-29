# BDSVM Option 1 (Threading) – astro-ph Status

## Dataset

astro-ph is a binary classification task from the SDCA paper (Shalev-Shwartz & Zhang, 2013),
originally used in the SVM-Perf benchmark (Joachims, 2006). It classifies astrophysics
abstracts from arXiv as astrophysics (+1) or not (−1).

## Status: Dataset Unavailable

Both official download sources return **HTTP 404** as of the date of this benchmark:

| URL | Status |
|---|---|
| `http://download.joachims.org/svm_perf/examples/example3.tar.gz` | 404 Not Found |
| `https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/astro-ph.bz2` | 404 Not Found |

The `load_astro_ph()` function in `data_utils/sdca_datasets.py` tries both URLs in order
and raises a `RuntimeError` with instructions to place the file manually if neither works.

## Manual Installation (if you have access to the file)

Place either of the following in the `data/` directory:

- `data/astro-ph.bz2` — LIBSVM bz2 format (single file, `load_svmlight_file` compatible)
- `data/svmperf_example3.tar.gz` — SVM-Perf tar.gz containing `train.dat` / `test.dat`

Then re-run:

```bash
python test_bdsvm_fl_threading.py --dataset astro-ph
```

## Alternative Sources

- OpenML dataset #1216: `fetch_openml("astro-ph", version=1)` (may differ slightly from
  the original SDCA benchmark split)
- Kaggle: search for "astro-ph LIBSVM" — community-uploaded copies may be available
