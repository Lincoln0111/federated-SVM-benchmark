"""Download SDCA benchmark datasets into the data/ directory.

Downloads rcv1 (train + test) and covtype from the LIBSVM repository.
Skips any file that already exists and has non-zero size.

Usage:
    python scripts/download_data.py
    python scripts/download_data.py --data-dir ./data
    python scripts/download_data.py --data-dir /path/to/data
"""
import argparse
import sys
from pathlib import Path

# Allow running from the repo root or from scripts/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmark import download_all  # noqa: E402


def main():
    ap = argparse.ArgumentParser(
        description="Download SDCA benchmark datasets (rcv1, covtype) from LIBSVM."
    )
    ap.add_argument(
        "--data-dir",
        default="./data",
        help="Directory to save datasets into (default: ./data)",
    )
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    print(f"Downloading datasets to: {data_dir.resolve()}")
    print("Skipping files that already exist.\n")

    download_all(str(data_dir), include_astro_ph=False)

    print("\nDone. Expected files in data/:")
    expected = [
        "rcv1_train.binary.bz2",
        "rcv1_test.binary.bz2",
        "covtype.libsvm.binary.scale.bz2",
    ]
    for fname in expected:
        path = data_dir / fname
        if path.exists():
            print(f"  [OK]     {fname}  ({path.stat().st_size / 1e6:.1f} MB)")
        else:
            print(f"  [MISSING] {fname}")
    print(
        "\nIf any file is missing and automatic download failed, download it manually from:"
        "\n  https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/"
        "\nand place it in the data/ directory."
    )


if __name__ == "__main__":
    main()
