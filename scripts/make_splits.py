"""
Create frozen train/val/test file splits for MDC2025-002.

Splits at file level (not event level), so all events within a file
stay in the same split. This guarantees per-disk graphs from the same
event are never leaked across splits.

Deterministic: seeded shuffle, reproducible output.

Usage:
    source setup_env.sh
    python3 scripts/make_splits.py
"""

import glob
import sys
from pathlib import Path

import numpy as np


SEED = 42
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
# TEST_FRAC = 1 - TRAIN_FRAC - VAL_FRAC = 0.15

DATA_PATTERN = "/pnfs/mu2e/tape/phy-nts/nts/mu2e/FlateMinusMix1BBTriggered/MDC2025-002/root/*/*/*.root"
SPLITS_DIR = Path(__file__).resolve().parents[1] / "splits"


def main():
    files = sorted(glob.glob(DATA_PATTERN))
    if not files:
        print("ERROR: No MDC2025-002 files found. Check path.", file=sys.stderr)
        sys.exit(1)

    n = len(files)
    print(f"Found {n} files")

    # Deterministic shuffle
    rng = np.random.default_rng(SEED)
    indices = rng.permutation(n)

    n_train = int(n * TRAIN_FRAC)
    n_val = int(n * VAL_FRAC)
    # Rest goes to test
    n_test = n - n_train - n_val

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    train_files = [files[i] for i in sorted(train_idx)]
    val_files = [files[i] for i in sorted(val_idx)]
    test_files = [files[i] for i in sorted(test_idx)]

    print(f"Split: {len(train_files)} train / {len(val_files)} val / {len(test_files)} test")

    # Verify no overlap
    all_split = set(train_files) | set(val_files) | set(test_files)
    assert len(all_split) == n, "File overlap detected!"
    assert len(set(train_files) & set(val_files)) == 0
    assert len(set(train_files) & set(test_files)) == 0
    assert len(set(val_files) & set(test_files)) == 0

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    for name, flist in [("train", train_files), ("val", val_files), ("test", test_files)]:
        path = SPLITS_DIR / f"{name}_files.txt"
        with open(path, "w") as f:
            for fp in flist:
                f.write(fp + "\n")
        print(f"  {path.name}: {len(flist)} files")

    print("\nDone. Split files are FROZEN — do not re-run.")


if __name__ == "__main__":
    main()
