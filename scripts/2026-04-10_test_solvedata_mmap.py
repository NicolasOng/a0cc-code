"""
Test that SolveData produces identical results across modes.
Outputs text files (one per mode) that can be diff'd.

Usage:
    python scripts/test_solvedata_mmap.py <solve_data_file> [num_samples] [--mode normal|mmap|both]

    num_samples: number of random indices to check (default: all entries)
    --mode:      which mode(s) to output (default: both)

Output:
    test_solvedata_normal.txt  (when mode is normal or both)
    test_solvedata_mmap.txt    (when mode is mmap or both)
"""

import sys
import os
import argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from cc.solvedata import SolveData


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("filename")
    parser.add_argument("num_samples", nargs="?", type=int, default=None)
    parser.add_argument("--mode", choices=["normal", "mmap", "both"], default="both")
    args = parser.parse_args()

    run_normal = args.mode in ("normal", "both")
    run_mmap = args.mode in ("mmap", "both")

    if run_normal:
        sd_normal = SolveData(args.filename)
    if run_mmap:
        sd_mmap = SolveData(args.filename, mmap=True)

    ref = sd_normal if run_normal else sd_mmap
    entries = ref.entries

    if args.num_samples is not None and args.num_samples < entries:
        rng = np.random.default_rng(42)
        indices = sorted(rng.choice(entries, size=args.num_samples, replace=False))
    else:
        indices = range(entries)

    f_normal = open("test_solvedata_normal.txt", "w") if run_normal else None
    f_mmap = open("test_solvedata_mmap.txt", "w") if run_mmap else None

    try:
        for i in indices:
            if f_normal:
                f_normal.write(f"{i} {sd_normal.get(int(i))}\n")
            if f_mmap:
                f_mmap.write(f"{i} {sd_mmap.get(int(i))}\n")
    finally:
        if f_normal:
            f_normal.close()
        if f_mmap:
            f_mmap.close()

    count = len(indices) if args.num_samples else entries
    files = []
    if run_normal:
        files.append("test_solvedata_normal.txt")
    if run_mmap:
        files.append("test_solvedata_mmap.txt")
    print(f"Checked {count} entries.")
    print(f"Output: {', '.join(files)}")


if __name__ == "__main__":
    main()
