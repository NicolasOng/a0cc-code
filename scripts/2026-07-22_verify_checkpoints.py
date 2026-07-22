"""Verify (and optionally delete) truncated model checkpoints after a crash.

A cluster crash can kill a train job mid-`save_model` (a plain, non-atomic
pickle.dump), leaving a truncated model_<i>.pkl. On resume,
a0.train.alphazero picks the HIGHEST-numbered checkpoint and pickle.load
dies on it — and in auto-resubmit mode the whole continuation chain burns
through MAX_CHAIN failed jobs without ever training. Deleting the truncated
file makes resume fall back to the previous checkpoint, costing one
iteration.

Stdlib only — no jax/flax needed — so it runs on the login node without a
venv. Integrity is checked by walking the pickle opcode stream with
pickletools.genops: a truncated file raises before (or never reaches) the
STOP opcode. This catches truncation, which is the crash failure mode; it
does not re-verify array contents byte-for-byte.

Only the highest-numbered checkpoint per trial is at risk (earlier ones were
written and closed long before the crash), but we scan downward until the
first good one, deleting any trailing bad files, just in case.

Usage:
    # report only (default)
    python3 scripts/2026-07-22_verify_checkpoints.py sweep-output/tdint25-6 \
        sweep-output/td_lambda16-3 sweep-output/td_lambda25-6

    # actually delete the corrupt trailing checkpoints
    python3 scripts/2026-07-22_verify_checkpoints.py --delete sweep-output/tdint25-6 ...

    # check every checkpoint, not just the trailing ones (slower)
    python3 scripts/2026-07-22_verify_checkpoints.py --all sweep-output/tdint25-6 ...
"""
import argparse
import os
import pickletools
import re
import sys

MODEL_RE = re.compile(r"^model_(\d+)\.pkl$")


def pickle_stream_ok(path: str) -> bool:
    """True if `path` contains a structurally complete pickle stream."""
    try:
        with open(path, "rb") as f:
            last_op = None
            for opcode, _arg, _pos in pickletools.genops(f):
                last_op = opcode.name
        return last_op == "STOP"
    except Exception:
        return False


def find_checkpoint_dirs(roots: list[str]) -> list[str]:
    """All directories under `roots` that contain model_<i>.pkl files."""
    dirs = []
    for root in roots:
        if not os.path.isdir(root):
            print(f"WARNING: not a directory, skipping: {root}", file=sys.stderr)
            continue
        for dirpath, _dirnames, filenames in os.walk(root):
            if any(MODEL_RE.match(f) for f in filenames):
                dirs.append(dirpath)
    return sorted(dirs)


def checkpoints_desc(dirpath: str) -> list[tuple[int, str]]:
    """(iteration, path) pairs in `dirpath`, highest iteration first."""
    ckpts = []
    for f in os.listdir(dirpath):
        m = MODEL_RE.match(f)
        if m:
            ckpts.append((int(m.group(1)), os.path.join(dirpath, f)))
    return sorted(ckpts, reverse=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("roots", nargs="+",
                        help="sweep dirs (or any parent of trial training dirs) to scan")
    parser.add_argument("--delete", action="store_true",
                        help="delete corrupt checkpoints (default: report only)")
    parser.add_argument("--all", action="store_true",
                        help="verify every checkpoint in each dir, not just the "
                             "trailing run of highest-numbered ones")
    args = parser.parse_args()

    ckpt_dirs = find_checkpoint_dirs(args.roots)
    print(f"Scanning {len(ckpt_dirs)} checkpoint dir(s)...")

    n_checked = 0
    bad: list[str] = []
    for dirpath in ckpt_dirs:
        for iteration, path in checkpoints_desc(dirpath):
            n_checked += 1
            if pickle_stream_ok(path):
                if not args.all:
                    break  # highest checkpoint is good -> resume is safe
            else:
                bad.append(path)
                size = os.path.getsize(path)
                if args.delete:
                    os.remove(path)
                    print(f"  DELETED corrupt checkpoint: {path} ({size} bytes, iter {iteration})")
                else:
                    print(f"  CORRUPT: {path} ({size} bytes, iter {iteration})")
                # keep scanning downward until we find a good checkpoint

    print(f"\nChecked {n_checked} file(s) across {len(ckpt_dirs)} dir(s): "
          f"{len(bad)} corrupt.")
    if bad and not args.delete:
        print("Re-run with --delete to remove them so training resumes from "
              "the previous checkpoint.")
    sys.exit(1 if (bad and not args.delete) else 0)


if __name__ == "__main__":
    main()
