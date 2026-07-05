"""Print how many training iterations remain for a (config, trial).

Used by the SLURM train wrapper to decide whether to self-resubmit a
continuation job (see slurm/train_a0_gpu.sh). Prints a single integer to
stdout:

    max(0, training_iterations - last_checkpoint_iteration)

0 means training is complete (nothing more to do). Kept dependency-light — no
jax / model imports — so it runs fast in the job preamble and doesn't need the
GPU env.

Usage:
    python -m a0.train.check_progress <config.json> [trial_num]
"""
import os

from config import config


def last_checkpoint_iteration(training_dir: str) -> int:
    """Highest i for which model_<i>.pkl exists in training_dir (0 if none)."""
    if not training_dir or not os.path.isdir(training_dir):
        return 0
    best = 0
    for f in os.listdir(training_dir):
        if f.startswith("model_") and f.endswith(".pkl"):
            try:
                best = max(best, int(f[len("model_"):-len(".pkl")]))
            except ValueError:
                continue
    return best


def main() -> None:
    # Mirror Config.training_dir's path WITHOUT going through the lazy property
    # (which would mkdir it); we only want to read, not create, the directory.
    training_dir = config.output_dir.rstrip("/") + "/training/"
    last = last_checkpoint_iteration(training_dir)
    remaining = max(0, config.training_iterations - last)
    print(remaining)


if __name__ == "__main__":
    main()
