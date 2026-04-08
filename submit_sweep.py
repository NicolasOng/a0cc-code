"""
Sweep orchestrator.

Reads a sweep spec (JSON), expands the HP grid (or list of points), writes one
config.json per HP point under output/<sweep>/<hp_id>/, builds tasks.txt and
hp_list.txt, and submits the train + combine + sweep-aggregate slurm jobs with
dependencies wired so the whole pipeline runs hands-off.

Stdlib only — no third-party imports — so it can run on the login node without
a venv.

Usage:
    python3 submit_sweep.py sweeps/example.json
    python3 submit_sweep.py sweeps/example.json --dry-run

Sweep spec format (see sweeps/example.json):
    {
        "name": "lr_lambda_sweep",
        "base_config": "config/config.json",
        "num_trials": 8,
        "grid": {
            "learning_rate": [1e-5, 5e-5, 1e-4],
            "td_lambda": [0.0, 0.5, 0.9]
        }
    }
or, for explicit points instead of a grid:
    {
        "name": "specific_points",
        "base_config": "config/config.json",
        "num_trials": 8,
        "points": [
            {"learning_rate": 1e-4, "td_lambda": 0.0},
            {"learning_rate": 5e-5, "td_lambda": 0.9}
        ]
    }
"""

import argparse
import hashlib
import itertools
import json
import os
import subprocess
import sys


SWEEP_ROOT = "output"
TRAIN_SCRIPT = "train_a0_gpu.sh"
COMBINE_SCRIPT = "combine_a0.sh"
AGGREGATE_SCRIPT = "aggregate_sweep.sh"  # built in Stage C/D


def load_sweep_spec(path: str) -> dict:
    with open(path) as f:
        spec = json.load(f)
    if "name" not in spec:
        raise ValueError("sweep spec missing 'name'")
    if "base_config" not in spec:
        raise ValueError("sweep spec missing 'base_config'")
    if "num_trials" not in spec:
        raise ValueError("sweep spec missing 'num_trials'")
    if ("grid" in spec) == ("points" in spec):
        raise ValueError("sweep spec must have exactly one of 'grid' or 'points'")
    return spec


def expand_points(spec: dict) -> list[dict]:
    """Return a list of HP-override dicts, one per point in the sweep."""
    if "points" in spec:
        return list(spec["points"])
    grid = spec["grid"]
    keys = list(grid.keys())
    value_lists = [grid[k] for k in keys]
    points = []
    for combo in itertools.product(*value_lists):
        points.append({k: v for k, v in zip(keys, combo)})
    return points


def hp_id_for(overrides: dict) -> str:
    """Stable short hash of an HP override dict — used as a directory name."""
    canonical = json.dumps(overrides, sort_keys=True)
    return hashlib.sha1(canonical.encode()).hexdigest()[:10]


def write_per_hp_config(base_config_path: str, overrides: dict, sweep_dir: str, hp_id: str) -> str:
    """Materialize a per-HP config.json that the existing config loader can consume.

    The output_dir is set to the per-HP directory, so the existing
    `config.output_dir.rstrip("/") + "/trial_" + str(trial_num) + "/"` logic
    will resolve trials to <sweep_dir>/<hp_id>/trial_<N>/.
    """
    with open(base_config_path) as f:
        cfg = json.load(f)
    cfg.update(overrides)
    hp_dir = os.path.join(sweep_dir, hp_id)
    os.makedirs(hp_dir, exist_ok=True)
    cfg["output_dir"] = hp_dir.rstrip("/") + "/"
    cfg["_sweep_overrides"] = overrides  # so we can recover what was swept
    config_path = os.path.join(hp_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(cfg, f, indent=2)
    return config_path


def write_tasks_file(tasks_file: str, hp_config_paths: list[str], num_trials: int) -> int:
    """Write the (config_path, trial_no) lines that the train slurm script reads."""
    n = 0
    with open(tasks_file, "w") as f:
        for cfg in hp_config_paths:
            for trial in range(1, num_trials + 1):
                f.write(f"{cfg} {trial}\n")
                n += 1
    return n


def write_hp_list_file(hp_list_file: str, hp_config_paths: list[str]) -> int:
    """Write one config-path per line for the combine job array."""
    with open(hp_list_file, "w") as f:
        for cfg in hp_config_paths:
            f.write(f"{cfg}\n")
    return len(hp_config_paths)


def sbatch(args: list[str], dry_run: bool) -> str | None:
    """Run sbatch with the given args, return the parsed job id (or None if dry-run)."""
    cmd = ["sbatch"] + args
    print("  $", " ".join(cmd))
    if dry_run:
        return None
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    out = result.stdout.strip()
    # sbatch prints "Submitted batch job <jobid>"
    job_id = out.split()[-1]
    print(f"    -> job {job_id}")
    return job_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sweep_spec")
    parser.add_argument("--dry-run", action="store_true",
                        help="materialize files but don't actually call sbatch")
    args = parser.parse_args()

    spec = load_sweep_spec(args.sweep_spec)
    sweep_name = spec["name"]
    base_config = spec["base_config"]
    num_trials = spec["num_trials"]

    points = expand_points(spec)
    print(f"Sweep '{sweep_name}': {len(points)} HP point(s) x {num_trials} trial(s) = {len(points) * num_trials} train task(s)")

    sweep_dir = os.path.join(SWEEP_ROOT, sweep_name)
    os.makedirs(sweep_dir, exist_ok=True)

    # 1. Materialize per-HP configs.
    hp_config_paths: list[str] = []
    for overrides in points:
        hp_id = hp_id_for(overrides)
        cfg_path = write_per_hp_config(base_config, overrides, sweep_dir, hp_id)
        hp_config_paths.append(cfg_path)
        print(f"  {hp_id}: {overrides}  ->  {cfg_path}")

    # 2. Write tasks.txt and hp_list.txt.
    tasks_file = os.path.join(sweep_dir, "tasks.txt")
    hp_list_file = os.path.join(sweep_dir, "hp_list.txt")
    n_tasks = write_tasks_file(tasks_file, hp_config_paths, num_trials)
    n_hps = write_hp_list_file(hp_list_file, hp_config_paths)
    print(f"Wrote {tasks_file} ({n_tasks} lines)")
    print(f"Wrote {hp_list_file} ({n_hps} lines)")

    # 3. Save a copy of the sweep spec next to the outputs for reproducibility.
    with open(os.path.join(sweep_dir, "sweep.json"), "w") as f:
        json.dump(spec, f, indent=2)

    # 4. Submit train array.
    print(f"\nSubmitting train array ({n_tasks} tasks)...")
    train_job = sbatch(
        [f"--array=1-{n_tasks}", TRAIN_SCRIPT, tasks_file],
        args.dry_run,
    )

    # 5. Submit combine array, dependent on train.
    print(f"\nSubmitting combine array ({n_hps} tasks)...")
    combine_dep_args = []
    if train_job is not None:
        combine_dep_args = [f"--dependency=afterany:{train_job}"]
    combine_job = sbatch(
        [f"--array=1-{n_hps}"] + combine_dep_args + [COMBINE_SCRIPT, hp_list_file],
        args.dry_run,
    )

    # 6. Submit sweep aggregate (Stage C/D).
    if os.path.exists(AGGREGATE_SCRIPT):
        print(f"\nSubmitting sweep aggregate...")
        aggregate_dep_args = []
        if combine_job is not None:
            aggregate_dep_args = [f"--dependency=afterany:{combine_job}"]
        sbatch(
            aggregate_dep_args + [AGGREGATE_SCRIPT, sweep_dir],
            args.dry_run,
        )
    else:
        print(f"\n(skipping sweep-aggregate submit: {AGGREGATE_SCRIPT} not found yet — added in Stage C/D)")

    print(f"\nDone. Sweep directory: {sweep_dir}")


if __name__ == "__main__":
    main()
