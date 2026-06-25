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
A grid axis may also be a list of dicts instead of scalars — a "coupled" axis
whose named params vary together rather than independently. The grid key is then
just a label. This lets you cross-product a coupled set against other axes
without exploding into the full cartesian product:
    {
        "name": "coupled_example",
        "base_config": "config/config.json",
        "num_trials": 8,
        "grid": {
            "C": [1],
            "AB": [{"A": 1, "B": 1}, {"A": 2, "B": 2}]
        }
    }
expands to just (A=1, B=1, C=1) and (A=2, B=2, C=1) — two points, not four.

Or, for explicit points instead of a grid:
    {
        "name": "specific_points",
        "base_config": "config/config.json",
        "num_trials": 8,
        "points": [
            {"learning_rate": 1e-4, "td_lambda": 0.0},
            {"learning_rate": 5e-5, "td_lambda": 0.9}
        ]
    }
Both `grid` and `points` may be specified together — the grid is expanded and
the explicit points are appended (deduplicated by HP id). Useful for adding a
baseline or one-off configuration alongside a grid without ballooning it into
the cartesian product.

Optional `slurm` block overrides the #SBATCH directives in the .sh scripts via
sbatch CLI flags (which always win over in-script #SBATCH lines). One sub-block
per stage; keys are passed through as `--<key>=<value>` to sbatch:
    {
        ...,
        "slurm": {
            "train":     {"time": "24:00:00", "cpus-per-task": 16, "mem-per-cpu": "4G", "gpus-per-node": 1},
            "combine":   {"time": "1:00:00",  "cpus-per-task": 4,  "mem-per-cpu": "4G"},
            "aggregate": {"time": "30:00",    "cpus-per-task": 2}
        }
    }

Local mode (no slurm) runs the same per-task / per-HP / per-sweep stages on
this machine, sequentially, by shelling out to python directly:
    python3 submit_sweep.py sweeps/foo.json --local
    python3 submit_sweep.py sweeps/foo.json --local --stages a0.eval3.plotting --skip-combine --skip-aggregate
"""

import argparse
import hashlib
import itertools
import json
import os
import subprocess
import sys


SWEEP_ROOT = "sweep-output"
TRAIN_SCRIPT = "slurm/train_a0_gpu.sh"
COMBINE_SCRIPT = "slurm/combine_a0.sh"
AGGREGATE_SCRIPT = "slurm/aggregate_sweep.sh"  # built in Stage C/D

# Default per-stage pipelines for --local mode. Mirror the python -m calls in
# slurm/train_a0_gpu.sh / slurm/combine_a0.sh / slurm/aggregate_sweep.sh. Overridden by --stages,
# --combine-stages, --aggregate-stages.
DEFAULT_TRAIN_STAGES = [
    "a0.train.alphazero",
    "a0.eval3.training_data",
    "a0.eval3.generate_datasets",
    "a0.eval3.dataset_evaluation",
    "a0.eval.player",
    "a0.eval3.plotting",
    "a0.eval3.extract_summary",
]
DEFAULT_COMBINE_STAGES = [
    "a0.eval.combine_merge",
    "a0.eval.combine_plot",
    "a0.eval3.combine",
    "a0.eval.combine_summary",
]
DEFAULT_AGGREGATE_STAGES = [
    "aggregate_sweep.py",
    "plot_sweep.py",
]


def load_sweep_spec(path: str) -> dict:
    with open(path) as f:
        spec = json.load(f)
    if "name" not in spec:
        raise ValueError("sweep spec missing 'name'")
    if "base_config" not in spec:
        raise ValueError("sweep spec missing 'base_config'")
    if "num_trials" not in spec:
        raise ValueError("sweep spec missing 'num_trials'")
    if "grid" not in spec and "points" not in spec:
        raise ValueError("sweep spec must have at least one of 'grid' or 'points'")
    return spec


def expand_points(spec: dict) -> list[dict]:
    """Return a list of HP-override dicts, one per point in the sweep.

    If both 'grid' and 'points' are present, the grid is expanded first and
    the explicit points are appended. Duplicates (same hp_id_for) are dropped,
    keeping the first occurrence.
    """
    points: list[dict] = []
    if "grid" in spec:
        grid = spec["grid"]
        keys = list(grid.keys())
        # Each grid key defines one axis; the cartesian product of all axes is
        # taken. An axis value is a list whose entries are normalized to
        # "settings dicts":
        #   - a scalar v     -> {key: v}      (this single param varies)
        #   - a dict {A:1,..} -> {A:1, ...}   (a *coupled* axis: the named params
        #     vary together and the grid key is just a label)
        # so e.g. grid = {"C": [1], "AB": [{"A":1,"B":1}, {"A":2,"B":2}]} yields
        # the two points {C:1,A:1,B:1} and {C:1,A:2,B:2} rather than all four.
        axes = []
        for k in keys:
            axis = [dict(v) if isinstance(v, dict) else {k: v} for v in grid[k]]
            axes.append(axis)
        for combo in itertools.product(*axes):
            merged: dict = {}
            for settings in combo:
                merged.update(settings)
            points.append(merged)
    if "points" in spec:
        points.extend(spec["points"])
    seen: set[str] = set()
    deduped: list[dict] = []
    for p in points:
        h = hp_id_for(p)
        if h in seen:
            continue
        seen.add(h)
        deduped.append(p)
    return deduped


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


def write_hp_index_file(hp_index_file: str, hp_ids: list[str], points: list[dict[str, object]]) -> None:
    """Write a human-readable hash -> HP overrides mapping for quick lookup.

    Format: one line per HP point, e.g.
        f2cd5e30ca   learning_rate=1e-05
        e3b74957d8   learning_rate=5e-05  td_lambda=0.5
    """
    with open(hp_index_file, "w") as f:
        for hp_id, overrides in zip(hp_ids, points):
            kv = "  ".join(f"{k}={v}" for k, v in overrides.items())
            f.write(f"{hp_id}   {kv}\n")


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


def slurm_flags_for(spec: dict, key: str) -> list[str]:
    """Translate spec['slurm'][key] (a dict of sbatch options) into ['--k=v', ...]
    overrides. sbatch CLI flags supersede in-script #SBATCH directives, so this
    is enough to retune time/cpus/memory/GPU per-sweep without touching the .sh."""
    settings = spec.get("slurm", {}).get(key, {}) or {}
    return [f"--{k}={v}" for k, v in settings.items()]


def stage_argv(stage: str, *positional: str) -> list[str]:
    """Build the argv to invoke a stage. Top-level scripts (e.g. aggregate_sweep.py)
    are run as `python <file>`; everything else as `python -m <module>`."""
    if stage.endswith(".py") and os.path.isfile(stage):
        return [sys.executable, stage, *positional]
    return [sys.executable, "-m", stage, *positional]


def run_stage_local(argv: list[str], dry_run: bool) -> None:
    print("  $", " ".join(argv))
    if dry_run:
        return
    subprocess.run(argv, check=True)


def run_local(
    args: argparse.Namespace,
    hp_config_paths: list[str],
    tasks_file: str,
    sweep_dir: str,
) -> None:
    """Local equivalent of the slurm pipeline: per-task stages, per-HP combine
    stages, and per-sweep aggregate stages — run sequentially in this process."""
    # 1. Per-(config, trial) stages.
    if args.skip_train_eval or not args.stages:
        print(f"\n[local] Skipping train-eval stages.")
    else:
        print(f"\n[local] Running {len(args.stages)} per-task stage(s) on tasks in {tasks_file}...")
        with open(tasks_file) as f:
            task_lines = [ln.strip() for ln in f if ln.strip()]
        for line in task_lines:
            cfg, trial = line.split()
            for stage in args.stages:
                run_stage_local(stage_argv(stage, cfg, trial), args.dry_run)

    # 2. Per-HP combine stages.
    if args.skip_combine or not args.combine_stages:
        print(f"\n[local] Skipping combine stages.")
    else:
        print(f"\n[local] Running {len(args.combine_stages)} combine stage(s) per HP...")
        for cfg in hp_config_paths:
            for stage in args.combine_stages:
                run_stage_local(stage_argv(stage, cfg), args.dry_run)

    # 3. Whole-sweep aggregate stages.
    if args.skip_aggregate or not args.aggregate_stages:
        print(f"\n[local] Skipping aggregate stages.")
    else:
        print(f"\n[local] Running {len(args.aggregate_stages)} aggregate stage(s) on sweep...")
        for stage in args.aggregate_stages:
            run_stage_local(stage_argv(stage, sweep_dir), args.dry_run)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sweep_spec")
    parser.add_argument("--dry-run", action="store_true",
                        help="materialize files but don't actually call sbatch (or python, "
                             "in --local mode) — just print what would run")
    parser.add_argument("--script", default=TRAIN_SCRIPT,
                        help=f"stage-1 .sh to run per task (default: {TRAIN_SCRIPT}). "
                             "Use slurm/eval_a0_gpu.sh to re-eval an existing sweep.")
    parser.add_argument("--reuse-configs", action="store_true",
                        help="skip rewriting per-HP config.json files; use whatever is on disk. "
                             "Required for re-eval-only runs against an already-trained sweep.")
    parser.add_argument("--local", action="store_true",
                        help="run all stages on this machine instead of submitting to slurm. "
                             "Skips the .sh wrappers (which assume cluster setup) and shells "
                             "out to python directly in the current environment.")
    parser.add_argument("--stages", nargs="+", default=DEFAULT_TRAIN_STAGES,
                        metavar="MODULE",
                        help="(--local only) per-(config, trial) stages to run. Each is a python "
                             "module (a0.eval3.plotting) or a top-level script (aggregate_sweep.py). "
                             f"Default: {' '.join(DEFAULT_TRAIN_STAGES)}.")
    parser.add_argument("--combine-stages", nargs="+", default=DEFAULT_COMBINE_STAGES,
                        metavar="MODULE",
                        help="(--local only) per-HP combine stages. "
                             f"Default: {' '.join(DEFAULT_COMBINE_STAGES)}.")
    parser.add_argument("--aggregate-stages", nargs="+", default=DEFAULT_AGGREGATE_STAGES,
                        metavar="MODULE",
                        help="(--local only) per-sweep aggregate stages. "
                             f"Default: {' '.join(DEFAULT_AGGREGATE_STAGES)}.")
    parser.add_argument("--skip-train-eval", action="store_true",
                        help="skip the per-(config, trial) train+eval pipeline. "
                             "Combine and aggregate stages still run (in slurm mode "
                             "they submit without a train dependency). Use with "
                             "--reuse-configs to re-run combine/aggregate over an "
                             "already-trained sweep.")
    parser.add_argument("--skip-combine", action="store_true",
                        help="skip combine stages. In slurm mode aggregate depends "
                             "on train instead (or runs immediately if --skip-train-eval).")
    parser.add_argument("--skip-aggregate", action="store_true",
                        help="skip aggregate stages.")
    args = parser.parse_args()

    spec = load_sweep_spec(args.sweep_spec)
    sweep_name = spec["name"]
    base_config = spec["base_config"]
    num_trials = spec["num_trials"]

    points = expand_points(spec)
    print(f"Sweep '{sweep_name}': {len(points)} HP point(s) x {num_trials} trial(s) = {len(points) * num_trials} train task(s)")

    sweep_dir = os.path.join(SWEEP_ROOT, sweep_name)
    os.makedirs(sweep_dir, exist_ok=True)

    # 1. Materialize per-HP configs (or reuse existing ones if --reuse-configs).
    hp_config_paths: list[str] = []
    hp_ids: list[str] = []
    for overrides in points:
        hp_id = hp_id_for(overrides)
        if args.reuse_configs:
            cfg_path = os.path.join(sweep_dir, hp_id, "config.json")
            if not os.path.exists(cfg_path):
                raise FileNotFoundError(
                    f"--reuse-configs set but no existing config at {cfg_path}. "
                    "Run without --reuse-configs to materialize it."
                )
        else:
            cfg_path = write_per_hp_config(base_config, overrides, sweep_dir, hp_id)
        hp_config_paths.append(cfg_path)
        hp_ids.append(hp_id)
        print(f"  {hp_id}: {overrides}  ->  {cfg_path}")

    # 2. Write tasks.txt, hp_list.txt, and a human-readable hp_index.txt.
    tasks_file = os.path.join(sweep_dir, "tasks.txt")
    hp_list_file = os.path.join(sweep_dir, "hp_list.txt")
    hp_index_file = os.path.join(sweep_dir, "hp_index.txt")
    n_tasks = write_tasks_file(tasks_file, hp_config_paths, num_trials)
    n_hps = write_hp_list_file(hp_list_file, hp_config_paths)
    write_hp_index_file(hp_index_file, hp_ids, points)
    print(f"Wrote {tasks_file} ({n_tasks} lines)")
    print(f"Wrote {hp_list_file} ({n_hps} lines)")
    print(f"Wrote {hp_index_file} ({n_hps} lines)")

    # 3. Save a copy of the sweep spec next to the outputs for reproducibility.
    with open(os.path.join(sweep_dir, "sweep.json"), "w") as f:
        json.dump(spec, f, indent=2)

    # 3c. Local mode short-circuits the slurm submission and runs everything
    # in this process. Skips the .sh wrappers (cluster-only setup) entirely.
    if args.local:
        run_local(args, hp_config_paths, tasks_file, sweep_dir)
        print(f"\nDone. Sweep directory: {sweep_dir}")
        return

    # 3b. Make a slurm logs dir co-located with the sweep so all per-task .out
    # files land here instead of cluttering the working directory.
    slurm_logs = os.path.join(sweep_dir, "slurm_logs")
    os.makedirs(slurm_logs, exist_ok=True)

    # In dry-run we use placeholder job IDs so the printed sbatch commands still
    # show the dependency wiring for stages that were submitted.
    placeholder_train = "<train_jid>"
    placeholder_combine = "<combine_jid>"

    # 4. Submit stage-1 array (train by default; slurm/eval_a0_gpu.sh for re-eval).
    train_job_id: str | None = None
    if args.skip_train_eval:
        print(f"\nSkipping train-eval array.")
    else:
        stage_name = os.path.splitext(os.path.basename(args.script))[0]
        print(f"\nSubmitting {stage_name} array ({n_tasks} tasks)...")
        train_job_id = sbatch(
            [f"--array=1-{n_tasks}",
             f"--output={slurm_logs}/{stage_name}_%A_%a.out",
             *slurm_flags_for(spec, "train"),
             args.script, tasks_file],
            args.dry_run,
        ) or (placeholder_train if args.dry_run else None)

    # 5. Submit combine array, dependent on train if train was submitted.
    combine_job_id: str | None = None
    if args.skip_combine:
        print(f"\nSkipping combine array.")
    else:
        print(f"\nSubmitting combine array ({n_hps} tasks)...")
        dep_flags = [f"--dependency=afterany:{train_job_id}"] if train_job_id else []
        combine_job_id = sbatch(
            [f"--array=1-{n_hps}",
             f"--output={slurm_logs}/combine_%A_%a.out",
             *dep_flags,
             *slurm_flags_for(spec, "combine"),
             COMBINE_SCRIPT, hp_list_file],
            args.dry_run,
        ) or (placeholder_combine if args.dry_run else None)

    # 6. Submit sweep aggregate (Stage C/D). Depends on combine if submitted,
    # else train if submitted, else runs immediately.
    if args.skip_aggregate:
        print(f"\nSkipping sweep aggregate.")
    elif os.path.exists(AGGREGATE_SCRIPT):
        print(f"\nSubmitting sweep aggregate...")
        dep_id = combine_job_id or train_job_id
        dep_flags = [f"--dependency=afterany:{dep_id}"] if dep_id else []
        sbatch(
            [f"--output={slurm_logs}/aggregate_%j.out",
             *dep_flags,
             *slurm_flags_for(spec, "aggregate"),
             AGGREGATE_SCRIPT, sweep_dir],
            args.dry_run,
        )
    else:
        print(f"\n(skipping sweep-aggregate submit: {AGGREGATE_SCRIPT} not found yet — added in Stage C/D)")

    print(f"\nDone. Sweep directory: {sweep_dir}")


if __name__ == "__main__":
    main()
