"""
Test whether weighted_buckets balancing causes value-head collapse in real
alphazero training, and whether the effect comes from the weight VALUES or
just the code path (self.weights being non-None).

Three conditions, 8 trials each:
  A) dataset_balance_method = "none"
  B) dataset_balance_method = "weighted_buckets"
  C) dataset_balance_method = "weighted_buckets_ones"
     (compute weights then force to 1.0 — isolates code path from values)

Uses real alphazero self-play + training with td_lambda=1.0 on a 6x6 board.
training_iterations=2 (collapse happens at iter 1).
max_collapse_retries=0 (one attempt per trial — we just measure collapse rate).

Usage:
    python scripts/2026-05-27_balance_collapse_a0.py
"""
import json
import os
import subprocess
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_CONFIG = os.path.join(REPO_ROOT, "config", "config36-6.json")
OUTPUT_BASE = os.path.join(REPO_ROOT, "sweep-output", "balance_collapse_a0")
PYTHON = sys.executable

NUM_TRIALS = 8
CONDITIONS = ["none", "weighted_buckets", "weighted_buckets_ones"]

OVERRIDES = {
    "td_lambda": 1.0,
    "alternative_target": "td_lambda",
    "training_iterations": 2,
    "max_collapse_retries": 0,
    "do_player_eval": False,
    "do_gt_evals": False,
    "num_workers": 32,
}


def make_config(condition: str) -> str:
    with open(BASE_CONFIG) as f:
        cfg = json.load(f)
    cfg.update(OVERRIDES)
    cfg["dataset_balance_method"] = condition
    cfg["output_dir"] = os.path.join(OUTPUT_BASE, condition) + "/"

    config_dir = os.path.join(OUTPUT_BASE, "configs")
    os.makedirs(config_dir, exist_ok=True)
    config_path = os.path.join(config_dir, f"config_{condition}.json")
    with open(config_path, "w") as f:
        json.dump(cfg, f, indent=2)
    return config_path


def run_trial(config_path: str, trial: int) -> dict:
    cmd = [PYTHON, "-m", "a0.train.alphazero", config_path, str(trial)]
    start = time.time()
    proc = subprocess.run(cmd, cwd=REPO_ROOT)
    elapsed = time.time() - start

    # read retry log
    with open(config_path) as f:
        cfg = json.load(f)
    retry_log = os.path.join(cfg["output_dir"], f"trial_{trial}", "logs", "retry_log.jsonl")
    collapsed = None
    if os.path.exists(retry_log):
        with open(retry_log) as f:
            entries = [json.loads(line) for line in f]
        if entries:
            last = entries[-1]
            collapsed = last["outcome"] != "succeeded"

    return {
        "trial": trial,
        "returncode": proc.returncode,
        "elapsed": round(elapsed, 1),
        "collapsed": collapsed,
    }


def main():
    os.makedirs(OUTPUT_BASE, exist_ok=True)
    all_results = {}

    for condition in CONDITIONS:
        print(f"\n{'='*60}", flush=True)
        print(f"CONDITION: {condition}", flush=True)
        print(f"{'='*60}", flush=True)

        config_path = make_config(condition)
        results = []

        for trial in range(1, NUM_TRIALS + 1):
            print(f"  Trial {trial}/{NUM_TRIALS} ...", end=" ", flush=True)
            result = run_trial(config_path, trial)
            results.append(result)
            status = "COLLAPSED" if result["collapsed"] else "ok"
            print(f"{status} ({result['elapsed']:.0f}s)", flush=True)

        n_collapsed = sum(1 for r in results if r["collapsed"])
        print(f"\n  {condition}: {n_collapsed}/{NUM_TRIALS} collapsed ({n_collapsed/NUM_TRIALS:.0%})", flush=True)
        all_results[condition] = results

    # save results
    results_path = os.path.join(OUTPUT_BASE, "results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # summary
    print(f"\n{'='*60}", flush=True)
    print("SUMMARY", flush=True)
    print(f"{'='*60}", flush=True)
    for condition in CONDITIONS:
        trials = all_results[condition]
        n_collapsed = sum(1 for r in trials if r["collapsed"])
        print(f"  {condition:30s}: {n_collapsed}/{len(trials)} collapsed ({n_collapsed/len(trials):.0%})", flush=True)
    print(f"\nResults saved to {results_path}", flush=True)


if __name__ == "__main__":
    main()
