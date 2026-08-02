"""
Per-HP scalar summary aggregation.

Reads `<config.output_dir>/trial_*/summary.json` (written by extract_summary.py),
computes mean / std / 95% CI for each metric across trials, and writes
`<config.output_dir>/summary.json` (the per-HP summary).

Globs for trial summaries rather than trusting `config.num_trials`, so partial
sweeps still aggregate cleanly.
"""

import glob
import json
import math
import os

import numpy as np
from scipy import stats

from config import config
from utils.log import get_logger, setup_logging
from a0.eval.trial_selection import select_trial_paths
logger = get_logger(__name__)


def aggregate_metric(values: list[float]) -> dict[str, float | int | None]:
    """Compute mean / std / 95% CI for a metric across trials."""
    clean = [v for v in values if v is not None and not math.isnan(v)]
    n = len(clean)
    if n == 0:
        return {"mean": None, "std": None, "ci95": None, "n": 0}

    arr = np.asarray(clean, dtype=np.float64)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if n >= 2 else 0.0
    if n >= 2:
        t_value = float(stats.t.ppf(0.975, df=n - 1))
        ci95 = t_value * std / math.sqrt(n)
    else:
        ci95 = None
    return {"mean": mean, "std": std, "ci95": ci95, "n": n}


def main():
    setup_logging(
        level=20,
        log_dir=config.log_dir,
        process_name="combine_summary",
    )

    pattern = f"{config.output_dir}trial_*/summary.json"
    trial_summary_paths = select_trial_paths(sorted(glob.glob(pattern)))
    if not trial_summary_paths:
        logger.error(f"No per-trial summaries found matching {pattern}")
        return

    logger.info(f"Aggregating {len(trial_summary_paths)} trial summaries from {config.output_dir}")

    # Load all per-trial summaries.
    per_trial: list[dict] = []
    trials_found: list[str] = []
    for p in trial_summary_paths:
        with open(p) as f:
            per_trial.append(json.load(f))
        # Recover the trial directory name (e.g. "trial_3") for the manifest.
        trials_found.append(os.path.basename(os.path.dirname(p)))

    # Collect every metric name we see across all trials.
    all_metric_names: list[str] = []
    seen: set[str] = set()
    for s in per_trial:
        for k in s.get("metrics", {}).keys():
            if k not in seen:
                seen.add(k)
                all_metric_names.append(k)

    # Aggregate.
    aggregated: dict[str, dict] = {}
    for name in all_metric_names:
        values = [s["metrics"].get(name) for s in per_trial]
        aggregated[name] = aggregate_metric(values)

    # Recover the HP overrides from the per-HP config we wrote at sweep submit time.
    hp_config = {}
    overrides = {}
    if os.path.exists(config.path):
        with open(config.path) as f:
            hp_config = json.load(f)
        overrides = hp_config.get("_sweep_overrides", {})

    out = {
        "hp_id": os.path.basename(os.path.normpath(config.output_dir)),
        "config_path": config.path,
        "overrides": overrides,
        "n_trials_found": len(per_trial),
        "trials_found": trials_found,
        "metrics": aggregated,
    }

    out_path = os.path.join(config.output_dir, "summary.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    logger.info(f"Wrote per-HP summary to {out_path}")
    for k, v in aggregated.items():
        logger.info(f"  {k}: mean={v['mean']} ci95={v['ci95']} n={v['n']}")


if __name__ == "__main__":
    main()
