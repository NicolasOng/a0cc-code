"""
Per-trial summary extraction (eval3).

Reads the per-trial eval pkls produced by the eval3 pipeline and writes
a small JSON of scalar metrics to `<config.output_dir>/summary.json`. The
config's `output_dir` is the trial dir (output/<sweep>/<hp>/trial_<N>/), so
each trial writes its own summary in parallel — no coordination needed.

To add or remove a metric, edit METRICS below.

Metric keys are kept identical to a0.eval.extract_summary so that downstream
consumers (combine_summary, etc.) keep working during the eval → eval3
migration. The underlying datasets changed names: `*_gtv` → `*_nd` (no-draws,
for value-head accuracy) and `*_nt_gtv` → `*_nt` (non-trivial, for policy-head
accuracy); semantically the metrics still measure the same thing.
"""

import json
import os

from a0.utils.plotting import load_series

from config import config
from utils.log import get_logger, setup_logging
logger = get_logger(__name__)


# Each entry: (metric_name, source_pkl_filename, series_y_key, reduction)
# reduction is "last", "max", or "last20pct" (mean of final 20% of samples).
METRICS: list[tuple[str, str, str, str]] = [
    ("seen_nd_value_acc_final",     "seen_nd_eval.pkl",         "value_accuracy",  "last"),
    ("seen_nd_value_acc_max",       "seen_nd_eval.pkl",         "value_accuracy",  "max"),
    ("seen_nd_value_acc_last20pct", "seen_nd_eval.pkl",         "value_accuracy",  "last20pct"),
    ("seen_nt_policy_acc_final",     "seen_nt_eval.pkl",         "policy_accuracy", "last"),
    ("seen_nt_policy_acc_max",       "seen_nt_eval.pkl",         "policy_accuracy", "max"),
    ("seen_nt_policy_acc_last20pct", "seen_nt_eval.pkl",         "policy_accuracy", "last20pct"),
    ("random_nd_value_acc_final",        "random_nd_eval.pkl",       "value_accuracy",  "last"),
    ("random_nd_value_acc_max",          "random_nd_eval.pkl",       "value_accuracy",  "max"),
    ("random_nd_value_acc_last20pct",    "random_nd_eval.pkl",       "value_accuracy",  "last20pct"),
    ("random_nt_policy_acc_final",    "random_nt_eval.pkl",       "policy_accuracy", "last"),
    ("random_nt_policy_acc_max",      "random_nt_eval.pkl",       "policy_accuracy", "max"),
    ("random_nt_policy_acc_last20pct","random_nt_eval.pkl",       "policy_accuracy", "last20pct"),
    ("gamedata_value_acc_final",      "gamedata_overall_acc.pkl", "Overall Value Accuracy",  "last"),
    ("gamedata_value_acc_last20pct",  "gamedata_overall_acc.pkl", "Overall Value Accuracy",  "last20pct"),
    ("gamedata_policy_acc_final",     "gamedata_overall_acc.pkl", "Overall Policy Accuracy", "last"),
    ("gamedata_policy_acc_last20pct", "gamedata_overall_acc.pkl", "Overall Policy Accuracy", "last20pct"),
]


def reduce_series_values(values: list[float], reduction: str) -> float | None:
    """Apply a reduction to a list of y-values, ignoring NaNs."""
    if not values:
        return None
    clean = [v for v in values if v == v]  # NaN != NaN
    if not clean:
        return None
    if reduction == "last":
        return clean[-1]
    if reduction == "max":
        return max(clean)
    if reduction == "last20pct":
        k = max(1, len(clean) // 5)
        tail = clean[-k:]
        return sum(tail) / len(tail)
    raise ValueError(f"unknown reduction: {reduction}")


def main():
    setup_logging(
        level=20,
        log_dir=config.log_dir,
        process_name="extract_summary",
    )

    metrics: dict[str, float | None] = {}
    series_cache: dict[str, object] = {}

    for name, pkl_fn, y_key, reduction in METRICS:
        path = os.path.join(config.eval_dir, pkl_fn)
        if pkl_fn not in series_cache:
            try:
                series_cache[pkl_fn] = load_series(path)
            except FileNotFoundError:
                logger.warning(f"Missing pkl {path}, metrics from it will be null")
                series_cache[pkl_fn] = None

        series = series_cache[pkl_fn]
        if series is None:
            metrics[name] = None
            continue

        if y_key not in series.ys:  # type: ignore[union-attr]
            logger.warning(f"Series {pkl_fn} has no key '{y_key}', skipping")
            metrics[name] = None
            continue

        metrics[name] = reduce_series_values(series.ys[y_key], reduction)  # type: ignore[union-attr]

    summary = {
        "trial_num": config.trial_num,
        "metrics": metrics,
    }

    out_path = os.path.join(config.output_dir, "summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Wrote per-trial summary to {out_path}")
    for k, v in metrics.items():
        logger.info(f"  {k} = {v}")


if __name__ == "__main__":
    main()
