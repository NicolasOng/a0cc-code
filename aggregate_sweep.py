"""
Sweep-level aggregation.

Reads all per-HP summaries (`<sweep_dir>/*/summary.json`, written by
combine_summary.py) and produces:

  <sweep_dir>/sweep_results.json   - structured: list of HP entries with metrics
  <sweep_dir>/sweep_results.md     - human-readable markdown table

Stdlib only — no third-party imports — so it can run anywhere.

Usage:
    python3 aggregate_sweep.py output/<sweep_name>/
"""

import argparse
import glob
import json
import os
import sys


def load_per_hp_summaries(sweep_dir: str) -> list[dict]:
    """Load every per-HP summary.json under sweep_dir."""
    paths = sorted(glob.glob(os.path.join(sweep_dir, "*", "summary.json")))
    summaries = []
    for p in paths:
        with open(p) as f:
            summaries.append(json.load(f))
    return summaries


def load_sweep_spec(sweep_dir: str) -> dict | None:
    spec_path = os.path.join(sweep_dir, "sweep.json")
    if not os.path.exists(spec_path):
        return None
    with open(spec_path) as f:
        return json.load(f)


def collect_metric_names(summaries: list[dict]) -> list[str]:
    """Stable ordered list of all metric names seen across all HP summaries."""
    seen: set[str] = set()
    names: list[str] = []
    for s in summaries:
        for k in s.get("metrics", {}).keys():
            if k not in seen:
                seen.add(k)
                names.append(k)
    return names


def collect_override_keys(summaries: list[dict]) -> list[str]:
    """Stable ordered list of all HP override keys seen across all HP summaries."""
    seen: set[str] = set()
    keys: list[str] = []
    for s in summaries:
        for k in s.get("overrides", {}).keys():
            if k not in seen:
                seen.add(k)
                keys.append(k)
    return keys


def fmt_metric(entry: dict | None) -> str:
    """Format an aggregated metric (mean ± ci95) for the markdown table."""
    if entry is None:
        return "—"
    mean = entry.get("mean")
    ci = entry.get("ci95")
    if mean is None:
        return "—"
    if ci is None:
        return f"{mean:.4f}"
    return f"{mean:.4f} ± {ci:.4f}"


def fmt_override(value) -> str:
    """Format an HP value for the markdown table."""
    if isinstance(value, float):
        # Use scientific for very small / very large; otherwise plain.
        if value != 0 and (abs(value) < 1e-3 or abs(value) >= 1e4):
            return f"{value:.1e}"
        return f"{value:g}"
    return str(value)


def write_sweep_json(sweep_dir: str, sweep_name: str, summaries: list[dict],
                     metric_names: list[str], override_keys: list[str],
                     spec: dict | None) -> str:
    """Write the structured sweep_results.json."""
    out = {
        "sweep_name": sweep_name,
        "spec": spec,
        "n_hp_points": len(summaries),
        "override_keys": override_keys,
        "metric_names": metric_names,
        "hp_points": summaries,
    }
    out_path = os.path.join(sweep_dir, "sweep_results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    return out_path


def write_sweep_md(sweep_dir: str, sweep_name: str, summaries: list[dict],
                   metric_names: list[str], override_keys: list[str]) -> str:
    """Write the markdown table."""
    lines: list[str] = []
    lines.append(f"# Sweep: {sweep_name}")
    lines.append("")
    lines.append(f"{len(summaries)} HP point(s)")
    lines.append("")

    headers = override_keys + metric_names
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    for s in summaries:
        overrides = s.get("overrides", {})
        metrics = s.get("metrics", {})
        row = []
        for k in override_keys:
            row.append(fmt_override(overrides.get(k, "—")))
        for m in metric_names:
            row.append(fmt_metric(metrics.get(m)))
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    out_path = os.path.join(sweep_dir, "sweep_results.md")
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sweep_dir")
    args = parser.parse_args()

    sweep_dir = args.sweep_dir.rstrip("/")
    if not os.path.isdir(sweep_dir):
        print(f"ERROR: not a directory: {sweep_dir}", file=sys.stderr)
        sys.exit(1)

    sweep_name = os.path.basename(sweep_dir)
    summaries = load_per_hp_summaries(sweep_dir)
    if not summaries:
        print(f"ERROR: no per-HP summaries found in {sweep_dir}/*/summary.json", file=sys.stderr)
        sys.exit(1)

    spec = load_sweep_spec(sweep_dir)
    metric_names = collect_metric_names(summaries)
    override_keys = collect_override_keys(summaries)

    json_path = write_sweep_json(sweep_dir, sweep_name, summaries, metric_names, override_keys, spec)
    md_path = write_sweep_md(sweep_dir, sweep_name, summaries, metric_names, override_keys)

    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    print(f"  {len(summaries)} HP points x {len(metric_names)} metrics")


if __name__ == "__main__":
    main()
