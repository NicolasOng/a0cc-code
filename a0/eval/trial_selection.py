"""Optional trial subsetting for the combine/merge steps.

Set the env var A0_COMBINE_TRIALS to restrict which `trial_<N>` dirs the
combine steps aggregate, e.g. "1-5", "1,2,3", or "1-5,8,10-12". Unset or
empty means "all trials" (the default). This lets a subset of seeds be
merged cleanly when a sweep dir holds more trials than were (re-)evaluated
with the current code — e.g. re-evaluating trials 1-5 of an existing
25-seed sweep without pulling in the other 20 trials' stale outputs.
"""
import os
import re

from utils.log import get_logger

logger = get_logger(__name__)

TRIALS_ENV_VAR = "A0_COMBINE_TRIALS"
_TRIAL_RE = re.compile(r"trial_(\d+)")


def parse_trial_spec(spec: str | None) -> set[int] | None:
    """Parse "1-5,8,10-12" into a set of ints; None if spec is empty/None."""
    if not spec:
        return None
    out: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.update(range(int(lo), int(hi) + 1))
        else:
            out.add(int(part))
    return out or None


def select_trial_paths(paths: list[str], env_var: str = TRIALS_ENV_VAR) -> list[str]:
    """Filter trial_* paths by the A0_COMBINE_TRIALS env spec (no-op if unset)."""
    spec = os.environ.get(env_var)
    wanted = parse_trial_spec(spec)
    if wanted is None:
        return paths
    kept = [p for p in paths
            if (m := _TRIAL_RE.search(p)) and int(m.group(1)) in wanted]
    logger.info(f"{env_var}={spec}: merging {len(kept)}/{len(paths)} trials")
    return kept
