#!/bin/bash
# Stage the run's solve_data file into $SLURM_TMPDIR (node-local NVMe) so that
# cc.solvedata.SolveData mmaps a fast LOCAL copy instead of faulting pages over
# the network filesystem. GT lookups are random-access, so this matters for the
# big solve files (e.g. 49-4 ~20GB). SolveData._resolve_staged_path prefers this
# copy automatically. No-op when not on a compute node (no $SLURM_TMPDIR) — then
# SolveData just mmaps the original path in place.
#
# Usage (after the venv is active, from the repo root):
#   bash slurm/stage_solve_data.sh <config.json>
CONFIG_FILE="$1"
if [ -z "${SLURM_TMPDIR:-}" ]; then
    echo "stage_solve_data: no \$SLURM_TMPDIR; skipping (SolveData will mmap in place)."
    exit 0
fi

# Resolve the solve_data path from the per-run config, falling back to the
# default config if this one doesn't override it.
get_solve() { python -c "import json,sys; print(json.load(open(sys.argv[1])).get('solve_data','') or '')" "$1" 2>/dev/null; }
SOLVE="$(get_solve "$CONFIG_FILE")"
[ -z "$SOLVE" ] && SOLVE="$(get_solve config/config.json)"

if [ -n "$SOLVE" ] && [ -f "$SOLVE" ]; then
    DEST="$SLURM_TMPDIR/$(basename "$SOLVE")"
    if [ -f "$DEST" ]; then
        echo "stage_solve_data: $(basename "$SOLVE") already staged."
    else
        echo "stage_solve_data: copying $SOLVE -> $DEST"
        time cp "$SOLVE" "$DEST"
    fi
else
    echo "stage_solve_data: no solve_data file found (got '$SOLVE'); SolveData will use its config path."
fi
exit 0
