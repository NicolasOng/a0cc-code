"""Plot the plateau MCTS-sweep: win rate vs search budget, one line per td_lambda.

Reads each HP combo's merged_player_sweep_results.pkl (produced by the combine
merge of a0.eval.player_sweep's per-trial outputs) and the sweep's hp_index.txt,
and overlays the combos so you can see whether low search budgets separate the
lambdas the way value accuracy does (heavy search masks value quality).

Self-contained (Series pickle shim) so it runs on the cluster or locally on
downloaded pkls, no a0 env needed.

Usage:
    python plot_player_sweep.py <sweep_dir> [--board 5]
"""
from __future__ import annotations

import argparse
import pickle
import sys
import types
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

LAMBDA_COLOR = {
    0.0:  "#d62728",   # red
    0.5:  "#ff7f0e",   # orange
    0.75: "#9acd32",   # yellowgreen
    0.9:  "#1f8a8a",   # teal
    1.0:  "#6a3d9a",   # purple
}


# --- pickle shim so we can load Series pkls without the a0 package ---------- #
class Series:
    x: list
    ys: dict

_shim = types.ModuleType("a0.utils.plotting"); _shim.Series = Series
_a0 = types.ModuleType("a0"); _u = types.ModuleType("a0.utils"); _a0.utils = _u; _u.plotting = _shim
sys.modules.setdefault("a0", _a0)
sys.modules.setdefault("a0.utils", _u)
sys.modules["a0.utils.plotting"] = _shim


def parse_hp_index(path: Path, board_size: int) -> dict[str, float]:
    """{hp_id: td_lambda} for the given board."""
    out = {}
    for line in path.read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        kv = dict(t.split("=", 1) for t in parts[1:] if "=" in t)
        if int(kv.get("board_size", -1)) == board_size and "td_lambda" in kv:
            out[parts[0]] = float(kv["td_lambda"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep_dir")
    ap.add_argument("--board", type=int, default=5, help="board_size to plot (default 5 = 25-6)")
    ap.add_argument("--side", choices=["overall", "p1", "p2"], default="overall",
                    help="which win rate to plot (default overall)")
    args = ap.parse_args()

    sweep = Path(args.sweep_dir)
    hp_to_td = parse_hp_index(sweep / "hp_index.txt", args.board)
    if not hp_to_td:
        raise SystemExit(f"no board_size={args.board} HPs with td_lambda in {sweep}/hp_index.txt")

    key = "winrate" if args.side == "overall" else f"winrate_{args.side}"
    board_tag = f"{args.board * args.board}-6"

    fig, ax = plt.subplots(figsize=(9, 6))
    plotted = 0
    for hp_id, td in sorted(hp_to_td.items(), key=lambda kv: kv[1]):
        pkl = sweep / hp_id / "eval" / "merged_player_sweep_results.pkl"
        if not pkl.exists():
            print(f"  [skip] lambda={td}: {pkl} missing")
            continue
        s = pickle.load(open(pkl, "rb"))
        x = np.asarray(s.x, dtype=float)
        y = np.asarray(s.ys[key], dtype=float)
        ci = np.asarray(s.ys.get(f"{key}_ci", np.zeros_like(y)), dtype=float)
        color = LAMBDA_COLOR.get(td, "black")
        ax.plot(x, y, marker="o", color=color, linewidth=1.8, label=fr"$\lambda$={td}")
        ax.fill_between(x, y - ci, y + ci, color=color, alpha=0.15, linewidth=0)
        plotted += 1

    if not plotted:
        raise SystemExit("no merged_player_sweep_results.pkl found for any combo — run the sweep + combine first")

    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted({int(v) for v in x}))
    ax.get_xaxis().set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_xlabel("MCTS samples per move (eval search budget)")
    ax.set_ylabel(f"Win rate vs baseline ({args.side}, plateau mean)")
    ax.set_title(f"[{board_tag}] Plateau win rate vs search budget, by td_lambda "
                 f"(low budget exposes value quality; shaded = 95% CI)")
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()

    out_dir = sweep / "sweep_plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"plateau_sweep_winrate_{args.side}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
