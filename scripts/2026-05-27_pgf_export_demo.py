"""
Demo: export a matplotlib figure as a .pgf file that LaTeX can \\input directly.

The PGF backend emits TikZ/PGF commands instead of a rasterized or
self-contained vector image. The resulting file can be included in a
LaTeX document with:

    \\input{sine_cosine.pgf}

Text and math in the figure are then typeset by LaTeX using the same
fonts as the surrounding document, so labels, ticks, and legends match
your paper exactly.

Requirements on the LaTeX side: a modern engine (pdflatex / lualatex /
xelatex) with the `pgf` package. Typical preamble:

    \\usepackage{pgf}
    \\usepackage{amsmath}

Run:
    python scripts/2026-05-27_pgf_export_demo.py
"""

import matplotlib

matplotlib.use("pgf")

import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    "font.family": "serif",
    "text.usetex": True,
    "pgf.rcfonts": False,
})

OUT_PATH = "sine_cosine.pgf"

x = np.linspace(0, 2 * np.pi, 400)

fig, ax = plt.subplots(figsize=(4.5, 2.8))
ax.plot(x, np.sin(x), label=r"$\sin(x)$")
ax.plot(x, np.cos(x), label=r"$\cos(x)$")
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$f(x)$")
ax.set_title(r"Sine and cosine")
ax.legend(loc="upper right")
ax.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(OUT_PATH)

print(f"Wrote {OUT_PATH}. In your LaTeX document, add \\input{{{OUT_PATH}}}.")
