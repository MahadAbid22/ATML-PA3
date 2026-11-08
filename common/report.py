"""Figure style and LaTeX table helpers shared by the analysis scripts.

Colours: a fixed, colour-vision-deficiency-checked categorical order; each entity keeps its
colour in every figure (colour follows the entity, never its rank). Lines also differ by
marker / dash so identity never relies on colour alone (print and CVD readers).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common.utils import REPORT  # noqa: E402

CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
NEUTRAL = "#6b6a66"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
DASHES = ["-", "--", "-.", ":", (0, (5, 1)), (0, (3, 1, 1, 1)), (0, (1, 1)), (0, (8, 2))]


def color_map(names: list[str]) -> dict:
    return {n: CATEGORICAL[i % len(CATEGORICAL)] for i, n in enumerate(names)}


def style():
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 300, "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
        "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.edgecolor": INK_2,
        "axes.labelcolor": INK, "xtick.color": INK_2, "ytick.color": INK_2, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
        "lines.linewidth": 1.6, "lines.markersize": 4, "legend.frameon": False, "pdf.fonttype": 42,
        "font.family": "DejaVu Sans", "axes.axisbelow": True,
    })


def clean_log_x(ax, ticks=(0.25, 0.5, 1, 2, 4, 8, 16, 32, 64)):
    """Log x-axis with plain-number major ticks inside the data range and no minor labels."""
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
    lo, hi = ax.get_xlim()
    t = [v for v in ticks if lo <= v <= hi]
    ax.xaxis.set_major_locator(FixedLocator(t))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(NullFormatter())


def savefig(fig, name: str):
    out = REPORT / "figures"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(out / f"{name}.png", bbox_inches="tight")
    plt.close(fig)


def fmt(x, nd=1, pct=False, sign=False):
    if x is None or (isinstance(x, float) and x != x):
        return "--"
    v = 100 * x if pct else x
    s = f"{v:+.{nd}f}" if sign else f"{v:.{nd}f}"
    return "$-$" + s[1:] if s.startswith("-") else s


def latex_table(header: list[str], rows: list[list[str]], caption: str, label: str, name: str,
                align: str | None = None, notes: str | None = None, size: str = "\\small") -> Path:
    """Write a booktabs table (\\input-able). Caption text is a placeholder for the author to edit."""
    align = align or ("l" + "r" * (len(header) - 1))
    lines = ["\\begin{table}[t]", "\\centering", size, f"\\begin{{tabular}}{{{align}}}", "\\toprule",
             " & ".join(header) + " \\\\", "\\midrule"]
    for r in rows:
        if r == "midrule":
            lines.append("\\midrule")
        else:
            lines.append(" & ".join(str(c) for c in r) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    if notes:
        lines.append(f"\\\\[2pt]{{\\footnotesize {notes}}}")
    lines += ["% Placeholder caption listing the contents only - replace with your own wording.",
              f"\\caption{{{caption}}}", f"\\label{{{label}}}", "\\end{table}"]
    out = REPORT / "tables"
    out.mkdir(parents=True, exist_ok=True)
    p = out / f"{name}.tex"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p
