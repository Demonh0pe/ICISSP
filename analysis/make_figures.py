"""Regenerate the paper's figures and tables under the corrected metric.

Reads the conference result CSVs, recovers per-class F1 by inverting the logged
confusion matrix (see recompute_metrics.py), and emits PDF + PNG figures plus
LaTeX tables ready to \\input.

Three metrics appear throughout, and keeping them side by side is the point:

  published      binary F1 of label 1. Label 1 is FIXED, so this scores the
                 model on recognising already-patched code. The paper prints
                 this as Macro-F1.
  macro          the true (F1_vulnerable + F1_fixed) / 2 the paper defines.
  vulnerable     F1 on the vulnerable class alone -- what a detector is for.

Usage:
    python analysis/make_figures.py --repo <main checkout> --out figures/

    # once new runs exist, they carry macro_f1 natively and need no inversion
    python analysis/make_figures.py --repo <main checkout> --out figures/ \
        --new-metrics runs/main/metrics.csv
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from recompute_metrics import PAPER_F1, SOURCES, invert, load, window_class_counts  # noqa: E402

try:
    from scipy.stats import wilcoxon
except ImportError:
    wilcoxon = None

# Validated categorical slots 1-3 (light mode, white surface, all-pairs):
# worst CVD dE 9.2, worst normal-vision dE 24.0. Aqua sits at 2.82:1 on white,
# so every series carries a direct label rather than relying on the swatch.
PUBLISHED, MACRO, VULN = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE, SURFACE = "#e1e0d9", "#c3c2b7", "#ffffff"

METRIC_LABEL = {
    "published": "Published (binary F1, FIXED class)",
    "macro": "Corrected Macro-F1",
    "vulnerable": "VULNERABLE-class F1",
}
COLOR = {"published": PUBLISHED, "macro": MACRO, "vulnerable": VULN}

GRANULARITY_FILES = [
    ("1 month", "result/metrics_log_1month.csv"),
    ("2 months", "result/metrics_log_2month.csv"),
    ("3 months", "result/metrics_log_3month.csv"),
    ("6 months", "result/metrics_log_halfyear.csv"),
    ("12 months", "result/metrics_log_wholeyear.csv"),
]


def style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "sans-serif", "font.size": 9,
        "axes.edgecolor": BASELINE, "axes.linewidth": 0.8,
        "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "grid.color": GRID, "grid.linewidth": 0.6,
        "legend.frameon": False, "legend.fontsize": 8,
        "figure.dpi": 150, "savefig.bbox": "tight",
    })


def recessive(ax, xgrid=True):
    ax.set_axisbelow(True)
    ax.grid(axis="x" if xgrid else "y", alpha=1.0)
    ax.grid(axis="y" if xgrid else "x", visible=False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def caption(fig, text, reserve_in=0.85):
    """Footnote below the axes, given a fixed strip of space in inches.

    Reserving the space first matters: text placed at a negative figure
    coordinate and left to bbox='tight' lands on the x-axis label. Reserving it
    as a *fraction* instead misbehaves in the other direction -- the same
    fraction is a much taller strip on a tall figure, which stranded the
    caption an inch below the axes.
    """
    fig.subplots_adjust(bottom=reserve_in / fig.get_figheight())
    fig.text(0.01, 0.012, text, fontsize=7.5, color=MUTED, ha="left", va="bottom")


def label_ends(ax, ends, min_gap):
    """Direct-label series at their final point, nudged apart if they collide."""
    ends = sorted(ends, key=lambda t: t[0])
    ys = [e[0] for e in ends]
    for i in range(1, len(ys)):
        if ys[i] - ys[i - 1] < min_gap:
            ys[i] = ys[i - 1] + min_gap
    for (value, colour, name), y in zip(ends, ys):
        # The label sits at the nudged y; a leader runs back to where the series
        # actually ends, so separating the text never misstates the data.
        # Both endpoints in the same (axes-fraction x, data y) system: mixing
        # "offset points" with "data" makes the y component an absolute data
        # value rather than a delta, which throws the label off the chart.
        ax.annotate(name,
                    xy=(1.0, value), xycoords=("axes fraction", "data"),
                    xytext=(1.03, y), textcoords=("axes fraction", "data"),
                    va="center", ha="left", fontsize=8, color=colour,
                    fontweight="bold", annotation_clip=False,
                    arrowprops=dict(arrowstyle="-", color=colour, lw=0.7,
                                    shrinkA=0, shrinkB=3))


def save(fig, out, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"))
    plt.close(fig)
    print(f"  {name}.pdf / .png")


def gather(repo):
    """method -> per-window frame with published / macro / vulnerable columns."""
    counts = window_class_counts(repo)
    res = {}
    for name, spec in SOURCES.items():
        fw = load(repo, spec)
        if fw is None:
            continue
        frame, residual = invert(fw, counts)
        if frame is None:
            continue
        res[name] = frame.rename(columns={
            "f1_fixed": "published", "macro_f1": "macro", "f1_vulnerable": "vulnerable"})
        res[name].attrs["residual"] = residual
    return res


def fig_metric_shift(res, out):
    """Each method's three metrics on one row -- the correction, per method."""
    order = sorted(res, key=lambda k: res[k]["macro"].mean())
    y = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(7.2, 0.46 * len(order) + 1.5))

    for i, name in enumerate(order):
        f = res[name]
        lo = min(f[m].mean() for m in COLOR)
        hi = max(f[m].mean() for m in COLOR)
        # Connector first, so the dots sit on top of it.
        ax.plot([lo, hi], [i, i], color=BASELINE, lw=1.2, zorder=1, solid_capstyle="round")

    for metric, colour in COLOR.items():
        vals = [res[n][metric].mean() for n in order]
        ax.scatter(vals, y, s=52, color=colour, zorder=3,
                   edgecolor=SURFACE, linewidth=1.4, label=METRIC_LABEL[metric])

    # Direct labels on the top row satisfy the relief rule for the sub-3:1 slot.
    top = len(order) - 1
    for metric, colour in COLOR.items():
        v = res[order[top]][metric].mean()
        ax.annotate(f"{v:.3f}", (v, top), textcoords="offset points", xytext=(0, 11),
                    ha="center", fontsize=7.5, color=INK2, fontweight="bold")

    ax.set_yticks(y, order)
    ax.set_xlabel("Mean forward F1 across bi-monthly windows")
    ax.set_title("Correcting the metric lowers every method by about 0.03")
    recessive(ax)
    ax.legend(loc="upper left", ncols=1)
    ax.margins(y=0.06)
    caption(fig, "Label 1 is FIXED, so the published number scores recognition of "
                 "already-patched code, not vulnerability detection.")
    save(fig, out, "fig1_metric_shift")


def fig_significance(res, out, baseline="Window-only"):
    """Per-window paired gap against the baseline, under each metric."""
    if baseline not in res:
        return
    methods = [m for m in ("Hybrid-CASR", "CASR", "Replay-1P", "Cumulative") if m in res]
    if not methods:
        return
    fig, axes = plt.subplots(1, len(methods), figsize=(2.5 * len(methods), 3.2), sharey=True)
    axes = np.atleast_1d(axes)

    for ax, name in zip(axes, methods):
        j = res[name].merge(res[baseline], on="window", suffixes=("_m", "_b"))
        for k, metric in enumerate(("published", "macro")):
            d = j[f"{metric}_m"] - j[f"{metric}_b"]
            ax.scatter(np.full(len(d), k) + np.random.RandomState(0).uniform(-.09, .09, len(d)),
                       d, s=14, color=COLOR[metric], alpha=0.55, zorder=2,
                       edgecolor="none")
            ax.plot([k - .22, k + .22], [d.mean()] * 2, color=INK, lw=2, zorder=4)
            if wilcoxon is not None and not np.allclose(j[f"{metric}_m"], j[f"{metric}_b"]):
                p = wilcoxon(j[f"{metric}_m"], j[f"{metric}_b"]).pvalue
                ax.annotate(f"p={p:.3f}" + ("*" if p < 0.05 else ""),
                            (k, ax.get_ylim()[1]), textcoords="offset points",
                            xytext=(0, -6), ha="center", fontsize=7.5,
                            color=INK if p < 0.05 else MUTED,
                            fontweight="bold" if p < 0.05 else "normal")
        ax.axhline(0, color=BASELINE, lw=1, zorder=1)
        ax.set_xticks([0, 1], ["published", "macro"], fontsize=8)
        ax.set_title(name, fontsize=9)
        recessive(ax, xgrid=False)
        ax.set_xlim(-0.5, 1.5)

    axes[0].set_ylabel(f"Per-window F1 minus {baseline}")
    fig.suptitle("Under the corrected metric the gains stop being significant",
                 x=0.0, ha="left", fontsize=10, fontweight="bold")
    fig.tight_layout()
    caption(fig, "Each dot is one evaluation window; the bar is the mean. "
                 "Wilcoxon signed-rank, paired on windows. * marks p < 0.05.")
    save(fig, out, "fig2_significance")


def fig_timeline(res, out, a="Hybrid-CASR", b="Window-only"):
    """Corrected Macro-F1 over the timeline for the paper's headline comparison."""
    if a not in res or b not in res:
        return
    j = res[a].merge(res[b], on="window", suffixes=("_a", "_b")).sort_values("window")
    x = np.arange(len(j))
    fig, ax = plt.subplots(figsize=(7.6, 3.2))
    ax.plot(x, j["macro_a"], color=MACRO, lw=2, solid_capstyle="round")
    ax.plot(x, j["macro_b"], color=PUBLISHED, lw=2, solid_capstyle="round")

    # The two series end within ~0.001 of each other, so unnudged end-labels
    # print on top of one another.
    span = float(np.nanmax([j["macro_a"].max(), j["macro_b"].max()])
                 - np.nanmin([j["macro_a"].min(), j["macro_b"].min()]))
    label_ends(ax, [(j["macro_a"].iloc[-1], MACRO, a),
                    (j["macro_b"].iloc[-1], PUBLISHED, b)], min_gap=0.08 * span)

    step = max(1, len(j) // 10)
    ax.set_xticks(x[::step], [w.replace("_", "\n") for w in j["window"][::step]], fontsize=7)
    ax.set_ylabel("Corrected Macro-F1")
    ax.set_title("The two methods track each other window by window")
    recessive(ax, xgrid=False)
    ax.margins(x=0.02)
    ax.set_xlim(-0.5, len(j) - 0.5)
    caption(fig, "Forward evaluation, one point per window. Right-hand labels are "
                 "nudged apart; the two series end 0.001 apart.")
    save(fig, out, "fig3_timeline")


def fig_granularity(repo, res, out):
    """Window size versus the proposed method, in the paper's own metric.

    The granularity runs log different window tags and no class counts, so the
    confusion matrix cannot be inverted for them. This figure therefore stays in
    the published metric -- which is the metric the paper's granularity claim was
    made in, so the comparison is still apples to apples.
    """
    rows = []
    for label, rel in GRANULARITY_FILES:
        path = os.path.join(repo, rel)
        if not os.path.exists(path):
            continue
        d = pd.read_csv(path)
        fw = d[d["direction"] == "forward"]
        rows.append((label, fw["f1"].mean(), fw["f1"].std(), len(fw)))
    if not rows:
        return

    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    x = np.arange(len(rows))
    means = [r[1] for r in rows]
    ax.errorbar(x, means, yerr=[r[2] for r in rows], fmt="o", ms=9, color=PUBLISHED,
                ecolor=BASELINE, elinewidth=1.4, capsize=4, zorder=3,
                markeredgecolor=SURFACE, markeredgewidth=1.4,
                label="Window-only baseline")
    for xi, m in zip(x, means):
        ax.annotate(f"{m:.3f}", (xi, m), textcoords="offset points", xytext=(0, 13),
                    ha="center", fontsize=7.5, color=INK2, fontweight="bold")

    if "Hybrid-CASR" in res:
        hy = res["Hybrid-CASR"]["published"].mean()
        ax.axhline(hy, color=MACRO, lw=2, ls=(0, (5, 3)), zorder=2)
        # Anchored to the left inside the axes: a right-anchored label at the
        # last tick was clipped off the canvas.
        ax.annotate(f"Hybrid-CASR at 2 months = {hy:.3f}",
                    xy=(0.02, hy), xycoords=("axes fraction", "data"),
                    xytext=(0, 7), textcoords="offset points",
                    ha="left", fontsize=8, color=MACRO, fontweight="bold")

    ax.set_xticks(x, [r[0] for r in rows])
    ax.set_xlabel("Temporal window granularity")
    ax.set_ylabel("Mean forward F1 (published metric)")
    ax.set_title("Retraining quarterly matches the proposed method, with no method")
    recessive(ax, xgrid=False)
    ax.margins(x=0.08, y=0.14)
    caption(fig, "Error bars are 1 SD across windows. Published metric only: the "
                 "granularity runs log no class counts, so the corrected metric cannot "
                 "be recovered for them.")
    save(fig, out, "fig4_granularity")


def write_tables(res, out):
    """LaTeX tables to \\input, mirroring the figures."""
    order = sorted(res, key=lambda k: -res[k]["macro"].mean())
    lines = [
        r"% Regenerated by analysis/make_figures.py -- do not hand-edit.",
        r"\begin{tabular}{lrrrr}", r"\toprule",
        r"Method & $n$ & Published & Macro-F1 & VULN-F1 \\", r"\midrule",
    ]
    for name in order:
        f = res[name]
        lines.append(f"{name} & {len(f)} & {f['published'].mean():.4f} & "
                     f"{f['macro'].mean():.4f} & {f['vulnerable'].mean():.4f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    path = os.path.join(out, "table_methods.tex")
    open(path, "w").write("\n".join(lines) + "\n")
    print(f"  table_methods.tex")

    if wilcoxon is None or "Window-only" not in res:
        return
    base = res["Window-only"]
    lines = [
        r"% Regenerated by analysis/make_figures.py -- do not hand-edit.",
        r"\begin{tabular}{lrrrr}", r"\toprule",
        r"Method vs.\ window-only & $\Delta$ pub. & $p$ & $\Delta$ macro & $p$ \\",
        r"\midrule",
    ]
    for name in order:
        if name == "Window-only":
            continue
        j = res[name].merge(base, on="window", suffixes=("_m", "_b"))
        if len(j) < 5:
            continue
        cells = []
        for metric in ("published", "macro"):
            x, y = j[f"{metric}_m"], j[f"{metric}_b"]
            if np.allclose(x, y):
                cells += ["--", "--"]
                continue
            p = wilcoxon(x, y).pvalue
            cells += [f"{x.mean() - y.mean():+.4f}",
                      (r"\textbf{" + f"{p:.3f}" + "}") if p < 0.05 else f"{p:.3f}"]
        lines.append(f"{name} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(out, "table_significance.tex"), "w").write("\n".join(lines) + "\n")
    print(f"  table_significance.tex")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True, help="checkout of main, containing result/")
    ap.add_argument("--out", default="figures")
    ap.add_argument("--new-metrics", help="metrics.csv from experiments/train.py")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    style()

    res = gather(args.repo)
    if not res:
        sys.exit(f"no result CSVs found under {args.repo}")
    flagged = [n for n, f in res.items() if f.attrs.get("residual", 0) > 1e-6]
    print(f"{len(res)} methods recovered" +
          (f"; inversion unverified for {', '.join(flagged)}" if flagged else ""))

    if args.new_metrics and os.path.exists(args.new_metrics):
        d = pd.read_csv(args.new_metrics)
        fw = d[d["direction"] == "forward"]
        for name, g in fw.groupby("method"):
            # New runs log the corrected metrics directly; no inversion needed.
            res[f"{name} (new)"] = g.rename(columns={
                "eval_window": "window", "f1_binary_pos1_LEGACY": "published",
                "macro_f1": "macro", "f1_vulnerable": "vulnerable"})[
                ["window", "published", "macro", "vulnerable"]]
        print(f"added {fw['method'].nunique()} method(s) from {args.new_metrics}")

    print("writing:")
    fig_metric_shift(res, args.out)
    fig_significance(res, args.out)
    fig_timeline(res, args.out)
    fig_granularity(args.repo, res, args.out)
    write_tables(res, args.out)
    print(f"\n-> {args.out}/")


if __name__ == "__main__":
    main()
