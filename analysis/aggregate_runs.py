"""Aggregate multi-seed runs from experiments/train.py into paper-ready numbers.

The conference version reported one run per method. Three Hybrid-CASR result
files survive in the repository with means of 0.6544, 0.6640 and 0.6671 -- a
spread of 0.0127 against a claimed improvement of 0.0164. At that ratio a
single run cannot establish anything, so this script is built around making
seed variance visible rather than averaging it away.

Two variance components are reported separately because they answer different
questions:

  window sd   spread across evaluation windows within one seed -- how much the
              method's performance depends on which period it is tested on.
  seed sd     spread of per-seed means -- how much depends on initialisation
              alone. If this approaches the gap between methods, the gap is
              not evidence.

Significance is computed on per-window means taken across seeds, paired by
window. Treating every (seed, window) pair as an independent sample would
multiply n by the seed count without adding independent information, which
inflates significance.

Usage:
    python analysis/aggregate_runs.py --metrics runs/main/metrics.csv --out figures/
    python analysis/aggregate_runs.py --metrics runs/main/metrics.csv --out figures/ \
        --metric f1_vulnerable --baseline window-only
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_figures import (BASELINE, INK, INK2, MACRO, MUTED,  # noqa: E402
                          PUBLISHED, SURFACE, caption, recessive, save, style)

try:
    from scipy.stats import wilcoxon
except ImportError:
    wilcoxon = None

METRICS = ["macro_f1", "f1_vulnerable", "f1_fixed", "f1_binary_pos1_LEGACY", "accuracy"]


def load(path):
    d = pd.read_csv(path)
    missing = {"method", "seed", "eval_window", "direction"} - set(d.columns)
    if missing:
        sys.exit(f"{path} is missing columns {sorted(missing)}; is it from train.py?")
    return d


def label_methods(d):
    """Qualify method names with the backbone when more than one is present.

    Otherwise the same method run on two models collapses into one group and
    the cross-architecture comparison silently averages the two together.
    """
    if "model" in d.columns and d["model"].nunique(dropna=True) > 1:
        d = d.copy()
        d["method"] = d["method"].astype(str) + " @" + d["model"].astype(str)
    return d


def summarise(fw, metric):
    """Per method: seed count, window count, mean, window sd, seed sd."""
    rows = []
    for method, g in fw.groupby("method"):
        per_seed = g.groupby("seed")[metric].mean()
        rows.append({
            "method": method,
            "seeds": g["seed"].nunique(),
            "windows": g["eval_window"].nunique(),
            "mean": per_seed.mean(),
            "window_sd": g.groupby("seed")[metric].std().mean(),
            "seed_sd": per_seed.std(ddof=1) if len(per_seed) > 1 else np.nan,
            "min_seed": per_seed.min(),
            "max_seed": per_seed.max(),
        })
    return pd.DataFrame(rows).sort_values("mean", ascending=False).reset_index(drop=True)


def window_means(fw, metric):
    """method -> Series indexed by eval_window, averaged over seeds."""
    return {m: g.groupby("eval_window")[metric].mean()
            for m, g in fw.groupby("method")}


def compare(wm, baseline, metric):
    """Each method against the baseline, paired by window.

    With more than one backbone present, every method is compared against the
    baseline *on its own backbone*. Comparing a Qwen method against a phi-2
    baseline would fold the architecture gap into every row and answer no
    question; what the study asks is whether the ordering holds within a model.
    """
    def base_for(method):
        if " @" in method:
            same = f"{baseline.split(' @')[0]} @{method.split(' @', 1)[1]}"
            if same in wm:
                return same
        return baseline if baseline in wm else None

    rows = []
    for method, series in wm.items():
        b = base_for(method)
        if b is None or method == b:
            continue
        joined = pd.concat([series, wm[b]], axis=1, join="inner").dropna()
        if len(joined) < 5:
            continue
        x, y = joined.iloc[:, 0], joined.iloc[:, 1]
        p = np.nan
        if wilcoxon is not None and not np.allclose(x, y):
            p = wilcoxon(x, y).pvalue
        rows.append({"method": method, "baseline": b, "n_windows": len(joined),
                     "delta": x.mean() - y.mean(), "p": p})
    if not rows:
        return None
    return pd.DataFrame(rows).sort_values("delta", ascending=False).reset_index(drop=True)


def fig_seed_spread(summary, out, metric, baseline=None):
    """Per-seed means against the between-method gaps."""
    import matplotlib.pyplot as plt

    s = summary[summary["seeds"] > 1]
    if s.empty:
        print("  (only one seed per method -- skipping the seed-spread figure)")
        return
    s = s.sort_values("mean")
    y = np.arange(len(s))
    fig, ax = plt.subplots(figsize=(7.0, 0.45 * len(s) + 1.6))

    ax.hlines(y, s["min_seed"], s["max_seed"], color=BASELINE, lw=3,
              zorder=1, capstyle="round")
    ax.scatter(s["mean"], y, s=58, color=MACRO, zorder=3,
               edgecolor=SURFACE, linewidth=1.4, label="mean over seeds")
    ax.scatter(s["min_seed"], y, s=26, color=PUBLISHED, zorder=2,
               edgecolor=SURFACE, linewidth=1.0, label="individual seed (min / max)")
    ax.scatter(s["max_seed"], y, s=26, color=PUBLISHED, zorder=2,
               edgecolor=SURFACE, linewidth=1.0)

    if baseline and baseline in set(s["method"]):
        # No label on the rule: the baseline is already a named row on the axis.
        b = float(s.loc[s["method"] == baseline, "mean"].iloc[0])
        ax.axvline(b, color=MUTED, lw=1, ls=(0, (4, 3)), zorder=0)

    ax.set_yticks(y, s["method"])
    ax.set_xlabel(f"{metric} (forward evaluation)")
    ax.set_title("Seed spread against the gaps between methods")
    recessive(ax)
    ax.legend(loc="lower right")
    ax.margins(y=0.08)
    caption(fig, "Bar spans the lowest and highest per-seed mean. Where a bar is as "
                 "wide as the distance to the next method, that ordering is not "
                 "established by these runs.")
    save(fig, out, "fig5_seed_spread")


def latex(summary, comparison, out, metric, baseline):
    lines = [r"% Regenerated by analysis/aggregate_runs.py -- do not hand-edit.",
             r"\begin{tabular}{lrrrr}", r"\toprule",
             r"Method & Seeds & " + metric.replace("_", r"\_") +
             r" & Window SD & Seed SD \\", r"\midrule"]
    for r in summary.itertuples():
        seed_sd = "--" if np.isnan(r.seed_sd) else f"{r.seed_sd:.4f}"
        lines.append(f"{r.method} & {r.seeds} & {r.mean:.4f} & "
                     f"{r.window_sd:.4f} & {seed_sd} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(out, "table_multiseed.tex"), "w").write("\n".join(lines) + "\n")
    print("  table_multiseed.tex")

    if comparison is None or comparison.empty:
        return
    lines = [r"% Regenerated by analysis/aggregate_runs.py -- do not hand-edit.",
             r"\begin{tabular}{lrrr}", r"\toprule",
             rf"Method vs.\ {baseline} & Windows & $\Delta$ & $p$ \\", r"\midrule"]
    for r in comparison.itertuples():
        p = "--" if np.isnan(r.p) else (rf"\textbf{{{r.p:.3f}}}" if r.p < 0.05 else f"{r.p:.3f}")
        lines.append(f"{r.method} & {r.n_windows} & {r.delta:+.4f} & {p} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(out, "table_multiseed_significance.tex"), "w").write("\n".join(lines) + "\n")
    print("  table_multiseed_significance.tex")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metrics", required=True, nargs="+",
                    help="one or more metrics.csv from experiments/train.py")
    ap.add_argument("--out", default="figures")
    ap.add_argument("--metric", default="macro_f1", choices=METRICS)
    ap.add_argument("--baseline", default="window-only")
    ap.add_argument("--granularity", type=int,
                    help="restrict to one window width (default: all present)")
    args = ap.parse_args()

    d = label_methods(pd.concat([load(p) for p in args.metrics], ignore_index=True))
    if args.granularity and "granularity" in d.columns:
        d = d[d["granularity"] == args.granularity]
    if d.empty:
        sys.exit("no rows after filtering")

    os.makedirs(args.out, exist_ok=True)
    style()

    fw = d[d["direction"] == "forward"]
    summary = summarise(fw, args.metric)

    print(f"metric: {args.metric}   runs: {len(args.metrics)} file(s)")
    print(f"{'method':16s} {'seeds':>5s} {'wins':>5s} {'mean':>8s} "
          f"{'window sd':>10s} {'seed sd':>8s} {'seed range':>16s}")
    for r in summary.itertuples():
        rng = "--" if np.isnan(r.seed_sd) else f"{r.min_seed:.4f}-{r.max_seed:.4f}"
        sd = "--" if np.isnan(r.seed_sd) else f"{r.seed_sd:.4f}"
        print(f"{r.method:16s} {r.seeds:5d} {r.windows:5d} {r.mean:8.4f} "
              f"{r.window_sd:10.4f} {sd:>8s} {rng:>16s}")

    single = summary[summary["seeds"] < 2]["method"].tolist()
    if single:
        print(f"\nonly one seed for: {', '.join(single)} -- no seed variance available")

    wm = window_means(fw, args.metric)
    baseline = args.baseline
    if baseline not in wm:
        cands = [k for k in wm if k.split(" @")[0] == baseline]
        if len(cands) == 1:
            baseline = cands[0]
        elif not cands:
            print(f"\nbaseline '{baseline}' not among {sorted(wm)}")
        # More than one candidate is the normal cross-backbone case: compare()
        # pairs each method with the baseline on its own backbone.
    comparison = compare(wm, baseline, args.metric)
    if comparison is None:
        print(f"\nbaseline '{baseline}' not in these runs; skipping comparisons")
    else:
        multi = comparison["baseline"].nunique() > 1
        print(f"\nPaired Wilcoxon on per-window means across seeds"
              + (", each vs the baseline on its own backbone:" if multi
                 else f" vs {baseline}:"))
        width = max(len(str(m)) for m in comparison["method"])
        for r in comparison.itertuples():
            p = "n/a" if np.isnan(r.p) else f"{r.p:.4f}"
            star = "*" if (not np.isnan(r.p) and r.p < 0.05) else " "
            vs = f"  vs {r.baseline}" if multi else ""
            print(f"  {r.method:{width}s} n={r.n_windows:3d}  d={r.delta:+.4f}  p={p}{star}{vs}")

        # The comparison worth stating plainly: an effect smaller than the
        # spread between seeds of the same configuration is not an effect.
        base_row = summary[summary["method"] == baseline]
        if not base_row.empty and not np.isnan(base_row["seed_sd"].iloc[0]):
            base_sd = float(base_row["seed_sd"].iloc[0])
            weak = [r.method for r in comparison.itertuples() if abs(r.delta) < base_sd]
            if weak:
                print(f"\n  Effects smaller than the baseline's own seed sd ({base_sd:.4f}): "
                      f"{', '.join(weak)}")

    bw = d[d["direction"].str.startswith("backward")]
    if not bw.empty:
        print("\nBackward retention (IBR), mean over seeds and windows:")
        ibr = bw.pivot_table(index="method", columns="direction",
                             values=args.metric, aggfunc="mean")
        print(ibr.round(4).to_string())
        ibr.round(4).to_csv(os.path.join(args.out, "ibr.csv"))

    print("\nwriting:")
    fig_seed_spread(summary, args.out, args.metric, baseline)
    latex(summary, comparison, args.out, args.metric, baseline)
    summary.to_csv(os.path.join(args.out, "summary_multiseed.csv"), index=False)
    print(f"  summary_multiseed.csv\n\n-> {args.out}/")


if __name__ == "__main__":
    main()
