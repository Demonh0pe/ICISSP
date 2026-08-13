"""Recompute the paper's headline metrics from the committed result CSVs.

Why this exists
---------------
Every `f1_score(...)` call in the notebooks uses sklearn's defaults, i.e.
`average='binary', pos_label=1`. The label map is

    {"VULNERABLE": 0, "FIXED": 1}

so the logged `f1` column is the F1 of the FIXED class alone -- not Macro-F1,
and not the vulnerable class. The paper reports those numbers as Macro-F1.

The CSVs log accuracy, precision and recall alongside f1, and the Hybrid-CASR
run additionally logs per-window class counts. That is enough to invert the
confusion matrix exactly:

    TP = recall * n1              FN = n1 - TP
    FP = TP * (1 - P) / P         TN = n0 - FP

from which the class-0 F1 and the true Macro-F1 follow. Class counts are a
property of the data, so the counts recovered from the Hybrid-CASR run apply to
every method evaluated on the same windows.

The inversion is not assumed correct -- it is checked. Reconstructed accuracy is
compared against the logged accuracy for every row, and any method whose
residual exceeds --tol is reported as unverified rather than silently trusted.

Usage:
    python analysis/recompute_metrics.py --repo /path/to/main/checkout
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

try:
    from scipy.stats import wilcoxon
except ImportError:
    wilcoxon = None

# Maps the paper's method names onto the result files. A (file, method) tuple
# selects one method out of a merged CSV.
SOURCES = {
    "Hybrid-CASR": "result/metrics_log_hybrid_casr_balanced__RUN_20250821-220210.csv",
    "Cumulative": ("result/metrics_log_merged.csv", "upper_baseline"),
    "Replay-1P": "result/metrics_log_replay_1p.csv",
    "CASR": "result/metrics_log_casr.csv",
    "LB-CL": "result/metrics_log_lbcl.csv",
    "Window-only": "result/metrics_log_2month.csv",
    "Replay-3P": "result/metrics_log_replay_3p.csv",
    "OLoRA": "result/metrics_log_olora.csv",
    "Zero-shot": "result/metrics_log_zero.csv",
}

# Table 4 of the conference version, for side-by-side comparison.
PAPER_F1 = {
    "Hybrid-CASR": 0.667, "Cumulative": 0.661, "Replay-1P": 0.659, "CASR": 0.659,
    "LB-CL": 0.651, "Window-only": 0.651, "Replay-3P": 0.622, "OLoRA": 0.599,
    "Zero-shot": 0.504,
}

COUNTS_FROM = "result/metrics_log_hybrid_casr_balanced__RUN_20250821-220210.csv"


def load(repo, spec, direction="forward"):
    path, method = (spec, None) if isinstance(spec, str) else spec
    full = os.path.join(repo, path)
    if not os.path.exists(full):
        return None
    d = pd.read_csv(full)
    if method is not None:
        d = d[d["method"] == method]
    return d[d["direction"] == direction].copy()


def window_class_counts(repo):
    """eval_window -> (n_vulnerable, n_fixed), recovered from the run that logs them.

    A window's counts appear on the row where that window is the *training*
    window (n0_current/n1_current describe the window's own data; n0_train/
    n1_train include replayed samples, so they are not usable here).
    """
    d = pd.read_csv(os.path.join(repo, COUNTS_FROM))
    return {r.train_window: (float(r.n0_current), float(r.n1_current))
            for r in d.itertuples()}


def invert(fw, counts):
    """Recover per-window class-0 F1 and Macro-F1. Returns (frame, max_residual)."""
    fw = fw[fw["eval_window"].isin(counts)].copy()
    if fw.empty:
        return None, np.inf
    n0 = np.array([counts[w][0] for w in fw["eval_window"]])
    n1 = np.array([counts[w][1] for w in fw["eval_window"]])
    P, R = fw["precision"].values, fw["recall"].values

    safe_P = np.where(P == 0, 1.0, P)
    TP = R * n1
    FN = n1 - TP
    FP = np.where(P > 0, TP * (1 - P) / safe_P, 0.0)
    TN = n0 - FP

    residual = np.abs((TP + TN) / (n0 + n1) - fw["accuracy"].values).max()

    def f1(p, r):
        return 2 * p * r / np.where(p + r == 0, 1.0, p + r)

    f1_fixed = f1(P, R)
    f1_vuln = f1(TN / np.where(TN + FN == 0, 1.0, TN + FN),
                 TN / np.where(TN + FP == 0, 1.0, TN + FP))

    return pd.DataFrame({
        "window": fw["eval_window"].values,
        "f1_fixed": f1_fixed,          # what the paper prints as "Macro-F1"
        "f1_vulnerable": f1_vuln,      # the class a detector is actually for
        "macro_f1": (f1_fixed + f1_vuln) / 2,
    }), residual


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="checkout containing result/")
    ap.add_argument("--tol", type=float, default=1e-6,
                    help="max accuracy residual before a method is flagged unverified")
    ap.add_argument("--baseline", default="Window-only")
    ap.add_argument("--out", help="optional CSV of per-window recomputed metrics")
    args = ap.parse_args()

    if not os.path.exists(os.path.join(args.repo, COUNTS_FROM)):
        sys.exit(f"class counts unavailable: {os.path.join(args.repo, COUNTS_FROM)} not found")
    counts = window_class_counts(args.repo)

    res, resid, missing = {}, {}, []
    for name, spec in SOURCES.items():
        fw = load(args.repo, spec)
        if fw is None:
            missing.append(name)
            continue
        frame, r = invert(fw, counts)
        if frame is not None:
            res[name], resid[name] = frame, r
    if missing:
        print(f"note: no result file for {', '.join(missing)}\n")

    print("=" * 78)
    print("Per-method means. 'residual' is max |reconstructed accuracy - logged accuracy|;")
    print("anything above the tolerance means the inversion is not trustworthy for that row.")
    print("=" * 78)
    print(f"{'method':13s} {'n':>3s} {'residual':>9s} {'f1(FIXED)':>10s} {'paper':>7s} "
          f"{'f1(VULN)':>9s} {'MacroF1':>8s} {'delta':>7s}")
    for name in sorted(res, key=lambda k: -res[k]["f1_fixed"].mean()):
        f = res[name]
        flag = "" if resid[name] <= args.tol else "  UNVERIFIED"
        macro = f["macro_f1"].mean()
        print(f"{name:13s} {len(f):3d} {resid[name]:9.1e} {f['f1_fixed'].mean():10.4f} "
              f"{PAPER_F1.get(name, float('nan')):7.3f} {f['f1_vulnerable'].mean():9.4f} "
              f"{macro:8.4f} {macro - f['f1_fixed'].mean():+7.4f}{flag}")

    print("\n" + "=" * 78)
    print("Ranking: as published vs under the metric the paper says it uses")
    print("=" * 78)
    pub = sorted(res, key=lambda k: -res[k]["f1_fixed"].mean())
    tru = sorted(res, key=lambda k: -res[k]["macro_f1"].mean())
    for i, (a, b) in enumerate(zip(pub, tru), 1):
        print(f"{i:2d}  {a:13s} {b:13s}{'   <-- moved' if a != b else ''}")

    if wilcoxon is None:
        print("\nscipy missing; skipping significance tests")
    elif args.baseline in res:
        print("\n" + "=" * 78)
        print(f"Paired Wilcoxon vs {args.baseline}, on windows both methods evaluated")
        print("=" * 78)
        base = res[args.baseline]
        print(f"{'method':13s} {'n':>3s} " + " ".join(
            f"{c:>22s}" for c in ("f1(FIXED) [published]", "Macro-F1 [corrected]")))
        for name in pub:
            if name == args.baseline:
                continue
            j = res[name].merge(base, on="window", suffixes=("_m", "_b"))
            if len(j) < 5:
                continue
            cells = []
            for col in ("f1_fixed", "macro_f1"):
                x, y = j[f"{col}_m"], j[f"{col}_b"]
                if np.allclose(x, y):
                    cells.append(f"{'identical':>22s}")
                    continue
                p = wilcoxon(x, y).pvalue
                cells.append(f"{x.mean() - y.mean():+.4f} p={p:.4f}{'*' if p < 0.05 else ' '}".rjust(22))
            print(f"{name:13s} {len(j):3d} " + " ".join(cells))
        print("\n* p < 0.05")

    if args.out:
        allrows = pd.concat([f.assign(method=n) for n, f in res.items()], ignore_index=True)
        allrows.to_csv(args.out, index=False)
        print(f"\nper-window metrics written to {args.out}")


if __name__ == "__main__":
    main()
