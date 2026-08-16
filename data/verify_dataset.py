"""Check whether a local dataset directory is the one that produced the published results.

The conference results were computed on bi-monthly JSONL splits that are too
large to move around, so the check is done by fingerprint instead: the
Hybrid-CASR run logged per-window class counts, and those counts are a property
of the data alone. If a directory reproduces them exactly, it is the same data.

The counts are compared *after* applying the notebooks' own filter, so this
script must keep that filter identical:

    df = df[df["prompt"].astype(str).str.len() > 10]
    df = df[df["response"].isin(["VULNERABLE", "FIXED"])]

Usage:
    python data/verify_dataset.py --data-dir /root/autodl-tmp/temporal_splits_by_time

Exit code 0 means the data is usable: either an exact match, or a rebuild close
enough to stand in. A rebuild is never silently equated with the original -- the
deviation is printed so it can be reported.
"""

import argparse
import json
import os
import sys

REF_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reference_counts.json")


def read_window(path):
    """Count classes in one JSONL window, applying the notebooks' filter.

    Returns (n_vulnerable, n_fixed, n_dropped, error_or_None). Parsed line by
    line rather than with pandas so a single malformed record is reported
    instead of taking the whole file down.
    """
    n_vuln = n_fixed = n_drop = 0
    bad_lines = 0
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                bad_lines += 1
                continue
            prompt, response = rec.get("prompt"), rec.get("response")
            if not isinstance(prompt, str) or len(prompt) <= 10:
                n_drop += 1
                continue
            if response == "VULNERABLE":
                n_vuln += 1
            elif response == "FIXED":
                n_fixed += 1
            else:
                n_drop += 1
    return n_vuln, n_fixed, n_drop, (f"{bad_lines} unparseable lines" if bad_lines else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, help="directory holding <window>.jsonl files")
    ap.add_argument("--reference", default=REF_PATH)
    ap.add_argument("--quiet", action="store_true", help="only print mismatches and the verdict")
    args = ap.parse_args()

    if not os.path.isdir(args.data_dir):
        sys.exit(f"not a directory: {args.data_dir}")
    ref = json.load(open(args.reference, encoding="utf-8"))["windows"]

    matched, mismatched, missing, extra = [], [], [], []
    totals = {"got": {}, "want": {}}
    seen = set()

    for window, want in sorted(ref.items()):
        path = os.path.join(args.data_dir, f"{window}.jsonl")
        seen.add(f"{window}.jsonl")
        if not os.path.exists(path):
            missing.append(window)
            continue
        got_v, got_f, dropped, warn = read_window(path)
        totals["got"][window] = got_v + got_f
        totals["want"][window] = want["vulnerable"] + want["fixed"]
        ok = (got_v == want["vulnerable"] and got_f == want["fixed"])
        (matched if ok else mismatched).append(window)
        if not args.quiet or not ok:
            mark = "ok  " if ok else "DIFF"
            note = ""
            if not ok:
                note = (f"   expected {want['vulnerable']}/{want['fixed']}"
                        f"  delta {got_v - want['vulnerable']:+d}/{got_f - want['fixed']:+d}")
            if warn:
                note += f"   [{warn}]"
            print(f"  {mark} {window}  vuln/fixed = {got_v}/{got_f}"
                  f"{f'  (+{dropped} filtered)' if dropped else ''}{note}")

    for name in sorted(os.listdir(args.data_dir)):
        if name.endswith(".jsonl") and name not in seen:
            extra.append(name)

    print()
    print(f"matched    {len(matched)}/{len(ref)}")
    if mismatched:
        print(f"mismatched {len(mismatched)}"
              + (f": {', '.join(mismatched)}" if len(mismatched) <= 8 else ""))
    if missing:
        print(f"missing    {len(missing)}: {', '.join(missing)}")
    if extra:
        # Extra windows are not a failure on their own: the reference covers the
        # 41 windows that appear in the logged run, and the timeline has 42.
        print(f"not in reference ({len(extra)}): {', '.join(extra)}")

    # Counting mismatched windows alone cannot distinguish a rebuild that is a
    # fraction of a percent light from genuinely different data, so report the
    # magnitude and let it decide.
    total_got = sum(totals["got"].values())
    total_want = sum(totals["want"].values())
    print()
    if total_want:
        overall = 100 * (total_got - total_want) / total_want
        per_window = sorted(100 * (totals["got"][w] / totals["want"][w] - 1)
                            for w in totals["want"] if totals["want"][w])
        median = per_window[len(per_window) // 2] if per_window else 0.0
        worst = max((abs(p) for p in per_window), default=0.0)
        low = sum(1 for p in per_window if p < 0)
        print(f"samples    {total_got:,} vs {total_want:,} expected "
              f"({total_got - total_want:+,}, {overall:+.2f}%)")
        print(f"per window median {median:+.2f}%, largest deviation {worst:.2f}%, "
              f"{low}/{len(per_window)} below reference")
    else:
        overall = worst = 0.0

    print()
    if missing:
        print("VERDICT: windows are missing -- this cannot stand in for the published data.")
        return 1
    if not mismatched:
        print("VERDICT: matches the published dataset on every reference window.")
        return 0
    if abs(overall) < 2 and worst < 10:
        print(f"VERDICT: a close rebuild -- {abs(overall):.2f}% off overall, no window "
              f"more than {worst:.1f}%.")
        print("  Consistent with a handful of source commits having become unavailable.")
        print("  Usable as a stand-in, but it is a rebuild, not the original: report the")
        print("  deviation, and anchor at least one run against a published configuration")
        print("  so the difference is measured rather than assumed.")
        return 0
    print(f"VERDICT: differs substantially ({overall:+.2f}% overall, up to {worst:.1f}% "
          f"in a window).")
    print("  Any run on it is a new experiment, not a reproduction -- say so in the paper.")
    print("  Same windows, different contents: likely a different dedup or language filter.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
