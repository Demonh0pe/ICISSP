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

Exit code 0 means every window matched and the data is safe to train on.
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
    seen = set()

    for window, want in sorted(ref.items()):
        path = os.path.join(args.data_dir, f"{window}.jsonl")
        seen.add(f"{window}.jsonl")
        if not os.path.exists(path):
            missing.append(window)
            continue
        got_v, got_f, dropped, warn = read_window(path)
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
        print(f"mismatched {len(mismatched)}: {', '.join(mismatched)}")
    if missing:
        print(f"missing    {len(missing)}: {', '.join(missing)}")
    if extra:
        # Extra windows are not a failure on their own: the reference covers the
        # 41 windows that appear in the logged run, and the timeline has 42.
        print(f"not in reference ({len(extra)}): {', '.join(extra)}")

    print()
    if mismatched or missing:
        print("VERDICT: this is NOT the dataset behind the published numbers.")
        print("  Any run on it is a new experiment, not a reproduction -- say so in the paper.")
        if mismatched and not missing:
            print("  Same windows, different contents: likely a different dedup or language filter.")
        return 1
    print("VERDICT: matches the published dataset on every reference window.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
