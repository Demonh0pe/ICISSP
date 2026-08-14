"""Build the temporal window JSONL files the trainer consumes, from CVEfixes.db.

Output matches what the notebooks read, so existing and new runs stay comparable:

    <out>/2018_M01-02.jsonl, <out>/2018_M03-04.jsonl, ...
    {"prompt": "<function source>", "response": "VULNERABLE"|"FIXED"}

Three choices here follow the paper rather than convenience, because each one
changes the numbers:

1. Instances are timestamped by **CVE disclosure date**, not commit date.
   Commits routinely precede disclosure by weeks under coordinated release, so
   using commit dates leaks future knowledge into training windows.

2. Deduplication keeps the **earliest** occurrence along the timeline. Keeping
   an arbitrary copy would let a function that first appears in 2019 be assigned
   to a 2022 window, breaking the forward-chaining guarantee that a window's
   test data was never seen during training.

3. A function whose "fix" left it byte-identical carries no signal and appears
   under both labels; both copies are dropped.

The NVD feeds under data/nvd/ can supply disclosure dates via --nvd-dir. Those
files are named by CVE ID year, not disclosure year, so the date is always read
from each record's publishedDate field and never inferred from the filename.

Usage:
    python data/build_temporal_splits.py --db CVEfixes.db \
        --out data/temporal_splits_by_time --granularity 2 --languages c,c++

    # the "open up the language filter" run
    python data/build_temporal_splits.py --db CVEfixes.db \
        --out data/splits_all --languages all

Verify the result against the published dataset with:
    python data/verify_dataset.py --data-dir <out>
"""

import argparse
import collections
import glob
import hashlib
import json
import os
import re
import sqlite3
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "experiments"))

_WS = re.compile(r"\s+")
# Block comments then line comments; used only for the dedup key, never for output.
_COMMENT = re.compile(r"/\*.*?\*/|//[^\n]*|#[^\n]*", re.S)


def norm_key(code, strip_comments):
    """Whitespace-insensitive dedup key, optionally comment-insensitive.

    The paper describes normalised hashes with whitespace *and* comments removed.
    Comment stripping is regex-based and therefore approximate on string
    literals containing // or #, so it is opt-in via --strip-comments.
    """
    if strip_comments:
        code = _COMMENT.sub(" ", code)
    return hashlib.sha1(_WS.sub(" ", code).strip().encode("utf-8", "replace")).hexdigest()


def window_tag(year, month, months):
    start = ((month - 1) // months) * months + 1
    if months == 1:
        return f"{year}_M{start:02d}"
    return f"{year}_M{start:02d}-{start + months - 1:02d}"


def check_schema(con):
    have = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    required = {"method_change": ["method_change_id", "file_change_id", "code", "before_change"],
                "file_change": ["file_change_id", "hash", "programming_language"],
                "fixes": ["cve_id", "hash"],
                "cve": ["cve_id", "published_date"]}
    problems = []
    for table, cols in required.items():
        if table not in have:
            problems.append(f"missing table '{table}'")
            continue
        actual = {r[1] for r in con.execute(f"PRAGMA table_info({table})")}
        problems += [f"{table}.{c} missing (found: {sorted(actual)})" for c in cols if c not in actual]
    if problems:
        sys.exit("CVEfixes schema mismatch:\n  " + "\n  ".join(problems)
                 + "\n\nLikely a different CVEfixes release; adjust the query below.")


def nvd_dates(nvd_dir):
    """cve_id -> publishedDate, read from NVD 1.1 feeds.

    Dates come from each record, never from the filename: nvdcve-1.1-2008.json
    holds CVE *IDs* from 2008, some disclosed years later.
    """
    out = {}
    files = sorted(glob.glob(os.path.join(nvd_dir, "**", "*.json"), recursive=True))
    for path in files:
        try:
            with open(path, encoding="utf-8") as fh:
                blob = json.load(fh)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        for item in blob.get("CVE_Items", []):
            try:
                cid = item["cve"]["CVE_data_meta"]["ID"]
                pub = item["publishedDate"]
            except (KeyError, TypeError):
                continue
            # Keep the earliest date if a CVE appears in more than one feed.
            if cid not in out or pub < out[cid]:
                out[cid] = pub
    print(f"NVD: {len(out)} disclosure dates from {len(files)} feed file(s)")
    return out


def build(args):
    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    check_schema(con)
    con.row_factory = sqlite3.Row

    tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    cve_cols = {r[1] for r in con.execute("PRAGMA table_info(cve)")}

    # CVEfixes marks withdrawn entries inconsistently across releases; filter on
    # whichever status column exists.
    status_col = next((c for c in ("status", "cve_status", "vuln_status") if c in cve_cols), None)
    rejected = set()
    if status_col and not args.keep_rejected:
        rejected = {r[0] for r in con.execute(
            f"SELECT cve_id FROM cve WHERE LOWER(COALESCE({status_col},'')) LIKE '%reject%'")}
        print(f"excluding {len(rejected)} rejected CVE(s) via cve.{status_col}")

    db_dates = {r["cve_id"]: r["published_date"]
                for r in con.execute("SELECT cve_id, published_date FROM cve "
                                     "WHERE published_date IS NOT NULL")}
    print(f"CVEfixes: {len(db_dates)} disclosure dates")
    dates = dict(db_dates)
    if args.nvd_dir:
        feed = nvd_dates(args.nvd_dir)
        added = sum(1 for k in feed if k not in dates)
        if args.prefer_nvd:
            dates.update(feed)
            print(f"NVD dates take precedence ({added} CVEs added)")
        else:
            for k, v in feed.items():
                dates.setdefault(k, v)
            print(f"NVD used only to fill gaps ({added} CVEs added)")

    rows = con.execute("""
        SELECT mc.method_change_id AS mid, mc.code AS code,
               mc.before_change AS before_change,
               fc.programming_language AS lang,
               group_concat(DISTINCT f.cve_id) AS cve_ids
        FROM method_change mc
        JOIN file_change fc ON fc.file_change_id = mc.file_change_id
        JOIN fixes f        ON f.hash = fc.hash
        GROUP BY mc.method_change_id
    """).fetchall()
    con.close()
    print(f"method_change rows joined: {len(rows)}")
    if not rows:
        sys.exit("No rows. Is this an empty or partially-built CVEfixes dump?")

    want_langs = None if args.languages == "all" else {
        s.strip().lower() for s in args.languages.split(",") if s.strip()}

    stats = collections.Counter()
    langs_seen = collections.Counter()
    candidates = []

    for r in rows:
        stats["total"] += 1
        flag = (r["before_change"] or "").strip().lower()
        if flag in ("true", "1"):
            label = "VULNERABLE"
        elif flag in ("false", "0"):
            label = "FIXED"
        else:
            stats["dropped_bad_flag"] += 1
            continue

        code = r["code"] or ""
        if len(code.strip()) <= args.min_chars:
            stats["dropped_too_short"] += 1
            continue
        if args.max_chars and len(code) > args.max_chars:
            stats["dropped_too_long"] += 1
            continue

        lang = (r["lang"] or "unknown").strip()
        langs_seen[lang.lower()] += 1
        if want_langs is not None and lang.lower() not in want_langs:
            stats["dropped_language"] += 1
            continue

        cve_ids = [c for c in (r["cve_ids"] or "").split(",") if c]
        if not args.keep_rejected and cve_ids and all(c in rejected for c in cve_ids):
            stats["dropped_rejected"] += 1
            continue

        # A method can belong to several CVEs; the earliest disclosure is the
        # first moment the world could have known about it.
        avail = sorted(dates[c] for c in cve_ids if c in dates)
        if not avail:
            stats["dropped_no_date"] += 1
            continue
        candidates.append((avail[0], code, label))

    # Earliest-first, so dedup naturally keeps the first appearance.
    candidates.sort(key=lambda t: t[0])
    seen = {}
    kept = []
    for pub, code, label in candidates:
        key = norm_key(code, args.strip_comments)
        if key in seen:
            if seen[key][1] != label:
                seen[key] = (seen[key][0], label, True)   # mark contradictory
                stats["dropped_contradictory"] += 1
            else:
                stats["dropped_duplicate"] += 1
            continue
        seen[key] = (len(kept), label, False)
        kept.append((pub, code, label, key))

    contradictory = {k for k, v in seen.items() if v[2]}
    before = len(kept)
    kept = [t for t in kept if t[3] not in contradictory]
    stats["dropped_contradictory"] += before - len(kept)

    if not kept:
        sys.exit("Everything was filtered out. Loosen --languages / --min-chars.")

    buckets = collections.defaultdict(list)
    for pub, code, label, _ in kept:
        m = re.match(r"(\d{4})-(\d{2})", str(pub))
        if not m:
            stats["dropped_unparseable_date"] += 1
            continue
        year, month = int(m.group(1)), int(m.group(2))
        if not (args.start_year <= year <= args.end_year):
            stats["dropped_out_of_range"] += 1
            continue
        buckets[window_tag(year, month, args.granularity)].append(
            {"prompt": code, "response": label})

    os.makedirs(args.out, exist_ok=True)
    counts = {}
    for tag, recs in sorted(buckets.items()):
        with open(os.path.join(args.out, f"{tag}.jsonl"), "w", encoding="utf-8") as fh:
            for rec in recs:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        counts[tag] = {"vulnerable": sum(r["response"] == "VULNERABLE" for r in recs),
                       "fixed": sum(r["response"] == "FIXED" for r in recs)}

    summary = {
        "db": os.path.abspath(args.db),
        "granularity_months": args.granularity,
        "languages": args.languages,
        "date_source": "nvd-preferred" if args.prefer_nvd else "cvefixes-preferred",
        "dropped": {k: v for k, v in sorted(stats.items()) if k.startswith("dropped")},
        "kept": sum(c["vulnerable"] + c["fixed"] for c in counts.values()),
        "windows": counts,
    }
    with open(os.path.join(args.out, "build_stats.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1, ensure_ascii=False)

    print(f"\nkept {summary['kept']} of {stats['total']}")
    for k, v in summary["dropped"].items():
        print(f"  {k:26s} {v}")
    print(f"\n{len(counts)} windows written to {args.out}")
    empty = [t for t, c in counts.items() if c["vulnerable"] == 0 or c["fixed"] == 0]
    for tag in sorted(counts)[:6]:
        c = counts[tag]
        n = c["vulnerable"] + c["fixed"]
        print(f"  {tag}  n={n:<6d} vuln={c['vulnerable']:<5d} fixed={c['fixed']:<5d} "
              f"({100 * c['vulnerable'] / max(1, n):.0f}% vuln)")
    if len(counts) > 6:
        print(f"  ... {len(counts) - 6} more (see build_stats.json)")
    if empty:
        print(f"\nWARNING: {len(empty)} window(s) have only one class and will "
              f"produce degenerate metrics: {', '.join(empty[:8])}")
    print("\nNext: python data/verify_dataset.py --data-dir " + args.out)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--db", required=True, help="path to CVEfixes.db")
    p.add_argument("--out", required=True, help="output directory for <window>.jsonl")
    p.add_argument("--granularity", type=int, default=2, help="window width in months")
    p.add_argument("--languages", default="c,c++",
                   help="'all' or comma-separated; the conference run used C/C++")
    p.add_argument("--start-year", type=int, default=2018)
    p.add_argument("--end-year", type=int, default=2024)
    p.add_argument("--nvd-dir", help="directory of NVD 1.1 feeds, to fill disclosure dates")
    p.add_argument("--prefer-nvd", action="store_true",
                   help="let NVD dates override CVEfixes rather than only fill gaps")
    p.add_argument("--min-chars", type=int, default=10,
                   help="matches the trainer's len(prompt) > 10 filter")
    p.add_argument("--max-chars", type=int, default=20000, help="0 disables")
    p.add_argument("--strip-comments", action="store_true",
                   help="ignore comments when deduplicating (approximate)")
    p.add_argument("--keep-rejected", action="store_true",
                   help="keep CVEs marked Rejected (the paper excludes them)")
    args = p.parse_args()
    if 12 % args.granularity:
        sys.exit(f"--granularity must divide 12, got {args.granularity}")
    if not os.path.exists(args.db):
        sys.exit(f"not found: {args.db}")
    build(args)


if __name__ == "__main__":
    main()
