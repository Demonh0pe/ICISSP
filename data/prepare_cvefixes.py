"""Build a method-level vulnerability-detection dataset from CVEfixes.db.

CVEfixes stores, for each CVE-fixing commit, the changed methods in both their
pre-fix and post-fix form (`method_change.before_change`). That gives a natural
binary label:

    before_change == 'True'   -> label 1  (vulnerable)
    before_change == 'False'  -> label 0  (fixed)

Two things this script is deliberate about, because both silently inflate
reported F1 and both are easy to get wrong:

1. Deduplication. The same method text recurs constantly across CVEfixes --
   vendored copies, backport commits across maintenance branches, one commit
   listed under several CVEs. Left in, identical code lands in train and test.
   Dedup is global and runs before splitting.

2. Grouped splitting. Splitting samples at random puts the pre-fix and post-fix
   version of the *same method* on opposite sides of the split: near-identical
   text, opposite labels. Splits here are always along whole groups (--split-by),
   never individual samples. The report prints how much repo overlap remains so
   the number can go in the paper instead of being assumed away.

Usage:
    python data/prepare_cvefixes.py --db CVEfixes.db --out data/processed

    # keep every language (the "open up the language filter" run)
    python data/prepare_cvefixes.py --db CVEfixes.db --out data/processed_all \
        --languages all

    # stricter split: no repository appears on both sides
    python data/prepare_cvefixes.py --db CVEfixes.db --out data/processed_repo \
        --split-by repo

Output: train.jsonl / val.jsonl / test.jsonl (one JSON object per line) plus
stats.json. Field names are configurable via --text-field / --label-field so the
output can be pointed at an existing trainer without editing the loader.
"""

import argparse
import collections
import hashlib
import json
import os
import random
import re
import sqlite3
import sys

# CVEfixes has shipped a few schema revisions; these are the columns this script
# actually reads. Checked up front so a mismatch is one clear message instead of
# an OperationalError from somewhere inside the join.
REQUIRED = {
    "method_change": ["method_change_id", "file_change_id", "name", "code", "before_change"],
    "file_change": ["file_change_id", "hash", "filename", "programming_language"],
    "fixes": ["cve_id", "hash"],
}

_WS = re.compile(r"\s+")


def check_schema(con):
    have = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    problems = []
    for table, cols in REQUIRED.items():
        if table not in have:
            problems.append(f"missing table '{table}'")
            continue
        actual = {r[1] for r in con.execute(f"PRAGMA table_info({table})")}
        for c in cols:
            if c not in actual:
                problems.append(f"{table}.{c} missing (found: {sorted(actual)})")
    if problems:
        sys.exit(
            "CVEfixes schema does not match what this script expects:\n  "
            + "\n  ".join(problems)
            + "\n\nThis usually means a different CVEfixes release. Adjust REQUIRED "
              "and the query in fetch_rows() to match your dump."
        )
    return have


def fetch_rows(con):
    """One row per changed method, with its CVE(s) and repo attached."""
    # A commit can be listed under several CVEs, which would multiply rows; the
    # group_concat collapses that back to one row per method.
    q = """
        SELECT mc.method_change_id           AS mid,
               mc.name                       AS method_name,
               mc.code                       AS code,
               mc.before_change              AS before_change,
               fc.programming_language       AS lang,
               fc.filename                   AS filename,
               fc.hash                       AS commit_hash,
               group_concat(DISTINCT f.cve_id) AS cve_ids
        FROM method_change mc
        JOIN file_change fc ON fc.file_change_id = mc.file_change_id
        JOIN fixes f        ON f.hash = fc.hash
        GROUP BY mc.method_change_id
    """
    con.row_factory = sqlite3.Row
    return con.execute(q).fetchall()


def repo_of(con):
    """commit hash -> repo_url. Lives on `fixes` in most releases, `commits` in some."""
    for table in ("fixes", "commits"):
        cols = {r[1] for r in con.execute(f"PRAGMA table_info({table})")}
        if {"hash", "repo_url"} <= cols:
            return {h: u for h, u in con.execute(
                f"SELECT DISTINCT hash, repo_url FROM {table} WHERE repo_url IS NOT NULL")}
    return {}


def cwe_of(con):
    """cve_id -> sorted list of CWE ids. Absent in trimmed dumps; not fatal."""
    tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if "cwe_classification" not in tables:
        return {}
    out = collections.defaultdict(set)
    for cve, cwe in con.execute("SELECT cve_id, cwe_id FROM cwe_classification"):
        if cwe:
            out[cve].add(cwe)
    return {k: sorted(v) for k, v in out.items()}


def norm(code):
    """Whitespace-insensitive key, so reindented copies dedup together."""
    return _WS.sub(" ", code).strip()


def build(args):
    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    check_schema(con)

    rows = fetch_rows(con)
    repos, cwes = repo_of(con), cwe_of(con)
    con.close()
    print(f"method_change rows joined: {len(rows)}")
    if not rows:
        sys.exit("No rows returned. Is this an empty or partially-built CVEfixes dump?")

    want_langs = None if args.languages == "all" else {
        l.strip().lower() for l in args.languages.split(",") if l.strip()
    }

    stats = collections.Counter()
    lang_counter = collections.Counter()
    unexpected_flags = collections.Counter()
    seen = {}
    samples = []

    for r in rows:
        stats["total"] += 1

        flag = (r["before_change"] or "").strip().lower()
        if flag in ("true", "1"):
            label = 1
        elif flag in ("false", "0"):
            label = 0
        else:
            unexpected_flags[r["before_change"]] += 1
            stats["dropped_bad_flag"] += 1
            continue

        code = r["code"] or ""
        if len(code.strip()) < args.min_chars:
            stats["dropped_too_short"] += 1
            continue
        if args.max_chars and len(code) > args.max_chars:
            stats["dropped_too_long"] += 1
            continue

        lang = (r["lang"] or "unknown").strip()
        lang_counter[lang.lower()] += 1
        if want_langs is not None and lang.lower() not in want_langs:
            stats["dropped_language"] += 1
            continue

        key = hashlib.sha1(norm(code).encode("utf-8", "replace")).hexdigest()
        if key in seen:
            # Same text under both labels means the "fix" did not change this
            # method; it carries no signal either way, so drop both copies.
            if seen[key]["label"] != label:
                seen[key]["contradictory"] = True
                # Counted as contradictory, not duplicate, so the two drop
                # reasons stay disjoint and can be summed.
                stats["dropped_contradictory"] += 1
            else:
                stats["dropped_duplicate"] += 1
            continue

        cve_ids = sorted((r["cve_ids"] or "").split(",")) if r["cve_ids"] else []
        rec = {
            "id": str(r["mid"]),
            "code": code,
            "label": label,
            "lang": lang,
            "method_name": r["method_name"],
            "filename": r["filename"],
            "cve_ids": cve_ids,
            "cwe_ids": sorted({c for cv in cve_ids for c in cwes.get(cv, [])}),
            "repo_url": repos.get(r["commit_hash"], ""),
            "commit_hash": r["commit_hash"],
            "contradictory": False,
        }
        seen[key] = rec
        samples.append(rec)

    before_contra = len(samples)
    samples = [s for s in samples if not s.pop("contradictory")]
    stats["dropped_contradictory"] += before_contra - len(samples)

    if unexpected_flags:
        print(f"  note: unrecognised before_change values {dict(unexpected_flags)}")
    if not samples:
        sys.exit("Every row was filtered out. Loosen --languages / --min-chars.")

    # --- grouped split -----------------------------------------------------
    def group_key(s):
        if args.split_by == "repo":
            return s["repo_url"] or f"__norepo__{s['commit_hash']}"
        if args.split_by == "commit":
            return s["commit_hash"]
        return s["cve_ids"][0] if s["cve_ids"] else f"__nocve__{s['commit_hash']}"

    by_group = collections.defaultdict(list)
    for s in samples:
        by_group[group_key(s)].append(s)

    groups = sorted(by_group)  # sort first so the shuffle is seed-reproducible
    random.Random(args.seed).shuffle(groups)

    n = len(groups)
    n_test = max(1, int(n * args.test_frac))
    n_val = max(1, int(n * args.val_frac))
    split_of = {}
    for g in groups[:n_test]:
        split_of[g] = "test"
    for g in groups[n_test:n_test + n_val]:
        split_of[g] = "val"
    for g in groups[n_test + n_val:]:
        split_of[g] = "train"

    splits = collections.defaultdict(list)
    for g, items in by_group.items():
        splits[split_of[g]].extend(items)

    # --- leakage report ----------------------------------------------------
    def repos_in(name):
        return {s["repo_url"] for s in splits[name] if s["repo_url"]}

    train_repos = repos_in("train")
    overlap_test = train_repos & repos_in("test")
    leaked = sum(1 for s in splits["test"] if s["repo_url"] in overlap_test)

    os.makedirs(args.out, exist_ok=True)
    for name in ("train", "val", "test"):
        rows_out = splits[name]
        random.Random(args.seed).shuffle(rows_out)
        with open(os.path.join(args.out, f"{name}.jsonl"), "w", encoding="utf-8") as fh:
            for s in rows_out:
                rec = dict(s)
                rec[args.text_field] = rec.pop("code")
                rec[args.label_field] = rec.pop("label")
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    summary = {
        "db": os.path.abspath(args.db),
        "filters": {
            "languages": args.languages, "min_chars": args.min_chars,
            "max_chars": args.max_chars,
        },
        "split": {
            "by": args.split_by, "seed": args.seed,
            "val_frac": args.val_frac, "test_frac": args.test_frac,
            "n_groups": n,
        },
        "dropped": {k: v for k, v in sorted(stats.items()) if k.startswith("dropped")},
        "kept": len(samples),
        "counts": {
            name: {
                "n": len(splits[name]),
                "vuln": sum(s["label"] for s in splits[name]),
                "groups": len({group_key(s) for s in splits[name]}),
            } for name in ("train", "val", "test")
        },
        "languages_seen": dict(lang_counter.most_common()),
        "repo_overlap_train_test": {
            "n_repos": len(overlap_test),
            "n_test_samples_in_shared_repos": leaked,
            "pct_of_test": round(100 * leaked / max(1, len(splits["test"])), 2),
        },
    }
    with open(os.path.join(args.out, "stats.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    print(f"\nkept {len(samples)} of {stats['total']}")
    for k, v in summary["dropped"].items():
        print(f"  {k:24s} {v}")
    print()
    for name in ("train", "val", "test"):
        c = summary["counts"][name]
        pct = 100 * c["vuln"] / max(1, c["n"])
        print(f"  {name:5s} n={c['n']:<7d} vuln={c['vuln']:<7d} ({pct:.1f}%)  groups={c['groups']}")
    ov = summary["repo_overlap_train_test"]
    print(f"\nrepo overlap train/test: {ov['n_repos']} repos, "
          f"{ov['n_test_samples_in_shared_repos']} test samples ({ov['pct_of_test']}%)")
    if args.split_by != "repo" and ov["pct_of_test"] > 0:
        print("  (use --split-by repo to drive this to zero; report it either way)")
    print(f"\nwrote {args.out}/{{train,val,test}}.jsonl + stats.json")
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--db", required=True, help="path to CVEfixes.db")
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--languages", default="all",
                   help="'all' (default) or a comma-separated list, e.g. 'c,c++,python'")
    p.add_argument("--split-by", default="cve", choices=["cve", "repo", "commit"],
                   help="grouping unit for the split; 'repo' is strictest (default: cve)")
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--test-frac", type=float, default=0.1)
    p.add_argument("--min-chars", type=int, default=32, help="drop stubs shorter than this")
    p.add_argument("--max-chars", type=int, default=20000, help="0 disables")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--text-field", default="code", help="output field name for the source text")
    p.add_argument("--label-field", default="label", help="output field name for the 0/1 label")
    args = p.parse_args()

    if not os.path.exists(args.db):
        sys.exit(f"not found: {args.db}")
    build(args)


if __name__ == "__main__":
    main()
