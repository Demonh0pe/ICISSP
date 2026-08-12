"""Self-contained test for prepare_cvefixes.py against a synthetic CVEfixes dump.

Builds a miniature database with the same schema as the real CVEfixes release and
asserts on the behaviours that are easy to break and expensive to notice later:
dedup, contradictory-pair removal, language filtering, and — most importantly —
that no split group ever straddles train/test.

    python data/test_prepare_cvefixes.py
"""

import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "prepare_cvefixes.py")

SCHEMA = """
CREATE TABLE fixes (cve_id TEXT, hash TEXT, repo_url TEXT);
CREATE TABLE file_change (
    file_change_id TEXT PRIMARY KEY, hash TEXT, filename TEXT,
    programming_language TEXT);
CREATE TABLE method_change (
    method_change_id TEXT PRIMARY KEY, file_change_id TEXT, name TEXT,
    code TEXT, before_change TEXT);
CREATE TABLE cwe_classification (cve_id TEXT, cwe_id TEXT);
"""

def build_db(path):
    con = sqlite3.connect(path)
    con.executescript(SCHEMA)
    mid = [0]

    def method(fcid, code, before, name="f"):
        mid[0] += 1
        con.execute("INSERT INTO method_change VALUES (?,?,?,?,?)",
                    (f"m{mid[0]}", fcid, name, code, before))

    def fixture(cve, commit, repo, fcid, lang="C", fname="a.c"):
        con.execute("INSERT INTO fixes VALUES (?,?,?)", (cve, commit, repo))
        con.execute("INSERT INTO file_change VALUES (?,?,?,?)", (fcid, commit, fname, lang))

    # 12 ordinary CVEs, each a vulnerable/fixed pair -- enough groups to split.
    for i in range(12):
        fixture(f"CVE-2020-{1000+i}", f"h{i}", f"https://github.com/org/repo{i}", f"fc{i}")
        method(f"fc{i}", f"int a{i}() {{ char b[8]; strcpy(b, input); return {i}; }}", "True")
        method(f"fc{i}", f"int a{i}() {{ char b[8]; strncpy(b, input, 7); return {i}; }}", "False")

    # Exact duplicate of CVE-2020-1000's vulnerable method, different CVE/repo.
    fixture("CVE-2021-7777", "hdup", "https://github.com/org/vendored", "fcdup")
    method("fcdup", "int a0() { char b[8]; strcpy(b, input); return 0; }", "True")

    # Same text reindented -- must dedup with the one above too.
    fixture("CVE-2021-7778", "hws", "https://github.com/org/reindent", "fcws")
    method("fcws", "int a0() {\n    char b[8];\n    strcpy(b, input);\n    return 0;\n}", "True")

    # Contradictory pair: identical text labelled both ways -> both dropped.
    fixture("CVE-2021-8888", "hc", "https://github.com/org/contra", "fcc")
    method("fcc", "void untouched_by_the_fix(void) { return; }", "True")
    method("fcc", "void untouched_by_the_fix(void) { return; }", "False")

    # Python row, for the language filter.
    fixture("CVE-2021-9999", "hpy", "https://github.com/org/pyproj", "fcpy", "Python", "a.py")
    method("fcpy", "def handler(req):\n    return eval(req.body)\n", "True")

    # Junk before_change value, and a stub below --min-chars.
    fixture("CVE-2021-6666", "hbad", "https://github.com/org/bad", "fcbad")
    method("fcbad", "int whatever(void) { return 1; }", "MAYBE")
    method("fcbad", "x;", "True")

    con.commit()
    con.close()


def run(db, out, *extra):
    cmd = [sys.executable, SCRIPT, "--db", db, "--out", out, *extra]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout, r.stderr)
        raise SystemExit(f"prepare_cvefixes.py failed: {' '.join(cmd)}")
    with open(os.path.join(out, "stats.json")) as fh:
        stats = json.load(fh)
    rows = {}
    for name in ("train", "val", "test"):
        with open(os.path.join(out, f"{name}.jsonl")) as fh:
            rows[name] = [json.loads(l) for l in fh]
    return stats, rows, r.stdout


def main():
    tmp = tempfile.mkdtemp()
    failures = []

    def check(cond, msg):
        print(("  ok   " if cond else "  FAIL ") + msg)
        if not cond:
            failures.append(msg)

    try:
        db = os.path.join(tmp, "CVEfixes.db")
        build_db(db)

        print("[default run: all languages, split by cve]")
        stats, rows, _ = run(db, os.path.join(tmp, "out"))
        allrows = rows["train"] + rows["val"] + rows["test"]
        codes = [r["code"] for r in allrows]

        check(stats["dropped"].get("dropped_bad_flag") == 1, "junk before_change dropped")
        check(stats["dropped"].get("dropped_too_short") == 1, "stub below --min-chars dropped")
        # exact duplicate + whitespace-variant duplicate
        check(stats["dropped"].get("dropped_duplicate") == 2, "both duplicate copies dropped")
        check(stats["dropped"].get("dropped_contradictory") == 2,
              "both members of the contradictory pair counted")
        # Drop reasons must be disjoint -- these totals go in the paper.
        check(stats["kept"] + sum(stats["dropped"].values()) == 31,
              f"kept + dropped == 31 rows inserted (got {stats['kept']} + "
              f"{sum(stats['dropped'].values())})")
        check(not any("untouched_by_the_fix" in c for c in codes),
              "contradictory method absent from output")
        check(len(codes) == len(set(codes)), "no duplicate code text across all splits")

        n_pairs = 12
        check(len(allrows) == n_pairs * 2 + 1, f"kept 25 samples, got {len(allrows)}")
        check(sum(r["label"] for r in allrows) == n_pairs + 1, "vulnerable count correct")
        check(all(r["label"] in (0, 1) for r in allrows), "labels are 0/1")
        check(any(r["cwe_ids"] == [] for r in allrows), "cwe_ids present as a list")
        check(all(r["repo_url"].startswith("https://") for r in allrows), "repo_url resolved")

        print("[grouped split integrity]")
        where = {}
        for name in ("train", "val", "test"):
            for r in rows[name]:
                for cve in r["cve_ids"]:
                    where.setdefault(cve, set()).add(name)
        straddling = {c: s for c, s in where.items() if len(s) > 1}
        check(not straddling, f"no CVE straddles splits (offenders: {straddling})")
        check(all(stats["counts"][n]["n"] > 0 for n in ("train", "val", "test")),
              "all three splits non-empty")

        print("[--languages c]")
        stats_c, rows_c, _ = run(db, os.path.join(tmp, "out_c"), "--languages", "c")
        allc = rows_c["train"] + rows_c["val"] + rows_c["test"]
        check(all(r["lang"].lower() == "c" for r in allc), "only C survives the filter")
        check(stats_c["dropped"].get("dropped_language") == 1, "the Python row was filtered")
        check("python" in stats_c["languages_seen"], "languages_seen records pre-filter counts")

        print("[--split-by repo]")
        stats_r, rows_r, _ = run(db, os.path.join(tmp, "out_r"), "--split-by", "repo")
        check(stats_r["repo_overlap_train_test"]["n_repos"] == 0,
              "repo split leaves zero train/test repo overlap")
        seen_repo = {}
        for name in ("train", "val", "test"):
            for r in rows_r[name]:
                seen_repo.setdefault(r["repo_url"], set()).add(name)
        check(all(len(v) == 1 for v in seen_repo.values()), "no repo straddles splits")

        print("[--text-field / --label-field renaming]")
        _, rows_f, _ = run(db, os.path.join(tmp, "out_f"),
                           "--text-field", "func", "--label-field", "target")
        one = (rows_f["train"] + rows_f["test"])[0]
        check("func" in one and "target" in one, "custom field names applied")
        check("code" not in one and "label" not in one, "default field names replaced")

        print("[determinism]")
        s1, r1, _ = run(db, os.path.join(tmp, "d1"), "--seed", "7")
        s2, r2, _ = run(db, os.path.join(tmp, "d2"), "--seed", "7")
        check(r1 == r2, "same seed reproduces identical splits")
        s3, r3, _ = run(db, os.path.join(tmp, "d3"), "--seed", "8")
        check(s3["counts"] != s1["counts"] or r3 != r1, "different seed changes the split")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print()
    if failures:
        print(f"{len(failures)} FAILED")
        for f in failures:
            print("  -", f)
        sys.exit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
