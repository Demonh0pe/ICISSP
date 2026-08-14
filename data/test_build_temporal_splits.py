"""Test build_temporal_splits.py against a synthetic CVEfixes dump and NVD feed.

Focuses on the properties that silently corrupt a temporal study: which window a
sample lands in, whether a duplicate keeps its earliest date, and whether output
is in the exact shape the trainer reads.

    python data/test_build_temporal_splits.py
"""

import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "build_temporal_splits.py")

SCHEMA = """
CREATE TABLE cve (cve_id TEXT PRIMARY KEY, published_date TEXT, status TEXT);
CREATE TABLE fixes (cve_id TEXT, hash TEXT, repo_url TEXT);
CREATE TABLE file_change (file_change_id TEXT PRIMARY KEY, hash TEXT,
                          filename TEXT, programming_language TEXT);
CREATE TABLE method_change (method_change_id TEXT PRIMARY KEY, file_change_id TEXT,
                            name TEXT, code TEXT, before_change TEXT);
"""

FAILURES = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        FAILURES.append(msg)


def build_db(path):
    con = sqlite3.connect(path)
    con.executescript(SCHEMA)
    n = [0]

    def cve(cid, date, status="Analyzed"):
        con.execute("INSERT OR IGNORE INTO cve VALUES (?,?,?)", (cid, date, status))

    def fix(cid, h, fcid, lang="C"):
        con.execute("INSERT INTO fixes VALUES (?,?,?)", (cid, h, f"https://x/{h}"))
        con.execute("INSERT OR IGNORE INTO file_change VALUES (?,?,?,?)", (fcid, h, "a.c", lang))

    def method(fcid, code, before):
        n[0] += 1
        con.execute("INSERT INTO method_change VALUES (?,?,?,?,?)",
                    (f"m{n[0]}", fcid, "f", code, before))

    # One vulnerable/fixed pair per bi-monthly window of 2018.
    for i, month in enumerate([1, 3, 5, 7, 9, 11]):
        cid = f"CVE-2018-{100 + i}"
        cve(cid, f"2018-{month:02d}-15T00:00Z")
        fix(cid, f"h{i}", f"fc{i}")
        method(f"fc{i}", f"int vuln{i}() {{ strcpy(buf, in); return {i}; }}", "True")
        method(f"fc{i}", f"int vuln{i}() {{ strncpy(buf, in, 7); return {i}; }}", "False")

    # Duplicate of the 2018-01 vulnerable method, disclosed in 2020. Dedup must
    # keep the 2018 copy, so nothing new should appear in the 2020 window.
    cve("CVE-2020-500", "2020-05-01T00:00Z")
    fix("CVE-2020-500", "hdup", "fcdup")
    method("fcdup", "int vuln0() { strcpy(buf, in); return 0; }", "True")

    # Rejected CVE -- excluded by default.
    cve("CVE-2019-900", "2019-03-01T00:00Z", status="Rejected")
    fix("CVE-2019-900", "hrej", "fcrej")
    method("fcrej", "void rejected_entry(void) { gets(b); }", "True")

    # Python row, for the language filter.
    cve("CVE-2019-777", "2019-07-04T00:00Z")
    fix("CVE-2019-777", "hpy", "fcpy", "Python")
    method("fcpy", "def h(r):\n    return eval(r.body)\n", "True")

    # Date only in the NVD feed, not in CVEfixes.
    con.execute("INSERT INTO fixes VALUES (?,?,?)", ("CVE-2019-321", "hnvd", "x"))
    con.execute("INSERT INTO file_change VALUES (?,?,?,?)", ("fcnvd", "hnvd", "a.c", "C"))
    method("fcnvd", "int only_dated_by_nvd(void) { return 1; }", "True")

    # Fix left this method unchanged: same text, both labels -> both dropped.
    cve("CVE-2018-700", "2018-03-02T00:00Z")
    fix("CVE-2018-700", "hc", "fcc")
    method("fcc", "void untouched_by_the_fix(void) { return; }", "True")
    method("fcc", "void untouched_by_the_fix(void) { return; }", "False")

    con.commit(); con.close()


def build_nvd(path):
    os.makedirs(path, exist_ok=True)
    # Filename year deliberately disagrees with publishedDate, as the real feeds do.
    json.dump({"CVE_Items": [
        {"cve": {"CVE_data_meta": {"ID": "CVE-2019-321"}}, "publishedDate": "2019-09-09T00:00Z"},
        {"cve": {"CVE_data_meta": {"ID": "CVE-2018-100"}}, "publishedDate": "2023-01-01T00:00Z"},
    ]}, open(os.path.join(path, "nvdcve-1.1-2008.json"), "w"))


def run(db, out, *extra):
    r = subprocess.run([sys.executable, SCRIPT, "--db", db, "--out", out, *extra],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout, r.stderr)
        raise SystemExit("build_temporal_splits.py failed")
    windows = {}
    for name in os.listdir(out):
        if name.endswith(".jsonl"):
            with open(os.path.join(out, name)) as fh:
                windows[name[:-6]] = [json.loads(l) for l in fh if l.strip()]
    return json.load(open(os.path.join(out, "build_stats.json"))), windows, r.stdout


def main():
    tmp = tempfile.mkdtemp()
    try:
        db = os.path.join(tmp, "CVEfixes.db")
        nvd = os.path.join(tmp, "nvd")
        build_db(db); build_nvd(nvd)

        print("[default: C/C++, bi-monthly]")
        stats, win, _ = run(db, os.path.join(tmp, "o1"))

        check(set(win) == {"2018_M01-02", "2018_M03-04", "2018_M05-06", "2018_M07-08",
                           "2018_M09-10", "2018_M11-12"},
              f"one window per disclosure period (got {sorted(win)})")
        check(all(len(v) == 2 for v in win.values()), "each window holds its vuln/fixed pair")

        rec = win["2018_M01-02"][0]
        check(set(rec) == {"prompt", "response"}, f"output fields are prompt/response (got {set(rec)})")
        check(all(r["response"] in ("VULNERABLE", "FIXED")
                  for v in win.values() for r in v), "responses use the notebooks' vocabulary")

        print("[temporal integrity]")
        check("2020_M05-06" not in win,
              "duplicate disclosed later does not create a 2020 window")
        check(stats["dropped"].get("dropped_duplicate") == 1, "the later duplicate was dropped")
        allcode = [r["prompt"] for v in win.values() for r in v]
        check(len(allcode) == len(set(allcode)), "no code text repeats across windows")

        print("[filters]")
        check(stats["dropped"].get("dropped_rejected") == 1, "rejected CVE excluded")
        check(not any("rejected_entry" in c for c in allcode), "rejected code absent from output")
        check(stats["dropped"].get("dropped_language") == 1, "Python row filtered by default")
        check(stats["dropped"].get("dropped_contradictory") == 2,
              "both copies of the unchanged method dropped")
        check(not any("untouched_by_the_fix" in c for c in allcode), "unchanged method absent")
        check(stats["dropped"].get("dropped_no_date") == 1,
              "row with no disclosure date dropped when NVD is not supplied")

        print("[--nvd-dir]")
        stats2, win2, _ = run(db, os.path.join(tmp, "o2"), "--nvd-dir", nvd)
        check(stats2["dropped"].get("dropped_no_date", 0) == 0, "NVD supplies the missing date")
        check("2019_M09-10" in win2,
              f"NVD-dated row lands in its publishedDate window (got {sorted(win2)})")
        check(any("only_dated_by_nvd" in r["prompt"] for r in win2.get("2019_M09-10", [])),
              "the NVD-dated method is the one in that window")
        check("2023_M01-02" not in win2,
              "CVEfixes date wins by default, so CVE-2018-100 stays in 2018")

        print("[--prefer-nvd]")
        _, win3, _ = run(db, os.path.join(tmp, "o3"), "--nvd-dir", nvd, "--prefer-nvd")
        check("2023_M01-02" in win3, "NVD date overrides CVEfixes when asked")

        print("[--languages all]")
        stats4, win4, _ = run(db, os.path.join(tmp, "o4"), "--languages", "all")
        check(stats4["dropped"].get("dropped_language", 0) == 0, "no language filtering")
        check(any("eval(r.body)" in r["prompt"] for v in win4.values() for r in v),
              "the Python sample now appears")

        print("[--granularity]")
        _, win5, _ = run(db, os.path.join(tmp, "o5"), "--granularity", "3")
        check(all("-" in t for t in win5), "quarterly tags keep the range form")
        check(set(win5) == {"2018_M01-03", "2018_M04-06", "2018_M07-09", "2018_M10-12"},
              f"quarterly windows (got {sorted(win5)})")
        _, win6, _ = run(db, os.path.join(tmp, "o6"), "--granularity", "1")
        check(all(t.count("-") == 0 for t in win6), "monthly tags have no range suffix")

        print("[verify_dataset.py reads the output]")
        ref = os.path.join(tmp, "ref.json")
        json.dump({"windows": {t: {"vulnerable": sum(r["response"] == "VULNERABLE" for r in v),
                                   "fixed": sum(r["response"] == "FIXED" for r in v)}
                               for t, v in win.items()}}, open(ref, "w"))
        r = subprocess.run([sys.executable, os.path.join(HERE, "verify_dataset.py"),
                            "--data-dir", os.path.join(tmp, "o1"), "--reference", ref, "--quiet"],
                           capture_output=True, text=True)
        check(r.returncode == 0, f"round-trips through the verifier (rc={r.returncode})")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED")
        for f in FAILURES:
            print("  -", f)
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
