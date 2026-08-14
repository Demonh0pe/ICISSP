"""Tests for build_from_patches.py, covering both reconstruction modes.

The faithful-mode assertions pin the *original* behaviour, flaws included, so
the conference dataset stays reproducible. The hunk-mode assertions check that
the flaws are actually gone rather than merely renamed.

    python data/test_build_from_patches.py
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "build_from_patches.py")

FAILURES = []

PATCH_MIXED = """From abc Mon Sep 17 00:00:00 2001
Subject: [PATCH] fix overflow

diff --git a/src/a.c b/src/a.c
index 111..222 100644
--- a/src/a.c
+++ b/src/a.c
@@ -10,7 +10,7 @@ int handle(char *in)
 int handle(char *in) {
     char buf[8];
-    strcpy(buf, in);
+    strncpy(buf, in, 7);
     return 0;
 }
diff --git a/docs/readme.md b/docs/readme.md
--- a/docs/readme.md
+++ b/docs/readme.md
@@ -1,2 +1,2 @@
-old docs
+new docs
"""

# Same C change as above, disclosed later: dedup must keep the earlier copy.
PATCH_DUP = """diff --git a/src/a.c b/src/a.c
--- a/src/a.c
+++ b/src/a.c
@@ -10,7 +10,7 @@
 int handle(char *in) {
     char buf[8];
-    strcpy(buf, in);
+    strncpy(buf, in, 7);
     return 0;
 }
"""

# Two hunks in one file: hunk mode must emit them separately.
PATCH_TWO_HUNKS = """diff --git a/src/b.c b/src/b.c
--- a/src/b.c
+++ b/src/b.c
@@ -5,3 +5,3 @@
 void first(void) {
-    gets(a);
+    fgets(a, 8, stdin);
 }
@@ -50,3 +50,3 @@
 void second(void) {
-    sprintf(b, s);
+    snprintf(b, 8, s);
 }
"""


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        FAILURES.append(msg)


def run(tmp, out, *extra):
    r = subprocess.run([sys.executable, SCRIPT, "--patches", os.path.join(tmp, "patches"),
                        "--nvd-dir", os.path.join(tmp, "nvd"), "--out", out, *extra],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout, r.stderr)
        raise SystemExit("build_from_patches.py failed")
    windows = {}
    for name in os.listdir(out):
        if name.endswith(".jsonl"):
            with open(os.path.join(out, name)) as fh:
                windows[name[:-6]] = [json.loads(l) for l in fh if l.strip()]
    return json.load(open(os.path.join(out, "build_stats.json"))), windows


def code_of(rec):
    return rec["prompt"].split(":\n", 1)[1]


def main():
    tmp = tempfile.mkdtemp()
    try:
        os.makedirs(os.path.join(tmp, "patches"))
        os.makedirs(os.path.join(tmp, "nvd"))
        for name, body in [("CVE-2019-1111_0.patch", PATCH_MIXED),
                           ("CVE-2020-2222_0.patch", PATCH_DUP),
                           ("CVE-2019-3333_0.patch", PATCH_TWO_HUNKS)]:
            open(os.path.join(tmp, "patches", name), "w").write(body)
        json.dump({"CVE_Items": [
            {"cve": {"CVE_data_meta": {"ID": "CVE-2019-1111"}}, "publishedDate": "2019-05-20T00:00Z"},
            {"cve": {"CVE_data_meta": {"ID": "CVE-2019-3333"}}, "publishedDate": "2019-09-02T00:00Z"},
            {"cve": {"CVE_data_meta": {"ID": "CVE-2020-2222"}}, "publishedDate": "2020-03-11T00:00Z"},
        ]}, open(os.path.join(tmp, "nvd", "feed.json"), "w"))

        print("[faithful mode reproduces the original, flaws intact]")
        stats, win = run(tmp, os.path.join(tmp, "of"), "--mode", "faithful")
        check(stats["dedup"] is False, "dedup off by default in faithful mode")
        recs = win.get("2019_M05-06", [])
        check(len(recs) == 2, f"one VULNERABLE and one FIXED per patch (got {len(recs)})")
        vuln = [r for r in recs if r["response"] == "VULNERABLE"][0]
        body = code_of(vuln)
        check("strcpy(buf, in);" in body, "removed C line present")
        check("old docs" in body,
              "markdown line merged into the same sample -- the original flaw is preserved")
        check("int handle(char *in) {" not in body, "context lines discarded, as in the original")
        check(win.get("2020_M03-04"), "duplicate patch produces its own sample when dedup is off")

        print("[hunk mode: contiguous code with context]")
        stats, win = run(tmp, os.path.join(tmp, "oh"), "--mode", "hunk", "--languages", "c,c++")
        check(stats["dedup"] is True, "dedup on by default in hunk mode")
        recs = win.get("2019_M05-06", [])
        vuln = [r for r in recs if r["response"] == "VULNERABLE"][0]
        fixed = [r for r in recs if r["response"] == "FIXED"][0]
        vb, fb = code_of(vuln), code_of(fixed)
        check("int handle(char *in) {" in vb and "return 0;" in vb, "context retained")
        check("old docs" not in vb, "markdown file filtered out by --languages")
        check(stats["dropped"].get("dropped_language") == 1, "the markdown hunk was counted")
        check("strcpy(buf, in);" in vb and "strncpy" not in vb, "before-image holds the old line")
        check("strncpy(buf, in, 7);" in fb and "strcpy(buf, in);" not in fb,
              "after-image holds the new line")
        shared = set(vb.split("\n")) & set(fb.split("\n"))
        check(len(shared) >= 3,
              "the two labels share their context, so the label is not readable "
              "from added-vs-removed surface form")

        print("[hunk mode: per-hunk granularity]")
        recs = win.get("2019_M09-10", [])
        check(len(recs) == 4, f"two hunks -> two labelled pairs (got {len(recs)})")
        bodies = [code_of(r) for r in recs]
        check(any("gets(a);" in b for b in bodies) and any("sprintf" in b for b in bodies),
              "both hunks represented")
        check(not any("gets(a);" in b and "sprintf" in b for b in bodies),
              "hunks are not merged into one sample")

        print("[dedup keeps the earliest occurrence]")
        check("2020_M03-04" not in win,
              "the 2020 duplicate does not create a later window")
        check(stats["dropped"].get("dropped_duplicate") == 2,
              f"both sides of the duplicate dropped (got {stats['dropped'].get('dropped_duplicate')})")

        print("[--no-dedup / --min-code-chars]")
        stats2, win2 = run(tmp, os.path.join(tmp, "on"), "--mode", "hunk", "--no-dedup")
        check("2020_M03-04" in win2, "--no-dedup restores the later duplicate")
        stats3, _ = run(tmp, os.path.join(tmp, "om"), "--mode", "hunk", "--min-code-chars", "10000")
        check(stats3["kept"] == 0, "--min-code-chars can reject everything (filter is live)")
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
