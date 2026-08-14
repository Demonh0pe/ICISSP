"""Re-download the commit patches the dataset is built from.

The original pipeline (vuln_LLM(phi2)_2morth.py) wrote patches to Colab's
ephemeral /content/patches, so they are gone while the inputs that produced
them -- the NVD feeds and github_links_*.json -- survive on disk. This rebuilds
that directory.

Each patch is one HTTP request, not a clone, so the whole set is hours rather
than days. Downloads are cached and resumable: rerun after an interruption and
only the missing ones are fetched.

Authentication matters. Unauthenticated GitHub allows ~60 requests/hour and
will stall a run of this size almost immediately; a token raises that to 5000.

    export GITHUB_TOKEN=ghp_...
    python data/fetch_patches.py --links data/nvd/github_links_2015_2024.json \
        --out data/patches

Rerun until "missing" reaches zero or stops falling. Some commits are
permanently gone -- deleted repos, force-pushed history, renamed owners -- and
those are recorded in fetch_failures.json rather than retried forever.
"""

import argparse
import concurrent.futures
import glob
import json
import os
import random
import re
import sys
import threading
import time
import urllib.error
import urllib.request

COMMIT_RE = re.compile(r"github\.com/([^/]+)/([^/]+)/commit/([0-9a-fA-F]{7,40})")
_print_lock = threading.Lock()


def parse_commit(url):
    """(owner, repo, sha) from a commit URL, or None if it is not one."""
    m = COMMIT_RE.search(url.split("#")[0])
    if not m:
        return None
    owner, repo, sha = m.groups()
    return owner, repo.removesuffix(".git"), sha


def fetch(url, token, timeout):
    req = urllib.request.Request(url, headers={
        "User-Agent": "icissp-dataset-rebuild",
        # The diff media type returns the same content as <url>.patch but goes
        # through the API, where a token actually raises the rate limit.
        "Accept": "application/vnd.github.diff",
        **({"Authorization": f"Bearer {token}"} if token else {}),
    })
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", "replace")


def check_token(token, timeout):
    """Fail before the run, not 24000 doomed requests into it.

    A bad token is not distinguishable from a slow network once downloads are
    under way -- every task just retries and errors -- so the limit is read up
    front and a rejected credential is fatal.
    """
    if not token:
        print("token: NOT SET -- ~60 requests/hour, which will not finish.")
        print("  Set GITHUB_TOKEN and rerun; downloads already on disk are kept.")
        return
    # A pasted-literally placeholder can still carry a real-looking prefix, so
    # check the body too: real tokens are ASCII alphanumerics and underscores.
    if not token.isascii() or not re.fullmatch(r"[A-Za-z0-9_]{20,}", token):
        sys.exit(f"GITHUB_TOKEN is not a valid token: {token[:12]!r}...\n"
                 "  A placeholder pasted literally from instructions will do this.\n"
                 "  Re-export the real value and rerun.")
    if not token.startswith(("ghp_", "github_pat_", "gho_", "ghs_")):
        print(f"token: unfamiliar prefix {token.split('_')[0]!r}; continuing")
    try:
        body = fetch("https://api.github.com/rate_limit", token, timeout)
    except urllib.error.HTTPError as e:
        if e.code == 401:
            sys.exit("GITHUB_TOKEN was rejected (HTTP 401). Generate a new one at "
                     "https://github.com/settings/tokens and re-export it.")
        print(f"could not verify token (HTTP {e.code}); continuing")
        return
    except Exception as e:
        print(f"could not verify token ({type(e).__name__}); continuing")
        return
    try:
        core = json.loads(body)["resources"]["core"]
        print(f"token: valid, {core['remaining']}/{core['limit']} requests remaining this hour")
        if core["limit"] < 1000:
            print("  WARNING: limit under 1000/hour -- this token is not raising the "
                  "anonymous cap. Check that it is valid and not expired.")
    except (json.JSONDecodeError, KeyError):
        print("token: accepted")


def download_one(task, args, token, state):
    """Returns one of: cached | ok | gone | error."""
    cve_id, index, url = task
    parsed = parse_commit(url)
    if not parsed:
        return "gone", "not a commit URL"
    owner, repo, sha = parsed

    path = os.path.join(args.out, f"{cve_id}_{index}.patch")
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return "cached", None

    api = f"https://api.github.com/repos/{owner}/{repo}/commits/{sha}"
    for attempt in range(args.retries):
        try:
            body = fetch(api, token, args.timeout)
            if args.max_bytes and len(body.encode()) > args.max_bytes:
                return "gone", f"patch larger than --max-bytes ({len(body)} chars)"
            tmp = path + ".part"
            with open(tmp, "w", encoding="utf-8") as fh:
                fh.write(body)
            os.replace(tmp, path)
            return "ok", None
        except urllib.error.HTTPError as e:
            # 404/451 are permanent: repo deleted, made private, or DMCA'd.
            if e.code in (404, 410, 451):
                return "gone", f"HTTP {e.code}"
            if e.code in (403, 429):
                # Secondary rate limit. Honour Retry-After when present.
                wait = int(e.headers.get("Retry-After") or 0) or min(60, 2 ** attempt * 5)
                reset = e.headers.get("X-RateLimit-Remaining")
                if reset == "0":
                    with _print_lock:
                        if not state["warned"]:
                            state["warned"] = True
                            print("\n  rate limit reached; sleeping. Set GITHUB_TOKEN "
                                  "if you have not." if not token else
                                  "\n  rate limit reached; sleeping.")
                time.sleep(wait + random.uniform(0, 2))
                continue
            if attempt == args.retries - 1:
                return "error", f"HTTP {e.code}"
            time.sleep(2 ** attempt)
        except Exception as e:  # timeouts, DNS, reset connections
            if attempt == args.retries - 1:
                return "error", f"{type(e).__name__}: {e}"
            time.sleep(2 ** attempt + random.uniform(0, 1))
    return "error", "retries exhausted"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--links", required=True, help="github_links_*.json")
    ap.add_argument("--out", required=True, help="directory for .patch files")
    ap.add_argument("--workers", type=int, default=4,
                    help="keep this low; GitHub throttles aggressive clients")
    ap.add_argument("--retries", type=int, default=4)
    ap.add_argument("--timeout", type=int, default=30)
    ap.add_argument("--max-bytes", type=int, default=5_000_000,
                    help="skip enormous patches (vendored trees, generated code); 0 disables")
    ap.add_argument("--limit", type=int, help="stop after N downloads (for a trial run)")
    ap.add_argument("--nvd-dir", help="NVD feeds, needed by --min-year/--max-year")
    ap.add_argument("--min-year", type=int,
                    help="skip CVEs disclosed before this year (requires --nvd-dir)")
    ap.add_argument("--max-year", type=int,
                    help="skip CVEs disclosed after this year (requires --nvd-dir)")
    args = ap.parse_args()

    if (args.min_year or args.max_year) and not args.nvd_dir:
        ap.error("--min-year/--max-year need --nvd-dir to read disclosure dates")

    # This is meant to be run under nohup, where stdout is a file and therefore
    # block-buffered: without this every progress line sits in the buffer and
    # the log looks empty while the run is in fact working.
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except AttributeError:
        pass

    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    check_token(token, args.timeout)
    os.makedirs(args.out, exist_ok=True)

    # Year filtering uses disclosure date, never the CVE ID or feed filename:
    # nvdcve-1.1-2008.json contains entries disclosed as late as 2023, so an
    # ID-based filter would drop patches the study needs.
    years = {}
    if args.nvd_dir:
        files = sorted(glob.glob(os.path.join(args.nvd_dir, "**", "*.json"), recursive=True))
        # The feeds run to ~1 GB and take minutes to parse, so the result is
        # cached: without it every rerun repeats the scan in silence and looks
        # like a hang.
        cache_path = os.path.join(args.nvd_dir, ".disclosure_years.json")
        stamp = str(sorted((os.path.basename(f), os.path.getsize(f)) for f in files))
        if os.path.exists(cache_path):
            try:
                cached = json.load(open(cache_path, encoding="utf-8"))
                if cached.get("stamp") == stamp:
                    years = {k: int(v) for k, v in cached["years"].items()}
                    print(f"{len(years)} disclosure years (cached)")
            except (json.JSONDecodeError, KeyError, ValueError):
                pass
        if not years:
            print(f"parsing {len(files)} NVD feed file(s) -- about a minute for the full set")
            for n, path in enumerate(files, 1):
                print(f"  [{n}/{len(files)}] {os.path.basename(path)}", flush=True)
                try:
                    blob = json.load(open(path, encoding="utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
                for item in blob.get("CVE_Items", []):
                    try:
                        cid = item["cve"]["CVE_data_meta"]["ID"]
                        pub = item.get("publishedDate", "")[:4]
                    except (KeyError, TypeError):
                        continue
                    if pub.isdigit() and (cid not in years or int(pub) < years[cid]):
                        years[cid] = int(pub)
            try:
                json.dump({"stamp": stamp, "years": years}, open(cache_path, "w"))
            except OSError:
                pass
            print(f"{len(years)} disclosure years loaded")

    entries = json.load(open(args.links, encoding="utf-8"))
    tasks = []
    skipped_year = skipped_undated = 0
    for entry in entries:
        cve_id = entry["cve_id"].replace(":", "-")
        if args.min_year or args.max_year:
            y = years.get(entry["cve_id"]) or years.get(cve_id)
            if y is None:
                # No date means it cannot be placed in a window later either.
                skipped_undated += 1
                continue
            if (args.min_year and y < args.min_year) or (args.max_year and y > args.max_year):
                skipped_year += 1
                continue
        for i, link in enumerate(entry.get("github_links", [])):
            if parse_commit(link):
                tasks.append((cve_id, i, link))
    # Deterministic order so reruns are comparable.
    tasks.sort()
    print(f"{len(entries)} CVE entries -> {len(tasks)} commit links")
    if skipped_year or skipped_undated:
        print(f"  skipped {skipped_year} CVE(s) outside the year range, "
              f"{skipped_undated} with no disclosure date")

    failures_path = os.path.join(os.path.dirname(os.path.abspath(args.out)),
                                 "fetch_failures.json")
    known_gone = set()
    if os.path.exists(failures_path):
        try:
            known_gone = {tuple(k) for k in json.load(open(failures_path))["permanent"]}
            print(f"skipping {len(known_gone)} known-permanent failures "
                  f"(delete {failures_path} to retry them)")
        except (json.JSONDecodeError, KeyError):
            pass
    tasks = [t for t in tasks if (t[0], t[1]) not in known_gone]
    if args.limit:
        tasks = tasks[: args.limit]

    counts = {"cached": 0, "ok": 0, "gone": 0, "error": 0}
    gone, errors = [], []
    state = {"warned": False}
    started = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(download_one, t, args, token, state): t for t in tasks}
        for n, fut in enumerate(concurrent.futures.as_completed(futures), 1):
            task = futures[fut]
            try:
                status, detail = fut.result()
            except Exception as e:
                status, detail = "error", f"{type(e).__name__}: {e}"
            counts[status] += 1
            if status == "gone":
                gone.append([task[0], task[1], detail])
            elif status == "error":
                errors.append([task[0], task[1], detail])
            if n % 200 == 0 or n == len(tasks):
                rate = n / max(1e-9, time.time() - started)
                eta = (len(tasks) - n) / max(1e-9, rate)
                with _print_lock:
                    print(f"  {n}/{len(tasks)}  ok={counts['ok']} cached={counts['cached']} "
                          f"gone={counts['gone']} err={counts['error']}  "
                          f"{rate:.1f}/s  eta {eta / 60:.0f}m")

    with open(failures_path, "w", encoding="utf-8") as fh:
        json.dump({"permanent": [g[:2] for g in gone] + [list(k) for k in known_gone],
                   "permanent_detail": gone, "transient": errors}, fh, indent=1)

    have = len([f for f in os.listdir(args.out) if f.endswith(".patch")])
    print(f"\ndownloaded {counts['ok']}, already had {counts['cached']}, "
          f"permanently gone {counts['gone']}, transient errors {counts['error']}")
    print(f"{have} .patch files in {args.out}")
    if counts["error"]:
        print(f"\n{counts['error']} transient failures -- rerun this command to retry them.")
        return 1
    print("\nNext: python data/build_from_patches.py --patches " + args.out +
          " --nvd-dir <cleaned NVD dir> --out data/temporal_splits_by_time")
    return 0


if __name__ == "__main__":
    sys.exit(main())
