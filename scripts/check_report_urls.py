#!/usr/bin/env python3
"""Check reachability of source URLs referenced in output reports.

Usage: python scripts/check_report_urls.py [output_dir]

This script iterates `output/report_*.json` and checks each `source.url`.
It prints per-report summaries and a final failure count. It uses `httpx`
if available, otherwise falls back to `requests` if installed.
"""
import sys
from pathlib import Path
import json
import time

try:
    import httpx
    _HAS_HTTPX = True
except Exception:
    httpx = None
    _HAS_HTTPX = False

def _is_local_archive(url: str) -> bool:
    return isinstance(url, str) and url.startswith("file://")


def check_url(url: str, timeout: float = 10.0) -> tuple[int | None, str | None]:
    try:
        if _HAS_HTTPX:
            r = httpx.head(url, follow_redirects=True, timeout=timeout)
            return r.status_code, None
        else:
            import requests
            r = requests.head(url, allow_redirects=True, timeout=timeout)
            return r.status_code, None
    except Exception as exc:
        return None, str(exc)

def main(output_dir: str = "output") -> int:
    out = Path(output_dir)
    if not out.exists():
        print(f"ERROR: output directory '{output_dir}' not found.")
        return 2

    files = sorted(out.glob("report_*.json"))
    if not files:
        print("No report_*.json files found in output directory.")
        return 1

    total_checked = 0
    total_failures = 0

    for path in files:
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"{path.name}: failed to read JSON: {exc}")
            continue

        print(f"Checking {path.name}...")
        failures = 0
        for src in report.get("sources", []):
            url = src.get("url")
            archived_url = src.get("archived_url") or (src.get("url_status") or {}).get("archived_url")
            if not url:
                print(f"  - source {src.get('source_id')} has no URL")
                failures += 1
                continue

            status, err = check_url(url)
            total_checked += 1
            if status and 200 <= status < 400:
                print(f"  - OK [{status}] {url}")
            elif archived_url:
                if _is_local_archive(archived_url):
                    print(f"  - OK [archived] {url} -> {archived_url}")
                else:
                    fallback_status, fallback_err = check_url(archived_url)
                    if fallback_status and 200 <= fallback_status < 400:
                        print(f"  - OK [archived {fallback_status}] {url} -> {archived_url}")
                    else:
                        failures += 1
                        total_failures += 1
                        print(f"  - ERROR [original status {status} archived status {fallback_status}] {url} -> {archived_url}")
            else:
                failures += 1
                total_failures += 1
                if status is None:
                    print(f"  - ERROR [exception] {url} -> {err}")
                else:
                    print(f"  - ERROR [status {status}] {url}")

        if failures:
            print(f"  -> {failures} url issue(s) in {path.name}")
        else:
            print(f"  -> all urls OK in {path.name}")
        print()
        # be polite rate-limit
        time.sleep(0.1)

    print(f"Checked {total_checked} urls across {len(files)} report(s). {total_failures} failures.")
    return 0 if total_failures == 0 else 3

if __name__ == "__main__":
    outdir = sys.argv[1] if len(sys.argv) > 1 else "output"
    raise SystemExit(main(outdir))
