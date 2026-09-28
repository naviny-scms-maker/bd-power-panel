#!/usr/bin/env python3
"""
Download BPDB Daily Generation Archive PDFs for a date range.

The BPDB archive page is:
https://misc.bpdb.gov.bd/daily-generation-archive

Usage:
    python bpdb_archive_download.py --start 2025-01-01 --end 2026-09-30

Optional:
    --columns "Page 1,Page 2,Page 3,Summary"
    --out bpdb_pdfs

Requirements:
    pip install requests beautifulsoup4 truststore
"""
try:
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    print("[WARN] truststore not installed (pip install truststore); using default SSL settings.")

import argparse
import re
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

ARCHIVE_URL = "https://misc.bpdb.gov.bd/daily-generation-archive"
BASE_URL = "https://misc.bpdb.gov.bd"
DEFAULT_COLUMNS = ["Page 1", "Page 2", "Page 3", "Summary"]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/153.0 Safari/537.36"
    )
}

# Examples supplied from the current BPDB archive:
# page_1_20260921_20260922114041.pdf
# page_2_20260921_20260922114041.pdf
# page_3_20260921_20260922114041.pdf
# summary_20260921_20260922114804.pdf
FILE_RE = re.compile(
    r"(page_[123]|summary)_(\d{8})_\d{14}\.pdf(?:$|\?)",   # tolerate ?query strings
    re.IGNORECASE,
)
COLUMN_KIND = {"page 1": "page_1", "page 2": "page_2", "page 3": "page_3", "summary": "summary"}

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    p.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    p.add_argument(
        "--columns",
        default="Page 1,Page 2,Page 3,Summary",
        help="Comma-separated archive columns to download",
    )
    p.add_argument("--out", default="bpdb_pdfs", help="Output directory")
    p.add_argument("--delay", type=float, default=2.0,
                   help="Seconds between requests (keep >= 2: the site discourages automated access)")
    return p.parse_args()

def get_session():
    s = requests.Session()
    s.headers.update(HEADERS)
    return s

def archive_page(session, page_number):
    url = ARCHIVE_URL if page_number == 1 else f"{ARCHIVE_URL}?page={page_number}"
    r = session.get(url, timeout=30)
    r.raise_for_status()
    return r.text, r.url

def extract_archive_links(html):
    soup = BeautifulSoup(html, "html.parser")
    results = {}

    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        full = urljoin(BASE_URL, href)
        m = FILE_RE.search(full)
        if not m:
            continue

        kind = m.group(1).lower()
        datestr = m.group(2)
        try:
            d = datetime.strptime(datestr, "%Y%m%d").date()
        except ValueError:
            continue

        results[(d, kind)] = full

    # Fallback: some entries use other file names (e.g. storage/daily_entry/2026-07-12.pdf).
    # Read the archive table row by row: the row gives the date, the column header gives the kind.
    for table in soup.find_all("table"):
        header = [th.get_text(strip=True).lower() for th in table.find_all("th")]
        if "date" not in header:
            continue
        for tr in table.find_all("tr"):
            cells = tr.find_all("td")
            if not cells:
                continue
            d = None
            for c in cells:
                t = c.get_text(strip=True)
                if re.fullmatch(r"\d{2}/\d{2}/\d{4}", t):
                    d = datetime.strptime(t, "%d/%m/%Y").date()
                    break
            if d is None:
                continue
            for i, c in enumerate(cells):
                kind = COLUMN_KIND.get(header[i]) if i < len(header) else None
                a = c.find("a", href=True)
                if kind and a and (d, kind) not in results:
                    href = urljoin(BASE_URL, a["href"].strip())
                    if href.lower().split("?")[0].endswith(".pdf"):
                        results[(d, kind)] = href
    return results

def iter_archive_pages(session, start, end, delay):
    # Current BPDB archive displays about 10 records per page. We first
    # scan pages until we have reached dates older than the requested range.
    page = 1
    seen = set()
    seen_pages = set()
    found = {}

    while page <= 1000:
        try:
            html, final_url = archive_page(session, page)
        except Exception as e:
            print(f"[ERROR] Archive page {page}: {e}")
            break

        links = extract_archive_links(html)
        if not links:
            print(f"[STOP] No PDF links found on archive page {page}.")
            break

        new_dates = set()
        for (d, kind), url in links.items():
            found[(d, kind)] = url
            new_dates.add(d)

        page_key = (min(new_dates), max(new_dates))
        if page_key in seen_pages:
            print(f"[STOP] Page {page} repeats an earlier page: pagination has changed. Send Claude the page-2 URL.")
            break
        seen_pages.add(page_key)
        unseen = new_dates - seen
        seen.update(new_dates)

        if unseen:
            print(
                f"[ARCHIVE] page={page:03d} "
                f"dates={min(unseen)}..{max(unseen)} "
                f"PDFs={len(links)}"
            )

        # Once the oldest date on the current page is before the start,
        # we have normally passed the required range.
        oldest = min(new_dates)
        newest = max(new_dates)

        if oldest < start:
            # We have reached/passed the beginning of the requested period.
            break

        page += 1
        time.sleep(delay)

    return found

def download_file(session, url, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists() and destination.stat().st_size > 0:
        print(f"[SKIP] {destination}")
        return True

    tmp = destination.with_suffix(destination.suffix + ".part")
    for attempt in range(1, 4):
        if _try_download(session, url, destination, tmp):
            return True
        print(f"[RETRY] attempt {attempt} failed for {url}")
        time.sleep(5 * attempt)
    return False

def _try_download(session, url, destination, tmp):
    try:
        with session.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            content_type = r.headers.get("Content-Type", "")
            if "pdf" not in content_type.lower() and not url.lower().endswith(".pdf"):
                print(f"[WARN] Unexpected content type for {url}: {content_type}")

            with open(tmp, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 64):
                    if chunk:
                        f.write(chunk)

        tmp.replace(destination)
        print(f"[OK]   {destination}")
        return True
    except Exception as e:
        if tmp.exists():
            tmp.unlink()
        print(f"[FAIL] {url} -> {e}")
        return False

def main():
    args = parse_args()

    start = datetime.strptime(args.start, "%Y-%m-%d").date()
    end = datetime.strptime(args.end, "%Y-%m-%d").date()
    if end < start:
        raise SystemExit("--end must be on or after --start")

    wanted = [x.strip().lower() for x in args.columns.split(",") if x.strip()]
    kind_for_column = {
        "page 1": "page_1",
        "page 2": "page_2",
        "page 3": "page_3",
        "summary": "summary",
    }

    unknown = [x for x in wanted if x not in kind_for_column]
    if unknown:
        raise SystemExit(
            "Unknown --columns value(s): "
            + ", ".join(unknown)
            + ". Use Page 1, Page 2, Page 3, Summary."
        )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    session = get_session()

    print(f"Requested period: {start} to {end}")
    print(f"Columns: {', '.join(args.columns.split(','))}")
    print("Reading BPDB archive pages...")

    links = iter_archive_pages(session, start, end, args.delay)

    total = 0
    success = 0
    missing = []
    failed = []

    current = start
    while current <= end:
        for column in wanted:
            kind = kind_for_column[column]
            url = links.get((current, kind))

            if not url:
                missing.append((current, column))
                continue

            # Save in a date-specific directory. Keep the original filename
            # so the source timestamp is preserved.
            filename = Path(url.split("?", 1)[0]).name
            destination = out / current.isoformat() / filename

            total += 1
            if download_file(session, url, destination):
                success += 1
            else:
                failed.append((current, column, url))
            time.sleep(args.delay)

        current += timedelta(days=1)

    report = out / "download_report.txt"
    with open(report, "w", encoding="utf-8") as f:
        f.write(f"Requested: {start} to {end}\n")
        f.write(f"Requested columns: {', '.join(args.columns.split(','))}\n")
        f.write(f"PDFs found/downloaded: {success}/{total}\n")
        f.write(f"Failed downloads (re-run the script to retry): {len(failed)}\n")
        for d, column, url in failed:
            f.write(f"FAILED  {d.isoformat()} - {column} - {url}\n")
        f.write(f"\nNo link in the archive: {len(missing)}\n")
        for d, column in missing:
            f.write(f"MISSING {d.isoformat()} - {column}\n")

    print("\n========== COMPLETE ==========")
    print(f"Downloaded/skipped successfully: {success}/{total}")
    print(f"Failed (re-run to retry): {len(failed)}")
    print(f"No link in archive: {len(missing)}")
    print(f"Output: {out.resolve()}")
    print(f"Report: {report.resolve()}")

if __name__ == "__main__":
    main()
