#!/usr/bin/env python3
"""
batch_ingest.py — sequential auto-ingestion for a folder of videos.

Ingests every video in a folder ONE AT A TIME (waits for each to finish before
starting the next, so the GPU isn't overloaded). Skips videos already indexed.
Video files are named <video_id>.mp4, so the video_id is taken from the filename.

Usage (Windows, from the project folder, with the app already running):
    python batch_ingest.py --folder "C:\\Users\\Click\\Downloads\\RAGY\\...\\youtube_videos"

Options:
    --folder PATH       folder containing the .mp4 files (required)
    --server URL        base URL of the running app (default http://localhost:8001)
    --timeout SECONDS   max wait per video before giving up (default 21600 = 6h)
    --poll SECONDS      how often to check progress (default 15)
    --ext .mp4,.mkv     comma-separated extensions to ingest (default .mp4)
    --language LANG     optional language hint passed to ingest (default: auto)
    --dry-run           list what WOULD be ingested, do nothing

The app must already be running (uvicorn). This script only drives it over HTTP.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from urllib import request as _req
from urllib.error import URLError, HTTPError


def _post(url: str, payload: dict, timeout: int = 60) -> dict:
    data = json.dumps(payload).encode("utf-8")
    r = _req.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with _req.urlopen(r, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _get(url: str, timeout: int = 30) -> dict:
    with _req.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def is_indexed(server: str, video_id: str) -> bool:
    try:
        s = _get(f"{server}/ingestion/status/{video_id}", timeout=30)
        return bool(s.get("indexed"))
    except Exception:
        return False


def wait_until_done(server: str, video_id: str, timeout: int, poll: int) -> bool:
    """Poll the status endpoint until indexed or timeout. Returns True if indexed."""
    start = time.time()
    last_chunks = -1
    while time.time() - start < timeout:
        try:
            s = _get(f"{server}/ingestion/status/{video_id}", timeout=30)
            if s.get("indexed"):
                return True
            ch = int(s.get("total_chunks", 0) or 0)
            if ch != last_chunks:
                last_chunks = ch
                mins = (time.time() - start) / 60
                print(f"      … still ingesting {video_id}  (chunks={ch}, {mins:.1f} min elapsed)")
        except Exception as e:
            print(f"      (status check failed, retrying: {e})")
        time.sleep(poll)
    return is_indexed(server, video_id)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True)
    ap.add_argument("--server", default="http://localhost:8001")
    ap.add_argument("--timeout", type=int, default=21600)
    ap.add_argument("--poll", type=int, default=15)
    ap.add_argument("--ext", default=".mp4")
    ap.add_argument("--language", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--benchmark-jsonl", default=None,
                    help="path to video_mmlu.jsonl — if given, only ingest videos whose "
                         "id is in the benchmark, and print a coverage report first")
    ap.add_argument("--limit", type=int, default=None,
                    help="ingest at most N videos (after filtering/sampling). Good for "
                         "a quick validation run, e.g. --limit 15")
    ap.add_argument("--sample", type=int, default=None,
                    help="randomly sample N videos to ingest (representative subset for "
                         "a paper). Uses --seed for reproducibility.")
    ap.add_argument("--seed", type=int, default=42, help="random seed for --sample")
    args = ap.parse_args()

    folder = Path(args.folder)
    if not folder.is_dir():
        print(f"ERROR: folder not found: {folder}")
        sys.exit(1)

    exts = {e if e.startswith(".") else "." + e for e in args.ext.lower().split(",")}
    videos = sorted(p for p in folder.iterdir() if p.suffix.lower() in exts)
    if not videos:
        print(f"No video files with extensions {exts} in {folder}")
        sys.exit(0)

    # Optional: restrict to benchmark IDs + coverage report
    bench_ids = None
    if args.benchmark_jsonl:
        bp = Path(args.benchmark_jsonl)
        if not bp.is_file():
            print(f"ERROR: benchmark jsonl not found: {bp}")
            sys.exit(1)
        bench_ids = set()
        for line in bp.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    bench_ids.add(json.loads(line)["video_id"])
                except Exception:
                    pass
        have = {p.stem for p in videos}
        present = bench_ids & have
        missing = bench_ids - have
        extra = have - bench_ids
        print("── Benchmark coverage ──────────────────────────────")
        print(f"  benchmark videos (jsonl):     {len(bench_ids)}")
        print(f"  files present for benchmark:   {len(present)}")
        print(f"  benchmark videos MISSING file: {len(missing)}")
        print(f"  files NOT in benchmark:        {len(extra)}")
        print("────────────────────────────────────────────────────")
        # restrict the ingest list to benchmark videos we have files for
        videos = sorted(p for p in videos if p.stem in bench_ids)
        if not videos:
            print("No benchmark videos with files to ingest.")
            sys.exit(0)

    # Optional sampling / limiting (for validation runs or paper subsets)
    if args.sample:
        import random
        rng = random.Random(args.seed)
        videos = sorted(rng.sample(videos, min(args.sample, len(videos))), key=lambda p: p.stem)
        print(f"Randomly sampled {len(videos)} videos (seed={args.seed}).")
    if args.limit:
        videos = videos[:args.limit]
        print(f"Limited to first {len(videos)} videos.")

    server = args.server.rstrip("/")
    print(f"Found {len(videos)} video file(s) in {folder}")
    print(f"Server: {server}\n")

    # quick connectivity check
    if not args.dry_run:
        try:
            _get(f"{server}/ingestion/status/__probe__", timeout=10)
        except (URLError, HTTPError) as e:
            # a 404/known response is fine; only a connection error is fatal
            if isinstance(e, URLError) and not isinstance(e, HTTPError):
                print(f"ERROR: cannot reach the app at {server}. Is uvicorn running?\n  {e}")
                sys.exit(1)

    done, skipped, failed = 0, 0, 0
    for i, vpath in enumerate(videos, 1):
        video_id = vpath.stem  # filename without extension = video_id
        print(f"[{i}/{len(videos)}] {video_id}")

        if is_indexed(server, video_id):
            print("      already indexed — skipping.")
            skipped += 1
            continue

        if args.dry_run:
            print("      (dry-run) would ingest.")
            continue

        payload = {"video_path": str(vpath), "video_id": video_id}
        if args.language:
            payload["language"] = args.language

        try:
            resp = _post(f"{server}/ingest", payload, timeout=120)
            status = resp.get("status", "")
            if status == "already_indexed":
                print("      already indexed — skipping.")
                skipped += 1
                continue
            print(f"      ingestion started… (this can take a while)")
        except Exception as e:
            print(f"      ERROR starting ingestion: {e}")
            failed += 1
            continue

        ok = wait_until_done(server, video_id, args.timeout, args.poll)
        if ok:
            print(f"      ✅ done: {video_id}\n")
            done += 1
        else:
            print(f"      ⚠️ timed out / not confirmed indexed: {video_id} "
                  f"(skipping to next)\n")
            failed += 1

    print("=" * 50)
    print(f"Finished. ingested={done}  skipped(already done)={skipped}  failed/timeout={failed}")


if __name__ == "__main__":
    main()