#!/usr/bin/env python3
"""
lvbench_auto.py — download LVBench videos AND ingest them sequentially.

ONE manual step you cannot skip: download video_info.meta.jsonl from
    https://huggingface.co/datasets/THUDM/LVBench/tree/main
(it's behind a dataset license agreement — no script can click "agree" for you).
Put it next to this script. Everything after that is automatic.

PREREQS:
    pip install yt-dlp requests

USAGE:
    # download 5 shortest videos + ingest each one fully before the next:
    python lvbench_auto.py --meta video_info.meta.jsonl --n 5 \
        --out-dir videos --api http://localhost:8000

    # download only (skip ingest):
    python lvbench_auto.py --meta video_info.meta.jsonl --n 5 --no-ingest

    # ingest existing downloads only (skip download):
    python lvbench_auto.py --meta video_info.meta.jsonl --n 5 --no-download

WHAT IT DOES:
    1. reads meta, sorts by duration, picks N SHORTEST videos
    2. downloads each via yt-dlp (<=720p) into --out-dir, skips ones present
    3. writes pilot_lvbench.jsonl (questions for bench_run.py)
    4. ingests each downloaded video ONE AT A TIME via /ingest, waiting for
       /ingestion/status to report 'ready' before starting the next
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time

import requests


# ── meta parsing ──────────────────────────────────────────────────────
def load_meta(path):
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def parse_question(qa, key, dur):
    raw = qa.get("question", "")
    stem = re.split(r"\n\([A-F]\)", raw)[0].strip()
    opts = dict(re.findall(r"\(([A-F])\)\s*([^\n]+)", raw))
    return {
        "video_id": key,
        "question": stem,
        "options": {k: v.strip() for k, v in opts.items()},
        "answer": str(qa.get("answer", "")).strip()[:1].upper(),
        "category": ",".join(qa.get("question_type", []) or []),
        "time_reference": qa.get("time_reference", ""),
        "duration_minutes": round(float(dur), 1),
    }


# ── download ──────────────────────────────────────────────────────────
def download_one(key, out_dir):
    for ext in ("mp4", "mkv", "webm"):
        p = os.path.join(out_dir, f"{key}.{ext}")
        if os.path.exists(p):
            print(f"    already downloaded: {p}")
            return p
    url = f"https://www.youtube.com/watch?v={key}"
    out_tmpl = os.path.join(out_dir, f"{key}.%(ext)s")
    cmd = [
        "yt-dlp",
        "-f", "bestvideo[height<=720]+bestaudio/best[height<=720]/best",
        "--merge-output-format", "mp4",
        "-o", out_tmpl,
        url,
    ]
    print(f"    downloading {key} ...")
    try:
        subprocess.run(cmd, check=True)
        for ext in ("mp4", "mkv", "webm"):
            p = os.path.join(out_dir, f"{key}.{ext}")
            if os.path.exists(p):
                return p
    except subprocess.CalledProcessError as e:
        print(f"    DOWNLOAD FAILED {key}: {e}")
    return None


# ── ingest ────────────────────────────────────────────────────────────
def ingest_one(api, video_path, poll_s=15, timeout_min=240):
    r = requests.post(
        f"{api}/ingest",
        json={"video_path": os.path.abspath(video_path)},
        timeout=60,
    )
    r.raise_for_status()
    data = r.json()
    vid = data.get("video_id")
    if data.get("status") == "already_indexed":
        print(f"    already indexed: {vid}")
        return vid, True, 0
    print(f"    ingesting video_id={vid} (this can take ~20 min) ...")
    deadline = time.time() + timeout_min * 60
    last = None
    while time.time() < deadline:
        try:
            s = requests.get(f"{api}/ingestion/status/{vid}", timeout=30).json()
            if s.get("indexed"):
                return vid, True, s.get("total_chunks", 0)
            st = s.get("status")
            if st != last:
                print(f"      status={st} ...", flush=True)
                last = st
        except Exception as e:
            print(f"      (status check retry: {e})", flush=True)
        time.sleep(poll_s)
    return vid, False, 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", default="video_info.meta.jsonl")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--out-dir", default="videos")
    ap.add_argument("--api", default="http://localhost:8000")
    ap.add_argument("--no-download", action="store_true")
    ap.add_argument("--no-ingest", action="store_true")
    ap.add_argument("--longest", action="store_true",
                    help="pick longest instead of shortest")
    ap.add_argument("--timeout-min", type=int, default=240)
    args = ap.parse_args()

    if not os.path.exists(args.meta):
        print(f"ERROR: {args.meta} not found.")
        print("Download it (one-time, requires HF login + dataset agreement):")
        print("  https://huggingface.co/datasets/THUDM/LVBench/tree/main")
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)
    entries = load_meta(args.meta)
    entries.sort(
        key=lambda e: e.get("video_info", {}).get("duration_minutes", 9999),
        reverse=args.longest,
    )
    chosen = entries[: args.n]

    print(f"Selected {len(chosen)} videos "
          f"({'longest' if args.longest else 'shortest'} first):")
    for e in chosen:
        d = e.get("video_info", {}).get("duration_minutes", 0)
        print(f"  {e['key']}  {d:.0f} min  {e.get('type')}  {len(e.get('qa', []))} q")
    print()

    # write questions
    all_q = []
    for e in chosen:
        dur = e.get("video_info", {}).get("duration_minutes", 0)
        for qa in e.get("qa", []):
            all_q.append(parse_question(qa, e["key"], dur))
    with open("pilot_lvbench.jsonl", "w", encoding="utf-8") as f:
        for q in all_q:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    print(f"Wrote pilot_lvbench.jsonl ({len(all_q)} questions)\n")

    # download + ingest, sequentially, per video
    summary = []
    for i, e in enumerate(chosen, 1):
        key = e["key"]
        print(f"[{i}/{len(chosen)}] {key}")
        path = None
        if not args.no_download:
            path = download_one(key, args.out_dir)
            if not path:
                summary.append((key, "download_failed", 0))
                print()
                continue
        else:
            for ext in ("mp4", "mkv", "webm"):
                p = os.path.join(args.out_dir, f"{key}.{ext}")
                if os.path.exists(p):
                    path = p
                    break
            if not path:
                summary.append((key, "no_file", 0))
                print("    no downloaded file found, skipping\n")
                continue

        if args.no_ingest:
            summary.append((key, "downloaded", 0))
            print()
            continue

        try:
            _vid, ok, chunks = ingest_one(
                args.api, path, timeout_min=args.timeout_min
            )
            summary.append((key, "done" if ok else "ingest_timeout", chunks))
            print(f"    {'DONE' if ok else 'TIMEOUT'} — {chunks} chunks\n")
        except Exception as ex:
            summary.append((key, f"ingest_error", 0))
            print(f"    INGEST ERROR: {ex}\n")

    print("=" * 55)
    print("SUMMARY")
    for key, status, chunks in summary:
        print(f"  {status:18s} {chunks:>5} chunks  {key}")
    if not args.no_ingest:
        done = sum(1 for _, s, _ in summary if s == "done")
        print(f"\n{done}/{len(chosen)} ingested. Next:")
        print("  python bench_run.py pilot_lvbench.jsonl")


if __name__ == "__main__":
    main()