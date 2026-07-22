#!/usr/bin/env python3
"""
eval_video_mmlu.py — Evaluate the RAGY pipeline against the Video-MMLU benchmark.

What it does
------------
1. Loads the Video-MMLU QA dataset (Video_MMLU_QA.tsv by default) and groups
   the questions by video_id.
2. Queries the *live* RAGY server (GET /videos) to learn which video_ids are
   actually ingested, and only evaluates those. Dataset videos that are NOT
   ingested are skipped and logged to skipped_videos.txt.

   NOTE ON IDS: Video-MMLU is keyed by YouTube ID (e.g. "Y8KMa8tJw-o"). The
   Video-MMLU videos were ingested into RAGY using that YouTube ID as the
   video_id, so matching is an exact id match. The LVBench videos (ingested
   with natural-language `<hash>_<title>` ids) simply won't match any dataset
   id and are therefore skipped automatically.
3. For each ingested video, loops through its questions one at a time, sends
   each to POST /ask with the correct video_id, and collects the answer.
4. Saves results incrementally (appended after every answer, so a crash never
   loses progress):
       results/video_mmlu_results.txt   — human-readable blocks
       results/video_mmlu_results.jsonl — one JSON object per line for scoring
   The run is resumable: on restart, (video_id, question) pairs already present
   in the .jsonl are skipped.

Usage
-----
    python eval_video_mmlu.py                       # defaults
    python eval_video_mmlu.py --server http://localhost:8000
    python eval_video_mmlu.py --limit-videos 20     # quick smoke test
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import requests

# ── Defaults ────────────────────────────────────────────────────────────────
DEFAULT_SERVER = "http://localhost:8000"
DEFAULT_QA_TSV = r"C:\Users\Click\Downloads\Video-MMLU\Video_MMLU_QA.tsv"
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = SCRIPT_DIR / "results"
DEFAULT_SKIPPED_FILE = SCRIPT_DIR / "skipped_videos.txt"


# ── Dataset loading ─────────────────────────────────────────────────────────
def load_dataset(qa_tsv: Path) -> "OrderedDict[str, List[Dict[str, str]]]":
    """Load Video_MMLU_QA.tsv and group rows by video_id (YouTube id, no .mp4).

    Columns: index, video, discipline, question, answer.
    Returns an ordered mapping: video_id -> [ {question, answer, discipline}, ... ]
    """
    from collections import OrderedDict

    grouped: "OrderedDict[str, List[Dict[str, str]]]" = OrderedDict()
    with open(qa_tsv, encoding="utf-8", newline="") as f:
        # It's a TSV — disable quote handling so stray quotes in answers don't
        # swallow whole fields.
        reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        for row in reader:
            raw_video = (row.get("video") or "").strip()
            if not raw_video:
                continue
            video_id = raw_video[:-4] if raw_video.lower().endswith(".mp4") else raw_video
            question = (row.get("question") or "").strip()
            answer = (row.get("answer") or "").strip()
            if not question:
                continue
            grouped.setdefault(video_id, []).append(
                {
                    "question": question,
                    "answer": answer,
                    "discipline": (row.get("discipline") or "").strip(),
                }
            )
    return grouped


# ── RAGY server interaction ─────────────────────────────────────────────────
def get_ingested_video_ids(server: str, timeout: int = 60) -> set:
    """Ask the live RAGY server which video_ids are ingested (GET /videos)."""
    url = f"{server.rstrip('/')}/videos"
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    ids = set()
    for v in data.get("videos", []):
        vid = str(v.get("video_id", "")).strip()
        if not vid:
            continue
        # Normalise a possible trailing extension, just in case.
        if vid.lower().endswith(".mp4"):
            vid = vid[:-4]
        ids.add(vid)
    return ids


def ask_ragy(server: str, video_id: str, question: str, timeout: int) -> Dict:
    """Send one question to POST /ask. Returns {answer, latency_seconds, error}."""
    url = f"{server.rstrip('/')}/ask"
    payload = {
        "question": question,
        "video_id": video_id,
        "confirmed": True,       # skip the clarification gate
        "llm": True,
        "use_cache": False,      # fresh answer every time for a fair eval
        "benchmark": "video_mmlu",
    }
    t0 = time.time()
    try:
        resp = requests.post(url, json=payload, timeout=timeout)
        latency = time.time() - t0
        resp.raise_for_status()
        data = resp.json()
        answer = data.get("answer")
        if answer is None and isinstance(data.get("llm"), dict):
            answer = data["llm"].get("answer")
        return {"answer": answer if answer is not None else "", "latency_seconds": round(latency, 3), "error": None}
    except Exception as e:  # noqa: BLE001 — record the error, keep going
        return {"answer": "", "latency_seconds": round(time.time() - t0, 3), "error": str(e)}


# ── Resume support ──────────────────────────────────────────────────────────
def load_completed(jsonl_path: Path) -> set:
    """Return the set of (video_id, question) already answered in the .jsonl."""
    done = set()
    if not jsonl_path.exists():
        return done
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            done.add((obj.get("video_id"), obj.get("question")))
    return done


# ── Persistence ─────────────────────────────────────────────────────────────
def append_result(txt_path: Path, jsonl_path: Path, record: Dict) -> None:
    """Append one result to both the human-readable .txt and the .jsonl."""
    block = (
        f"VIDEO_ID: {record['video_id']}\n"
        f"QUESTION: {record['question']}\n"
        f"GT_ANSWER: {record['gt_answer']}\n"
        f"RAGY_ANSWER: {record['ragy_answer']}\n"
        f"---\n"
    )
    with open(txt_path, "a", encoding="utf-8") as f:
        f.write(block)
        f.flush()

    line = {
        "video_id": record["video_id"],
        "question": record["question"],
        "gt_answer": record["gt_answer"],
        "ragy_answer": record["ragy_answer"],
        "latency_seconds": record["latency_seconds"],
    }
    if record.get("error"):
        line["error"] = record["error"]
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(line, ensure_ascii=False) + "\n")
        f.flush()


# ── Main ────────────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate RAGY on the Video-MMLU benchmark.")
    ap.add_argument("--server", default=DEFAULT_SERVER, help="RAGY base URL")
    ap.add_argument("--qa-tsv", default=DEFAULT_QA_TSV, help="Path to Video_MMLU_QA.tsv")
    ap.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR), help="Output dir")
    ap.add_argument("--skipped-file", default=str(DEFAULT_SKIPPED_FILE), help="Skipped log")
    ap.add_argument("--timeout", type=int, default=300, help="Per-question HTTP timeout (s)")
    ap.add_argument("--limit-videos", type=int, default=0, help="Only eval first N ingested videos (0 = all)")
    args = ap.parse_args()

    qa_tsv = Path(args.qa_tsv)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    txt_path = results_dir / "video_mmlu_results.txt"
    jsonl_path = results_dir / "video_mmlu_results.jsonl"
    skipped_path = Path(args.skipped_file)

    if not qa_tsv.exists():
        print(f"ERROR: dataset not found: {qa_tsv}", file=sys.stderr)
        return 2

    # 1. Load + group dataset.
    print(f"Loading dataset: {qa_tsv}")
    grouped = load_dataset(qa_tsv)
    total_q = sum(len(v) for v in grouped.values())
    print(f"  {len(grouped)} videos, {total_q} questions in dataset")

    # 2. Which videos are actually ingested?
    print(f"Querying ingested videos: {args.server}/videos")
    try:
        ingested = get_ingested_video_ids(args.server)
    except Exception as e:  # noqa: BLE001
        print(f"ERROR: could not reach RAGY at {args.server} ({e}).", file=sys.stderr)
        print("       Start the server first: uvicorn api.main:app --port 8000", file=sys.stderr)
        return 3
    print(f"  {len(ingested)} videos ingested in RAGY")

    eval_videos = [vid for vid in grouped if vid in ingested]
    skipped_videos = [vid for vid in grouped if vid not in ingested]

    # Log skipped (not-ingested) dataset videos.
    with open(skipped_path, "w", encoding="utf-8") as f:
        for vid in skipped_videos:
            f.write(vid + "\n")
    print(f"  {len(eval_videos)} dataset videos are ingested -> will evaluate")
    print(f"  {len(skipped_videos)} dataset videos NOT ingested -> logged to {skipped_path.name}")

    if not eval_videos:
        print("\nNothing to evaluate: none of the dataset video_ids match ingested "
              "video_ids. Confirm the Video-MMLU videos were ingested with the "
              "YouTube ID as their video_id.")
        return 0

    if args.limit_videos > 0:
        eval_videos = eval_videos[: args.limit_videos]
        print(f"  --limit-videos: restricting to first {len(eval_videos)} videos")

    # Resume: skip already-answered (video_id, question) pairs.
    completed = load_completed(jsonl_path)
    if completed:
        print(f"  resuming: {len(completed)} answers already recorded, will skip those")

    # 3 + 4 + 5. Loop videos -> questions, ask, save incrementally.
    total_videos = len(eval_videos)
    answered = 0
    errors = 0
    for vi, video_id in enumerate(eval_videos, start=1):
        questions = grouped[video_id]
        m = len(questions)
        for qi, q in enumerate(questions, start=1):
            print(f"video {vi}/{total_videos} — question {qi}/{m}  [{video_id}]")
            if (video_id, q["question"]) in completed:
                continue  # already done in a previous run

            res = ask_ragy(args.server, video_id, q["question"], args.timeout)
            if res["error"]:
                errors += 1
                print(f"    ! error: {res['error']}")

            append_result(
                txt_path,
                jsonl_path,
                {
                    "video_id": video_id,
                    "question": q["question"],
                    "gt_answer": q["answer"],
                    "ragy_answer": res["answer"],
                    "latency_seconds": res["latency_seconds"],
                    "error": res["error"],
                },
            )
            answered += 1

    print("\nDone.")
    print(f"  answered this run : {answered}")
    print(f"  errors            : {errors}")
    print(f"  results (txt)     : {txt_path}")
    print(f"  results (jsonl)   : {jsonl_path}")
    print(f"  skipped videos    : {skipped_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
