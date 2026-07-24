"""
export_to_supabase.py — one-time local export of the 12 demo videos.

Reads chunks + BGE-M3 dense vectors out of the local store and pushes them
into Supabase (demo_videos + video_chunks).

RUN THIS LOCALLY, from the project root, with the venv active:
    pip install supabase
    python export_to_supabase.py --dry-run     # inspect first, writes nothing
    python export_to_supabase.py               # actually upload

Nothing in your local system is modified — this only READS metadata.pkl
and the FAISS index.
"""

import argparse
import os
import pickle
import sys
from pathlib import Path

# Load .env from the project root so SUPABASE_SERVICE_KEY (and friends) are
# available via os.environ below. Must run BEFORE the CONFIG block reads them.
# No-op if python-dotenv isn't installed or .env is absent.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ----------------------------------------------------------------------
# CONFIG — fill these in
# ----------------------------------------------------------------------
SUPABASE_URL = "https://zievhmocgbrrsltnyfkf.supabase.co"

# IMPORTANT: use the SERVICE ROLE key here, not the anon key.
# Supabase dashboard -> Settings -> API -> service_role (secret).
# This key bypasses RLS so the script can insert. NEVER put it in frontend code.
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY", "PASTE_SERVICE_ROLE_KEY_HERE")

STORAGE_DIR = Path("api/data/faiss_store")
METADATA_PKL = STORAGE_DIR / "metadata.pkl"

# internal video_id  ->  (youtube_id, nice title, is_truncated)
VIDEOS = {
    "509f803e_A_Brief_History_of_Geologic_Time":        ("rWp5ZpJAIAE", "A Brief History of Geologic Time", False),
    "74280c28_How_to_manage_your_emotions":             ("Uew5BbvmLks", "How to manage your emotions", False),
    "7b7d5ff0_What_happens_if_you_cut_down_all_of_a_ci":("zarll9bx6FI", "What happens if you cut down all of a city's trees", False),
    "b5438048_The_benefits_of_doing_nothing____6_Minut":("Y681hXWwhQY", "The benefits of doing nothing - 6 Minute English", False),
    "dc28b907_MIT_Introduction_to_Deep_Learning__2025_":("alfdI7S6wCY", "MIT Introduction to Deep Learning 2025", True),   # frame-capped
    "db98802f_Learn_Arabic_Podcast_Nature__Simple_Arab":("iTuheortQL0", "Learn Arabic Podcast - Nature (Simple Arabic)", False),
    "cSj82GG6MX4": ("cSj82GG6MX4", "", False),
    "cXhSXGyC5F4": ("cXhSXGyC5F4", "", False),
    "j7qVcc6X8g4": ("j7qVcc6X8g4", "", False),
    "0YqZIHFmVzg": ("0YqZIHFmVzg", "", False),
    "-O3J0FkiRsk": ("-O3J0FkiRsk", "", False),
    "dxmm__4p-I8": ("dxmm__4p-I8", "", False),
}

BATCH = 200          # rows per insert
EMBED_DIM = 1024     # BGE-M3 dense


# ----------------------------------------------------------------------
def load_meta():
    """
    Load the chunk map out of metadata.pkl.

    The local store is nested: {'meta': {chunk_id: chunk}, 'video_chunks': {...}}.
    Older/flat stores are {chunk_id: chunk} directly. Handle both by unwrapping
    the 'meta' key when present, so callers always get {chunk_id: chunk}.
    """
    if not METADATA_PKL.exists():
        sys.exit(f"ERROR: {METADATA_PKL} not found. Run from the project root.")
    with open(METADATA_PKL, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, dict) and "meta" in data and isinstance(data["meta"], dict):
        return data["meta"]
    return data


def load_vectors(chunk_ids):
    """
    Pull the BGE-M3 dense vector for each chunk_id out of the local FAISS
    cross-language ('*') index, which mirrors every chunk.
    Returns {chunk_id: [float] * 1024}
    """
    import numpy as np
    import faiss

    # the '*' index is written with a 'star' filename on Windows
    candidates = [
        STORAGE_DIR / "faiss_star.index",
        STORAGE_DIR / "faiss_*.index",
        STORAGE_DIR / "faiss_all.index",
    ]
    idx_path = next((p for p in candidates if p.exists()), None)
    ids_path = idx_path.with_suffix(".ids") if idx_path else None

    if not idx_path or not ids_path or not ids_path.exists():
        print("WARNING: cross-language FAISS index not found; "
              "no embeddings will be exported.")
        print(f"  looked for: {[str(c) for c in candidates]}")
        return {}

    print(f"  reading vectors from {idx_path.name}")
    index = faiss.read_index(str(idx_path))
    with open(ids_path, "rb") as f:
        id_map = pickle.load(f)

    # This store's .ids is a schema dict:
    #   {'str_to_int': {chunk_id: int}, 'int_to_str': {int: chunk_id}, ...}
    # Older stores are a flat map ({chunk_id: int} or {int: chunk_id}). Resolve
    # both to chunk_to_int = {chunk_id: faiss_int_id}.
    if isinstance(id_map, dict) and "str_to_int" in id_map:
        chunk_to_int = id_map["str_to_int"]
    else:
        sample_k = next(iter(id_map))
        if isinstance(sample_k, str):
            chunk_to_int = id_map
        else:
            chunk_to_int = {v: k for k, v in id_map.items()}

    wanted = {cid: chunk_to_int[cid] for cid in chunk_ids if cid in chunk_to_int}
    print(f"  {len(wanted)}/{len(chunk_ids)} chunks found in the index")

    out = {}
    for cid, int_id in wanted.items():
        try:
            vec = index.reconstruct(int(int_id))
            out[cid] = np.asarray(vec, dtype="float32").tolist()
        except Exception:
            pass          # some index types can't reconstruct; skipped
    print(f"  reconstructed {len(out)} vectors")
    return out


def clean_metadata(md):
    """Drop dead absolute frame paths and the bulky duplicated CLIP vector."""
    if not isinstance(md, dict):
        return {}
    md = dict(md)
    md.pop("clip_embedding", None)
    fp = md.get("frame_path")
    if isinstance(fp, str) and ("\\" in fp or fp.startswith("/")):
        # keep only the filename; the old absolute path points at a dead machine
        md["frame_file"] = Path(fp.replace("\\", "/")).name
        md.pop("frame_path", None)
    return md


def build_rows(meta, vectors):
    rows = []
    skipped_no_vec = 0
    for cid, c in meta.items():
        vid = c.get("video_id")
        if vid not in VIDEOS:
            continue
        emb = vectors.get(cid)
        if emb is None or len(emb) != EMBED_DIM:
            skipped_no_vec += 1
            continue
        kp = c.get("key_points") or []
        if not isinstance(kp, list):
            kp = []
        rows.append({
            "chunk_id":      cid,
            "video_id":      vid,
            "text":          c.get("text") or "",
            "original_text": c.get("original_text") or "",
            "search_text":   c.get("search_text") or c.get("text") or "",
            "start_time":    float(c.get("start_time") or 0.0),
            "end_time":      float(c.get("end_time") or 0.0),
            "language":      c.get("language") or "unknown",
            "speaker":       c.get("speaker") or "SPEAKER_00",
            "topic":         c.get("topic") or "general",
            "chunk_type":    c.get("chunk_type") or "speech",
            "key_points":    kp,
            "metadata":      clean_metadata(c.get("metadata")),
            "embedding":     emb,
        })
    if skipped_no_vec:
        print(f"  NOTE: {skipped_no_vec} chunks skipped (no usable 1024-d vector)")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="inspect only; do not write to Supabase")
    args = ap.parse_args()

    print("1) loading local metadata ...")
    meta = load_meta()
    print(f"   {len(meta)} chunks in the local store")

    target_ids = [cid for cid, c in meta.items() if c.get("video_id") in VIDEOS]
    print(f"   {len(target_ids)} chunks belong to the {len(VIDEOS)} demo videos")

    per_video = {}
    for cid in target_ids:
        per_video.setdefault(meta[cid]["video_id"], 0)
        per_video[meta[cid]["video_id"]] += 1
    for vid, (yt, title, trunc) in VIDEOS.items():
        n = per_video.get(vid, 0)
        flag = "  [TRUNCATED]" if trunc else ""
        warn = "  <-- NOT FOUND IN STORE" if n == 0 else ""
        print(f"     {n:5}  {vid[:48]:48} -> {yt}{flag}{warn}")

    print("\n2) reading embeddings from local FAISS ...")
    vectors = load_vectors(target_ids)

    print("\n3) building rows ...")
    rows = build_rows(meta, vectors)
    print(f"   {len(rows)} rows ready")

    if args.dry_run:
        print("\nDRY RUN — nothing written. Sample row:")
        if rows:
            s = dict(rows[0])
            s["embedding"] = f"[float x {len(s['embedding'])}]"
            for k, v in s.items():
                print(f"   {k}: {str(v)[:90]}")
        return

    if "PASTE_SERVICE_ROLE" in SUPABASE_SERVICE_KEY:
        sys.exit("ERROR: set SUPABASE_SERVICE_KEY (env var or edit the file).")

    print("\n4) uploading to Supabase ...")
    from supabase import create_client
    sb = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

    video_rows = [{
        "video_id": vid, "youtube_id": yt, "title": title,
        "is_truncated": trunc, "n_chunks": per_video.get(vid, 0),
    } for vid, (yt, title, trunc) in VIDEOS.items()]
    sb.table("demo_videos").upsert(video_rows).execute()
    print(f"   demo_videos: {len(video_rows)} rows")

    for i in range(0, len(rows), BATCH):
        batch = rows[i:i + BATCH]
        sb.table("video_chunks").upsert(batch).execute()
        print(f"   video_chunks: {min(i + BATCH, len(rows))}/{len(rows)}")

    print("\nDONE.")


if __name__ == "__main__":
    main()
