"""
upload_frames.py — push the 12 demo videos' frames to Supabase Storage,
then write each frame's public URL back into video_chunks.metadata.frame_url.

RUN LOCALLY from the project root:
    pip install supabase pillow
    python upload_frames.py --dry-run      # inspect, uploads nothing
    python upload_frames.py                # upload + patch metadata

Prereq: create a PUBLIC bucket named "frames" in Supabase -> Storage.

Frames are resized to max 480px wide before upload (they only ever render as
thumbnails). ~1,993 frames at ~13 KB each is ~27 MB raw; resizing keeps it
well under the free-tier storage cap.
"""

import argparse
import os
import pickle
import sys
from io import BytesIO
from pathlib import Path

SUPABASE_URL = "https://zievhmocgbrrsltnyfkf.supabase.co"
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InppZXZobW9jZ2JycnNsdG55ZmtmIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc4NDYxOTQ0NCwiZXhwIjoyMTAwMTk1NDQ0fQ.dXdpGG8Fa3EIJgfZm0jSKti737uxFbBx-9WwzpgoeBU")

STORAGE_DIR = Path("api/data/faiss_store")
FRAMES_ROOT = Path("api/data/frames")          # <video_id>/<frame>.jpg
BUCKET = "frames"
MAX_WIDTH = 480
JPEG_QUALITY = 78

VIDEOS = {
    "509f803e_A_Brief_History_of_Geologic_Time",
    "74280c28_How_to_manage_your_emotions",
    "7b7d5ff0_What_happens_if_you_cut_down_all_of_a_ci",
    "b5438048_The_benefits_of_doing_nothing____6_Minut",
    "dc28b907_MIT_Introduction_to_Deep_Learning__2025_",
    "db98802f_Learn_Arabic_Podcast_Nature__Simple_Arab",
    "cSj82GG6MX4", "cXhSXGyC5F4", "j7qVcc6X8g4",
    "0YqZIHFmVzg", "-O3J0FkiRsk", "dxmm__4p-I8",
}


def collect():
    """-> [(chunk_id, video_id, local_path, storage_key)]"""
    with open(STORAGE_DIR / "metadata.pkl", "rb") as f:
        top = pickle.load(f)

    # metadata.pkl is {"meta": {chunk_id: chunk}, "video_chunks": ...};
    # the actual chunks live under "meta".
    meta = top.get("meta", top) if isinstance(top, dict) else top

    items, missing = [], 0
    for cid, c in meta.items():
        vid = c.get("video_id")
        if vid not in VIDEOS:
            continue
        if (c.get("chunk_type") or "") != "visual":
            continue
        m = c.get("metadata") or {}
        raw = m.get("frame_path") or m.get("frame_file")
        if not raw:
            continue
        name = Path(str(raw).replace("\\", "/")).name
        local = FRAMES_ROOT / vid / name
        if not local.exists():
            missing += 1
            continue
        items.append((cid, vid, local, f"{vid}/{name}"))
    return items, missing


def shrink(path: Path) -> bytes:
    from PIL import Image
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    if img.width > MAX_WIDTH:
        h = int(img.height * MAX_WIDTH / img.width)
        img = img.resize((MAX_WIDTH, h), Image.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=JPEG_QUALITY, optimize=True)
    return buf.getvalue()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-upload", action="store_true",
                    help="only patch metadata URLs, assume files already uploaded")
    args = ap.parse_args()

    print("1) collecting frames ...")
    items, missing = collect()
    print(f"   {len(items)} frames found on disk")
    if missing:
        print(f"   {missing} referenced frames MISSING on disk (skipped)")

    per_video = {}
    for _, vid, _, _ in items:
        per_video[vid] = per_video.get(vid, 0) + 1
    for vid, n in sorted(per_video.items(), key=lambda x: -x[1]):
        print(f"      {n:5}  {vid[:50]}")

    raw_bytes = sum(p.stat().st_size for _, _, p, _ in items)
    print(f"\n   raw size on disk: {raw_bytes/1e6:.1f} MB")

    if args.dry_run:
        print("\nDRY RUN — nothing uploaded.")
        if items:
            cid, vid, local, key = items[0]
            print(f"   sample local : {local}")
            print(f"   sample key   : {key}")
            print(f"   public URL   : {SUPABASE_URL}/storage/v1/object/public/{BUCKET}/{key}")
            try:
                print(f"   resized      : {len(shrink(local))/1024:.0f} KB")
            except Exception as e:
                print(f"   resize failed: {e}")
        return

    if "PASTE_SERVICE_ROLE" in SUPABASE_SERVICE_KEY:
        sys.exit("ERROR: set SUPABASE_SERVICE_KEY first.")

    from supabase import create_client
    sb = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

    if not args.skip_upload:
        print("\n2) uploading to Storage ...")
        done = fail = 0
        for i, (cid, vid, local, key) in enumerate(items, 1):
            try:
                data = shrink(local)
                sb.storage.from_(BUCKET).upload(
                    key, data,
                    {"content-type": "image/jpeg", "upsert": "true"},
                )
                done += 1
            except Exception as e:
                # upsert on an existing object can still raise on some versions
                if "exists" in str(e).lower() or "duplicate" in str(e).lower():
                    done += 1
                else:
                    fail += 1
                    if fail <= 5:
                        print(f"   ! {key}: {str(e)[:90]}")
            if i % 200 == 0:
                print(f"   {i}/{len(items)}")
        print(f"   uploaded {done}, failed {fail}")

    print("\n3) patching video_chunks.metadata.frame_url ...")
    base = f"{SUPABASE_URL}/storage/v1/object/public/{BUCKET}"
    patched = 0
    for i, (cid, vid, local, key) in enumerate(items, 1):
        try:
            cur = sb.table("video_chunks").select("metadata").eq("chunk_id", cid).single().execute()
            md = (cur.data or {}).get("metadata") or {}
            md["frame_url"] = f"{base}/{key}"
            sb.table("video_chunks").update({"metadata": md}).eq("chunk_id", cid).execute()
            patched += 1
        except Exception as e:
            if patched < 5:
                print(f"   ! {cid}: {str(e)[:90]}")
        if i % 200 == 0:
            print(f"   {i}/{len(items)}")
    print(f"   patched {patched} rows")
    print("\nDONE.")


if __name__ == "__main__":
    main()
