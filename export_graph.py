"""
export_graph.py — push the 12 demo videos' knowledge subgraph to Supabase.

Reads api/data/graph_store/graph.pkl (NetworkX MultiDiGraph, 67 MB) and writes
graph_nodes / graph_node_videos / graph_edges.

Modality is DERIVED: each edge's evidence carries chunk_id; we look that chunk
up in the local metadata store and read its chunk_type (speech | visual).
An edge whose evidence spans both becomes 'mixed'.

RUN LOCALLY from the project root:
    python export_graph.py --dry-run
    python export_graph.py
"""

import argparse
import os
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

SUPABASE_URL = "https://zievhmocgbrrsltnyfkf.supabase.co"
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InppZXZobW9jZ2JycnNsdG55ZmtmIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc4NDYxOTQ0NCwiZXhwIjoyMTAwMTk1NDQ0fQ.dXdpGG8Fa3EIJgfZm0jSKti737uxFbBx-9WwzpgoeBU")

GRAPH_PKL = Path("api/data/graph_store/graph.pkl")
META_PKL  = Path("api/data/faiss_store/metadata.pkl")

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

BATCH = 400


def jsonable(v):
    if isinstance(v, set):
        return sorted(str(x) for x in v)
    if isinstance(v, (list, tuple)):
        return [jsonable(x) for x in v]
    if isinstance(v, dict):
        return {str(k): jsonable(x) for k, x in v.items()}
    return v


def load_chunk_modality():
    """chunk_id -> (chunk_type, language)"""
    if not META_PKL.exists():
        print(f"   WARNING: {META_PKL} missing; modality will be 'unknown'")
        return {}
    with open(META_PKL, "rb") as f:
        top = pickle.load(f)
    # metadata.pkl is {"meta": {chunk_id: chunk}, "video_chunks": ...};
    # the actual chunks live under "meta".
    meta = top.get("meta", top) if isinstance(top, dict) else top
    out = {}
    for cid, c in meta.items():
        if not isinstance(c, dict):
            continue
        out[cid] = (c.get("chunk_type") or "unknown", c.get("language") or "unknown")
    return out


def derive_modality(evidence, chunk_lookup):
    """speech / visual / mixed / unknown  + dominant language"""
    kinds, langs = set(), Counter()
    for ev in evidence or []:
        cid = ev.get("chunk_id")
        if not cid:
            continue
        kind, lang = chunk_lookup.get(cid, (None, None))
        if kind:
            kinds.add("visual" if kind == "visual" else
                      "speech" if kind == "speech" else kind)
        if lang and lang != "unknown":
            langs[lang] += 1
    if not kinds:
        modality = "unknown"
    elif len(kinds) > 1:
        modality = "mixed"
    else:
        modality = next(iter(kinds))
    language = langs.most_common(1)[0][0] if langs else None
    return modality, language


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not GRAPH_PKL.exists():
        sys.exit(f"ERROR: {GRAPH_PKL} not found. Run from the project root.")

    print("1) loading graph ...")
    with open(GRAPH_PKL, "rb") as f:
        data = pickle.load(f)
    G = data.get("graph") if isinstance(data, dict) else data
    print(f"   {G.number_of_nodes()} nodes / {G.number_of_edges()} edges total")

    print("2) loading chunk modality lookup ...")
    chunk_lookup = load_chunk_modality()
    print(f"   {len(chunk_lookup)} chunks")

    print("3) selecting the 12 demo videos ...")

    # ---- edges: one row per (edge, video-with-evidence) ----
    edge_rows = []
    node_needed = set()
    per_video_edges = Counter()

    for u, v, key, attrs in G.edges(keys=True, data=True):
        evid = attrs.get("evidence") or []
        by_video = defaultdict(list)
        for ev in evid:
            vid = ev.get("video_id")
            if vid in VIDEOS:
                by_video[vid].append(ev)
        if not by_video:
            continue

        relation = attrs.get("type") or key or "related_to"
        for vid, evs in by_video.items():
            modality, language = derive_modality(evs, chunk_lookup)
            first = min(evs, key=lambda e: e.get("timestamp") or 0.0)
            edge_rows.append({
                "video_id":   vid,
                "source":     str(u),
                "target":     str(v),
                "relation":   str(relation),
                "description": (attrs.get("description") or "")[:600],
                "weight":     float(attrs.get("weight") or 1.0),
                "n_evidence": len(evs),
                "modality":   modality,
                "language":   language,
                "timestamp":  float(first.get("timestamp") or 0.0),
                "end_time":   float(first.get("end_time") or 0.0) or None,
                "speaker":    first.get("speaker"),
                "snippet":    (first.get("snippet") or "")[:300],
                "evidence":   jsonable(evs[:8]),
            })
            per_video_edges[vid] += 1
            node_needed.add(str(u))
            node_needed.add(str(v))

    # ---- nodes ----
    node_rows, node_video_rows = [], []
    per_video_nodes = Counter()
    for n, attrs in G.nodes(data=True):
        n = str(n)
        vids = {str(x) for x in (attrs.get("video_ids") or set())} & VIDEOS
        if not vids and n not in node_needed:
            continue
        node_rows.append({
            "node_id":    n,
            "name":       attrs.get("name") or n,
            "node_type":  attrs.get("type") or "OTHER",
            "description": (attrs.get("description") or "")[:600],
            "aliases":    jsonable(attrs.get("aliases") or []),
            "n_mentions": int(attrs.get("n_mentions") or 0),
            "video_ids":  sorted(vids),
        })
        for vid in vids:
            n_m = sum(1 for m in (attrs.get("mentions") or [])
                      if m.get("video_id") == vid)
            node_video_rows.append({
                "node_id": n, "video_id": vid, "n_mentions": n_m,
            })
            per_video_nodes[vid] += 1

    print(f"   nodes: {len(node_rows)}  node-video links: {len(node_video_rows)}")
    print(f"   edges: {len(edge_rows)}\n")
    for vid in sorted(VIDEOS, key=lambda v: -per_video_edges[v]):
        print(f"     {per_video_nodes[vid]:5} nodes | {per_video_edges[vid]:5} edges | {vid[:46]}")

    mod = Counter(r["modality"] for r in edge_rows)
    rel = Counter(r["relation"] for r in edge_rows)
    print(f"\n   modality: {dict(mod)}")
    print(f"   top relations: {dict(rel.most_common(8))}")

    if args.dry_run:
        print("\nDRY RUN — nothing written.")
        if edge_rows:
            s = dict(edge_rows[0]); s["evidence"] = f"[{len(s['evidence'])} dicts]"
            print("   sample edge:")
            for k, v_ in s.items():
                print(f"      {k}: {str(v_)[:80]}")
        return

    if "PASTE_SERVICE_ROLE" in SUPABASE_SERVICE_KEY:
        sys.exit("ERROR: set SUPABASE_SERVICE_KEY first.")

    from supabase import create_client
    sb = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

    print("\n4) uploading ...")
    for i in range(0, len(node_rows), BATCH):
        sb.table("graph_nodes").upsert(node_rows[i:i+BATCH]).execute()
        print(f"   nodes {min(i+BATCH, len(node_rows))}/{len(node_rows)}")
    for i in range(0, len(node_video_rows), BATCH):
        sb.table("graph_node_videos").upsert(node_video_rows[i:i+BATCH]).execute()
        print(f"   node-videos {min(i+BATCH, len(node_video_rows))}/{len(node_video_rows)}")

    # edges have a serial pk -> clear the demo videos first, then insert
    for vid in VIDEOS:
        sb.table("graph_edges").delete().eq("video_id", vid).execute()
    for i in range(0, len(edge_rows), BATCH):
        sb.table("graph_edges").insert(edge_rows[i:i+BATCH]).execute()
        print(f"   edges {min(i+BATCH, len(edge_rows))}/{len(edge_rows)}")

    print("\nDONE.")


if __name__ == "__main__":
    main()
