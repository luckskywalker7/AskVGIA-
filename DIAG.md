# AskVGIA Cloud — Diagnosis

Scope: why the dashboard rendered empty panels when served by `qa_service.py`
(cloud build) even though the same frontend works against `api/main.py` (local
build). All findings below were reproduced against the live Supabase project,
not inferred from reading code.

Ground rule for every fix: **`static/app/` and `project/` were not touched.**
The frontend is fixed; the backend was made to match what it already asks for.

---

## Summary

| # | Symptom | Root cause | Fixed? |
|---|---------|-----------|--------|
| 1 | Graph blank / "isolated nodes, no edges" | `get_video_graph` returns edges pointing at nodes it didn't return; d3 throws and aborts the render | ✅ cloud-side |
| 2 | Graph genuinely sparse on 4 videos | Only the `speaker→video` edge was ever extracted | ❌ needs local re-ingestion |
| 3 | Transcript tab empty | `demo_videos` has no `summary` / `keywords` column | ✅ derived from chunks |
| 4 | Timeline tab empty | `POST /search` did not exist | ✅ implemented |
| 5 | Chapters tab empty | `GET /video/{id}/chapters` did not exist | ✅ implemented |
| 6 | No image keywords on video cards | `/videos` never returned them | ✅ implemented |

---

## The data we actually have

Established by probing PostgREST directly.

**`demo_videos`** — 12 rows, columns:
```
video_id, youtube_id, title, is_truncated, n_chunks
```
Notably **absent**: `summary`, `keywords`, `topics`, `duration_seconds`,
`chapters`, `metadata`. `export_to_supabase.py:233-236` only ever wrote those
five fields, so every dashboard panel that reads video-level metadata had
nothing to read.

**`video_chunks`** — 2 529 rows (536 `speech`, 1 993 `visual`), columns:
```
chunk_id, video_id, text, original_text, search_text, start_time, end_time,
language, speaker, topic, chunk_type, key_points, metadata, embedding
```

`metadata` on visual chunks is rich — this is what made fixes 3 and 6 possible
without a re-export. Coverage on the MIT lecture (342 visual chunks):

| field | populated |
|---|---|
| `caption` | 342 |
| `short_caption` | 342 |
| `ocr_text` | 340 |
| `screen_text` | 340 |
| `tags` | 325 |
| `board_text` | 174 |

`tags` sample: `['university', 'autumn', 'academic', 'building', 'people']`

`key_points` is **empty on every row checked** — don't build anything on it.
`topic` has 44 distinct values on the MIT video but 342/400 chunks are
`'general'`.

**`graph_nodes`** (1 175 rows): `node_id, name, node_type, description,
aliases, n_mentions, video_ids, metadata`
**`graph_edges`** (3 638 rows): `id, video_id, source, target, relation,
weight, snippet, speaker, modality, evidence, timestamp, …`

Tables that do **not** exist: `video_graph`, `entities`, `chapters`.

---

## 1. Graph renders blank — dangling edges

**Severity: high.** This was the "ugly and weird" one, and it wasn't a styling
problem — the graph was never drawn at all.

`api.js:907` does:

```js
d3.forceSimulation(nodes)
  .force('link', d3.forceLink(edges).id(d => d.id) …)
```

`d3.forceLink` resolves every `edge.source` / `edge.target` against the node
`id` set. If one is missing it throws `Error: missing: <id>` — and because the
throw happens inside `renderGraphViz`, the whole render aborts and the panel
keeps whatever was on screen. **One bad edge blanks a 588-node graph.**

The RPC selects nodes and edges by different keys: nodes via
`graph_nodes.video_ids @> video`, edges via `graph_edges.video_id = p_video_id`.
An entity whose `video_ids` never got the video appended still has its edges
returned. The unresolved ids are all junk extractions — `'questions'`, `'test'`,
`'problem'`, `'split'`, `'a'`, `'r'`, `'3 4 5'`, `'02'`, `'x'`.

Measured before the fix — **7 of 12 videos were throwing**:

```
video                               nodes  edges  dangling  verdict
dc28b907_MIT_Introduction_to_Deep_    588   1872         3  D3 THROWS
-O3J0FkiRsk                            58     99         1  D3 THROWS
cSj82GG6MX4                           145    471        22  D3 THROWS
dxmm__4p-I8                           111    406        16  D3 THROWS
j7qVcc6X8g4                            91    307         1  D3 THROWS
cXhSXGyC5F4                           124    382        15  D3 THROWS
0YqZIHFmVzg                            34     51         0  ok
db98802f_Learn_Arabic_Podcast_Natu     34     46         0  ok
```

**Fix** — `/graph/video/{id}` drops edges whose endpoints aren't in the node
set, logs what it dropped, then removes nodes left with no edges (isolated
nodes are the "scattered confetti" look). Isolated nodes are kept if that would
otherwise empty the graph, so a sparse-but-real graph still renders instead of
falling back to the "no graph data" placeholder.

After: **all 12 videos report 0 dangling.** MIT renders 570 nodes / 1 869 edges.

---

## 2. Four videos have a genuinely empty graph — NOT fixable here

Distinct from #1. These have no dangling edges; extraction simply never
produced any real relations:

| video | nodes | edges |
|---|---:|---:|
| `509f803e_A_Brief_History_of_Geologic_Time` | 33 | 1 |
| `74280c28_How_to_manage_your_emotions` | 19 | 1 |
| `7b7d5ff0_What_happens_if_you_cut_down_all…` | 20 | 1 |
| `b5438048_The_benefits_of_doing_nothing…` | 21 | 1 |

The single edge is always the trivial one:

```json
{"source":"speaker_00","target":"74280c28_how_to_manage_your_emotions",
 "relation":"speaks_in","weight":16}
```

So 20-30 entities were extracted but **zero relations between them**. After the
isolated-node filter these now render as 2 connected nodes rather than a cloud
of disconnected dots — honest, but still nearly empty.

**This is an ingestion gap and cannot be repaired from the cloud service.**
The relation-extraction pass has to be re-run locally for these four videos and
`export_graph.py` re-run. Note all four are the file-ingested videos
(`<hash>_<title>`); the YouTube-id videos all have real graphs, which suggests
the two paths didn't run the same extraction stage.

---

## 3. Transcript tab — "No summary available"

`api.js:692-694` reads `data.summary`, `data.keywords`, `data.topics` from
`/video/{id}`. `demo_videos` has none of those columns, so `summary` was `''`
and line 711 rendered the empty-state message.

**Fix** — `/video/{id}` now derives them from `video_chunks`:

- `summary` — leading speech text (falls back to frame captions for silent video)
- `keywords` — most frequent `metadata.tags` across visual chunks, minus a
  stop-list of tags that describe the recording rather than the subject
  (`video`, `screenshot`, `background`, `person`, `blurry`, …)
- `topics` — most frequent `topic` values, excluding `'general'`
- `duration_seconds` — max `end_time`

Result for the MIT lecture:

```
summary : "Good afternoon, everyone, and thank you for joining us today.
           My name is Alexander Amini, and together with Av…"
keywords: deep learning, education, mit, lecture, technology, presentation,
          lecturer, academic, neural networks, mathematics, lectures, classroom
topics  : Neural Network Layers, Neural Network Basics, Deep Learning Progress…
duration: 2333.98
```

---

## 4. Timeline tab — "No timeline data available"

`api.js:716` calls `POST /search` with `{video_id, query, top_k}` and reads
`.results || .chunks`. **The route didn't exist** → `.catch(() => null)` →
empty list → empty-state message.

**Fix** — implemented `POST /search`. Important detail: the Timeline tab wants
chunks in **time order, not relevance order** (it filters out visual chunks and
lists timestamps), so this deliberately does not run the hybrid search. It
returns speech chunks ordered by `start_time` under both `results` and `chunks`.

Also **deduplicated**: the store contains overlapping speech chunks — the
opening line appeared twice at `9.3s` — which rendered as repeated rows.

Before / after for the first three entries:

```
before:  9.3s Good afternoon, everyone…   after:  9.3s Good afternoon, everyone…
         9.3s Good afternoon, everyone…          20.2s This is MIT Introduction…
        20.2s This is MIT Introduction…          30.4s And I think probably a…
```

---

## 5. Chapters tab — "No chapters yet"

`api.js:753` calls `GET /video/{id}/chapters` expecting
`{chapters: [{index, title, start_time, end_time, summary, thumbnail_url}]}`.
**The route didn't exist.**

**Fix** — implemented, deriving chapters from `topic` transitions in time
order. Chunks with `topic='general'` extend the open chapter instead of
starting a meaningless one. Each chapter gets a thumbnail from the first member
chunk carrying `metadata.frame_url`, and a blurb from the first chunk with text.

**Tuning note:** a fixed 20 s minimum produced **164 chapters** for the
39-minute lecture, because topics flip every few seconds in the ingested data.
The minimum is now derived from duration (`max(45s, duration/24)`), giving
**28 chapters** — readable at any video length.

```
Ch1     9.3s -   96.2s  MIT Deep Learning Course        thumb=True
Ch2    87.8s -  181.0s  Deep Learning Progress          thumb=True
Ch3   170.9s -  259.4s  Deep Learning Course            thumb=True
Ch4   252.3s -  347.3s  Static Script Limitations       thumb=True
```

---

## 6. Image keywords on the video cards

Requested position: under the title, above the chunk count. That is exactly the
`${summary}` slot in `_videoCardHTML`:

```
api.js:1096   <h3>  … title …  </h3>
api.js:1097   ${summary}                 ← keywords go here
api.js:1098   <p class="video-meta"> … status / chunks … </p>
```

Since `api.js` is off-limits, `/videos` now populates `summary` with the top
visual tags (and also returns a proper `keywords` array for any consumer that
wants it). Tags are sampled from up to 150 visual chunks per video, fetched
concurrently with `asyncio.gather` and cached, so the grid doesn't fan out into
12 full-table reads.

```
MIT Introduction to Deep Learning   education · lecture · deep learning · technology · mit
A Brief History of Geologic Time    lecture · presentation · dark background · geology
How to manage your emotions         cartoon · character · cartoon-style · animation
What happens if you cut down all…   trees · landscape · greenery · cartoon · tree
```

---

## Verification

All routes exercised against live Supabase via `TestClient`:

```
/video/{id}            200  summary ✓  keywords 12 ✓  topics ✓  duration ✓
POST /search           200  50 results, 0 duplicates, time-ordered ✓
/video/{id}/chapters   200  28 chapters, thumbnails ✓
/graph/video/{id}      200  12/12 videos, 0 dangling edges ✓
/videos                200  12 videos, keywords on every card ✓
```

---

## Still open

**Requires local re-ingestion (cannot be fixed in the cloud service):**

- Relation extraction for the four videos in §2.
- `key_points` is empty on every chunk — nothing consumes it today, but it was
  presumably meant to.
- 1 993 visual vs 536 speech chunks. Frame sampling is aggressive; `dedupe_visual`
  in `/ask` compensates at query time, but the underlying redundancy is why
  topics flip every few seconds and why chapter tuning was needed.

**Deliberately not implemented** (all degrade safely):

- `POST /ask/stream` — 404 is caught at `api.js:1245`, which falls back to
  `/ask`. Answers arrive complete instead of streaming token-by-token.
- `/upload`, `/ingest`, `/ingestion/status` — ingestion stays local by design.
- `/video/{id}` DELETE — the delete button on the cards will fail.

**Caveats:**

- Derived metadata is cached in-process for the process lifetime. The corpus is
  static (ingestion is local), so this is safe — but a re-export won't be picked
  up until the service restarts. On Render's free tier the instance sleeps after
  ~15 min idle, so this resolves itself in practice.
- Video-card thumbnails come from `/vision/frames`, which only returns chunks
  with `metadata.frame_url`. Videos whose frames were never uploaded by
  `upload_frames.py` still show the film-icon placeholder — missing data, not a
  bug. Check with `curl "…/vision/frames/<video_id>?limit=1"`.
