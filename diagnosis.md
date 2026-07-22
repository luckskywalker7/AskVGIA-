# Graph Extraction Timeout Diagnosis

**Date:** 2026-06-28  
**Symptom:** "Ollama RELATION call TIMEOUT after 92.1s (timeout=90s)" and "ENTITY call TIMEOUT → heuristic fallback" repeating over 20+ minutes for a single video, producing a sparse graph.

---

## 1. Exact timeout values and where they are set

`entity_timeout_s = 90` and `relation_timeout_s = 90` — both set correctly from `settings.graph_extract_timeout_s` / `settings.graph_extract_relation_timeout_s` (both 90 in `config/settings.py:258-259`). `api/main.py:351-353` reads those and passes them into `ExtractorConfig`. The `ExtractorConfig` dataclass hard-codes the same 90 s defaults, so the settings path is the effective one.

---

## 2. Prompt size per call — measured, not estimated

| Call | Prompt chars | Tokens in (measured by Ollama) | Tokens out |
|------|-------------|-------------------------------|------------|
| Entity | 1,847 | **409** | 380 |
| Relation | 2,045 | **477** | 389 |

The prompts are tiny — 400–500 tokens in, 380–390 tokens out. **Prompt size is not the problem.**

---

## 3. `think: false` — set correctly, not emitting reasoning tokens

Both `_call_entity` and `_call_relation` send `"think": False` in the payload (`core/graph_extractor.py:501, 566`), and both prompts start with `/nothink`. Measured calls returned **0 thinking tags** and the response was clean JSON. qwen3:8b is NOT blowing the time budget on reasoning.

---

## 4. `num_ctx` — NOT set; model defaults to 40,960 tokens

The graph extractor payload (`core/graph_extractor.py:499-507, 564-572`) sends only `temperature`, `num_predict`, and `top_p`. **There is no `num_ctx`.**

```python
# graph_extractor.py _call_entity payload — what is actually sent:
"options": {
    "temperature": self.cfg.temperature,
    "num_predict": self.cfg.entity_num_predict,  # 500
    "top_p": 0.9,
}
# ← no num_ctx
```

Confirmed via `qwen3.context_length = 40960` from Ollama's `/api/show` endpoint. This forces Ollama to pre-allocate a ~3 GB KV cache buffer for every single call (40,960 ctx × 8 KV heads × 128 dim × 2 K+V × 36 layers × fp16 ≈ 3 GB), even though the actual prompt + output fits in under 1,000 tokens.

**The VLM client already applies the correct fix** (`core/vlm_client.py:236`):
```python
"num_ctx": 4096,
# comment: "Smaller num_ctx = less KV-cache = faster and lower VRAM. 4096 is plenty for one frame + prompt."
```
The graph extractor was never given the same treatment.

---

## 5. Actual timed measurements (RTX 5060 Ti, 16 GB VRAM, qwen3:8b Q4_K_M)

```
Cold load (model not in VRAM):
  Entity call:  14.5 s total
    load_duration:          6.63 s
    prompt_eval_duration:   0.37 s   (409 tokens prefill)
    eval_duration:          5.44 s   (380 tokens generated)

Hot cache (model already in VRAM, no contention):
  Relation call:  8.2 s total
    load_duration:          0.29 s
    prompt_eval_duration:   0.31 s   (477 tokens prefill)
    eval_duration:          5.51 s   (389 tokens generated)

  Entity call:    5.5 s total
    load_duration:          0.16 s
    prompt_eval_duration:   0.08 s   (140 tokens prefill)
    eval_duration:          3.15 s   (227 tokens generated)
```

**In isolation, calls are 5–14 s. They can only hit 90 s when something else is competing for the GPU.**

---

## 6. Root cause — GPU queue starvation from concurrent ingest

`api/main.py:113` sets:

```python
_ingest_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="ingest")
```

Two videos can ingest simultaneously. The ingest pipeline is sequential *within* one video (vision → graph), but when two videos run concurrently:

- **Video A**: VisionIndexer calls `qwen2.5vl:7b` for each keyframe — each call takes **~20–30 s** and is submitted one at a time (`vlm_max_concurrency=1`).
- **Video B**: GraphExtractor starts sending entity calls to `qwen3:8b`.

Ollama serialises all GPU work on a single device. When Video B's entity call arrives while Video A's VLM frame is still mid-generation, it queues behind it. If **two or three VLM frame calls** are already in-flight or queued when graph extraction begins, the entity call waits `2–3 × 30 s = 60–90 s` in the queue — right at the timeout boundary.

Additionally, the missing `num_ctx` matters even in single-video scenarios: Ollama must tear down a 4,096-token KV cache buffer (from VLM's `num_ctx=4096`) and re-allocate a 3 GB buffer for the default `num_ctx=40960` before the first token of the graph extraction call generates. On a system that just finished processing 1,000 VLM frames, that buffer churn adds latency to the first call in each graph extraction session.

---

## Summary

| # | Finding | Evidence |
|---|---------|----------|
| 1 | Timeout = 90 s for both entity and relation calls | `config/settings.py:258-259`, `api/main.py:352-353` |
| 2 | Prompts are tiny (~450 tokens in, ~380 tokens out) | Measured: `prompt_eval_count=409/477` |
| 3 | `think: False` + `/nothink` work correctly — no reasoning overhead | 0 `<think>` tags in measured responses |
| 4 | **`num_ctx` not set → 40,960-token default → ~3 GB KV buffer per call** | `core/graph_extractor.py:499-507`; VLM client shows the correct pattern at `core/vlm_client.py:236` |
| 5 | Isolated hot-cache call = 5–8 s; well within 90 s | Measured above |
| 6 | **Concurrent ingest (max_workers=2) queues entity calls behind 20–30 s VLM frame calls, hitting the 90 s wall** | `api/main.py:113`; VLM load measured at 5.87 s with inference ~20-30 s/frame |

---

## Proposed fixes (not yet applied)

1. **Add `"num_ctx": 2048` to graph extractor payloads** — prompts are <1,000 tokens total; 2,048 is ample headroom and reduces KV cache from ~3 GB to ~150 MB per call. Apply in both `_call_entity` and `_call_relation` in `core/graph_extractor.py`, mirroring the pattern already in `core/vlm_client.py:236`.

2. **Serialise Ollama access between VLM and graph phases** — reduce `_ingest_executor` to `max_workers=1`, or add a per-video mutex so graph extraction only starts after the VLM phase of that video (and all concurrent videos) has fully drained the Ollama queue. This eliminates the queue-starvation scenario.
