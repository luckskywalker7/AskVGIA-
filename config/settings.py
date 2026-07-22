from __future__ import annotations
from typing import List, Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8",
        case_sensitive=False, extra="ignore",
    )

    # ── Application ───────────────────────────────────────────────────
    app_name: str = "Agentic Video Search"
    app_version: str = "3.0.0"
    environment: str = "development"
    debug: bool = True

    # ── API / CORS ────────────────────────────────────────────────────
    cors_origins: List[str] = Field(default_factory=lambda: ["*"])

    # ── Transcription / WhisperX ─────────────────────────────────────
    whisper_model: str = "medium"
    whisper_device: str = "cuda"          # ~3.0 GB VRAM (float16)
    whisper_compute_type: str = "float16"
    audio_dir: str = "data/audio"

    # ── BGE-M3 Embedding device ───────────────────────────────────────
    # KEEP on CPU.
    # Reason: during indexing qwen2.5vl:7b occupies ~8.5 GB.
    # WhisperX takes ~3.0 GB. That leaves only ~4.5 GB free.
    # BGE-M3 on CUDA needs ~2.2 GB — too tight, causes OOM.
    # On CPU, BGE-M3 embedding is still fast enough (~50 ms/chunk).
    bge_device: str = "cpu"

    # ── Diarization / Pyannote ────────────────────────────────────────
    pyannote_enabled: bool = True
    pyannote_min_speakers: int = 1
    pyannote_max_speakers: int = 5
    hf_token: Optional[str] = None
    huggingface_token: Optional[str] = None

    # ── Storage (FAISS + BM25) ────────────────────────────────────────
    storage_dir: str = "data/faiss_store"
    faiss_index_type: str = "hnsw"
    faiss_hnsw_m: int = 32
    faiss_hnsw_ef_construction: int = 200
    # CHANGED: 128 → 256 — better recall at search time, costs ~5 ms extra.
    # Pure CPU operation, no VRAM involved.
    faiss_hnsw_ef_search: int = 256
    faiss_nprobe: int = 10
    bge_dense_dim: int = 1024

    # Language-specific indexes
    faiss_language_indexes: List[str] = Field(
        default_factory=lambda: ["ar", "en", "fr"]
    )

    # BM25
    bm25_enabled: bool = True
    bm25_k1: float = 1.5
    bm25_b: float = 0.75

    # Score fusion
    rrf_k: int = 60
    fusion_ar_dense: float = 0.35
    fusion_ar_sparse: float = 0.65
    fusion_en_dense: float = 0.65
    fusion_en_sparse: float = 0.35
    fusion_fr_dense: float = 0.50
    fusion_fr_sparse: float = 0.50
    fusion_default_dense: float = 0.50
    fusion_default_sparse: float = 0.50

    # Score cache
    score_cache_size: int = 500
    score_cache_ttl: float = 1800.0

    # Snapshot
    snapshot_auto_save: bool = True
    snapshot_interval_chunks: int = 50

    # Embedding cache TTL — 30 days
    embedding_cache_ttl_seconds: float = 2592000.0

    # ── Semantic Query Cache ──────────────────────────────────────────
    query_cache_enabled: bool = True
    # CHANGED: 0.92 → 0.95 — stricter cosine threshold before serving a cached
    # answer. Stops "did X mention Y?" from returning the cached answer for
    # "did X mention Z?" — they score ~0.93 cosine but need different answers.
    query_cache_similarity: float = 0.95
    query_cache_max_entries: int = 10_000
    query_cache_ttl_days: float = 30.0
    query_cache_llm_fallback_threshold: float = 0.75

    # ── Qdrant (kept for backward compat, unused with FAISS) ─────────
    qdrant_mode: str = "local"
    qdrant_path: str = "data/qdrant"
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "video_chunks"
    qdrant_timeout: float = 30.0
    qdrant_prefer_grpc: bool = False

    # ── Embeddings / Search ───────────────────────────────────────────
    nomic_dense_dim: int = 768
    nomic_enabled: bool = True
    colbert_enabled: bool = True
    hyde_enabled: bool = True
    embedding_cache_enabled: bool = True

    # ── Chunking ──────────────────────────────────────────────────────
    chunking_semantic_enabled: bool = True
    chunking_max_words: int = 220
    chunking_min_words: int = 40
    chunking_overlap_seconds: float = 3.0
    chunking_max_duration_seconds: float = 35.0
    chunking_min_duration_seconds: float = 8.0
    chunking_split_on_speaker_change: bool = True
    chunking_merge_short_segments: bool = True
    chunking_max_gap_seconds: float = 1.5

    # ── Languages ─────────────────────────────────────────────────────
    supported_languages: List[str] = Field(
        default_factory=lambda: ["ar", "en", "fr", "es", "de", "it", "pt"]
    )
    default_language: str = "en"

    # ── Vision / OCR ──────────────────────────────────────────────────
    vision_enabled: bool = True
    # CHANGED: 2.0 → 8.0 s — for hour-long lectures, a 2 s grid hits the
    # vision_max_frames cap at ~26 min and the rest of the video is never
    # ingested (confirmed: a 69-min lecture only indexed 0–26 min). Slides are
    # held far longer than 8 s, and scene-change detection still catches every
    # slide transition between interval ticks, so 8 s gives full-video coverage.
    vision_frame_interval_seconds: float = 8.0
    # Hard cap on frames sent to the VLM per video.
    # CHANGED: 800 → 1000.  At 800 + 2 s interval the loop stopped at ~26 min on
    # a 69-min lecture (the second half was never ingested). With an 8 s interval
    # the full hour needs ~520 interval frames PLUS scene-change frames; 1000
    # gives headroom so long lectures are not truncated again. NOTE: ingest time
    # scales with frame count (~27 s/frame on the 16 GB GPU), so a full 1-hour
    # lecture can still take several hours until content-aware dedup is added to
    # collapse duplicate-slide frames BEFORE the VLM. Lower this (or the interval)
    # if you need faster ingest at the cost of coverage.
    vision_max_frames: int = 1000
    # Structural similarity gate in KeyframeExtractor.  Frames more similar
    # than this threshold to the previously-kept frame are skipped.
    # CHANGED: not set previously → 0.97.  The old hard-coded default of 0.92
    # was rejecting valid cinematic frames (slow pans, talking-head shots) that
    # are intentionally visually consistent.  0.97 only drops near-perfect
    # duplicates (e.g. two consecutive frames of a static title card).
    vision_structural_sim_threshold: float = 0.97
    # pHash distance gate for frame deduplication (0–256, higher = looser).
    # CHANGED: was hardcoded at 5 in vlm_contracts.py and not settable.
    # 5/256 is so tight that frames 5 s apart in the same scene hash-collide
    # and get dropped, producing just 1 frame/video on long films.
    # 18 ≈ 93% visual similarity — drops true duplicates without killing
    # distinct shots from the same scene.
    vision_dedup_phash_distance: int = 18
    ocr_enabled: bool = True
    ocr_backend: str = "easyocr"
    ocr_languages: List[str] = Field(default_factory=lambda: ["ar", "en", "fr"])

    # ── Vision-Language Model (VLM) ───────────────────────────────────
    vision_model_enabled: bool = True

    # VLM = vision model for FRAME CAPTIONING only.
    # Do NOT change to qwen3:8b — qwen3 is text-only, it cannot see images.
    # qwen2.5vl:7b uses ~8.5 GB VRAM (Q4_K_M via Ollama).
    # Ollama loads it on demand and unloads it when the chat model is needed.
    # The two models are NEVER resident in VRAM simultaneously.
    vlm_primary_model: str = "qwen2.5vl:7b"
    vlm_fallback_models: list = ["qwen2.5vl:7b"]

    vlm_backend: str = "ollama"
    vlm_ollama_url: str = "http://localhost:11434"
    vlm_device: str = "auto"
    vlm_dtype: str = "auto"
    vlm_openai_base_url: str = ""
    vlm_openai_api_key: str = ""
    vlm_max_retries: int = 2
    # CHANGED: 90 → 120 s — more time for complex frames (slides with dense text)
    vlm_timeout_s: float = 120.0
    # KEEP at 1 — two concurrent VLM calls would double VRAM usage to ~17 GB → OOM.
    # Sequential is safe: one frame at a time, full 8.5 GB available per frame.
    vlm_max_concurrency: int = 1

    # ── Keyframe extraction ───────────────────────────────────────────
    # CHANGED: 2.0 → 8.0 s — must match vision_frame_interval_seconds above.
    # VisionIndexer reads THIS field; the one above is for older code paths.
    # Both must always be the same value. See note above: 2 s truncated long
    # lectures at the frame cap; 8 s + scene-change detection covers the full video.
    vision_frame_interval_s: float = 8.0

    # CHANGED: 0.30 → 0.25 — catch more scene changes (slide transitions,
    # cut scenes) without being so sensitive that static lecture shots
    # generate redundant frames.
    vision_scene_threshold: float = 0.25

    # ── CLIP (multimodal embeddings) ──────────────────────────────────
    vlm_clip_enabled: bool = True
    vlm_clip_model: str = "sentence-transformers/clip-ViT-B-32"
    vlm_clip_dim: int = 512
    # KEEP on CPU.
    # CLIP is only ~0.6 GB, but during indexing qwen2.5vl:7b + WhisperX already
    # fill 11.5 GB. Adding CLIP on CUDA pushes into OOM territory.
    # On CPU, embedding 500 frames takes ~30 s — acceptable one-time cost.
    vlm_clip_device: str = "cpu"

    # ── Visual CLIP persistent index ─────────────────────────────────
    visual_clip_collection_name: str = "visual_clip"
    visual_clip_store_dir: str = "data/faiss_store/visual_clip"
    visual_clip_faiss_index_type: str = "flat"
    visual_clip_rebuild_threshold: int = 100
    vision_batch_vqa_max_questions: int = 8

    # ── LLM (chat / answer synthesis) ────────────────────────────────
    ollama_base_url: str = "http://localhost:11434"

    # ★ CHANGED: qwen3:1.7b → qwen3:8b
    # This is the single biggest accuracy improvement.
    # qwen3:8b: follows instructions precisely, no self-contradictions,
    # handles Arabic/French/English equally well, gives YES/NO without drift.
    #
    # VRAM budget during chat (no indexing):
    #   qwen3:8b via Ollama    ~5.5 GB  (Q4_K_M)
    #   WhisperX (idle/unloaded)  0 GB  (Ollama unloads between requests)
    #   BGE-M3 on CPU             0 GB VRAM
    #   CLIP on CPU               0 GB VRAM
    #   Reranker on CPU           0 GB VRAM
    #   PyTorch CUDA context   ~0.5 GB
    #   ─────────────────────────────
    #   Total during chat:     ~6.0 GB  ← well within 16 GB
    #
    # Before starting the server, pull the model once:
    #   ollama pull qwen3:8b
    ollama_chat_model: str = "qwen3:8b"

    # Vision model for frame captioning — keep separate from chat model.
    ollama_vision_model: str = "qwen2.5vl:7b"

    # ── Answering ─────────────────────────────────────────────────────
    # CHANGED: 4 → 6 — qwen3:8b can reason over more evidence reliably.
    answer_max_evidence: int = 6
    # CHANGED: 8 → 10
    answer_max_context_chunks: int = 10
    answer_require_citations: bool = True

    # ── GraphRAG ─────────────────────────────────────────────────────
    graph_enabled: bool = True
    graph_store_dir: str = "data/graph_store"

    graph_extraction_model: Optional[str] = None   # falls back to ollama_chat_model
    graph_extract_max_text_chars: int = 900
    graph_extract_temperature: float = 0.1
    graph_extract_num_predict: int = 600
    graph_extract_timeout_s: int = 90
    graph_extract_relation_timeout_s: int = 90
    graph_extract_use_heuristic_fallback: bool = True
    graph_max_entities_per_chunk: int = 10
    graph_max_relations_per_chunk: int = 8

    # Community detection
    graph_run_communities: bool = True
    graph_community_summarize: bool = True
    graph_max_communities: int = 50
    graph_community_min_size: int = 2
    graph_extractor_max_workers: int = 1

    # Search
    graph_local_hops: int = 1
    graph_local_max_nodes: int = 25
    graph_global_top_k: int = 3
    graph_default_mode: str = "auto"

    # ── Reranker ──────────────────────────────────────────────────────
    # KEEP on CPU.
    # Cross-encoder reranker needs ~0.4 GB — safe on CPU.
    # Moving to CUDA would collide with qwen2.5vl:7b during indexing.
    # On CPU: ~3 s/chunk × 8 chunks = 24 s. Acceptable — runs in parallel
    # with the Ollama response stream so the user doesn't feel it.
    reranker_device: str = "cpu"

    # ── Properties ────────────────────────────────────────────────────

    @property
    def qdrant_url(self) -> str:
        if self.qdrant_mode == "local":
            return f"local://{self.qdrant_path}"
        return f"http://{self.qdrant_host}:{self.qdrant_port}"

    @property
    def effective_hf_token(self) -> Optional[str]:
        return self.hf_token or self.huggingface_token

    def get_fusion_weights(self, language: str) -> tuple:
        lang = language.lower()[:2]
        if lang == "ar":
            return self.fusion_ar_dense, self.fusion_ar_sparse
        elif lang == "en":
            return self.fusion_en_dense, self.fusion_en_sparse
        elif lang == "fr":
            return self.fusion_fr_dense, self.fusion_fr_sparse
        return self.fusion_default_dense, self.fusion_default_sparse


settings = Settings()