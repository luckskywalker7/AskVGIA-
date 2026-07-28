// ========== ASKVGIA — Backend API Integration ==========
// Connects the new frontend to the FastAPI backend at /

const API = {
  // ── Core fetch helpers ────────────────────────────────
  async get(path) {
    const r = await fetch(path);
    if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
    return r.json();
  },
  async post(path, body) {
    const r = await fetch(path, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!r.ok) {
      let msg = `${r.status}`;
      try { const d = await r.json(); msg = d.detail || msg; } catch (_) {}
      throw new Error(msg);
    }
    return r.json();
  },
  async delete(path) {
    const r = await fetch(path, { method: 'DELETE' });
    if (!r.ok) throw new Error(`${r.status}`);
    return r.json();
  },

  // ── Videos ───────────────────────────────────────────
  async listVideos() {
    const d = await this.get('/videos');
    return d.videos || [];
  },
  async getVideo(id) {
    return this.get(`/video/${id}`);
  },
  async deleteVideo(id) {
    return this.delete(`/video/${id}`);
  },

  // ── Upload ───────────────────────────────────────────
  async uploadFile(file, language, onProgress) {
    return new Promise((resolve, reject) => {
      const fd = new FormData();
      fd.append('file', file);
      if (language) fd.append('language', language);

      const xhr = new XMLHttpRequest();
      xhr.open('POST', '/upload');
      xhr.upload.onprogress = (e) => {
        if (e.lengthComputable && onProgress) onProgress(e.loaded / e.total);
      };
      xhr.onload = () => {
        if (xhr.status >= 200 && xhr.status < 300) {
          try { resolve(JSON.parse(xhr.responseText)); }
          catch (_) { resolve({}); }
        } else {
          let msg = xhr.status;
          try { msg = JSON.parse(xhr.responseText).detail || msg; } catch (_) {}
          reject(new Error(msg));
        }
      };
      xhr.onerror = () => reject(new Error('Network error'));
      xhr.send(fd);
    });
  },

  // ── Ingest (index) ───────────────────────────────────
  async ingest(videoId) {
    return this.post('/ingest', { video_id: videoId });
  },

  // ── Ingestion status polling ──────────────────────────
  async ingestionStatus(videoId) {
    return this.get(`/ingestion/status/${videoId}`);
  },

  // ── Ask / Q&A ────────────────────────────────────────
  async ask(videoId, question, opts = {}) {
    return this.post('/ask', {
      video_id: videoId,
      question,
      confirmed: true,
      use_llm: opts.useLLM !== false,
      use_rerank: opts.useRerank !== false,
      // FIX-4: larger pool so visual chunks are not crowded out by audio
      top_k: opts.topK || 16,
      // FIX-4: always enable vision so on-screen articles/text frames are retrieved
      vision: opts.vision !== false,
    });
  },

  // ── Ask with streaming ───────────────────────────────
  async* askStream(videoId, question, opts = {}) {
    const r = await fetch('/ask/stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        video_id: videoId,
        question,
        confirmed: true,
        use_llm: opts.useLLM !== false,
        use_rerank: opts.useRerank !== false,
        // FIX-4: larger pool so visual chunks are not crowded out by audio
        top_k: opts.topK || 16,
        // FIX-4: always enable vision so on-screen articles/text frames are retrieved
        vision: opts.vision !== false,
      }),
    });
    if (!r.ok) throw new Error(`${r.status}`);
    const reader = r.body.getReader();
    const decoder = new TextDecoder();
    let buf = '';
    let currentEvent = null;
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      const lines = buf.split('\n');
      buf = lines.pop();
      for (const line of lines) {
        if (line.startsWith('event: ')) {
          currentEvent = line.slice(7).trim();
        } else if (line.startsWith('data: ')) {
          const payload = line.slice(6).trim();
          if (payload && payload !== '[DONE]') {
            try {
              const parsed = JSON.parse(payload);
              if (currentEvent === 'results') {
                yield { _event: 'results', results: Array.isArray(parsed) ? parsed : [] };
              } else {
                yield parsed;
              }
            } catch (_) { yield { token: payload }; }
          }
          currentEvent = null;
        } else if (line.trim() === '') {
          currentEvent = null;
        }
      }
    }
  },

  // ── Health ───────────────────────────────────────────
  async health() {
    return this.get('/health');
  },
};

// ========== HISTORY STORE (server-side, per-user, 7-day retention) ==========
// Entries live in Supabase `ask_history`. Row-level security scopes them to the
// signed-in user and hides anything older than 7 days — but only if the request
// carries that user's token, so every call below sends one. Signed out means no
// header, and the backend answers with an empty history instead of an error.
async function _historyHeaders() {
  try {
    if (typeof sb === 'undefined') return {};
    const { data } = await sb.auth.getSession();
    const token = data?.session?.access_token;
    return token ? { Authorization: 'Bearer ' + token } : {};
  } catch (_) { return {}; }
}

// A Supabase row (snake_case columns, ISO created_at) in the shape the History
// page renders. videoTitle isn't a column — it rides along in metadata.
function _historyEntry(row) {
  const md = row.metadata || {};
  return {
    id: row.id,
    videoId: row.video_id,
    videoTitle: md.video_title || row.video_id,
    question: row.question || '',
    answer: row.answer || '',
    ts: Date.parse(row.created_at || '') || Date.now(),
  };
}

const HistoryStore = {
  // Fetch this user's entries from server (last 7 days, newest first)
  async list() {
    try {
      const r = await fetch('/history', { headers: await _historyHeaders() });
      if (!r.ok) return [];
      const d = await r.json();
      const rows = Array.isArray(d) ? d : (d.entries || []);
      return rows.map(_historyEntry);
    } catch (_) { return []; }
  },

  // Save one entry to server (fire-and-forget, returns promise)
  async save({ videoId, videoTitle, question, answer }) {
    try {
      await fetch('/history', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', ...(await _historyHeaders()) },
        body: JSON.stringify({
          video_id: videoId,
          question,
          answer,
          metadata: { video_title: videoTitle || '' },
        }),
      });
    } catch (_) {}
  },

  // Delete a single entry by id
  async delete(id) {
    try {
      await fetch(`/history/${id}`, {
        method: 'DELETE',
        headers: await _historyHeaders(),
      });
    } catch (_) {}
  },

  // Wipe all entries
  async clearAll() {
    try {
      await fetch('/history', {
        method: 'DELETE',
        headers: await _historyHeaders(),
      });
    } catch (_) {}
  },
};

// ========== STATE ==========
const State = {
  videos: [],           // all videos from /videos
  currentVideo: null,   // selected video object
  pollingTimer: null,
  chaptersCache: {},    // videoId → chapters array (session cache)
};

// Sticky "last selected video" — persists across page navigation (history.html,
// my-videos.html, index.html) so switching panels doesn't lose the selection.
const SELECTED_VIDEO_KEY = 'askvgia_selected_video';
function setSelectedVideoId(id) {
  try { localStorage.setItem(SELECTED_VIDEO_KEY, id); } catch (_) {}
}
function clearSelectedVideoId() {
  try { localStorage.removeItem(SELECTED_VIDEO_KEY); } catch (_) {}
}
function getSelectedVideoId() {
  try { return localStorage.getItem(SELECTED_VIDEO_KEY); } catch (_) { return null; }
}

// ========== DOM HELPERS ==========
function $(id) { return document.getElementById(id); }
function setStatus(msg, type = 'info') {
  const el = $('status-bar');
  if (!el) return;
  el.textContent = msg;
  el.className = `status-bar ${type} visible`;
}
function clearStatus() {
  const el = $('status-bar');
  if (el) { el.className = 'status-bar'; el.textContent = ''; }
}
function esc(s) {
  return String(s || '').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}
// Answer prose in its pre-wrap wrapper, so the model's paragraph breaks survive.
function _proseHTML(s) {
  return '<div class="answer-prose">' + esc(s) + '</div>';
}
function cleanVideoLabel(s) {
  // Remove leading hex IDs (e.g. "20c622da ")
  let clean = String(s || '').replace(/^[0-9a-f]{8,}\s+/i, '');
  // Remove "YTDown com" / "YTDown.com" watermark (case-insensitive)
  clean = clean.replace(/YTDown[\s.]?com\s*/gi, '');
  // Remove a media file extension — only known ones, because titles now come
  // from YouTube and a blind /\.[^.]+$/ ate real endings ("… | 6.S191" → "… | 6")
  clean = clean.replace(/\.(mp4|mkv|webm|mov|avi|m4v|mpe?g|mp3|m4a|wav)$/i, '');
  return clean.trim() || s;
}
function fmt(sec) {
  const s = Math.max(0, Math.round(sec));
  const m = Math.floor(s / 60), h = Math.floor(m / 60);
  if (h > 0) return `${h}:${String(m%60).padStart(2,'0')}:${String(s%60).padStart(2,'0')}`;
  return `${String(m%60).padStart(2,'0')}:${String(s%60).padStart(2,'0')}`;
}
// ========== CHUNK PANEL RENDERER ==========
function _detectTimestampQuestion(text) {
  return /(\d{1,2}:\d{2})|(\bmin(ute)?s?\b.*\d)|(\d+\s*min(ute)?s?\b)/i.test(text);
}

function _extractAskedSeconds(text) {
  // MM:SS
  let m = text.match(/(?:at|around|in|@|min(?:ute)?s?\s+)(\d{1,2}):(\d{2})/i);
  if (m) return parseInt(m[1]) * 60 + parseInt(m[2]);
  // "min 1" or "minute 1" or "1 min"
  m = text.match(/(?:min(?:ute)?s?\s+(\d{1,2}))|(\d{1,2})\s*min(?:ute)?s?\b/i);
  if (m) return parseInt(m[1] || m[2]) * 60;
  return null;
}

function _closestChunk(chunks, targetSeconds) {
  if (targetSeconds === null) return null;
  let best = null, bestDist = Infinity;
  for (const c of chunks) {
    const ts = c.start_time != null ? c.start_time : (c.start != null ? c.start : null);
    if (ts === null) continue;
    const dist = Math.abs(ts - targetSeconds);
    if (dist < bestDist) { bestDist = dist; best = c; }
  }
  return best;
}

/**
 * Filter and re-rank result chunks using LLM citations.
 *
 * The vector search returns N chunks that are semantically *similar* to the
 * question, but that doesn't mean they're all relevant to the *answer* the
 * LLM generated. The LLM emits a `citations` list in its `done` event that
 * contains the actual chunks it drew from.
 *
 * Strategy:
 *   1. If citations exist, keep only chunks whose start_time falls within
 *      2 s of a cited timestamp.  Cited chunks appear first (in video order),
 *      followed by any remaining non-cited chunks.
 *   2. If no citations (LLM fallback / extractive mode), return all chunks
 *      unchanged so the UI still has something to show.
 */
function _filterChunksByCitations(chunks, citations) {
  if (!citations || citations.length === 0) return chunks;

  const citedTimes = citations
    .map(c => c.start_time != null ? c.start_time : (c.start != null ? c.start : null))
    .filter(t => t !== null);

  if (citedTimes.length === 0) return chunks;

  const TOLERANCE = 2.0; // seconds

  const cited   = [];
  const uncited = [];

  for (const chunk of chunks) {
    const ts = chunk.start_time != null ? chunk.start_time : (chunk.start != null ? chunk.start : null);
    if (ts === null) { uncited.push(chunk); continue; }
    const isCited = citedTimes.some(ct => Math.abs(ts - ct) <= TOLERANCE);
    if (isCited) cited.push(chunk); else uncited.push(chunk);
  }

  // Sort cited chunks by video order
  cited.sort((a, b) => (a.start_time ?? a.start ?? 0) - (b.start_time ?? b.start ?? 0));

  // Return cited first; include up to 3 uncited so the panel isn't empty
  // when the LLM synthesized from paraphrased context (start_times may drift)
  return cited.length > 0 ? [...cited, ...uncited.slice(0, 3)] : chunks;
}

/**
 * Replace bare MM:SS / H:MM:SS timestamps in prose with clickable spans
 * that call seekVideoTo(seconds).
 */
function _linkifyTimestamps(htmlText) {
  // Match timestamps like 02:39 or 1:02:39 that are NOT already inside an HTML tag
  return htmlText.replace(/(?<![">])\b(\d{1,2}):(\d{2})(?::(\d{2}))?\b(?![^<]*>)/g, (match, p1, p2, p3) => {
    let sec = parseInt(p1) * 60 + parseInt(p2);
    if (p3 !== undefined) sec = parseInt(p1) * 3600 + parseInt(p2) * 60 + parseInt(p3);
    return `<span class="ts-link" onclick="seekVideoTo(${sec})" title="Jump to ${match}" style="color:#a5b4fc;cursor:pointer;text-decoration:underline;text-decoration-style:dotted;font-weight:600">${match}</span>`;
  });
}

/**
 * Strip raw JSON/dict/metadata leakage from visual frame text.
 * Handles patterns like:
 *   {'text': 'Hello', 'authority': 'Joanne', 'time': '...'}
 *   {frame: 0 text: My cousin ... authority: 0.0 language: en confidence: 0.83}
 */
function cleanVisualText(raw) {
  if (!raw) return '';
  const s = raw.trim();
  const hasTextKey = s.includes("'text'") || s.includes('"text"');
  if (s.startsWith('{') || s.startsWith('[') || hasTextKey) {
    // Try quoted: 'text': 'Hello world'
    const m = s.match(/['"](text)['"]\s*:\s*['"]([^'"]{4,})['"]/);
    if (m) return m[2].trim();
    // Try unquoted: text: Hello world authority: 0.0
    const m2 = s.match(/text\s*:\s*(.+?)(?:\s+\w+\s*[:{]|$)/i);
    if (m2) {
      const cand = m2[1].trim().replace(/^['"]|['"]$/g, '');
      if (cand.length > 4) return cand;
    }
    return '';
  }
  // Strip trailing {frame: ...} noise
  let out = s.replace(/\s*\{[^}]{0,200}\}\s*$/, '').trim();
  // Strip trailing metadata key:value pairs
  out = out.replace(/\s+(authority|confidence|language|score|confidence_level)\s*:\s*\S+.*$/i, '').trim();
  return out;
}

function renderChunkPanel(answerEl, answerText, chunks, videoId, question) {
  if (!answerEl) return;
  const audioChunks   = chunks.filter(c => (c.chunk_type || 'speech') !== 'visual');
  const visualChunks  = chunks.filter(c => (c.chunk_type || 'speech') === 'visual');

  audioChunks.sort((a, b) => (a.start_time ?? a.start ?? 0) - (b.start_time ?? b.start ?? 0));
  visualChunks.sort((a, b) => (a.start_time ?? a.start ?? 0) - (b.start_time ?? b.start ?? 0));

  let chunksHTML = '';
  if (audioChunks.length || visualChunks.length) {

    // Best display text for a chunk — covers all field variants including
    // visual-specific fields flattened by the backend into display_text.
    const getChunkText = (c, isVisual) => {
      if (isVisual) {
        // Each candidate is passed through cleanVisualText to strip raw JSON/metadata leaks
        const candidates = [
          c.display_text, c.short_caption, c.caption,
          c.ocr_text, c.description, c.label, c.text, c.content
        ];
        for (const raw of candidates) {
          if (!raw) continue;
          const cleaned = cleanVisualText(String(raw));
          if (cleaned.length > 3) return cleaned;
        }
        return '';
      }
      return (c.text || c.content || c.caption || '').trim();
    };

    const makeItem = (c, icon, color, isVisual = false, highlight = false) => {
      const ts    = c.start_time != null ? c.start_time : (c.start != null ? c.start : null);
      const tsStr = ts != null ? fmt(ts) : '--:--';
      const rawText = getChunkText(c, isVisual);
      const text  = esc(rawText.slice(0, 140));
      const clickAttr = ts != null
        ? `onclick="seekVideoTo(${ts})" title="Jump to ${tsStr}" style="cursor:pointer"`
        : `title="${esc(c.chunk_type || 'chunk')}"`;
      const frameImg = isVisual && c.chunk_id
        ? `<img src="/vision/frame/${esc(c.chunk_id)}" alt="frame at ${tsStr}" loading="lazy"
             style="width:120px;height:68px;object-fit:cover;border-radius:6px;flex-shrink:0;
                    border:1px solid ${highlight ? 'rgba(134,239,172,0.7)' : 'rgba(134,239,172,0.30)'};display:block"
             onerror="this.style.display='none'">`
        : '';
      const highlightStyle = highlight ? 'border:1.5px solid rgba(134,239,172,0.6);background:rgba(134,239,172,0.07);' : '';
      if (isVisual) {
        return `<div class="chunk-item" ${clickAttr} style="flex-direction:row;align-items:center;gap:8px;padding:6px 8px;${highlightStyle}">
          ${frameImg}
          <div style="display:flex;flex-direction:column;gap:3px;min-width:0">
            <span class="chunk-ts" style="color:${color};font-size:10px;font-weight:600;white-space:nowrap">
              <i class="${icon}" style="font-size:9px;margin-right:3px"></i>${tsStr}
              ${highlight ? '<span style="font-size:9px;color:rgba(134,239,172,0.9);margin-left:4px;font-weight:700">SOURCE</span>' : ''}
            </span>
            ${text ? `<span class="chunk-text" style="font-size:10.5px;color:var(--text-secondary);line-height:1.35;overflow:hidden;display:-webkit-box;-webkit-line-clamp:3;-webkit-box-orient:vertical">${text}${rawText.length > 140 ? '…' : ''}</span>` : ''}
          </div>
        </div>`;
      }
      const highlightAudioStyle = highlight ? 'border:1.5px solid rgba(165,180,252,0.6);background:rgba(165,180,252,0.08);' : '';
      return `<div class="chunk-item" ${clickAttr} style="${highlightAudioStyle}">
        <div style="display:flex;flex-direction:column;gap:2px;min-width:0">
          <span class="chunk-ts" style="color:${color};font-size:11px;font-weight:600">
            <i class="${icon}" style="font-size:9px;margin-right:3px"></i>${tsStr}
            ${highlight ? '<span style="font-size:9px;color:rgba(165,180,252,0.95);margin-left:4px;font-weight:700">CITED</span>' : ''}
          </span>
          <span class="chunk-text" style="font-size:12px;color:var(--text-secondary);line-height:1.5">${text}${rawText.length > 140 ? '…' : ''}</span>
        </div>
      </div>`;
    };

    // Find the audio chunk that matches a timestamp mentioned in the answer text,
    // so it gets pinned to the top with a CITED badge.
    const answerTimestamps = [...answerText.matchAll(/\b(\d{1,2}):(\d{2})(?::(\d{2}))?\b/g)].map(m => {
      let s = parseInt(m[1]) * 60 + parseInt(m[2]);
      if (m[3] !== undefined) s = parseInt(m[1]) * 3600 + parseInt(m[2]) * 60 + parseInt(m[3]);
      return s;
    });
    let citedAudioChunk = null;
    if (answerTimestamps.length && audioChunks.length) {
      let bestDist = Infinity;
      for (const c of audioChunks) {
        const ts = c.start_time != null ? c.start_time : (c.start != null ? c.start : null);
        if (ts === null) continue;
        for (const at of answerTimestamps) {
          const dist = Math.abs(ts - at);
          if (dist < bestDist && dist <= 5) { bestDist = dist; citedAudioChunk = c; }
        }
      }
    }

    // Put cited chunk first, then the rest in time order
    const orderedAudio = citedAudioChunk
      ? [citedAudioChunk, ...audioChunks.filter(c => c !== citedAudioChunk)]
      : audioChunks;

    // Highest-scored visual frame gets SOURCE badge — it's what the answer was sourced from
    const topScore = (c) => c.rerank_score ?? c.score ?? 0;
    const sourceFrame = visualChunks.length
      ? visualChunks.reduce((best, c) => topScore(c) > topScore(best) ? c : best, visualChunks[0])
      : null;

    let sections = '';
    if (visualChunks.length) {
      sections += `<div class="chunk-group-label" style="font-size:10px;font-weight:700;color:var(--text-muted);text-transform:uppercase;letter-spacing:.06em;margin:8px 0 4px">
        <i class="fa-solid fa-image" style="margin-right:4px"></i>Visual frames
      </div>`;
      sections += visualChunks.slice(0, 6).map(c =>
        makeItem(c, 'fa-solid fa-eye', '#86efac', true, c === sourceFrame)
      ).join('');
    }
    if (audioChunks.length) {
      sections += audioChunks.length ? `<div class="chunk-group-label" style="font-size:10px;font-weight:700;color:var(--text-muted);text-transform:uppercase;letter-spacing:.06em;margin:${visualChunks.length ? '12px' : '8px'} 0 4px">
        <i class="fa-solid fa-waveform-lines" style="margin-right:4px"></i>Audio segments
      </div>` : '';
      sections += orderedAudio.slice(0, 8).map(c => makeItem(c, 'fa-solid fa-microphone', '#a5b4fc', false, c === citedAudioChunk)).join('');
    }

    chunksHTML = `<div id="ask-chunks-panel" style="margin-top:14px;border-top:1px solid rgba(99,102,241,0.18);padding-top:10px">
      <div style="font-size:11px;font-weight:600;color:var(--text-muted);margin-bottom:6px;display:flex;align-items:center;gap:6px">
        <i class="fa-solid fa-film"></i> Referenced segments — click to jump
      </div>
      <div style="display:flex;flex-direction:column;gap:5px">
        ${sections}
      </div>
    </div>`;
  }

  const noVisualHint = (audioChunks.length > 0 && visualChunks.length === 0)
    ? `<div style="margin-top:10px;font-size:11px;color:var(--text-muted);display:flex;align-items:center;gap:5px;opacity:.7">
        <i class="fa-solid fa-circle-info" style="font-size:10px"></i>
        No visual frames found. If this video contains on-screen text or articles, try <strong>Re-index</strong> to enable frame analysis.
      </div>`
    : '';

  // ── PINNED TRANSCRIPT BOX (timestamp questions only) ──────────────
  // When the user asked "what did they say at X:XX", inject a pinned
  // box showing the raw transcript line closest to that timestamp,
  // BEFORE the LLM prose. This way users always see the ground truth.
  let pinnedBox = '';
  if (question && _detectTimestampQuestion(question) && audioChunks.length) {
    const askedSec = _extractAskedSeconds(question);
    const closest = _closestChunk(audioChunks, askedSec);
    if (closest) {
      const ts = closest.start_time != null ? closest.start_time : (closest.start != null ? closest.start : null);
      const tsStr = ts != null ? fmt(ts) : '--:--';
      const rawTxt = (closest.text || closest.content || closest.caption || '').trim();
      const clickAttr = ts != null
        ? `onclick="seekVideoTo(${ts})" title="Jump to ${tsStr}" style="cursor:pointer"`
        : '';
      if (rawTxt) {
        pinnedBox = `
          <div ${clickAttr} style="
            display:flex;align-items:flex-start;gap:10px;
            margin-bottom:12px;padding:10px 14px;
            background:rgba(99,102,241,0.10);
            border:1.5px solid rgba(99,102,241,0.35);
            border-radius:10px;
          ">
            <i class="fa-solid fa-quote-left" style="color:#a5b4fc;font-size:14px;margin-top:2px;flex-shrink:0"></i>
            <div style="display:flex;flex-direction:column;gap:4px;min-width:0">
              <span style="font-size:10px;font-weight:700;color:#a5b4fc;text-transform:uppercase;letter-spacing:.06em">
                <i class="fa-solid fa-microphone" style="font-size:9px;margin-right:3px"></i>Transcript at
                <span style="font-family:var(--font-mono)">${tsStr}</span>
                <span style="font-size:9px;color:rgba(134,239,172,0.9);margin-left:6px;font-weight:700">GROUND TRUTH — click to jump</span>
              </span>
              <span style="font-size:13px;color:var(--text-primary);line-height:1.6">"${esc(rawTxt)}"</span>
            </div>
          </div>`;
      }
    }
  }

  // The prose gets its own wrapper: esc() leaves newlines as-is, so without a
  // pre-wrap container every paragraph break in the model's answer collapses and
  // the whole thing renders as one block. Scoped to the prose so the generated
  // markup around it (pinned box, chunk list) keeps normal whitespace handling.
  answerEl.innerHTML = pinnedBox
    + '<div class="answer-prose">' + _linkifyTimestamps(esc(answerText)) + '</div>'
    + chunksHTML + noVisualHint;

  if (!document.getElementById('chunk-item-style')) {
    const s = document.createElement('style');
    s.id = 'chunk-item-style';
    s.textContent = `.chunk-item{display:flex;align-items:flex-start;gap:8px;padding:5px 8px;border-radius:7px;border:1px solid rgba(99,102,241,0.12);background:rgba(99,102,241,0.05);transition:background .15s}.chunk-item:hover{background:rgba(99,102,241,0.15);border-color:rgba(99,102,241,0.3)}`;
    document.head.appendChild(s);
  }
}


// ========== YOUTUBE PLAYER ==========
// The source video files no longer exist — every demo video is on YouTube and
// /videos returns a youtube_id.
//
// Reliability note: we do NOT let `new YT.Player('yt-el', {videoId})` build the
// player, because that path tells the IFrame API to REPLACE our styled <iframe>
// with a fresh one it constructs — which drops our width:100% styling and, on an
// empty iframe with no src, often renders nothing. Instead we set the iframe's
// src to the /embed URL ourselves (so the video is visible immediately, with no
// dependency on the JS API loading) and then ATTACH a YT.Player to that existing
// iframe purely to enable seekTo() for timestamp clicks.
let ytPlayer = null;        // YT.Player attached to the existing iframe
let ytPlayerReady = false;  // true once the player's onReady has fired
let ytApiReady = false;     // true once the IFrame JS API is available
let ytCurrentId = null;     // id currently loaded in the iframe
let ytBlockedId = null;     // id YouTube refused to embed (poster shown instead)

// Error codes from the IFrame API that mean "this will never play in an embed":
//   101 / 150 — the uploader disabled playback on other sites
//   100       — video removed or private
//   2         — the id itself is rejected
const YT_UNEMBEDDABLE = [101, 150, 100, 2];

function ytEmbedSrc(id) {
  return 'https://www.youtube.com/embed/' + id +
         '?enablejsapi=1&rel=0&modestbranding=1&playsinline=1';
}

// Called by the youtube/iframe_api script once it has finished loading.
window.onYouTubeIframeAPIReady = function () {
  ytApiReady = true;
  attachYouTubePlayer();
};

// Resolve a YouTube id from a video record, or null if it isn't a YT video.
function ytExtractId(v) {
  if (v.youtube_id) return v.youtube_id;
  if (v.metadata && v.metadata.youtube_id) return v.metadata.youtube_id;
  if (v.video_id && /^[\w-]{11}$/.test(v.video_id)) return v.video_id;
  return null;
}

// Best available poster image for a video (card thumbnail or player fallback).
// /videos sends the key `thumbnail` (qa_service.py:813) while the pages were
// reading `thumbnail_url` — accept either, then derive the poster straight from
// the YouTube id, since the source files no longer exist and the extract-a-frame
// path has no local file_url to work with.
function videoThumbUrl(v) {
  if (!v) return null;
  if (v.thumbnail_url) return v.thumbnail_url;
  if (v.thumbnail)     return v.thumbnail;
  const yt = ytExtractId(v);
  return yt ? 'https://i.ytimg.com/vi/' + yt + '/mqdefault.jpg' : null;
}

// Attach a YT.Player to the existing #yt-el iframe (which already has a src).
// Self-healing: works no matter when the API arrives, and injects the API
// script if the page never included it.
function attachYouTubePlayer() {
  const ytEl = $('yt-el');
  if (ytPlayer || !ytEl || !ytEl.src) return;   // nothing to attach to yet
  if (typeof YT !== 'undefined' && YT.Player) {
    ytApiReady = true;
    ytPlayer = new YT.Player(ytEl, {
      events: {
        onReady: function () { ytPlayerReady = true; },
        // Some videos (university channels especially) have embedding turned
        // off. Without this the tab just shows YouTube's grey "unavailable"
        // box and every timestamp click silently dies.
        onError: function (e) {
          if (YT_UNEMBEDDABLE.indexOf(e && e.data) !== -1) showYouTubeFallback(ytCurrentId);
        },
      },
    });
    return;
  }
  // API not here yet — poll for it, and inject it if it's missing entirely.
  if (!window.__ytPoll) {
    window.__ytPoll = setInterval(function () {
      if (typeof YT !== 'undefined' && YT.Player) {
        clearInterval(window.__ytPoll);
        window.__ytPoll = null;
        attachYouTubePlayer();
      }
    }, 150);
  }
  if (!document.querySelector('script[src*="youtube.com/iframe_api"]')) {
    var s = document.createElement('script');
    s.src = 'https://www.youtube.com/iframe_api';
    document.head.appendChild(s);
  }
}

// YouTube refused to embed this video (see YT_UNEMBEDDABLE). Replace the dead
// grey "unavailable" box with the video's poster frame and a button out to
// YouTube, so the panel still reads as *this* video instead of an error.
function showYouTubeFallback(videoId) {
  const id = videoId || ytCurrentId;
  if (!id) return;
  ytBlockedId = id;

  const ytEl = $('yt-el');
  const thumbEl = $('video-thumb');
  const overlayEl = $('video-overlay');
  const player = $('video-player');
  if (ytEl) ytEl.style.display = 'none';
  if (overlayEl) overlayEl.style.display = 'none';

  // hqdefault is the largest size guaranteed to exist for every video.
  if (thumbEl) {
    thumbEl.src = 'https://i.ytimg.com/vi/' + id + '/hqdefault.jpg';
    thumbEl.style.display = 'block';
  }

  let fb = $('yt-fallback');
  if (!fb && player) {
    fb = document.createElement('div');
    fb.id = 'yt-fallback';
    fb.className = 'yt-fallback';
    player.appendChild(fb);
  }
  if (!fb) return;
  fb.innerHTML =
      '<a class="yt-watch-btn" id="yt-watch-btn" target="_blank" rel="noopener"'
    + ' href="https://www.youtube.com/watch?v=' + encodeURIComponent(id) + '">'
    +   '<i class="fa-brands fa-youtube"></i> Regarder sur YouTube'
    + '</a>'
    + '<span class="yt-fallback-note">La lecture externe est désactivée pour cette vidéo.</span>';
  fb.style.display = 'flex';
}

// Back to a normal embed (called when switching to a different video).
function clearYouTubeFallback() {
  ytBlockedId = null;
  const fb = $('yt-fallback');
  if (fb) fb.style.display = 'none';
  const thumbEl = $('video-thumb');
  if (thumbEl) thumbEl.style.display = 'none';
  const ytEl = $('yt-el');
  if (ytEl) ytEl.style.display = 'block';
}

// Mount (or switch) the YouTube player to a given video id.
function mountYouTube(videoId) {
  const ytEl = $('yt-el');
  if (!ytEl || videoId === ytCurrentId) return;
  clearYouTubeFallback();   // previous video may have been blocked
  ytCurrentId = videoId;

  if (ytPlayer && ytPlayerReady && ytPlayer.loadVideoById) {
    // Player is live — switch smoothly without reloading the iframe.
    ytPlayer.loadVideoById(videoId);
  } else {
    // Show the video immediately as a plain embed (no JS-API dependency),
    // then attach the player once the API is available.
    ytEl.src = ytEmbedSrc(videoId);
    attachYouTubePlayer();
  }
}

// Is the YouTube player active AND ready to accept seek commands?
function ytIsActive() {
  const ytEl = $('yt-el');
  return !!(ytEl && ytEl.style.display !== 'none' &&
            ytPlayer && ytPlayerReady && ytPlayer.seekTo);
}

// Seek video player to a given second
function seekVideoTo(sec) {
  // Blocked video: nothing to seek. Point the watch button at the requested
  // second instead, so the click still leads somewhere. Deliberately does NOT
  // open a tab on its own — a popup per citation click would be hostile (and
  // blocked anyway); the user clicks through when they want to.
  if (ytBlockedId) {
    const btn = $('yt-watch-btn');
    if (btn) {
      btn.href = 'https://www.youtube.com/watch?v=' + encodeURIComponent(ytBlockedId) +
                 '&t=' + Math.floor(sec) + 's';
      btn.innerHTML = '<i class="fa-brands fa-youtube"></i> Regarder sur YouTube à ' + fmt(sec);
    }
    const c = $('video-player-container');
    if (c && c.scrollIntoView) c.scrollIntoView({ behavior: 'smooth', block: 'center' });
    return;
  }
  if (ytIsActive()) {
    ytPlayer.seekTo(sec, true);
    ytPlayer.playVideo();
    const c = $('video-player-container');
    if (c && c.scrollIntoView) c.scrollIntoView({ behavior: 'smooth', block: 'center' });
    return;
  }
  // YouTube iframe is on screen but the JS API hasn't attached yet — reload the
  // embed at the requested time so the click still jumps the video.
  const ytEl = $('yt-el');
  if (ytEl && ytEl.style.display !== 'none' && ytCurrentId) {
    ytEl.src = ytEmbedSrc(ytCurrentId) + '&start=' + Math.floor(sec) + '&autoplay=1';
    const c = $('video-player-container');
    if (c && c.scrollIntoView) c.scrollIntoView({ behavior: 'smooth', block: 'center' });
    return;
  }
  const vid = document.querySelector('video');
  if (vid) { vid.currentTime = sec; vid.play(); }
}


// ========== VIDEO LIST ==========
async function loadVideoList() {
  const sel = $('video-select');
  if (!sel) return;
  try {
    setStatus('Loading videos…', 'info');
    State.videos = await API.listVideos();
    clearStatus();

    sel.innerHTML = '<option value="">— select a video —</option>';

    if (State.videos.length === 0) {
      // Store returned empty — could be first launch or store not ready yet.
      // Retry once after a short delay before giving up.
      await new Promise(r => setTimeout(r, 1200));
      State.videos = await API.listVideos();
    }

    State.videos.forEach(v => {
      const id = v.video_id || v.id;
      const label = cleanVideoLabel(v.title || v.filename || id);
      const chunks = v.total_chunks || 0;
      const ready = chunks > 0 ? ` (${chunks} chunks)` : ' (not indexed)';
      const opt = document.createElement('option');
      opt.value = id;
      opt.textContent = label + ready;
      sel.appendChild(opt);
    });

    if (State.videos.length === 0) {
      const opt = document.createElement('option');
      opt.disabled = true;
      opt.textContent = 'No videos yet — upload one above';
      sel.appendChild(opt);
    }

    // Update my-videos page rec grid if visible
    renderRecommendations();
    renderMyVideosList();

  } catch (e) {
    sel.innerHTML = '<option value="">— could not load videos —</option>';
    setStatus('Could not load videos: ' + e.message, 'error');
  }
}

// ========== SELECT VIDEO ==========
function selectVideoById(id) {
  const v = State.videos.find(v => (v.video_id || v.id) === id);
  if (v) selectVideo(v);
}

function selectVideo(v) {
  const id = v.video_id || v.id;
  State.currentVideo = v;
  setSelectedVideoId(id);

  // Clear chat thread when switching videos
  const thread = $('chat-thread');
  if (thread) thread.innerHTML = '';
  const clearChatBtnEl = $('chat-clear-btn');
  if (clearChatBtnEl) clearChatBtnEl.className = 'chat-clear-btn';

  // Update title
  const titleEl = $('video-title');
  if (titleEl) titleEl.textContent = cleanVideoLabel(v.title || v.filename || id);

  // Enable delete button
  const delBtn = $('btn-delete-video');
  if (delBtn) delBtn.style.display = '';

  // Enable re-index button
  const ingestBtn = $('btn-ingest');
  if (ingestBtn) ingestBtn.disabled = false;

  // Load video file if available
  loadVideoPlayer(v);

  // Load transcript / chunks
  loadTranscript(id, v);

  // Drop the video you just picked out of "More in your library" — the filter
  // lives in renderRecommendations(), but it only reads State.currentVideo, so
  // it has to run again on every switch and not just when the list is loaded.
  renderRecommendations();

  // Videos with a near-empty graph (≤2 edges) get no Graph tab at all
  updateGraphTabVisibility(v);

  // Load knowledge-graph viz (fetched + cached, drawn once the Graph tab is visible)
  loadGraphViz(id);

  // Stop any existing polling
  stopPolling();

  // Start polling if not ready
  const chunks = v.total_chunks || 0;
  if (chunks === 0) {
    setStatus('Video not yet indexed. Click Re-index to process it.', 'info');
  } else {
    setStatus(`Ready — ${chunks} indexed chunks`, 'success');
    setTimeout(clearStatus, 2500);
  }
}

// ========== VIDEO PLAYER ==========
function loadVideoPlayer(v) {
  const videoEl    = $('video-el');
  const thumbEl    = $('video-thumb');
  const overlayEl  = $('video-overlay');
  const container  = $('video-player-container');
  const ytEl       = $('yt-el');
  if (!videoEl) return;

  // Always stop any active stream first to avoid ghost 206 requests
  videoEl.pause();
  videoEl.removeAttribute('src');
  videoEl.load();

  // YouTube branch — the demo videos live on YouTube, not as local files.
  const ytId = ytExtractId(v);
  if (ytId) {
    videoEl.style.display = 'none';
    if (thumbEl)   thumbEl.style.display = 'none';
    if (overlayEl) overlayEl.style.display = 'none';
    if (ytEl)      ytEl.style.display = 'block';
    if (container) container.classList.add('has-video');
    mountYouTube(ytId);
    return;
  }
  // Not a YouTube video — hide the iframe and fall through to the existing logic.
  if (ytEl) ytEl.style.display = 'none';
  if (ytPlayer && ytPlayer.pauseVideo) ytPlayer.pauseVideo();

  if (v.has_file && v.file_url) {
    videoEl.src = v.file_url;
    videoEl.style.display = 'block';
    if (thumbEl)    thumbEl.style.display = 'none';
    if (overlayEl)  overlayEl.style.display = 'none';
    if (container)  container.classList.add('has-video');
  } else {
    videoEl.style.display = 'none';
    // Show thumbnail if available, otherwise hide the broken unsplash placeholder
    if (thumbEl) {
      const poster = videoThumbUrl(v);
      if (poster) {
        thumbEl.src = poster;
        thumbEl.style.display = 'block';
      } else {
        // No file and no thumbnail — show a plain placeholder instead of broken image
        thumbEl.style.display = 'none';
      }
    }
    if (overlayEl)  overlayEl.style.display = 'none';
    if (container)  container.classList.remove('has-video');
  }
}

// ========== TRANSCRIPT ==========
async function loadTranscript(videoId, v) {
  const transcriptPanel = $('transcript');
  const timelinePanel   = $('timeline');
  const chaptersPanel   = $('chapters');
  if (!transcriptPanel) return;

  const chunks = v.total_chunks || 0;
  if (chunks === 0) {
    transcriptPanel.innerHTML = '<div style="text-align:center;color:var(--text-muted);padding:24px;font-size:13px">Video not indexed yet. Click Re-index to process.</div>';
    return;
  }

  transcriptPanel.innerHTML = '<div style="text-align:center;color:var(--text-muted);padding:24px;font-size:13px"><i class="fa-solid fa-spinner fa-spin" style="margin-right:6px"></i>Loading…</div>';

  try {
    const data = await API.get(`/video/${videoId}`);
    const summary  = data.summary || '';
    const keywords = data.keywords || [];
    const topics   = data.topics  || [];

    // ── Transcript tab → show the video summary from metadata ──────────
    if (summary) {
      transcriptPanel.innerHTML = `
        <div style="padding:16px 4px;line-height:1.8;font-size:14px;color:var(--text-primary)">
          ${esc(summary)}
        </div>
        ${keywords.length ? `
        <div style="margin-top:16px;padding-top:14px;border-top:1px solid var(--border-color)">
          <div style="font-size:11px;font-weight:600;color:var(--text-muted);text-transform:uppercase;letter-spacing:.06em;margin-bottom:8px">Keywords</div>
          <div style="display:flex;flex-wrap:wrap;gap:6px">
            ${keywords.map(k => `<span style="background:rgba(99,102,241,0.12);color:#a5b4fc;border:1px solid rgba(99,102,241,0.25);border-radius:20px;padding:3px 10px;font-size:12px">${esc(k)}</span>`).join('')}
          </div>
        </div>` : ''}`;
    } else {
      transcriptPanel.innerHTML = '<div style="text-align:center;color:var(--text-muted);padding:24px;font-size:13px">No summary available for this video.</div>';
    }

    // ── Timeline tab → timestamps from chunks ──────────────────────────
    if (timelinePanel) {
      const chunkData = await API.post('/search', { video_id: videoId, query: 'video content', top_k: 50 }).catch(() => null);
      const chunks_list = chunkData ? (chunkData.results || chunkData.chunks || []) : [];
      const speechChunks = chunks_list.filter(c => (c.chunk_type || 'speech') !== 'visual');

      if (speechChunks.length) {
        speechChunks.sort((a, b) => (a.start_time ?? a.start ?? 0) - (b.start_time ?? b.start ?? 0));
        timelinePanel.innerHTML = speechChunks.slice(0, 500).map(c => {
          const ts = c.start_time ?? c.start ?? 0;
          const text = c.text || c.content || '';
          return `
          <div class="transcript-item" data-ts="${ts}" role="button" tabindex="0" title="Jump to ${fmt(ts)}">
            <span class="transcript-time">${fmt(ts)}</span>
            <p class="transcript-text">${esc(text.slice(0, 120))}${text.length > 120 ? '…' : ''}</p>
          </div>`;
        }).join('');

        // The whole row is the target, and it goes through seekVideoTo() — the
        // old handler drove #video-el directly, which no longer has a src now
        // that every video is a YouTube embed, so nothing happened on click.
        timelinePanel.querySelectorAll('.transcript-item[data-ts]').forEach(el => {
          const jump = () => seekVideoTo(parseFloat(el.dataset.ts));
          el.addEventListener('click', jump);
          el.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); jump(); }
          });
        });
      } else {
        timelinePanel.innerHTML = '<div style="text-align:center;color:var(--text-muted);padding:24px;font-size:13px">No timeline data available.</div>';
      }
    }

    // ── Chapters tab → rich chapter cards from /video/{id}/chapters ───
    if (chaptersPanel) {
      // Use session cache to avoid re-fetching chapters every time
      let chapters = State.chaptersCache[videoId] || null;
      if (chapters === null) {
        chaptersPanel.innerHTML = '<div style="text-align:center;color:var(--text-muted);padding:24px;font-size:13px"><i class="fa-solid fa-spinner fa-spin" style="margin-right:6px"></i>Loading chapters…</div>';
      }
      try {
        if (chapters === null) {
          const chapData = await API.get(`/video/${videoId}/chapters`);
          chapters = chapData.chapters || [];
          State.chaptersCache[videoId] = chapters; // store in session cache
        }
        if (chapters.length) {
          chaptersPanel.innerHTML = chapters.map((ch, i) => {
            const ts = fmt(ch.start_time || 0);
            const dur = ch.end_time && ch.start_time != null
              ? fmt(ch.end_time - ch.start_time)
              : null;
            const thumb = ch.thumbnail_url
              ? `<img src="${esc(ch.thumbnail_url)}" alt="${esc(ch.title)}"
                     style="width:100%;height:100%;object-fit:cover;border-radius:6px;"
                     onerror="this.parentElement.innerHTML='<i class=\\'fa-solid fa-film\\' style=\\'color:#6366f1;font-size:20px\\'></i>'">`
              : `<i class="fa-solid fa-film" style="color:#6366f1;font-size:20px;opacity:.5"></i>`;
            return `
              <div class="chapter-card" data-start="${ch.start_time || 0}"
                   style="display:flex;gap:12px;align-items:flex-start;padding:12px 4px;
                          border-bottom:1px solid var(--border-color);cursor:pointer;
                          transition:background .15s;border-radius:8px;"
                   onmouseenter="this.style.background='rgba(99,102,241,0.07)'"
                   onmouseleave="this.style.background=''"
                   onclick="(function(el){
                     var videoEl=document.getElementById('video-el');
                     if(videoEl&&videoEl.src){videoEl.currentTime=${ch.start_time||0};videoEl.play();}
                   })(this)">
                <div style="flex-shrink:0;width:96px;height:54px;border-radius:6px;
                            background:linear-gradient(135deg,#1e1b4b,#312e81);
                            display:flex;align-items:center;justify-content:center;
                            overflow:hidden;position:relative;">
                  ${thumb}
                  <span style="position:absolute;bottom:3px;right:4px;background:rgba(0,0,0,.75);
                               color:#fff;font-size:10px;font-weight:600;padding:1px 4px;border-radius:3px;">
                    ${ts}
                  </span>
                </div>
                <div style="flex:1;min-width:0;">
                  <div style="display:flex;align-items:center;gap:8px;margin-bottom:3px;">
                    <span style="font-size:10px;font-weight:700;color:#6366f1;text-transform:uppercase;
                                 letter-spacing:.06em;flex-shrink:0;">Ch ${ch.index}</span>
                    ${dur ? `<span style="font-size:10px;color:var(--text-muted)">${dur}</span>` : ''}
                  </div>
                  <p style="margin:0 0 4px;font-size:13px;font-weight:600;
                             line-height:1.3;color:var(--text-primary);
                             white-space:nowrap;overflow:hidden;text-overflow:ellipsis;"
                     title="${esc(ch.title)}">${esc(ch.title)}</p>
                  ${ch.summary ? `<p style="margin:0;font-size:11px;color:var(--text-muted);
                                            line-height:1.4;display:-webkit-box;
                                            -webkit-line-clamp:2;-webkit-box-orient:vertical;
                                            overflow:hidden">${esc(ch.summary)}</p>` : ''}
                </div>
              </div>`;
          }).join('');
        } else {
          chaptersPanel.innerHTML = '<div style="text-align:center;color:var(--text-muted);padding:24px;font-size:13px"><i class="fa-solid fa-book-open" style="display:block;font-size:28px;opacity:.3;margin-bottom:10px"></i>No chapters yet — index this video to generate them.</div>';
        }
      } catch(e) {
        chaptersPanel.innerHTML = `<div style="text-align:center;color:#fca5a5;padding:24px;font-size:13px">Could not load chapters: ${esc(e.message)}</div>`;
      }
    }

  } catch (e) {
    transcriptPanel.innerHTML = `<div style="text-align:center;color:#fca5a5;padding:24px;font-size:13px">Error: ${esc(e.message)}</div>`;
  }
}

// ========== GRAPH VIZ (per-video knowledge graph) ==========
State.graphCache = {};        // videoId → {nodes, edges} response (legacy renderer)
State.graphVideoId = null;    // video id the graph panel is currently showing
State.graphRenderedId = null; // video id already drawn — guards against redraws

const GRAPH_TYPE_COLORS = {
  PERSON: '#f472b6', ORG: '#60a5fa', ORGANIZATION: '#60a5fa', LOCATION: '#34d399',
  GPE: '#34d399', EVENT: '#fbbf24', PRODUCT: '#a78bfa', WORK_OF_ART: '#f87171',
  SPEAKER: '#22d3ee', SCENE: '#94a3b8', OTHER: '#9ca3af',
};
function graphColorFor(type) {
  return GRAPH_TYPE_COLORS[(type || 'OTHER').toUpperCase()] || GRAPH_TYPE_COLORS.OTHER;
}

// A graph with 1–2 edges renders as an empty-looking canvas, so the tab is
// hidden for those videos. `has_graph` comes from the demo_videos_ext view; if
// it is missing (older payload) we fail open and keep the tab visible.
function updateGraphTabVisibility(v) {
  const btn = document.querySelector('.tab[data-tab="graph"]');
  if (!btn) return;

  const edges = v && v.graph_edges;
  const hide = (v && v.has_graph === false) ||
               (typeof edges === 'number' && edges <= 2);

  if (!hide) {
    btn.style.display = '';
    return;
  }

  btn.style.display = 'none';
  // Was the hidden tab the open one? Fall back to the default (Transcript).
  if (btn.classList.contains('active')) {
    const fallback = document.querySelector('.tab[data-tab="transcript"]');
    if (fallback) fallback.click();
  }
}

// Record which video the Graph tab should show. The drawing itself is done by
// graph-render.js (window.renderVGIAGraph), which fetches /graph/video/{id}
// itself — so there is nothing to pre-fetch or cache here any more.
function loadGraphViz(videoId) {
  State.graphVideoId = videoId;
  State.graphRenderedId = null;   // new video selected → force a redraw
  renderGraphVizIfActive();
}

// Draw the current video's graph, but only once the Graph tab is actually
// visible (an SVG force layout drawn into a display:none panel measures 0×0)
// and only once per video — main.js:17 calls this on every switch to the tab.
function renderGraphVizIfActive() {
  const panel = $('graph');
  if (!panel || !panel.classList.contains('active')) return;

  const container = $('graph-viz-container');
  const currentVideoId = State.graphVideoId;
  if (!container || !currentVideoId) return;
  if (State.graphRenderedId === currentVideoId) return;   // already on screen

  if (window.renderVGIAGraph && currentVideoId) {
    window.renderVGIAGraph(currentVideoId, container);
    // Set only after the call returns, so a throw leaves the tab retryable.
    State.graphRenderedId = currentVideoId;
  }
}

function renderGraphViz(data) {
  const container = $('graph-viz-container');
  const legend = $('graph-viz-legend');
  const tooltip = $('graph-viz-tooltip');
  if (!container) return;

  const nodes = (data.nodes || []).map(n => ({ ...n }));
  const edges = (data.edges || []).map(e => ({ ...e }));

  if (!nodes.length) {
    container.innerHTML = '<div style="text-align:center;color:var(--text-muted);padding:36px 16px;font-size:13px"><i class="fa-solid fa-diagram-project" style="font-size:22px;opacity:0.4;display:block;margin-bottom:10px;"></i>No graph data for this video yet — try re-indexing.</div>';
    if (legend) legend.style.display = 'none';
    return;
  }
  if (typeof d3 === 'undefined') {
    container.innerHTML = '<div style="text-align:center;color:var(--text-muted);padding:36px 16px;font-size:13px">Graph library failed to load.</div>';
    return;
  }

  container.innerHTML = '';
  const width  = container.clientWidth  || 600;
  const height = container.clientHeight || 420;

  if (legend) {
    const usedTypes = [...new Set(nodes.map(n => (n.type || 'OTHER').toUpperCase()))];
    legend.style.display = 'flex';
    legend.innerHTML = usedTypes.map(t => `
      <span style="display:inline-flex;align-items:center;gap:5px;font-size:11px;color:var(--text-muted)">
        <span style="width:9px;height:9px;border-radius:50%;background:${graphColorFor(t)};display:inline-block"></span>${esc(t)}
      </span>`).join('');
  }

  const svg = d3.select(container).append('svg')
    .attr('width', width).attr('height', height)
    .attr('viewBox', [0, 0, width, height]);
  const g = svg.append('g');
  svg.call(d3.zoom().scaleExtent([0.3, 4]).on('zoom', (ev) => g.attr('transform', ev.transform)));

  const sim = d3.forceSimulation(nodes)
    .force('link', d3.forceLink(edges).id(d => d.id).distance(90).strength(0.5))
    .force('charge', d3.forceManyBody().strength(-220))
    .force('center', d3.forceCenter(width / 2, height / 2))
    .force('collide', d3.forceCollide().radius(d => 8 + Math.sqrt(d.n_mentions || 1) * 3));

  const link = g.append('g')
    .attr('stroke', '#6b7280').attr('stroke-opacity', 0.5)
    .selectAll('line').data(edges).join('line')
    .attr('stroke-width', d => Math.max(1, Math.min(4, d.weight || 1)));

  const node = g.append('g')
    .selectAll('circle').data(nodes).join('circle')
    .attr('r', d => 6 + Math.sqrt(d.n_mentions || 1) * 2.5)
    .attr('fill', d => graphColorFor(d.type))
    .attr('stroke', '#111827').attr('stroke-width', 1.2)
    .style('cursor', 'pointer')
    .call(d3.drag()
      .on('start', (ev, d) => { if (!ev.active) sim.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; })
      .on('drag', (ev, d) => { d.fx = ev.x; d.fy = ev.y; })
      .on('end', (ev, d) => { if (!ev.active) sim.alphaTarget(0); d.fx = null; d.fy = null; }))
    .on('mouseenter', (ev, d) => {
      if (!tooltip) return;
      tooltip.innerHTML = `<strong>${esc(d.name)}</strong><br><span style="opacity:.7">${esc(d.type)}</span>${d.description ? `<br>${esc(d.description.slice(0, 160))}` : ''}`;
      tooltip.style.display = 'block';
    })
    .on('mousemove', (ev) => {
      if (!tooltip) return;
      tooltip.style.left = (ev.clientX + 14) + 'px';
      tooltip.style.top  = (ev.clientY + 14) + 'px';
    })
    .on('mouseleave', () => { if (tooltip) tooltip.style.display = 'none'; });

  const label = g.append('g')
    .selectAll('text').data(nodes).join('text')
    .text(d => d.name)
    .attr('font-size', 10)
    .attr('fill', '#9ca3af')
    .attr('dx', 10).attr('dy', 4)
    .style('pointer-events', 'none');

  sim.on('tick', () => {
    link.attr('x1', d => d.source.x).attr('y1', d => d.source.y)
        .attr('x2', d => d.target.x).attr('y2', d => d.target.y);
    node.attr('cx', d => d.x).attr('cy', d => d.y);
    label.attr('x', d => d.x).attr('y', d => d.y);
  });
}

// ========== RECOMMENDATIONS ==========
function renderRecommendations() {
  const recSec  = $('recommendations');
  const recGrid = $('rec-grid');
  if (!recSec || !recGrid) return;

  const others = State.videos.filter(v => (v.video_id || v.id) !== (State.currentVideo?.video_id || State.currentVideo?.id));
  if (!others.length) { recSec.style.display = 'none'; return; }
  recSec.style.display = '';

  const cards = others.slice(0, 5).map(v => {
    const id     = v.video_id || v.id;
    const label  = cleanVideoLabel(v.title || v.filename || id);
    const chunks = v.total_chunks || 0;

    // Duration badge
    let durLabel = chunks > 0 ? chunks + ' chunks' : 'not indexed';
    if (v.duration_seconds != null) {
      const s = Math.round(v.duration_seconds);
      const m = Math.floor(s / 60), h = Math.floor(m / 60);
      durLabel = h > 0
        ? `${h}:${String(m % 60).padStart(2,'0')}:${String(s % 60).padStart(2,'0')}`
        : `${m}:${String(s % 60).padStart(2,'0')}`;
    }

    // Poster first — every YouTube-backed video resolves one, and .rec-thumb img
    // already carries the sizing + hover zoom in style.css:932.
    // Local files keep the placeholder below: no <video src> is set in HTML, the
    // src lives in data-src and is only assigned on first hover, to avoid
    // Chrome/Firefox probing the stream endpoint on page load.
    const poster = videoThumbUrl(v);
    const thumbContent = poster
      ? `<img src="${esc(poster)}" alt="${esc(label)}" loading="lazy"
             onerror="this.style.display='none'"/>`
      : v.has_file && v.file_url
      ? `<div class="rec-thumb-placeholder" data-video-src="${esc(v.file_url)}"
              style="width:100%;height:100%;background:rgba(99,102,241,0.12);display:flex;align-items:center;justify-content:center;position:relative;overflow:hidden;">
           <i class="fa-solid fa-film" style="color:#6366f1;font-size:24px;opacity:.6;pointer-events:none"></i>
         </div>`
      : `<div style="width:100%;height:100%;background:rgba(99,102,241,0.15);display:flex;align-items:center;justify-content:center;">
           <i class="fa-solid fa-film" style="color:#6366f1;font-size:24px"></i>
         </div>`;

    return `
      <div class="rec-card" style="cursor:pointer" onclick="selectVideoAndSwitch('${esc(id)}')">
        <div class="rec-thumb">
          ${thumbContent}
          <span class="rec-duration">${durLabel}</span>
        </div>
        <div class="rec-info">
          <h4>${esc(label)}</h4>
          ${v.summary ? `<p style="-webkit-line-clamp:2;display:-webkit-box;-webkit-box-orient:vertical;overflow:hidden;">${esc(v.summary.slice(0, 80))}…</p>` : ''}
        </div>
      </div>`;
  }).join('');

  // Stop any existing rec-thumbnail streams before replacing the DOM
  recGrid.querySelectorAll('video.rec-thumb-video').forEach(vid => {
    vid.pause();
    vid.removeAttribute('src');
    vid.load();
  });

  recGrid.innerHTML = cards;

  // On hover: replace the placeholder div with a real <video> element, only then
  // assign src — this is the only way to guarantee no probing requests on load.
  recGrid.querySelectorAll('.rec-thumb-placeholder').forEach(placeholder => {
    const videoSrc = placeholder.dataset.videoSrc;
    if (!videoSrc) return;
    let loaded = false;
    placeholder.addEventListener('mouseenter', () => {
      if (loaded) return;
      loaded = true;
      const vid = document.createElement('video');
      vid.src = videoSrc;
      vid.preload = 'metadata';
      vid.muted = true;
      vid.playsInline = true;
      vid.style.cssText = 'position:absolute;inset:0;width:100%;height:100%;object-fit:cover;';
      vid.addEventListener('loadedmetadata', () => { vid.currentTime = 0.5; }, { once: true });
      placeholder.appendChild(vid);
    });
  });
}

function selectVideoAndSwitch(id) {
  const sel = $('video-select');
  if (sel) sel.value = id;
  selectVideoById(id);
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

// ========== MY-VIDEOS PAGE LIST ==========

// Cache des thumbnails déjà chargés { video_id: url_or_null }
const _thumbCache = {};

async function _loadThumbnailUrl(videoId) {
  if (videoId in _thumbCache) return _thumbCache[videoId];
  try {
    const data = await API.get(`/vision/frames/${videoId}?limit=1`);
    const frames = data.frames || [];
    if (frames.length) {
      const chunkId = frames[0].chunk_id || frames[0].id;
      const url = `/vision/frame/${chunkId}`;
      _thumbCache[videoId] = url;
      return url;
    }
  } catch (_) {}
  _thumbCache[videoId] = null;
  return null;
}

function _videoCardHTML(v, thumbUrl) {
  const id     = v.video_id || v.id;
  const label  = cleanVideoLabel(v.title || v.filename || id);
  const chunks = v.total_chunks || 0;
  const dur    = v.duration_seconds ? fmt(v.duration_seconds) : (chunks > 0 ? chunks + ' chunks' : '—');

  const thumbHTML = thumbUrl
    ? `<img src="${thumbUrl}" alt="${esc(label)}" loading="lazy"
         style="width:100%;height:100%;object-fit:cover;border-radius:inherit;"
         onerror="this.parentElement.innerHTML='<i class=\\"fa-solid fa-film\\" style=\\"color:#6366f1;font-size:32px\\"></i>';this.parentElement.style.display='flex';this.parentElement.style.alignItems='center';this.parentElement.style.justifyContent='center';">`
    : `<div style="width:100%;height:100%;display:flex;align-items:center;justify-content:center;">
         <i class="fa-solid fa-film" style="color:#6366f1;font-size:32px;opacity:.5"></i>
       </div>`;

  const statusBadge = chunks > 0
    ? `<span style="color:#22c55e;font-size:12px;font-weight:600"><i class="fa-solid fa-circle-check" style="margin-right:4px"></i>Indexed</span>`
    : `<span style="color:#f59e0b;font-size:12px;font-weight:600"><i class="fa-solid fa-clock" style="margin-right:4px"></i>Not indexed</span>`;

  const summary = v.summary ? `<p style="font-size:11px;color:var(--text-muted);margin:4px 0 0;line-height:1.4;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden">${esc(v.summary)}</p>` : '';

  return `
    <div class="video-card" id="vcard-${esc(id)}">
      <div class="video-card-thumb" style="position:relative;overflow:hidden;border-radius:12px 12px 0 0;aspect-ratio:16/9;background:#111;">
        ${thumbHTML}
        <span class="video-duration" style="position:absolute;bottom:6px;right:8px;background:rgba(0,0,0,.75);color:#fff;font-size:11px;font-weight:600;padding:2px 6px;border-radius:4px;">${dur}</span>
        <div class="video-overlay-card" style="position:absolute;inset:0;display:flex;align-items:center;justify-content:center;background:rgba(0,0,0,.4);opacity:0;transition:opacity .2s;border-radius:12px 12px 0 0;"
             onmouseenter="this.style.opacity=1" onmouseleave="this.style.opacity=0">
          <button class="play-btn-sm" onclick="window.location.href='/app/index.html?v=${esc(id)}'" style="width:48px;height:48px;border-radius:50%;background:#6366f1;border:none;cursor:pointer;display:flex;align-items:center;justify-content:center;">
            <i class="fa-solid fa-play" style="color:#fff;font-size:18px;margin-left:3px"></i>
          </button>
        </div>
      </div>
      <div class="video-card-body">
        <h3 style="font-size:14px;font-weight:600;line-height:1.4;margin:0 0 4px;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden">${esc(label)}</h3>
        ${summary}
        <p class="video-meta" style="margin:6px 0 8px;display:flex;align-items:center;gap:8px;">${statusBadge}</p>
        <div class="video-card-actions">
          <a href="/app/index.html?v=${esc(id)}" class="btn-ask">
            <i class="fa-solid fa-wand-magic-sparkles"></i> Ask AI
          </a>
          <button class="btn-icon" onclick="confirmDelete('${esc(id)}','${esc(label).replace(/'/g,'\\\'')}')" title="Delete">
            <i class="fa-solid fa-trash"></i>
          </button>
        </div>
      </div>
    </div>`;
}

async function renderMyVideosList() {
  const grid = document.getElementById('videos-grid');
  if (!grid) return;
  // If my-videos.html's own inline script is managing the grid (has loading/empty/error
  // state elements), do NOT render here — the inline script owns the lifecycle.
  if (document.getElementById('videos-loading')) return;

  if (!State.videos.length) {
    grid.innerHTML = '<div style="text-align:center;color:var(--text-muted);padding:40px;font-size:13px"><i class="fa-solid fa-film" style="font-size:32px;opacity:.2;display:block;margin-bottom:12px"></i>No videos yet. Upload one to get started.</div>';
    return;
  }

  // Rendu initial immédiat avec placeholder (sans attendre les thumbnails)
  grid.innerHTML = State.videos.map(v => _videoCardHTML(v, null)).join('');

  // Chargement asynchrone des thumbnails une par une → mise à jour DOM instantanée
  State.videos.forEach(async v => {
    const id = v.video_id || v.id;
    const card = document.getElementById(`vcard-${id}`);
    if (!card) return;
    const url = await _loadThumbnailUrl(id);
    if (!url) return;
    // Remplacer seulement le contenu du thumb
    const thumbDiv = card.querySelector('.video-card-thumb');
    if (!thumbDiv) return;
    const img = document.createElement('img');
    img.src = url;
    img.alt = cleanVideoLabel(v.title || v.filename || id);
    img.loading = 'lazy';
    img.style.cssText = 'width:100%;height:100%;object-fit:cover;border-radius:inherit;position:absolute;inset:0;';
    img.onerror = () => img.remove();
    // Insérer l'image en premier (derrière l'overlay)
    thumbDiv.style.position = 'relative';
    thumbDiv.insertBefore(img, thumbDiv.firstChild);
    // Enlever le placeholder icône s'il existe
    const icon = thumbDiv.querySelector('div > i.fa-film');
    if (icon && icon.parentElement !== thumbDiv) icon.parentElement.remove();
  });
}

async function confirmDelete(id, label) {
  if (!confirm(`Delete "${label}"?`)) return;
  try {
    await API.deleteVideo(id);
    State.videos = State.videos.filter(v => (v.video_id || v.id) !== id);
    renderMyVideosList();
  } catch (e) {
    alert('Delete failed: ' + e.message);
  }
}

// ========== ASK / Q&A ==========
async function sendQuestion(question) {
  if (!State.currentVideo) {
    setStatus('Please select a video first.', 'error');
    return;
  }
  const videoId = State.currentVideo.video_id || State.currentVideo.id;
  const chunks  = State.currentVideo.total_chunks || 0;
  if (chunks === 0) {
    setStatus('Video is not indexed yet. Please click Re-index first.', 'error');
    return;
  }

  const thread    = $('chat-thread');
  const clearBtn  = $('chat-clear-btn');
  const loadingEl = $('ask-loading');
  const textarea  = $('ask-textarea');

  loadingEl.className = 'ask-loading visible';
  if (textarea) textarea.disabled = true;

  // Append user question bubble
  const questionBubble = document.createElement('div');
  questionBubble.className = 'chat-question';
  questionBubble.innerHTML = `<i class="fa-solid fa-circle-user"></i><span>${esc(question)}</span>`;
  thread.appendChild(questionBubble);

  // Append answer bubble (streaming into it)
  const answerEl = document.createElement('div');
  answerEl.className = 'ask-answer';
  answerEl.innerHTML = '<span style="opacity:.4">…</span>';
  thread.appendChild(answerEl);

  // Show clear button
  if (clearBtn) clearBtn.className = 'chat-clear-btn visible';

  // Scroll to the new answer
  thread.scrollTop = thread.scrollHeight;

  try {
    let fullText = '';

    try {
      let resultChunks = [];
      let llmCitations = [];   // citations from the LLM done event
      for await (const chunk of API.askStream(videoId, question)) {
        if (chunk._event === 'results') {
          resultChunks = chunk.results || [];
          console.log('[chunks] received', resultChunks.length, 'chunks from results event', resultChunks.slice(0,2));
          // Render immediately with all chunks; will be refined once citations arrive
          renderChunkPanel(answerEl, fullText, resultChunks, videoId, question);
          thread.scrollTop = thread.scrollHeight;
          continue;
        }
        if (chunk.token)  { fullText += chunk.token; answerEl.innerHTML = _proseHTML(fullText); }
        if (chunk.text)   { fullText = chunk.text;   answerEl.innerHTML = _proseHTML(fullText); }
        if (chunk.answer) { fullText = chunk.answer; }
        // Capture citations from the done event
        if (chunk.done && chunk.citations && chunk.citations.length) {
          llmCitations = chunk.citations;
        }
        if (chunk.sources || chunk.chunks) {
          const sources = chunk.sources || chunk.chunks || [];
          if (sources.length && !resultChunks.length) {
            renderChunkPanel(answerEl, fullText, sources, videoId, question);
          }
        }
        thread.scrollTop = thread.scrollHeight;
      }
      // Re-render with citation-filtered chunks now that the stream is done
      if (resultChunks.length) {
        const filteredChunks = _filterChunksByCitations(resultChunks, llmCitations);
        renderChunkPanel(answerEl, fullText, filteredChunks, videoId, question);
      }

      if (fullText.trim()) {
        await HistoryStore.save({
          videoId,
          videoTitle: cleanVideoLabel(State.currentVideo?.title || State.currentVideo?.filename || videoId),
          question,
          answer: fullText,
        });
      }
    } catch (_streamErr) {
      // Fallback to non-streaming
      const data = await API.ask(videoId, question);
      fullText = data.answer || data.response || JSON.stringify(data);
      const sources = data.chunks || data.sources || [];
      renderChunkPanel(answerEl, fullText, sources, videoId, question);
      if (fullText.trim()) {
        await HistoryStore.save({
          videoId,
          videoTitle: cleanVideoLabel(State.currentVideo?.title || State.currentVideo?.filename || videoId),
          question,
          answer: fullText,
        });
      }
    }

  } catch (e) {
    answerEl.innerHTML = `<span style="color:#fca5a5"><i class="fa-solid fa-triangle-exclamation"></i> Error: ${esc(e.message)}</span>`;
  } finally {
    loadingEl.className = 'ask-loading';
    if (textarea) { textarea.disabled = false; textarea.value = ''; textarea.focus(); }
    thread.scrollTop = thread.scrollHeight;
  }
}

// ========== UPLOAD ON DASHBOARD ==========
async function handleDashboardUpload(files) {
  if (!files.length) return;
  const file = files[0];
  const card = $('upload-status-card');
  const nameEl = $('upload-status-name');
  const fillEl = $('upload-fill');
  const labelEl = $('upload-status-label');

  if (card) { card.className = 'upload-status-card visible'; }
  if (nameEl) nameEl.textContent = file.name;
  if (fillEl) fillEl.style.width = '0%';
  if (labelEl) labelEl.textContent = 'Uploading…';

  try {
    const result = await API.uploadFile(file, null, (pct) => {
      const p = Math.round(pct * 90);
      if (fillEl) fillEl.style.width = p + '%';
      if (labelEl) labelEl.textContent = p + '% — uploading';
    });

    if (fillEl) fillEl.style.width = '95%';
    if (labelEl) labelEl.textContent = 'Processing…';

    const videoId = result.video_id;
    if (videoId) {
      // Start ingestion
      await API.ingest(videoId).catch(() => {});
      if (fillEl) fillEl.style.width = '100%';
      if (labelEl) labelEl.textContent = 'Indexing in background…';

      // Refresh video list
      await loadVideoList();

      // Auto-select the new video
      const sel = $('video-select');
      if (sel) { sel.value = videoId; selectVideoById(videoId); }

      setTimeout(() => { if (card) card.className = 'upload-status-card'; }, 4000);
    }
  } catch (e) {
    if (labelEl) labelEl.textContent = 'Error: ' + e.message;
    if (fillEl) { fillEl.style.width = '100%'; fillEl.style.background = '#ef4444'; }
    setTimeout(() => { if (card) card.className = 'upload-status-card'; }, 4000);
  }
}

// ========== POLLING (ingestion status) ==========
function startPolling(videoId) {
  stopPolling();
  State.pollingTimer = setInterval(async () => {
    try {
      const s = await API.ingestionStatus(videoId);
      const done = s.status === 'ready' || s.status === 'done' || (s.total_chunks && s.total_chunks > 0);
      if (done) {
        stopPolling();
        setStatus('Indexing complete!', 'success');
        await loadVideoList();
        selectVideoById(videoId);
        setTimeout(clearStatus, 3000);
      } else {
        const pct = s.progress_pct || s.progress || 0;
        setStatus(`Indexing… ${pct > 0 ? Math.round(pct) + '%' : 'working'}`, 'info');
      }
    } catch (_) {}
  }, 3000);
}

function stopPolling() {
  if (State.pollingTimer) { clearInterval(State.pollingTimer); State.pollingTimer = null; }
}

// ========== INIT ==========
document.addEventListener('DOMContentLoaded', async () => {

  // Load video list
  await loadVideoList();

  // Resolve video to select, in priority order:
  //  1. ?v= URL param (explicit deep link)
  //  2. sessionStorage 'selectedVideo' (one-shot: set by history.html / my-videos.html
  //     when the user clicks a specific video there)
  //  3. localStorage sticky selection (keeps the Ask panel in sync with whatever
  //     video was last selected, even after navigating to other pages and back)
  const urlParam = new URLSearchParams(window.location.search).get('v');
  const savedId  = sessionStorage.getItem('selectedVideo');
  const stickyId = getSelectedVideoId();
  const autoId   = urlParam || savedId || stickyId || null;
  const sel = $('video-select');

  if (autoId && sel) {
    // The one-shot session hint is only meant to fire once
    if (savedId) sessionStorage.removeItem('selectedVideo');

    // Try exact match first, then prefix match (handles minor ID variations)
    const matched = State.videos.find(v =>
      (v.video_id || v.id) === autoId ||
      (v.video_id || v.id).startsWith(autoId) ||
      autoId.startsWith(v.video_id || v.id)
    );
    if (matched) {
      const matchedId = matched.video_id || matched.id;
      sel.value = matchedId;
      selectVideoById(matchedId);
    }
    // Clean ?v= from the address bar without triggering a reload
    if (urlParam) {
      const clean = window.location.pathname + window.location.hash;
      history.replaceState(null, '', clean);
    }
  }

  // Video selector change
  if (sel) {
    sel.addEventListener('change', () => {
      const id = sel.value;
      if (id) selectVideoById(id);
    });
  }

  // Re-index button
  const ingestBtn = $('btn-ingest');
  if (ingestBtn) {
    ingestBtn.addEventListener('click', async () => {
      if (!State.currentVideo) return;
      const id = State.currentVideo.video_id || State.currentVideo.id;
      // Invalidate cached chapters so they rebuild after re-indexing
      delete State.chaptersCache[id];
      ingestBtn.disabled = true;
      setStatus('Sending re-index request…', 'info');
      try {
        await API.ingest(id);
        setStatus('Re-indexing started — please wait', 'info');
        startPolling(id);
      } catch (e) {
        setStatus('Ingest error: ' + e.message, 'error');
        ingestBtn.disabled = false;
      }
    });
  }

  // Delete button
  const delBtn = $('btn-delete-video');
  if (delBtn) {
    delBtn.addEventListener('click', async () => {
      if (!State.currentVideo) return;
      const id    = State.currentVideo.video_id || State.currentVideo.id;
      const label = cleanVideoLabel(State.currentVideo.title || State.currentVideo.filename || id);
      if (!confirm(`Delete "${label}"?`)) return;
      try {
        await API.deleteVideo(id);
        State.currentVideo = null;
        if (getSelectedVideoId() === id) clearSelectedVideoId();
        State.videos = State.videos.filter(v => (v.video_id || v.id) !== id);
        await loadVideoList();
        const titleEl = $('video-title');
        if (titleEl) titleEl.textContent = 'Select a video above';
        delBtn.style.display = 'none';
        setStatus('Video deleted.', 'success');
        setTimeout(clearStatus, 3000);
      } catch (e) {
        setStatus('Delete error: ' + e.message, 'error');
      }
    });
  }

  // Upload new video button (dashboard)
  const uploadBtn = $('btn-upload-new');
  const fileInput = $('file-input');
  if (uploadBtn && fileInput) {
    uploadBtn.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', (e) => {
      if (e.target.files.length) handleDashboardUpload(e.target.files);
      fileInput.value = '';
    });
  }

  // Ask send button
  // Sidebar collapse toggle now lives in main.js (loaded on every page) so it
  // works on Settings too. Handled there to avoid binding the click twice.

  // Clear chat button
  const clearChatBtn = $('chat-clear-btn');
  if (clearChatBtn) {
    clearChatBtn.addEventListener('click', () => {
      const thread = $('chat-thread');
      if (thread) thread.innerHTML = '';
      clearChatBtn.className = 'chat-clear-btn';
    });
  }

  const askBtn  = $('ask-send-btn');
  const askArea = $('ask-textarea');
  if (askBtn && askArea) {
    askBtn.addEventListener('click', () => {
      const q = askArea.value.trim();
      if (q) sendQuestion(q);
    });
    askArea.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
        e.preventDefault();
        const q = askArea.value.trim();
        if (q) sendQuestion(q);
      }
    });
  }

  // Question cards
  document.querySelectorAll('.question-card, .chip[data-question]').forEach(card => {
    card.addEventListener('click', () => {
      const q = card.dataset.question || card.querySelector('.q-text')?.textContent.trim();
      if (q && askArea) {
        askArea.value = q;
        askArea.focus();
      }
    });
  });

  // My-videos page: render the list
  renderMyVideosList();

  // Upload page: hook file input
  const uploadArea = document.getElementById('upload-area');
  if (uploadArea) {
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(ev =>
      uploadArea.addEventListener(ev, e => { e.preventDefault(); e.stopPropagation(); }));
    ['dragenter', 'dragover'].forEach(ev =>
      uploadArea.addEventListener(ev, () => uploadArea.classList.add('dragover')));
    ['dragleave', 'drop'].forEach(ev =>
      uploadArea.addEventListener(ev, () => uploadArea.classList.remove('dragover')));
    uploadArea.addEventListener('drop', e => handleUploadPageFiles(e.dataTransfer.files));
    uploadArea.addEventListener('click', () => {
      const inp = document.createElement('input');
      inp.type = 'file'; inp.accept = 'video/*';
      inp.onchange = e => handleUploadPageFiles(e.target.files);
      inp.click();
    });
  }
});

// ========== UPLOAD PAGE ==========
async function handleUploadPageFiles(files) {
  if (!files.length) return;
  const uploadList = document.getElementById('upload-list');

  Array.from(files).forEach(async file => {
    let item;
    if (uploadList) {
      item = document.createElement('div');
      item.className = 'upload-item';
      item.innerHTML = `
        <div class="upload-thumb"><i class="fa-solid fa-film"></i></div>
        <div class="upload-details">
          <h4>${esc(file.name)}</h4>
          <p>${(file.size / 1024 / 1024).toFixed(1)} MB</p>
          <div class="upload-progress-bar">
            <div class="upload-progress-fill" style="width:0%"></div>
          </div>
        </div>
        <span class="upload-status">0%</span>`;
      uploadList.prepend(item);
    }

    const fill   = item?.querySelector('.upload-progress-fill');
    const status = item?.querySelector('.upload-status');

    try {
      const result = await API.uploadFile(file, null, (pct) => {
        const p = Math.round(pct * 90);
        if (fill)   fill.style.width = p + '%';
        if (status) status.textContent = p + '%';
      });

      if (fill)   fill.style.width = '95%';
      if (status) status.textContent = 'Indexing…';

      const videoId = result.video_id;
      if (videoId) {
        await API.ingest(videoId).catch(() => {});
        if (fill)   fill.style.width = '100%';
        if (status) status.textContent = 'Done';
        await loadVideoList();
      }
    } catch (e) {
      if (status) status.textContent = 'Error';
      if (fill)   { fill.style.width = '100%'; fill.style.background = '#ef4444'; }
    }
  });
}