from BigQueryGPT import app as app
#!/usr/bin/env python3
# api/index.py
# BigQueryGPT – ultra-light FastAPI + pure-Python TF-IDF, Vercel-friendly

import os, io, csv, json, math, time
from typing import List, Dict, Optional, Tuple
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import requests

# -----------------------------
# Config / Env
# -----------------------------
HF_MODEL = os.getenv("HF_MODEL", "google/flan-t5-small").strip()
HF_TOKEN = (os.getenv("HF_API_KEY") or os.getenv("HF_API_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN") or "").strip()
HF_TIMEOUT = int(os.getenv("HF_TIMEOUT", "4"))   # seconds
MAX_NEW_TOKENS = int(os.getenv("HF_MAX_NEW_TOKENS", "64"))
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "3"))
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "1400"))

UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "/tmp/uploads"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Tiny TF-IDF (pure Python)
# -----------------------------
def _tokenize(text: str) -> List[str]:
    # very basic tokenizer
    t = "".join(ch.lower() if ch.isalnum() or ch in "/._-" else " " for ch in text or "")
    return [tok for tok in t.split() if tok]

class TinyTfidf:
    def __init__(self, docs: List[str]):
        self.docs = docs[:]  # store original text
        self.N = len(docs)
        self.df = {}        # term -> document frequency
        self.idf = {}       # term -> idf
        self.doc_tf = []    # list[dict(term->tf)]
        # build
        for text in docs:
            toks = _tokenize(text)
            tf: Dict[str, float] = {}
            seen = set()
            for w in toks:
                tf[w] = tf.get(w, 0.0) + 1.0
                if w not in seen:
                    self.df[w] = self.df.get(w, 0) + 1
                    seen.add(w)
            # l2 normalize
            norm = math.sqrt(sum(v*v for v in tf.values())) or 1.0
            for k in tf:
                tf[k] /= norm
            self.doc_tf.append(tf)
        for w, c in self.df.items():
            # add-one smoothing to avoid div-by-zero in tiny corpora
            self.idf[w] = math.log((self.N + 1) / (c + 1)) + 1.0

    def search(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        if not self.docs:
            return []
        q_tf: Dict[str, float] = {}
        for w in _tokenize(query):
            q_tf[w] = q_tf.get(w, 0.0) + 1.0
        # weight with idf
        for w in list(q_tf.keys()):
            q_tf[w] *= self.idf.get(w, 0.0)
        q_norm = math.sqrt(sum(v*v for v in q_tf.values())) or 1.0
        for k in q_tf:
            q_tf[k] /= q_norm

        scores = []
        for i, tf in enumerate(self.doc_tf):
            score = 0.0
            for w, qv in q_tf.items():
                score += qv * (tf.get(w, 0.0) * self.idf.get(w, 0.0))
            scores.append((i, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

# -----------------------------
# Simple readers (CSV/JSON only)
# -----------------------------
def read_csv_text(file_bytes: bytes) -> List[str]:
    out = []
    try:
        content = file_bytes.decode("utf-8", errors="ignore")
        reader = csv.DictReader(io.StringIO(content))
        for row in reader:
            parts = [f"{k}: {v}" for k, v in row.items() if str(v).strip()]
            if parts:
                out.append(" | ".join(parts))
    except Exception:
        # fallback: treat as plain text
        out.append(file_bytes.decode("utf-8", errors="ignore"))
    return out

def read_json_text(file_bytes: bytes) -> List[str]:
    out = []
    try:
        data = json.loads(file_bytes.decode("utf-8", errors="ignore"))
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    parts = [f"{k}: {v}" for k, v in item.items() if str(v).strip()]
                    if parts:
                        out.append(" | ".join(parts))
                else:
                    out.append(str(item))
        elif isinstance(data, dict):
            parts = [f"{k}: {v}" for k, v in data.items() if str(v).strip()]
            if parts:
                out.append(" | ".join(parts))
        else:
            out.append(str(data))
    except Exception:
        out.append(file_bytes.decode("utf-8", errors="ignore"))
    return out

# -----------------------------
# HF text-generation helper
# -----------------------------
def hf_generate(prompt: str, max_new_tokens: int = MAX_NEW_TOKENS) -> str:
    url = f"https://api-inference.huggingface.co/models/{HF_MODEL}"
    headers = {"Content-Type": "application/json"}
    if HF_TOKEN:
        headers["Authorization"] = f"Bearer {HF_TOKEN}"

    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": max_new_tokens, "temperature": 0.0},
        "options": {"wait_for_model": True},
    }

    try:
        r = requests.post(url, headers=headers, json=payload, timeout=HF_TIMEOUT)
        if r.status_code != 200:
            return f"[HF ERROR] {r.status_code}: {r.text[:200]}"
        data = r.json()
        if isinstance(data, list) and data and "generated_text" in data[0]:
            return data[0]["generated_text"]
        return str(data)
    except Exception as e:
        return f"[HF ERROR] {e}"

# -----------------------------
# Global state (in-memory)
# -----------------------------
STATE: Dict[str, object] = {
    "docs": [],         # list[str]
    "tfidf": None,      # TinyTfidf
    "n_docs": 0,
}

# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(title="BigQueryGPT (Lite on Vercel)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"], allow_credentials=True,
)

# ---------- Routes ----------

@app.get("/", response_class=HTMLResponse)
def home():
    return HTMLResponse(f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>BigQueryGPT</title>
  <style>
    body {{ font-family: Inter, Arial; background:#0b1220; color:#eaf6ff; margin:0; }}
    .wrap {{ max-width: 940px; margin: 24px auto; padding: 0 16px; }}
    .zone {{ margin-top:14px; border:2px dashed #27b3ff; padding:18px; border-radius:12px; text-align:center; }}
    .btn {{ background:#27b3ff; color:#051020; border:none; padding:10px 14px; border-radius:10px; cursor:pointer; }}
    .bar {{ display:flex; gap:8px; align-items:center; margin-top:16px; }}
    input[type=text] {{ flex:1; border-radius:10px; border:1px solid #17324a; background:#0b1a2c; color:#eaf6ff; padding:10px 12px; }}
    .chat {{ margin-top:18px; background:#081422; border:1px solid #132b41; border-radius:12px; padding:12px; min-height: 120px; }}
    .msg {{ padding:8px 10px; border-radius:10px; margin:8px 0; max-width:80%; }}
    .user {{ background:#0f2c4a; margin-left:auto; }}
    .bot {{ background:#0a2036; margin-right:auto; }}
    .hint {{ color:#8ed3ff; font-size:0.9rem; }}
  </style>
</head>
<body>
<div class="wrap">
  <h2>💬 BigQueryGPT</h2>
  <p class="hint">1) Upload CSV/JSON → 2) Build Index → 3) Ask questions. (Model: {HF_MODEL})</p>

  <div class="zone" ondrop="dropHandler(event)" ondragover="event.preventDefault()" onclick="fileInput.click()">
    Drop files here or click to select
    <input id="fileInput" type="file" multiple style="display:none" onchange="uploadFiles(this.files)"/>
  </div>
  <pre id="files" class="hint"></pre>

  <div class="bar">
    <button class="btn" onclick="buildIndex()">Build Index</button>
    <span id="buildOut" class="hint"></span>
  </div>

  <div class="chat" id="chatBox">
    <div id="msgs"></div>
    <div class="bar">
      <input id="q" type="text" placeholder="Ask something about your uploaded data..."/>
      <button class="btn" onclick="ask()">Ask</button>
    </div>
  </div>
</div>

<script>
const fileInput = document.getElementById('fileInput');
const filesOut  = document.getElementById('files');
const msgs      = document.getElementById('msgs');
const qInput    = document.getElementById('q');

qInput.addEventListener('keydown', (e) => {{
  if (e.key === 'Enter') {{ e.preventDefault(); ask(); }}
}});

function dropHandler(ev) {{
  ev.preventDefault();
  uploadFiles(ev.dataTransfer.files);
}}

async function uploadFiles(fs){{
  let fd = new FormData();
  for (let f of fs) fd.append('files', f);
  const r = await fetch('/upload', {{ method:'POST', body: fd }});
  const j = await r.json();
  filesOut.textContent = (j.message || JSON.stringify(j));
}}

async function buildIndex(){{
  document.getElementById('buildOut').textContent = "Building...";
  const r = await fetch('/build_index', {{ method:'POST' }});
  const j = await r.json();
  document.getElementById('buildOut').textContent = j.status ? ("Index ready ("+j.n_docs+" docs)") : (j.error||JSON.stringify(j));
}}

function addMsg(txt, who) {{
  const d = document.createElement('div');
  d.className = 'msg ' + (who==='user' ? 'user' : 'bot');
  d.textContent = txt;
  msgs.appendChild(d);
  msgs.scrollTop = msgs.scrollHeight;
}}

async function ask(){{
  const q = qInput.value.trim();
  if (!q) return;
  addMsg(q, 'user');
  addMsg('Thinking...', 'bot');
  const r = await fetch('/ask?q=' + encodeURIComponent(q));
  const j = await r.json();
  let a = j.answer;
  if (typeof a !== 'string') a = JSON.stringify(a, null, 2);
  msgs.lastChild.textContent = a;
}}
</script>
</body></html>
""")

@app.post("/upload")
async def upload(files: List[UploadFile] = File(...)):
    count = 0
    for f in files:
        data = await f.read()
        dest = UPLOAD_DIR / f.filename
        with open(dest, "wb") as out:
            out.write(data)
        count += 1
    return {"status": "ok", "message": f"{count} file(s) uploaded."}

@app.post("/build_index")
def build_index():
    docs: List[str] = []
    for p in UPLOAD_DIR.glob("*"):
        try:
            b = p.read_bytes()
            ext = p.suffix.lower()
            if ext == ".csv":
                docs += read_csv_text(b)
            elif ext == ".json":
                docs += read_json_text(b)
            else:
                # skip heavy formats on Vercel
                continue
        except Exception:
            continue

    if not docs:
        raise HTTPException(400, detail="No supported files found. Upload CSV or JSON.")

    # chunk long rows a bit (shorter context helps latency)
    def chunk(s: str, n=600, overlap=150):
        out=[]; i=0
        while i < len(s):
            j=min(i+n, len(s)); out.append(s[i:j])
            if j>=len(s): break
            i=max(0, j-overlap)
        return out

    expanded = []
    for d in docs:
        for ch in chunk(d):
            expanded.append(ch)

    STATE["docs"] = expanded
    STATE["tfidf"] = TinyTfidf(expanded)
    STATE["n_docs"] = len(expanded)
    return {"status":"ok","n_docs":len(expanded)}

@app.get("/ask")
def ask(q: str = Query(..., min_length=1), top_k: int = 5, concise: bool = True):
    tfidf: Optional[TinyTfidf] = STATE.get("tfidf")  # type: ignore
    docs: List[str] = STATE.get("docs") or []        # type: ignore
    if not tfidf or not docs:
        raise HTTPException(400, detail="Index not built. Upload files and click Build Index.")

    hits = tfidf.search(q, top_k=min(top_k, RETRIEVAL_K))
    context = "\n".join([docs[i] for i, _ in hits])[:MAX_CONTEXT_CHARS]
    prompt = (f"Context:\n{context}\n\nQuestion: {q}\n\n"
              f"Answer in one or two short sentences. If unknown, say you don't know.")

    ans = hf_generate(prompt, max_new_tokens=MAX_NEW_TOKENS)
    if ans.startswith("[HF ERROR]"):
        # fallback: return the top snippets
        snippets = [docs[i] for i,_ in hits][:2]
        ans = "Top matches:\n- " + "\n- ".join(snippets) if snippets else "No relevant context."
    else:
        if concise:
            parts = [s.strip() for s in ans.split(".") if s.strip()]
            ans = (". ".join(parts[:2]) + ('.' if parts else '')) or ans

    # include tiny debug of scores to help UX
    ret = [{"text": docs[i], "score": float(s)} for i, s in hits]
    return {"answer": ans, "retrieved": ret}

@app.get("/api/health")
def health():
    return {
        "ok": "up",
        "hf_model": HF_MODEL,
        "token_present": bool(HF_TOKEN),
        "indexed_docs": STATE.get("n_docs", 0),
    }
