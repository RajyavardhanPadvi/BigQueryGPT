# api/index.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from typing import List, Dict
import re

app = FastAPI(title="BigQueryGPT Slim")

# --- Minimal in-memory store
DOCS: list[str] = []
VECTORS: list[list[float]] = []
VOCAB: dict[str, int] = {}

def tokenize(s: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", s.lower())

def build_vocab(texts: list[str]) -> dict[str, int]:
    vocab = {}
    for t in texts:
        for tok in tokenize(t):
            if tok not in vocab:
                vocab[tok] = len(vocab)
    return vocab

def tfidf_vectors(texts: list[str], vocab: dict[str, int]) -> list[list[float]]:
    import math
    df = [0]*len(vocab)
    docs_toks = [tokenize(t) for t in texts]
    for toks in docs_toks:
        seen = set(toks)
        for tok in seen:
            df[vocab[tok]] += 1
    N = len(texts)
    idf = [math.log((N+1)/(d+1)) + 1.0 for d in df]

    vecs = []
    for toks in docs_toks:
        counts = [0]*len(vocab)
        for tok in toks:
            counts[vocab[tok]] += 1
        maxc = max(counts) or 1
        tf = [c/maxc for c in counts]
        vecs.append([tf[i]*idf[i] for i in range(len(vocab))])
    return vecs

def cosine(a: list[float], b: list[float]) -> float:
    import math
    dot = sum(x*y for x,y in zip(a,b))
    na = math.sqrt(sum(x*x for x in a)) or 1e-9
    nb = math.sqrt(sum(y*y for y in b)) or 1e-9
    return dot/(na*nb)

@app.get("/api/health")
def health():
    return {"ok": "up", "indexed_docs": len(DOCS)}

@app.post("/upload")
async def upload(files: List[UploadFile] = File(...)):
    texts = []
    for f in files:
        data = await f.read()
        try:
            texts.append(data.decode("utf-8", errors="ignore"))
        except Exception:
            raise HTTPException(400, detail=f"Cannot decode {f.filename}")
    DOCS.clear()
    DOCS.extend(texts)
    global VOCAB, VECTORS
    VOCAB = build_vocab(DOCS)
    VECTORS = tfidf_vectors(DOCS, VOCAB)
    return {"status": "ok", "n_docs": len(DOCS)}

@app.get("/build_index")
def build_index():
    # no-op (index is built on upload), kept for UI compatibility
    return {"status": "ok", "n_docs": len(DOCS)}

@app.get("/ask")
def ask(q: str, top_k: int = 3):
    if not DOCS:
        raise HTTPException(400, detail="Index not built yet. Upload files first.")
    vocab = VOCAB
    if not vocab:
        raise HTTPException(500, detail="Empty vocab.")
    # vectorize query
    q_counts = [0]*len(vocab)
    for tok in tokenize(q):
        if tok in vocab:
            q_counts[vocab[tok]] += 1
    # simple tf weight
    maxc = max(q_counts) or 1
    q_vec = [c/maxc for c in q_counts]

    sims = [cosine(q_vec, v) for v in VECTORS]
    order = sorted(range(len(DOCS)), key=lambda i: sims[i], reverse=True)[:top_k]
    hits = [{"score": sims[i], "snippet": DOCS[i][:200]} for i in order]
    answer = hits[0]["snippet"] if hits else "No match."
    return {"answer": answer, "retrieved": hits}

@app.get("/", response_class=HTMLResponse)
def home():
    html = """
<!doctype html><html><head>
<meta charset="utf-8"/>
<title>BigQueryGPT Slim</title>
<style>
body{font-family:Inter,Arial;background:#0b1220;color:#e7f0ff;margin:0}
.wrap{max-width:900px;margin:24px auto;padding:0 16px}
.zone{margin-top:14px;border:2px dashed #4ea1ff;padding:18px;border-radius:12px;text-align:center}
.btn{background:#4ea1ff;border:none;padding:8px 12px;border-radius:8px;cursor:pointer}
.chat{margin-top:20px;background:#0e1a2b;border:1px solid #1f3550;border-radius:10px;padding:12px}
.msg{margin:8px 0;padding:8px;border-radius:8px;max-width:80%}
.user{margin-left:auto;background:#13314d}
.bot{margin-right:auto;background:#0c2238}
.row{display:flex;gap:8px;align-items:center}
input[type=text]{flex:1;border-radius:8px;border:1px solid #2a4a6c;background:#0b1b2b;color:#e7f0ff;padding:8px}
</style></head><body>
<div class="wrap">
  <h2>BigQueryGPT Slim</h2>
  <div class="zone" onclick="fileInput.click()" ondragover="event.preventDefault()" ondrop="dropHandler(event)">
    Drop text/CSV files here or click to select
    <input id="fileInput" type="file" multiple style="display:none" onchange="uploadFiles(this.files)"/>
  </div>
  <p id="info"></p>
  <div class="chat"><div id="msgs"></div>
    <div class="row" style="margin-top:8px">
      <input id="q" type="text" placeholder="Ask..."/>
      <button class="btn" onclick="ask()">Ask</button>
    </div>
  </div>
</div>
<script>
const fileInput = document.getElementById('fileInput');
const info = document.getElementById('info');
const msgs = document.getElementById('msgs');
document.getElementById('q').addEventListener('keydown', e=>{ if(e.key==='Enter'){ ask(); }});

function addMsg(t, who){ const d=document.createElement('div'); d.className='msg '+(who==='user'?'user':'bot'); d.textContent=t; msgs.appendChild(d); msgs.scrollTop=msgs.scrollHeight; }

async function uploadFiles(fs){
  let fd = new FormData();
  for (let f of fs) fd.append('files', f);
  const r = await fetch('/upload', {method:'POST', body: fd});
  const j = await r.json();
  info.textContent = j.status ? `Index ready (${j.n_docs} docs)` : JSON.stringify(j);
}

function dropHandler(ev){ ev.preventDefault(); uploadFiles(ev.dataTransfer.files); }

async function ask(){
  const q = document.getElementById('q').value.trim();
  if(!q) return;
  addMsg(q,'user'); addMsg('Thinking...','bot');
  const r = await fetch('/ask?q='+encodeURIComponent(q));
  const j = await r.json();
  msgs.lastChild.textContent = (typeof j.answer==='string') ? j.answer : JSON.stringify(j,null,2);
}
</script></body></html>
"""
    return HTMLResponse(html)
