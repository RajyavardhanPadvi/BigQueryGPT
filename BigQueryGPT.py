#!/usr/bin/env python3
# BigQueryGPT — session-scoped RAG with optional dataset_rag_full, Turnstile, MLflow

import os, re, json, math, uuid, traceback
from pathlib import Path
from typing import List, Dict, Optional
from collections import Counter

import pandas as pd
import requests

from fastapi import FastAPI, UploadFile, File, HTTPException, Body, Query, Response, Request, Form
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.gzip import GZipMiddleware
from urllib.parse import quote

# ---------------- Env & config ----------------
def _clean(s: str) -> str:
    if not isinstance(s, str): return ""
    return re.sub(r"[\u200b\u200c\u200d\u2060\uFEFF]", "", s).strip()

HF_MODEL = _clean(os.getenv("HF_MODEL") or "google/flan-t5-small")
HF_USE_TOKEN = (_clean(os.getenv("HF_USE_TOKEN") or "0") == "1")
HF_TOKEN = _clean(os.getenv("HF_API_KEY") or os.getenv("HF_API_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN") or "")

TURNSTILE_SITEKEY = _clean(os.getenv("TURNSTILE_SITEKEY") or "")
TURNSTILE_SECRET  = _clean(os.getenv("TURNSTILE_SECRET")  or "")

MLFLOW_TRACKING_URI = _clean(os.getenv("MLFLOW_TRACKING_URI") or "")
HAS_MLFLOW = False
try:
    if MLFLOW_TRACKING_URI:
        import mlflow
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        HAS_MLFLOW = True
except Exception:
    HAS_MLFLOW = False

DEBUG = os.getenv("DEBUG", "0") == "1"

# Per-session uploads under /tmp (never read server folders the user didn’t upload)
BASE_ROOT = Path(os.getenv("UPLOAD_ROOT", "/tmp")) / "bqgpt"
BASE_ROOT.mkdir(parents=True, exist_ok=True)

# Timeouts & limits
HF_TIMEOUT_SEC = int(_clean(os.getenv("HF_TIMEOUT") or "3"))
MAX_NEW_TOKENS = int(_clean(os.getenv("HF_MAX_NEW_TOKENS") or "64"))
RETRIEVAL_K = int(_clean(os.getenv("RETRIEVAL_K") or "3"))
MAX_CONTEXT_CHARS = int(_clean(os.getenv("MAX_CONTEXT_CHARS") or "1200"))
CHUNK_ROWS = int(_clean(os.getenv("CHUNK_ROWS") or "5000"))
MAX_ROWS_PER_FILE = int(_clean(os.getenv("MAX_ROWS_PER_FILE") or "50000"))
MAX_FILES_PER_SESSION = int(_clean(os.getenv("MAX_FILES_PER_SESSION") or "12"))

# Prefer your local dataset_rag_full if present
USING_EXTERNAL = False
try:
    from dataset_rag_full import (
        read_generic as dg_read_generic,
        dataframe_to_docs as dg_dataframe_to_docs,
        TfidfVectorStore as DG_TfidfVectorStore,
    )
    USING_EXTERNAL = True
except Exception:
    dg_read_generic = dg_dataframe_to_docs = DG_TfidfVectorStore = None

# ---------------- Turnstile helpers ----------------
def _is_localhost(host_header: str) -> bool:
    host = (host_header or "").split(":")[0].lower().strip()
    return host in {"localhost", "127.0.0.1"}

def turnstile_required(request: Request) -> bool:
    return bool(TURNSTILE_SITEKEY and TURNSTILE_SECRET and not _is_localhost(request.headers.get("host")))

def verify_turnstile_token(token: Optional[str]) -> bool:
    if not TURNSTILE_SECRET:
        return True
    tok = (token or "").strip()
    if not tok:
        return False
    try:
        r = requests.post(
            "https://challenges.cloudflare.com/turnstile/v0/siteverify",
            data={"secret": TURNSTILE_SECRET, "response": tok},
            timeout=5,
        )
        return bool(r.json().get("success"))
    except Exception:
        return False

# ---------------- Tiny TF-IDF (fallback) ----------------
def _tokenize(s: str) -> List[str]:
    return [t for t in re.sub(r"[^a-zA-Z0-9]+", " ", str(s)).lower().split() if t]

class TinyTfidf:
    def __init__(self, texts: List[str]):
        self.docs = texts
        toks = [_tokenize(d) for d in texts]
        self.N = len(toks)
        from collections import Counter
        df = Counter()
        for ts in toks: df.update(set(ts))
        self.idf = {t: math.log((1 + self.N) / (1 + c)) + 1.0 for t, c in df.items()}
        self.vecs = []
        for ts in toks:
            tf = Counter(ts); L = max(1, len(ts))
            self.vecs.append({t: (tf[t]/L) * self.idf.get(t, 0.0) for t in tf})

    def _qvec(self, q: str) -> Dict[str, float]:
        ts = _tokenize(q)
        if not ts: return {}
        from collections import Counter
        tf = Counter(ts); L = len(ts)
        return {t: (tf[t]/L) * self.idf.get(t, 0.0) for t in tf}

    @staticmethod
    def _cos(a: Dict[str, float], b: Dict[str, float]) -> float:
        if not a or not b: return 0.0
        keys = set(a) | set(b)
        dot = sum(a.get(k,0.0)*b.get(k,0.0) for k in keys)
        na = math.sqrt(sum(v*v for v in a.values())); nb = math.sqrt(sum(v*v for v in b.values()))
        return dot/(na*nb) if na and nb else 0.0

    def search(self, q: str, k: int = 5):
        qv = self._qvec(q)
        scores = [(i, self._cos(qv, self.vecs[i])) for i in range(len(self.docs))]
        scores.sort(key=lambda x: x[1], reverse=True)
        return [{"text": self.docs[i], "score": s} for i, s in scores[:max(1,k)]]

# ---------------- LLM (HF short timeout) ----------------
def _hf_api_generate(prompt: str, max_new_tokens: int = MAX_NEW_TOKENS, temperature: float = 0.0) -> str:
    model_clean = HF_MODEL.replace("\\", "/").replace(" ", "")
    url = f"https://api-inference.huggingface.co/models/{quote(model_clean, safe='/-_.')}"
    payload = {"inputs": prompt, "parameters": {"max_new_tokens": max_new_tokens, "temperature": temperature},
               "options": {"wait_for_model": False}}
    headers = {"Content-Type": "application/json"}
    if HF_USE_TOKEN and HF_TOKEN: headers["Authorization"] = f"Bearer {HF_TOKEN}"
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=HF_TIMEOUT_SEC)
    except Exception as e:
        return f"[HF ERROR] Request failed: {e}"
    if r.status_code != 200: return f"[HF ERROR] {r.status_code} · {r.text[:160].replace(chr(10),' ')}"
    data = r.json()
    if isinstance(data, list) and data and isinstance(data[0], dict) and "generated_text" in data[0]: return data[0]["generated_text"]
    if isinstance(data, list) and data and isinstance(data[0], str): return data[0]
    if isinstance(data, dict) and "generated_text" in data: return data["generated_text"]
    return str(data)

def answer_with_fallbacks(context: str, question: str, concise: bool = True, fast: bool = True) -> str:
    prompt = (f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer in 1–2 short sentences."
              if concise else f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer briefly.")
    if not fast:
        ans = _hf_api_generate(prompt)
        if ans and not ans.startswith("[HF ERROR]"):
            if concise:
                parts = [s.strip() for s in ans.split(".") if s.strip()]
                ans = (". ".join(parts[:2]) + ('.' if parts else '')) or ans
            return ans
    lines = [ln.strip() for ln in context.split("\n") if ln.strip()]
    return "Top matches:\n- " + "\n- ".join(lines[:2]) if lines else "No matching context."

# ---------------- Readers (safe & chunked) ----------------
def _chunk_text(text: str, chunk_size=500, overlap=100):
    if not text: return []
    out=[]; i=0; L=len(text)
    while i < L:
        j=min(i+chunk_size, L); out.append(text[i:j])
        if j>=L: break
        i=max(0, j-overlap)
    return out

def df_to_docs(df: pd.DataFrame) -> List[Dict]:
    docs=[]
    if list(df.columns)==["text"]:
        for _, t in df["text"].fillna("").astype(str).items():
            for ch in _chunk_text(t): docs.append({"text": ch})
        return docs
    for _, row in df.iterrows():
        parts=[f"{c}: {row[c]}" for c in df.columns if pd.notna(row[c]) and str(row[c]).strip()]
        for ch in _chunk_text(" | ".join(parts)): docs.append({"text": ch})
    return docs

def stream_csv_as_docs(path: Path, max_rows: int = MAX_ROWS_PER_FILE, chunk_rows: int = CHUNK_ROWS):
    rows = 0
    for chunk in pd.read_csv(path, low_memory=False, chunksize=chunk_rows):
        if rows >= max_rows: break
        need = max_rows - rows
        if len(chunk) > need: chunk = chunk.iloc[:need]
        for d in df_to_docs(chunk): yield d
        rows += len(chunk)
    if rows == 0:
        df = pd.read_csv(path, low_memory=False, nrows=max_rows)
        for d in df_to_docs(df): yield d

def read_any_to_docs(path: Path, input_format: str = "", header: bool = False) -> List[Dict]:
    ext = (input_format or path.suffix.lstrip(".")).lower()
    docs: List[Dict] = []
    try:
        if USING_EXTERNAL and dg_read_generic:
            # delegate to your dataset_rag_full reader if available
            df = dg_read_generic(str(path), fmt=ext)
            return df_to_docs(df)
        if ext == "csv":
            for d in stream_csv_as_docs(path): docs.append(d)
        elif ext == "json":
            df = pd.read_json(path); docs += df_to_docs(df)
        elif ext == "xlsx":
            df = pd.read_excel(path); docs += df_to_docs(df)
        elif ext == "parquet":
            df = pd.read_parquet(path); docs += df_to_docs(df)
        elif ext in {"txt","md"}:
            text = path.read_text(encoding="utf-8", errors="ignore")
            for ch in _chunk_text(text): docs.append({"text": ch})
        else:
            df = pd.read_csv(path, low_memory=False, nrows=MAX_ROWS_PER_FILE); docs += df_to_docs(df)
    except Exception:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
            for ch in _chunk_text(text): docs.append({"text": ch})
        except Exception:
            pass
    return docs

# ---------------- App scaffold ----------------
app = FastAPI(title="BigQueryGPT")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.add_middleware(GZipMiddleware, minimum_size=512)

SESSIONS: Dict[str, Dict[str, object]] = {}  # {sid: {"docs":[...], "vstore":..., "params": {...}}}

@app.get("/", response_class=FileResponse)
def root():
    p = Path("index.html")
    if p.exists(): return FileResponse("index.html")
    return PlainTextResponse("BigQueryGPT: index.html missing", status_code=200)

@app.head("/")
def root_head(): return Response(status_code=200)

@app.get("/health")
def health():
    return {"ok":"up","sessions":len(SESSIONS),"turnstile":bool(TURNSTILE_SITEKEY and TURNSTILE_SECRET),"mlflow":("on" if HAS_MLFLOW else "off")}

@app.head("/health")
def health_head(): return Response(status_code=200)

def _new_sid() -> str: return uuid.uuid4().hex
def _safe_name(name: str) -> str: return re.sub(r"[^a-zA-Z0-9_.-]+", "_", name)[:200]

# -------- Upload (multipart) --------
@app.post("/upload")
async def upload(files: List[UploadFile] = File(...), sid: Optional[str] = Form(None), cf_token: Optional[str] = Form(None), request: Request = None):
    if turnstile_required(request):
        if not verify_turnstile_token(cf_token):
            raise HTTPException(403, detail="CAPTCHA failed.")
    sid = sid or _new_sid()
    sdir = BASE_ROOT / sid
    sdir.mkdir(parents=True, exist_ok=True)
    count = 0
    for f in files:
        name = _safe_name(f.filename or f"file_{count}.dat")
        dest = sdir / name
        with open(dest, "wb") as out:
            out.write(await f.read())
        count += 1
        if count >= MAX_FILES_PER_SESSION: break
    SESSIONS.pop(sid, None)  # clear old index for this sid
    return {"status": "ok", "sid": sid, "message": f"{count} file(s) uploaded."}

# -------- Build index (JSON) --------
@app.post("/build_index")
def build_index(payload: Dict = Body(...), request: Request = None):
    if turnstile_required(request):
        tok = payload.get("cf_token", "")
        if not verify_turnstile_token(tok):
            raise HTTPException(403, detail="CAPTCHA failed.")
    sid = payload.get("sid", "")
    if not sid: raise HTTPException(400, detail="Missing session id (sid).")
    sdir = BASE_ROOT / sid
    if not sdir.exists(): raise HTTPException(400, detail="Invalid session.")

    # Local-style flags (like your CLI): --input-format, --header, --model, --concise, --tts
    input_format = str(payload.get("input_format", "")).lower().strip()
    header       = bool(payload.get("header", False))
    model        = str(payload.get("model", HF_MODEL))
    concise      = bool(payload.get("concise", True))
    tts          = bool(payload.get("tts", False))

    files = list(sdir.glob("*.*"))[:MAX_FILES_PER_SESSION]
    if not files: raise HTTPException(400, detail="No files uploaded for this session.")

    docs: List[Dict] = []
    for f in files:
        try:
            ds = read_any_to_docs(f, input_format=input_format, header=header)
            docs.extend(ds)
        except Exception as e:
            if DEBUG: print("[build_index] skip", f.name, e)

    if not docs: raise HTTPException(400, detail="No supported data found in files.")

    # Prefer your external vector store if available
    if USING_EXTERNAL and DG_TfidfVectorStore:
        vstore = DG_TfidfVectorStore(docs)
    else:
        vstore = TinyTfidf([d["text"] for d in docs])

    # Save in-memory & cache to disk (per session)
    SESSIONS[sid] = {"docs": docs, "vstore": vstore, "params": {
        "input_format": input_format, "header": header, "model": model, "concise": concise, "tts": tts
    }}
    try:
        (sdir / "index.json").write_text(json.dumps({"docs": docs}), encoding="utf-8")
    except Exception:
        pass

    # Optional MLflow logging
    if HAS_MLFLOW:
        try:
            import mlflow
            with mlflow.start_run(run_name=f"bqgpt_build_{sid[:6]}") as run:
                mlflow.log_param("sid", sid)
                mlflow.log_param("n_files", len(files))
                mlflow.log_param("n_docs", len(docs))
                mlflow.log_param("input_format", input_format or "auto")
                mlflow.log_param("header", header)
                mlflow.log_param("model", model)
                mlflow.log_param("concise", concise)
                mlflow.log_param("tts", tts)
        except Exception as e:
            if DEBUG: print("[mlflow] error:", e)

    return {"status": "ok", "sid": sid, "n_docs": len(docs)}

# -------- Ask --------
@app.get("/ask")
def ask(q: str, sid: str, top_k: int = 3, concise: bool = True, fast: int = 1, cf_token: str = "", request: Request = None):
    if turnstile_required(request):
        if not verify_turnstile_token(cf_token):
            raise HTTPException(403, detail="CAPTCHA failed.")
    if not sid: raise HTTPException(400, detail="Missing session id (sid).")
    if not q or not q.strip(): raise HTTPException(400, detail="Empty question")

    sess = SESSIONS.get(sid)
    if not sess:
        # try load cached index for this sid
        sdir = BASE_ROOT / sid
        cache = sdir / "index.json"
        if cache.exists():
            try:
                obj = json.loads(cache.read_text(encoding="utf-8"))
                docs = obj.get("docs") or []
                vstore = TinyTfidf([d["text"] for d in docs])
                SESSIONS[sid] = sess = {"docs": docs, "vstore": vstore, "params": {}}
            except Exception:
                raise HTTPException(400, detail="Index not built for this session.")
        else:
            raise HTTPException(400, detail="Index not built for this session.")

    vstore = sess["vstore"]
    if hasattr(vstore, "search"):
        hits = vstore.search(q, k=min(max(1, top_k), RETRIEVAL_K))
        context = "\n".join([h["text"] for h in hits])[:MAX_CONTEXT_CHARS]
    else:
        hits = []
        context = ""
    ans = answer_with_fallbacks(context, q, concise=concise, fast=(fast == 1))
    return {"answer": ans, "retrieved": hits[:top_k]}

# -------- Global exception guard -> always JSON (no 502) --------
@app.exception_handler(Exception)
async def _catch_all(request: Request, exc: Exception):
    if DEBUG:
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        print("[ERROR]", request.method, request.url.path, "->", exc, "\n", tb)
    return JSONResponse({"status": "error", "detail": str(exc)}, status_code=500)
