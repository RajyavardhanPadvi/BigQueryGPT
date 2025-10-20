#!/usr/bin/env python3
"""
dataset_rag_full.py — single-file pipeline (MLflow-integrated)
Features:
 - read csv/json/parquet/xlsx/pdf/docx/odt
 - chunking + TF-IDF or SBERT+FAISS vector store
 - irrelevance detection (skip LLM for unrelated queries)
 - small-talk fallback
 - Ollama local generation (streaming)
 - terminal UI (rich), Streamlit/Gradio hooks
 - MLflow & LoRA stubs
 - Databricks notebook helper
"""

import os
import sys
import argparse
import json
import time
import tempfile
from pathlib import Path
from typing import List, Dict
import mlflow

import re
import requests
import pandas as pd
import numpy as np
import joblib


ORIGINAL_INPUT_FILES = None
# Optional heavy libs detection
HAS_FAISS = False
HAS_SBER = False
try:
    import faiss
    HAS_FAISS = True
except Exception:
    HAS_FAISS = False

try:
    from sentence_transformers import SentenceTransformer
    HAS_SBER = True
except Exception:
    HAS_SBER = False

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Rich UI
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text
    console = Console()
except Exception:
    console = None

# Optional doc readers
try:
    import docx
except Exception:
    docx = None

try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

try:
    from odf.opendocument import load as odf_load
    from odf import text as odf_text, teletype
except Exception:
    odf_load = None

# Optional MLflow / TTS
try:
    import mlflow
    HAS_MLFLOW = True
except Exception:
    mlflow = None
    HAS_MLFLOW = False

try:
    import pyttsx3
    HAS_TTS = True
except Exception:
    HAS_TTS = False

# Optional agents (these files are expected to exist in project folder)
try:
    from intent_agent import guess_intent
    from semantic import load_workspaces
    from column_prune import prune_columns
except Exception:
    # Graceful fallback if helper modules are missing
    def guess_intent(q):
        return "general", 0.0
    def load_workspaces(path="workspaces.yaml"):
        return {"general": {"tables": [], "examples": []}}
    def prune_columns(df, query, top_k=10):
        # return df as-is and zero scores
        scores = [(c, 0.0) for c in df.columns.tolist()]
        return df, scores

# ---------------------------------------------------------------------------
# MLflow helpers (robust; optional)
# ---------------------------------------------------------------------------
def init_mlflow(tracking_uri: str = None, experiment_name: str = "/Shared/dataspark"):
    """
    Initialize MLflow connection and ensure experiment exists.
    Use env var MLFLOW_TRACKING_URI or pass tracking_uri.
    """
    if not HAS_MLFLOW:
        return False
    try:
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        else:
            env_uri = os.getenv("MLFLOW_TRACKING_URI")
            if env_uri:
                mlflow.set_tracking_uri(env_uri)
        mlflow.set_experiment(experiment_name)
        return True
    except Exception:
        return False

def _save_artifacts_and_log(vstore, docs, input_name="input", run_name="dataspark_index"):
    """
    Save vectorizer/docs to temp dir and log artifacts with mlflow.
    Returns run_id or raises.
    """
    tmp = tempfile.mkdtemp(prefix="dataspark_art_")
    vec_path = os.path.join(tmp, "tfidf.joblib")
    docs_path = os.path.join(tmp, "docs.json")
    # vectorizer may not exist for some vstore types; handle gracefully
    try:
        joblib.dump(getattr(vstore, "vectorizer", None), vec_path)
    except Exception:
        # create an empty placeholder
        with open(vec_path, "wb") as f:
            f.write(b"")
    with open(docs_path, "w", encoding="utf-8") as f:
        json.dump(getattr(vstore, "docs", docs), f, ensure_ascii=False)

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_param("input_file", input_name)
        mlflow.log_metric("n_docs", len(docs))
        try:
            mlflow.log_artifact(vec_path, artifact_path="artifacts")
        except Exception:
            pass
        try:
            mlflow.log_artifact(docs_path, artifact_path="artifacts")
        except Exception:
            pass
        return run.info.run_id

def safe_mlflow_log_index(input_name, docs, vstore, run_name="dataspark_index"):
    if not HAS_MLFLOW:
        return None
    try:
        return _save_artifacts_and_log(vstore, docs, input_name=input_name, run_name=run_name)
    except Exception:
        return None

# Optional tracing wrapper for Ollama call (uses @mlflow.trace if available)
def ask_model_traced(context: str, question: str, model: str = "phi"):
    """
    Calls the local Ollama query and attempts to record a trace if MLflow tracing is available.
    Returns the raw response string.
    """
    if HAS_MLFLOW and hasattr(mlflow, "trace"):
        @mlflow.trace
        def _inner(ctx, q, m):
            return query_ollama_local(ctx, q, model=m, concise=False)
        try:
            out = _inner(context, question, model)
            # _inner may return a dict/str depending on generation; ensure string
            return out if isinstance(out, str) else str(out)
        except Exception:
            # fallback
            return query_ollama_local(context, question, model=model, concise=False)
    else:
        return query_ollama_local(context, question, model=model, concise=False)
    
mlflow.set_tracking_uri("http://127.0.0.1:5000")
experiment_name = "/Shared/dataspark"

try:
    mlflow.set_experiment(experiment_name)
except Exception:
    exp_id = mlflow.create_experiment(experiment_name, artifact_location="./mlruns")
    mlflow.set_experiment(experiment_name)


# ---------------------------------------------------------------------------
# File readers -> pandas.DataFrame
# ---------------------------------------------------------------------------
def read_generic(path: str, fmt: str = "csv", header=True, infer=True) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input not found: {path}")
    fmt = fmt.lower()
    if fmt == "csv":
        return pd.read_csv(path, header=0 if header else None)
    if fmt == "json":
        return pd.read_json(path)
    if fmt == "parquet":
        return pd.read_parquet(path)
    if fmt == "xlsx":
        return pd.read_excel(path)
    if fmt == "pdf":
        if fitz is None:
            raise ImportError("PyMuPDF (fitz) not installed. pip install pymupdf")
        doc = fitz.open(path)
        pages = [pg.get_text() for pg in doc]
        return pd.DataFrame({"text": pages})
    if fmt == "docx":
        if docx is None:
            raise ImportError("python-docx not installed. pip install python-docx")
        D = docx.Document(path)
        paras = [p.text for p in D.paragraphs if p.text.strip()]
        return pd.DataFrame({"text": paras})
    if fmt == "odt":
        if odf_load is None:
            raise ImportError("odfpy not installed. pip install odfpy")
        o = odf_load(path)
        paras = o.getElementsByType(odf_text.P)
        texts = [teletype.extractText(p) for p in paras]
        texts = [t for t in texts if t.strip()]
        return pd.DataFrame({"text": texts})
    raise ValueError(f"Unsupported format: {fmt}")

# ---------------------------------------------------------------------------
# Dynammic Loader
# ---------------------------------------------------------------------------
def build_and_run_dynamic(model="phi", concise=True, mlflow_exp: str = "/Shared/dataspark"):
    print(" Starting DataSpark dynamic mode...")
    query = input("Ask your question: ").strip()
    intent, score = guess_intent(query)
    print(f"Detected domain: {intent} (score={score:.2f})")

    workspaces = load_workspaces()
    tables = workspaces.get(intent, {}).get("tables", [])
    if not tables:
        print("No predefined tables found for this domain; please specify dataset path(s).")
        paths = input("Enter comma-separated file paths: ").split(",")
        tables = [p.strip() for p in paths if p.strip()]

    print(f"Candidate tables: {tables}")

    dfs = []
    for t in tables:
        if not os.path.exists(t):
            print(f" File not found: {t} — skipping.")
            continue
        # determine format by extension
        ext = t.split(".")[-1].lower()
        fmt = ext if ext in {"csv", "json", "parquet", "xlsx", "pdf", "docx", "odt"} else "csv"
        try:
            df = read_generic(t, fmt=fmt)
        except Exception as e:
            print(f"Failed to read {t}: {e}; skipping.")
            continue
        # if it's a DataFrame with a single 'text' column (e.g., PDF), don't prune columns
        if "text" in df.columns and df.shape[1] == 1:
            pruned_df = df
            scores = [("text", 1.0)]
        else:
            pruned_df, scores = prune_columns(df, query, top_k=10)
        print(f"Using columns for {t}: {[c for c, _ in scores[:10]]}")
        dfs.append(pruned_df)

    if not dfs:
        raise RuntimeError("No data available after pruning.")
    df = pd.concat(dfs, ignore_index=True, sort=False)

    docs = dataframe_to_docs(df, chunking=True)
    vstore = TfidfVectorStore(docs)

    # Safe MLflow logging
    mlflow_ok = init_mlflow(experiment_name=mlflow_exp) if HAS_MLFLOW else False
    run_id = None
    if mlflow_ok:
        try:
            input_files_param = ORIGINAL_INPUT_FILES or ",".join(tables)
            run_id = safe_mlflow_log_index(input_files_param, docs, vstore, run_name="dataspark_index_dynamic")
            print(f"Logged MLflow run: {run_id}")
        except Exception as e:
            print(f"⚠️ MLflow logging skipped: {e}")

    # Retrieval & ask model
    question = query
    retrieved = vstore.search(question, top_k=5)
    # debug: show what was retrieved
    print("Retrieved (score, snippet):")
    for r in retrieved:
        print(f" - {r['score']:.4f} {r['text'][:200].replace('\\n',' ')}")

    # Use traced caller if possible
    ans = ask_model_traced("\n".join([r["text"] for r in retrieved]), question, model=model)
    # enforce concise behavior
    final_ans = _shorten_answer(ans, max_sentences=2) if concise else ans
    print(" Answer:", final_ans)
    return {"df": df, "vstore": vstore, "mlflow_run": run_id}

# ---------------------------------------------------------------------------
# Chunking utilities
# ---------------------------------------------------------------------------
def chunk_text(text: str, chunk_size: int = 800, overlap: int = 200) -> List[str]:
    if not text:
        return []
    chunks = []
    start = 0
    L = len(text)
    while start < L:
        end = min(start + chunk_size, L)
        chunks.append(text[start:end])
        if end >= L:
            break
        start = end - overlap
        if start < 0:
            start = 0
    return chunks

def dataframe_to_docs(df: pd.DataFrame, text_cols: List[str] = None, chunking: bool = True, chunk_size: int = 800, overlap: int = 200) -> List[Dict]:
    text_cols = text_cols or list(df.columns)
    docs = []
    for idx, row in df.iterrows():
        parts = []
        for c in text_cols:
            if c not in row:
                continue
            val = row[c]
            if pd.isna(val):
                continue
            parts.append(f"{c}: {val}")
        row_text = " | ".join(parts)
        if chunking:
            for ch in chunk_text(row_text, chunk_size=chunk_size, overlap=overlap):
                docs.append({"text": ch, "meta": {"_row": int(idx)}})
        else:
            docs.append({"text": row_text, "meta": {"_row": int(idx)}})
    return docs

# ---------------------------------------------------------------------------
# Vector stores
# ---------------------------------------------------------------------------
class BaseVectorStore:
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        raise NotImplementedError()

class TfidfVectorStore(BaseVectorStore):
    def __init__(self, docs: List[Dict]):
        self.docs = docs
        self.texts = [d["text"] for d in docs]
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.vectors = self.vectorizer.fit_transform(self.texts)
    def search(self, query: str, top_k: int = 5):
        qv = self.vectorizer.transform([query])
        sims = cosine_similarity(qv, self.vectors).flatten()
        idxs = sims.argsort()[-top_k:][::-1]
        results = []
        for i in idxs:
            results.append({"score": float(sims[i]), "text": self.docs[i]["text"], "meta": self.docs[i]["meta"]})
        return results

# Faiss+SBERT omitted here for brevity; you can plug in your FaissSentenceVectorStore if available.

# ---------------------------------------------------------------------------
# Ollama local query
# ---------------------------------------------------------------------------
def query_ollama_local(context: str, question: str, model: str = "phi", concise: bool = True, timeout: int = 60) -> str:
    url = "http://localhost:11434/api/generate"
    if concise:
        style = "Answer in ONE short paragraph (max 2–3 sentences). If not found in dataset reply: 'Not found in dataset.'"
    else:
        style = "Answer with a clear explanation using the provided context. If not found in dataset reply: 'Not found in dataset.'"
    payload = {"model": model, "prompt": f"Context:\\n{context}\\n\\nQuestion: {question}\\n\\n{style}\\nAnswer:"}
    try:
        r = requests.post(url, json=payload, timeout=timeout, stream=True)
        output = ""
        for line in r.iter_lines():
            if line:
                try:
                    part = json.loads(line.decode("utf-8"))
                    output += part.get("response", "")
                except Exception:
                    try:
                        output += line.decode("utf-8")
                    except Exception:
                        pass
        return output.strip()
    except Exception as e:
        return f"[LLM ERROR] {e}"

# ---------------------------------------------------------------------------
# Relevance detection
# ---------------------------------------------------------------------------
def is_relevant(q: str, vstore: BaseVectorStore, rel_threshold: float = 0.08):
    try:
        res = vstore.search(q, top_k=1)
        if not res:
            return False, 0.0
        best = float(res[0].get("score", 0.0))
        return (best >= rel_threshold), best
    except Exception:
        return False, 0.0
    
# ---------------------------------------------------------------------------
# Embedded Universal Agents (Intent, Column Prune, Semantic Layer)
# These make the system independent of topic or external files (like YAML or .py)
# ---------------------------------------------------------------------------

def guess_intent(query: str):
    """
    Universal, topic-agnostic intent detector.
    Classifies queries based on *type of question* rather than domain.
    Works for any dataset.
    """
    q = query.lower().strip()

    # Simple NLP-style rule-based intent classification
    if any(w in q for w in ["how many", "total", "count", "number of"]):
        return "quantitative", 0.9
    elif any(w in q for w in ["average", "mean", "maximum", "minimum", "highest", "lowest", "sum", "rate"]):
        return "analytical", 0.85
    elif any(w in q for w in ["list", "show", "display", "find all", "which", "give me", "what are"]):
        return "descriptive", 0.8
    elif any(w in q for w in ["compare", "trend", "change", "difference", "growth", "increase", "decrease"]):
        return "comparative", 0.8
    elif any(w in q for w in ["hi", "hello", "hey", "thanks", "bye", "greetings"]):
        return "conversational", 0.6
    else:
        return "general", 0.4


def prune_columns(df: pd.DataFrame, query: str, top_k: int = 10):
    """
    Automatically selects the most relevant columns based on semantic similarity
    between the query and column names using TF-IDF. Works for any dataset.
    """
    cols = df.columns.tolist()
    if not cols:
        return df, []

    from sklearn.feature_extraction.text import TfidfVectorizer
    import numpy as np

    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(cols + [query])
    sims = (matrix[:-1] @ matrix[-1].T).toarray().ravel()

    ranked = sorted(zip(cols, sims), key=lambda x: x[1], reverse=True)
    top_cols = [c for c, _ in ranked[:top_k]]

    # Return the pruned DataFrame and the similarity scores
    return df[top_cols], ranked


def load_workspaces(path=None):
    """
    Universal workspace loader (no YAML, no domain lock-in).
    Dynamically discovers datasets in the working directory or ./data/
    and logs them to MLflow for full experiment traceability.
    """
    import glob

    workspace = {"general": {"tables": [], "examples": []}}

    # Look for data files
    possible_dirs = [os.getcwd(), os.path.join(os.getcwd(), "data")]
    supported_exts = ("*.csv", "*.json", "*.xlsx", "*.parquet")

    found_files = []
    for d in possible_dirs:
        for ext in supported_exts:
            found_files.extend(glob.glob(os.path.join(d, ext)))

    # Populate workspace with detected files
    workspace["general"]["tables"] = [os.path.basename(f) for f in found_files]
    workspace["general"]["examples"] = [
        "Show me a summary of the dataset.",
        "What are the top entries?",
        "Find trends in the data.",
        "Which column has the highest values?",
        "List all unique categories.",
        "Show average or total values by group."
    ]

    # ✅ Log detected datasets to MLflow for experiment traceability
    if HAS_MLFLOW and found_files:
        try:
            mlflow.set_tag("input_files", ", ".join(os.path.basename(f) for f in found_files))
        except Exception:
            pass

    # Also print to console for user clarity
    if found_files:
        print(f"Detected datasets: {', '.join(os.path.basename(f) for f in found_files)}")
    else:
        print("No datasets found in current directory or .")

    return workspace


# ---------------------------------------------------------------------------
# Terminal chat loop (fixed TTS placement)
# ---------------------------------------------------------------------------
SMALLTALK = {"hi", "hello", "hey", "yo", "sup", "good morning", "good evening", "good afternoon", "hiya"}

def is_smalltalk(q: str) -> bool:
    """Return True if the query is a casual greeting or small-talk."""
    qn = q.strip().lower()
    return any(qn.startswith(s) for s in SMALLTALK)

def _shorten_answer(answer: str, max_sentences: int = 2) -> str:
    """Return up to max_sentences sentences from answer (naive sentence splitter)."""
    if not answer:
        return ""
    # Split on ., ?, ! followed by space or end — keep punctuation.
    parts = re.split(r'(?<=[\\.?\\!])\\s+', answer.strip())
    if len(parts) <= max_sentences:
        return answer.strip()
    return " ".join(parts[:max_sentences]).strip()

def terminal_chat_loop(vstore: BaseVectorStore, model="phi", concise=True, tts=False,
                       rel_threshold=0.08, voice_gender="female"):
    """Interactive terminal chat with RAG + guaranteed TTS + concise-answer truncation.

    Usage: terminal_chat_loop(vstore, model='phi', concise=True, tts=True, voice_gender='female')
    """
    if console:
        console.print("\\n[bold cyan]Vector RAG Chatbot Ready[/bold cyan] — type your question or exit\\n")
    else:
        print("Vector RAG Chatbot Ready — type your question or exit")

    # show chosen TTS voice once (best-effort)
    chosen_voice_name = None
    if tts and HAS_TTS:
        try:
            tmp = pyttsx3.init()
            voices = tmp.getProperty("voices")
            preferred = (voice_gender or "female").lower()
            chosen = None
            if preferred == "default":
                chosen = voices[0] if voices else None
            else:
                for v in voices:
                    vn = (v.name or "").lower()
                    vi = (getattr(v, "id", "") or "").lower()
                    if preferred == "female" and ("female" in vn or "zira" in vn or "samantha" in vn or "mary" in vn):
                        chosen = v; break
                    if preferred == "male" and ("male" in vn or "david" in vn or "mark" in vn or "paul" in vn):
                        chosen = v; break
                if not chosen:
                    chosen = voices[0] if voices else None
            if chosen:
                chosen_voice_name = getattr(chosen, "name", getattr(chosen, "id", "unknown"))
            tmp.stop()
            del tmp
        except Exception:
            chosen_voice_name = None

    if tts and chosen_voice_name:
        if console:
            console.print(f"[bold magenta]TTS voice selected:[/bold magenta] {chosen_voice_name}")
        else:
            print("TTS voice selected:", chosen_voice_name)

    def _show_panel(msg: str, title: str = "[green]BigQueryGPT[/green]", style: str = "bold cyan"):
        txt = Text(msg.rstrip(), overflow="fold", no_wrap=False)
        panel_width = None
        try:
            if console:
                panel_width = max(40, console.size.width - 4)
        except Exception:
            panel_width = None
        panel = Panel.fit(txt, title=title, border_style="bright_magenta", width=panel_width)
        if console:
            console.print(panel)
        else:
            print(f"{title}\\n{msg}\\n")

    def _speak(text_to_say: str):
        if not (tts and HAS_TTS and text_to_say):
            return
        try:
            local_engine = pyttsx3.init()
            voices = local_engine.getProperty("voices")
            preferred = (voice_gender or "female").lower()
            chosen = None
            for v in voices:
                vn = (v.name or "").lower()
                if preferred == "female" and ("female" in vn or "zira" in vn or "samantha" in vn or "mary" in vn):
                    chosen = v; break
                if preferred == "male" and ("male" in vn or "david" in vn or "mark" in vn or "paul" in vn):
                    chosen = v; break
            if chosen:
                try:
                    local_engine.setProperty("voice", chosen.id)
                except Exception:
                    pass
            local_engine.say(text_to_say)
            local_engine.runAndWait()
            local_engine.stop()
            del local_engine
        except Exception:
            # don't crash TTS failures — just continue
            pass

    while True:
        try:
            q = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            goodbye_text = "Goodbye!"
            _show_panel(goodbye_text)
            _speak(goodbye_text)
            break

        if not q:
            continue

        if q.lower() in ("exit", "quit", "bye"):
            goodbye_text = "Goodbye! Talk later."
            _show_panel(goodbye_text)
            _speak(goodbye_text)
            break

        # quick smalltalk
        if is_smalltalk(q):
            greeting = "Hi there! Ask me something about your dataset."
            _show_panel(greeting)
            _speak(greeting)
            continue

        # relevance check
        is_rel, best_score = is_relevant(q, vstore, rel_threshold)
        if not is_rel:
            reply = f"That question doesn’t seem related to the dataset (score={best_score:.3f}). Try another query."
            _show_panel(reply, style="bold yellow")
            _speak(reply)
            continue

        # retrieve, ask LLM
        retrieved = vstore.search(q, top_k=5)
        context = "\n".join([r["text"] for r in retrieved])
        raw_answer = query_ollama_local(context, q, model=model, concise=concise)

        # enforce concise post-processing (max 2 sentences) if requested
        final_answer = _shorten_answer(raw_answer, max_sentences=2) if concise else raw_answer

        # ensure we don't print extra blank lines or stray characters
        final_answer = final_answer.strip() or "No response."

        # show & speak
        _show_panel("🤖 " + final_answer)
        _speak(final_answer)

# ---------------------------------------------------------------------------
# Minimal pipeline builder and CLI (with MLflow logging)
# ---------------------------------------------------------------------------
def build_and_run(input_path: str, input_format: str = "csv", header: bool = True,
                  chunking: bool = True, chunk_size: int = 800, chunk_overlap: int = 200,
                  embedding_backend: str = "auto", model: str = "phi", ui: str = "terminal",
                  concise: bool = True, mlflow_exp: str = "/Shared/dataspark", tts: bool = False,
                  limit_rows: int = 5000, relevance_threshold: float = 0.08, voice_gender: str = "female"):

    df = read_generic(input_path, fmt=input_format, header=header, infer=True)
    if limit_rows and len(df) > limit_rows:
        df = df.head(limit_rows)

    docs = dataframe_to_docs(df, text_cols=None, chunking=chunking, chunk_size=chunk_size, overlap=chunk_overlap)
    if not docs:
        raise RuntimeError("No docs extracted from data")

    # Use TF-IDF vector store (Faiss/SBERT optional)
    vstore = TfidfVectorStore(docs)
    print(f"Vector store built: tfidf. Docs: {len(docs)}")

    # MLflow logging (safe)
    mlflow_run_id = None
    if HAS_MLFLOW:
        try:
            init_mlflow(experiment_name=mlflow_exp)
            input_files_param = ORIGINAL_INPUT_FILES or input_path
            mlflow_run_id = safe_mlflow_log_index(input_files_param, docs, vstore, run_name="dataspark_index_build")

            if mlflow_run_id:
                print(f"Logged MLflow run: {mlflow_run_id}")
        except Exception as e:
            print(f" MLflow logging skipped: {e}")

    # Launch
    if ui == "terminal":
        terminal_chat_loop(vstore, model=model, concise=concise, tts=tts,
                           rel_threshold=relevance_threshold, voice_gender=voice_gender)
    else:
        raise ValueError("Only terminal UI supported in this simplified version.")

    return {"df": df, "vstore": vstore, "mlflow_run_id": mlflow_run_id}

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=False, help="Path to input file (optional in --dynamic mode)")
    p.add_argument("--input-format", default="csv", choices=["csv","json","parquet","xlsx","pdf","docx","odt"])
    p.add_argument("--header", action="store_true")
    p.add_argument("--chunking", action="store_true")
    p.add_argument("--chunk-size", type=int, default=800)
    p.add_argument("--overlap", type=int, default=200)
    p.add_argument("--model", default="phi")
    p.add_argument("--tts", action="store_true")
    p.add_argument("--voice-gender", default="female", choices=["male","female","default"])
    p.add_argument("--limit-rows", type=int, default=5000)
    p.add_argument("--relevance-threshold", type=float, default=0.08)
    p.add_argument("--concise", action="store_true", help="Force concise (short) answers.")
    p.add_argument("--dynamic", action="store_true", help="Enable dynamic intent/workspace-driven mode.")
    p.add_argument("--mlflow-exp", default="/Shared/dataspark", help="MLflow experiment name (optional)")
    return p.parse_args()

def main():
    args = parse_args()
    if args.dynamic:
        build_and_run_dynamic(model=args.model, concise=args.concise, mlflow_exp=args.mlflow_exp)
        return

    if not args.input:
        raise ValueError("Please provide --input path or use --dynamic mode.")
    
    if os.path.isdir(args.input):
        folder = args.input
        print(f"Detected folder input: {folder}")
        csv_files = [str(p) for p in Path(folder).glob("*.csv")]
        if not csv_files:
            raise ValueError("No CSV files found in the folder.")
        global ORIGINAL_INPUT_FILES
        ORIGINAL_INPUT_FILES = ",".join([os.path.basename(f) for f in csv_files])
        dfs = [pd.read_csv(f) for f in csv_files]
        merged_df = pd.concat(dfs, ignore_index=True)
        merged_path = os.path.join(folder, "_merged_temp.csv")
        merged_df.to_csv(merged_path, index=False)
        args.input = merged_path

    build_and_run(
        input_path=args.input,
        input_format=args.input_format,
        header=args.header,
        chunking=args.chunking,
        chunk_size=args.chunk_size,
        chunk_overlap=args.overlap,
        model=args.model,
        ui="terminal",
        concise=args.concise,
        mlflow_exp=args.mlflow_exp,
        tts=args.tts,
        limit_rows=args.limit_rows,
        relevance_threshold=args.relevance_threshold,
        voice_gender=args.voice_gender
    )

if __name__ == "__main__":
    main()
