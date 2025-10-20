#!/usr/bin/env python3
# dataset.py — PySpark + Ollama + TF-IDF Vector RAG chatbot (dataset-agnostic, Rich UI)

import os
import sys
import argparse
import requests
import json
import pandas as pd
import time
import openpyxl  
import fitz  
import docx  
from odf import text, teletype
from odf.opendocument import load
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from pyspark.sql import SparkSession
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

console = Console()

# ---- Windows fixes -----------------------------------------------------------
if os.name == "nt":
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
    os.environ["TEMP"] = r"C:\\Temp"
    os.environ["TMP"] = r"C:\\Temp"
    os.environ["SPARK_LOCAL_DIRS"] = r"C:\\Temp\\spark"

# ---- Spark session -----------------------------------------------------------
def build_spark(app_name: str, shuffle: int):
    return (
        SparkSession.builder
        .appName(app_name)
        .master("local[*]")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.shuffle.partitions", str(shuffle))
        .getOrCreate()
    )

# ---- Input Reader ------------------------------------------------------------
def read_input(spark, path, fmt, header=False, infer=False):
    fmt = fmt.lower()
    if fmt == "csv":
        return (
            spark.read.option("header", str(header).lower())
            .option("inferSchema", str(infer).lower())
            .csv(path)
        )
    if fmt == "json":
        return spark.read.json(path)
    if fmt == "parquet":
        return spark.read.parquet(path)
    if fmt == "xlsx":
        # Load with pandas, then convert to Spark DataFrame
        pdf = pd.read_excel(path)
        return spark.createDataFrame(pdf)
    if fmt == "pdf":
        try:
            import fitz  
            doc = fitz.open(path)
            text = "\n".join([page.get_text() for page in doc])
        except ImportError:
            raise ImportError("Please install PyMuPDF: pip install pymupdf")
        return spark.createDataFrame(pd.DataFrame({"text": [text]}))
    if fmt == "docx":
        try:
            from docx import Document
            doc = Document(path)
            text = "\n".join([p.text for p in doc.paragraphs])
        except ImportError:
            raise ImportError("Please install python-docx: pip install python-docx")
        return spark.createDataFrame(pd.DataFrame({"text": [text]}))
    if fmt == "odt":
        try:
            from odf import text, teletype
            from odf.opendocument import load 
            doc = load(path)
            texts = doc.getElementsByType(text.P)
            text_content = "\n".join([teletype.extractText(t) for t in texts])
        except ImportError:
            raise ImportError("Please install odfpy: pip install odfpy")
        return spark.createDataFrame(pd.DataFrame({"text": [text_content]}))
    raise ValueError(f"Unsupported format: {fmt}")

# ---- Vector Store (TF-IDF) ---------------------------------------------------
class VectorStore:
    def __init__(self, df):
        self.df = df
        # Limit rows for speed
        self.pdf = df.limit(5000).toPandas().fillna("")
        # Flatten each row into a text string
        self.docs = self.pdf.astype(str).agg(" | ".join, axis=1).tolist()

        # Build TF-IDF vectors
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.vectors = self.vectorizer.fit_transform(self.docs)

    def search(self, query, top_k=5):
        """Return top-k most relevant rows to query"""
        q_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self.vectors).flatten()
        top_idx = sims.argsort()[-top_k:][::-1]
        return [self.docs[i] for i in top_idx]

# ---- Ollama Query (softer grounding) ----------------------------------------
def query_ollama(context, question, model="phi", concise=False):
    url = "http://localhost:11434/api/generate"

    if concise:
        style = (
            "Answer in ONE short paragraph (max 2–3 sentences). "
            "If the answer cannot be found in the dataset, reply exactly: 'Not found in dataset.'"
        )
    else:
        style = (
            "Give a clear but detailed explanation using only the dataset context. "
            "You may use reasoning and examples, but stay focused on the data."
        )

    payload = {
        "model": model,
        "prompt": (
            f"You are a dataset assistant. "
            f"ONLY use the context below to answer.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            f"{style}\nAnswer:"
        )
    }

    response = requests.post(url, json=payload, stream=True)
    output = ""
    for line in response.iter_lines():
        if line:
            try:
                data = json.loads(line.decode("utf-8"))
                output += data.get("response", "")
            except json.JSONDecodeError:
                pass
    return output.strip()



# ---- Chatbot Loop ------------------------------------------------------------
def rag_chatbot(vstore, model="phi", concise=False):
    console.print(
        "\n🤖 [bold cyan]Vector RAG Chatbot Ready[/bold cyan] — type your question, or 'exit' to quit\n",
        style="bold green"
    )

    while True:
        user_q = console.input("[bold yellow]You:[/bold yellow] ")

        if user_q.lower() in ["exit", "quit", "bye"]:
            console.print("[bold red]🤖: Goodbye![/bold red]\n")
            break

        # Retrieve relevant rows
        retrieved = vstore.search(user_q, top_k=5)
        context = "\n".join(retrieved)

        # Ask Ollama
        answer = query_ollama(context, user_q, model=model, concise=concise)

        panel = Panel.fit(
            Text("🤖 " + answer, style="bold cyan"),
            title="[green]AI Assistant[/green]",
            border_style="bright_magenta"
        )
        console.print(panel)


# ---- CLI ---------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--input-format", default="csv", choices=["csv", "json", "parquet", "xlsx", "pdf", "docx", "odt"])
    p.add_argument("--header", action="store_true")
    p.add_argument("--infer-schema", action="store_true")
    p.add_argument("--shuffle-partitions", type=int, default=200)
    p.add_argument("--model", default="phi", help="Ollama model (default: phi, fast)")
    p.add_argument("--concise", action="store_true", help="Force short answers (max 2–3 sentences)")
    return p.parse_args()


# ---- Main --------------------------------------------------------------------
def main():
    args = parse_args()
    spark = build_spark("DatasetRAG", args.shuffle_partitions)

    df = read_input(spark, args.input, args.input_format, args.header, args.infer_schema)
    vstore = VectorStore(df)

    print("Done ✅ Dataset loaded, vector index built.\n")
    rag_chatbot(vstore, model=args.model)

    spark.stop()

if __name__ == "__main__":
    main()

