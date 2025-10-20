#!/usr/bin/env python3
# chatbot.py — load Spark outputs and chat with Ollama

import os
import json
import requests
import pandas as pd

# ---- Ollama RAG helper ------------------------------------------------------
def query_ollama(context, question, model="llama2"):
    """Send context + question to Ollama for natural language Q&A."""
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model,
        "prompt": f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
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

def rag_chatbot(base_path="out", model="llama2"):
    """Interactive chatbot that answers questions using saved Spark outputs + Ollama."""
    dau_path = os.path.join(base_path, "dau")
    top_path = os.path.join(base_path, "top_products")

    # Load from parquet
    dau = pd.read_parquet(dau_path)
    top = pd.read_parquet(top_path)

    # Build context
    ctx_dau = dau.head(20).to_string(index=False)
    ctx_top = top.head(20).to_string(index=False)
    context = f"DAU & Revenue Trends:\n{ctx_dau}\n\nTop Products:\n{ctx_top}"

    print("\n[🤖 Ollama Data Chatbot Ready — type your question, or 'exit' to quit]\n")

    while True:
        user_q = input("You: ")
        if user_q.lower() in ["exit", "quit", "bye"]:
            print("🤖: Goodbye!")
            break
        answer = query_ollama(context, user_q, model=model)
        print(f"🤖: {answer}\n")


if __name__ == "__main__":
    rag_chatbot("out", model="llama2")
