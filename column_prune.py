# column_prune.py
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np

def column_relevance_scores(df: pd.DataFrame, query: str, sample_per_col: int = 50):
    texts = []
    cols = []
    for c in df.columns:
        series = df[c].dropna().astype(str)
        if len(series) == 0:
            texts.append("")
            cols.append(c)
            continue
        sample = series.sample(min(len(series), sample_per_col)).tolist()
        col_text = " ".join(sample)
        texts.append(col_text)
        cols.append(c)
    vect = TfidfVectorizer(stop_words="english")
    try:
        vect.fit(texts + [query])
        qv = vect.transform([query])
        mat = vect.transform(texts)
        sims = (mat @ qv.T).toarray().flatten()
    except Exception:
        sims = [0.0] * len(cols)
    return list(zip(cols, sims))

def prune_columns(df: pd.DataFrame, query: str, top_k: int = 10):
    scores = column_relevance_scores(df, query)
    scores.sort(key=lambda x: x[1], reverse=True)
    selected = [c for c, s in scores[:top_k]]
    # Ensure selected exist in df
    selected = [c for c in selected if c in df.columns]
    return df[selected], scores
