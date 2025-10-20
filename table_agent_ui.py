# table_agent_ui.py - Streamlit snippet integrating intent, semantic, and table selection
import streamlit as st
import pandas as pd
from intent_agent import guess_intent
from semantic import load_workspaces
from column_prune import prune_columns
import os

st.set_page_config(page_title="DataSpark - Table Agent", layout="wide")
st.title("DataSpark — Table Agent UI (Demo)")

workspaces = load_workspaces()

st.markdown("Upload one or more CSV files (they will be listed as candidate tables). For demo, try the sample CSVs in your repo.")
uploaded = st.file_uploader("Upload CSV(s)", accept_multiple_files=True, type=["csv"])
file_map = {}
if uploaded:
    for u in uploaded:
        bytes_data = u.read()
        tmp = os.path.join("/tmp", u.name)
        with open(tmp, "wb") as f:
            f.write(bytes_data)
        file_map[u.name] = tmp

query = st.text_input("Ask a question to detect intent and suggest tables")
if st.button("Suggest Tables") and query.strip():
    intent, score = guess_intent(query)
    st.write(f"Detected domain: **{intent}** (score {score:.2f})")
    # suggested from workspace
    suggested = workspaces.get(intent, {}).get("tables", [])
    st.write("Suggested workspace tables:", suggested)

    # show uploaded tables and allow selection
    st.write("Uploaded tables:")
    selected = []
    for fname, path in file_map.items():
        cols = pd.read_csv(path, nrows=2).columns.tolist()
        if st.checkbox(f"Use {fname} (cols: {cols})", value=True, key=fname):
            selected.append((fname, path))

    if not selected:
        st.warning("No tables selected — please select at least one table.")
    else:
        st.success(f"Selected {len(selected)} tables. Pruning columns and previewing...")
        for fname, path in selected:
            df = pd.read_csv(path)
            pruned, scores = prune_columns(df, query, top_k=10)
            st.write(f"### {fname} — top columns:")
            st.write(pruned.head(5))
            st.write("Scores:", scores[:10])
