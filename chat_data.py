import streamlit as st
import requests
import json
import nest_asyncio
import uuid
import re
from llama_index.core import VectorStoreIndex
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.schema import TextNode

nest_asyncio.apply()

st.set_page_config(page_title="DNB Lab Chat", layout="wide")
st.title("DNB Lab Chat")

GITHUB_JSON_URL = "https://raw.githubusercontent.com/anketaube/DNBLabChat/main/dnblab_index.json"

def is_valid_id(id_value):
    return isinstance(id_value, str) and id_value.strip() != ""

def extract_url_from_text(text):
    match = re.search(r'(https?://[^\s]+)', text)
    if match:
        return match.group(1)
    return ""

def load_nodes_from_json(data):
    nodes = []
    for entry in data:
        text = entry.get("text", "")
        id_val = entry.get("id")
        metadata = entry.get("metadata", {})
        if not isinstance(text, str) or not text.strip():
            continue
        if not is_valid_id(id_val):
            id_val = str(uuid.uuid4())
        if not metadata.get("source"):
            url = extract_url_from_text(text)
            metadata["source"] = url if url else ""
        node = TextNode(
            text=text,
            metadata=metadata,
            id_=id_val
        )
        nodes.append(node)
    return nodes

def fetch_index_from_github():
    r = requests.get(GITHUB_JSON_URL)
    if r.status_code == 200:
        data = r.json()
        nodes = load_nodes_from_json(data)
        if not nodes:
            st.error("Der geladene Index enthält keine gültigen Text-Chunks.")
            return None, []
        index = VectorStoreIndex(nodes)
        return index, nodes
    else:
        st.error("Konnte Index nicht von GitHub laden.")
        return None, []

def index_to_rich_json(nodes):
    export = []
    for node in nodes:
        if is_valid_id(node.node_id) and node.text and node.text.strip():
            export.append({
                "id": node.node_id,
                "text": node.text,
                "metadata": node.metadata,
            })
    return json.dumps(export, ensure_ascii=False, indent=2)

# --- Session-State-Initialisierung ---
for key, default in [
    ("index", None),
    ("nodes", []),
    ("chat_history", []),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# --- Nur beim Start: Index aus GitHub laden ---
if not st.session_state.nodes:
    with st.spinner("Lade Index aus GitHub..."):
        index, nodes = fetch_index_from_github()
        if index:
            st.session_state.index = index
            st.session_state.nodes = nodes
            st.success(f"Index aus GitHub geladen ({len(nodes)} Chunks).")

# --- SIDEBAR: Download des aktuellen Index ---
with st.sidebar:
    st.header("Index herunterladen")
    if st.session_state.nodes:
        json_data = index_to_rich_json(st.session_state.nodes)
        st.download_button(
            label="Aktuellen Index als JSON herunterladen",
            data=json_data,
            file_name="dnblab_index.json",
            mime="application/json"
        )

# --- HAUPTBEREICH: Chat ---
st.header("Chat mit dem Index")
st.write("Stelle Fragen zum geladenen Index. Quellen werden bei der Antwort angezeigt. Nachhaken ist möglich.")

if st.session_state.index:
    # Nur Eingabefeld, solange noch keine Antwort im Verlauf
    if not st.session_state.chat_history:
        user_input = st.text_input("Frage an den Index stellen:")
        if user_input:
            response = st.session_state.index.as_query_engine(similarity_top_k=3).query(user_input)
            antwort = response.response if hasattr(response, "response") else str(response)
            unique_sources = set(
                node.metadata.get('source', 'unbekannt')
                for node in response.source_nodes
                if node.metadata.get('source', '').strip()
            )
            st.session_state.chat_history.append({
                "frage": user_input,
                "antwort": antwort,
                "quellen": list(unique_sources)
            })
            st.write("Antwort:")
            st.write(antwort)
            st.write("Quellen:")
            if unique_sources:
                for source in unique_sources:
                    st.write(f"- {source}")
            else:
                st.write("Keine Quelle im Chunk-Metadatum gefunden.")
    else:
        # Nach erster Antwort: Chatverlauf und Nachhaken
        user_input = st.text_input("Frage an den Index stellen:", key="maininput")
        if user_input:
            response = st.session_state.index.as_query_engine(similarity_top_k=3).query(user_input)
            antwort = response.response if hasattr(response, "response") else str(response)
            unique_sources = set(
                node.metadata.get('source', 'unbekannt')
                for node in response.source_nodes
                if node.metadata.get('source', '').strip()
            )
            st.session_state.chat_history.append({
                "frage": user_input,
                "antwort": antwort,
                "quellen": list(unique_sources)
            })
            st.write("Antwort:")
            st.write(antwort)
            st.write("Quellen:")
            if unique_sources:
                for source in unique_sources:
                    st.write(f"- {source}")
            else:
                st.write("Keine Quelle im Chunk-Metadatum gefunden.")

        st.subheader("Chatverlauf dieser Sitzung")
        for i, entry in enumerate(st.session_state.chat_history):
            st.markdown(f"**Frage {i+1}:** {entry['frage']}")
            st.markdown(f"**Antwort:** {entry['antwort']}")
            st.markdown("**Quellen:**")
            for source in entry["quellen"]:
                st.markdown(f"- {source}")

        # Nachhaken-Feld
        st.subheader("Nachhaken zur letzten Antwort")
        followup = st.text_input("Nachhaken (bezieht sich auf die letzte Antwort):", key="followup")
        if followup:
            last_answer = st.session_state.chat_history[-1]["antwort"]
            followup_prompt = f"Vorherige Antwort: {last_answer}\nNachhaken: {followup}"
            response = st.session_state.index.as_query_engine(similarity_top_k=3).query(followup_prompt)
            antwort = response.response if hasattr(response, "response") else str(response)
            unique_sources = set(
                node.metadata.get('source', 'unbekannt')
                for node in response.source_nodes
                if node.metadata.get('source', '').strip()
            )
            st.write("Antwort auf Nachhaken:")
            st.write(antwort)
            st.write("Quellen:")
            if unique_sources:
                for source in unique_sources:
                    st.write(f"- {source}")
            else:
                st.write("Keine Quelle im Chunk-Metadatum gefunden.")
            st.session_state.chat_history.append({
                "frage": followup_prompt,
                "antwort": antwort,
                "quellen": list(unique_sources)
            })
else:
    st.info("Der Index wird beim Start automatisch aus GitHub geladen. Sobald er bereit ist, kannst du den Chat nutzen.")
