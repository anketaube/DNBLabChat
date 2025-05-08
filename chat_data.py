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

def create_rich_nodes(urls):
    from llama_index.readers.web import TrafilaturaWebReader
    documents = TrafilaturaWebReader().load_data(urls)
    parser = SimpleNodeParser()
    nodes = []
    for url, doc in zip(urls, documents):
        doc.metadata["source"] = url
        doc.metadata["title"] = doc.metadata.get("title", "")
        for node in parser.get_nodes_from_documents([doc]):
            node_id = node.node_id if is_valid_id(node.node_id) else str(uuid.uuid4())
            if node.text and node.text.strip():
                chunk_metadata = dict(node.metadata)
                chunk_metadata["source"] = url
                nodes.append(TextNode(
                    text=node.text,
                    metadata=chunk_metadata,
                    id_=node_id
                ))
    return nodes

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
    ("index_source", "github"),
    ("chat_history", []),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# --- Initiales Laden aus GitHub beim Start (nur, wenn noch kein Index geladen ist) ---
if not st.session_state.nodes and st.session_state.index_source == "github":
    with st.spinner("Lade Index aus GitHub..."):
        index, nodes = fetch_index_from_github()
        if index:
            st.session_state.index = index
            st.session_state.nodes = nodes
            st.success(f"Index aus GitHub geladen ({len(nodes)} Chunks).")

# --- SIDEBAR: Index-Funktionen ---
with st.sidebar:
    st.header("Index-Funktionen")
    st.write("1. **URLs indexieren und als neue JSON speichern**\n2. **URLs zum bestehenden Index hinzufügen**\n3. **Kombinierten Index herunterladen**")
    st.markdown("---")

    # 1. Neue JSON nur aus URLs bauen
    st.subheader("Nur neue URLs zu JSON (ohne Zusammenführen)")
    urls_json_input = st.text_area("Neue URLs für eigenen Index (eine pro Zeile):", key="json_urls")
    urls_json = [u.strip() for u in urls_json_input.split('\n') if u.strip()]
    if urls_json and st.button("Neuen Index aus diesen URLs als JSON erstellen"):
        with st.spinner("Erzeuge neuen Index..."):
            nodes = create_rich_nodes(urls_json)
            if nodes:
                json_data = index_to_rich_json(nodes)
                st.download_button(
                    label="Nur neuen Index als JSON herunterladen",
                    data=json_data,
                    file_name="dnblab_index_neu.json",
                    mime="application/json"
                )
            else:
                st.warning("Keine gültigen Chunks aus den URLs extrahiert.")

    st.markdown("---")

    # 2. URLs zum bestehenden Index hinzufügen
    st.subheader("Neue URLs zum bestehenden Index hinzufügen")
    urls_input = st.text_area("Neue URLs (eine pro Zeile):", key="add_urls")
    urls = [u.strip() for u in urls_input.split('\n') if u.strip()]
    if urls and st.button("Neue URLs indexieren und hinzufügen"):
        with st.spinner("Neue URLs werden indexiert..."):
            new_nodes = create_rich_nodes(urls)
            existing_ids = set(n.node_id for n in st.session_state.nodes)
            existing_texts = set(n.text for n in st.session_state.nodes)
            combined_nodes = st.session_state.nodes.copy()
            added = 0
            for node in new_nodes:
                if node.node_id not in existing_ids and node.text not in existing_texts:
                    combined_nodes.append(node)
                    added += 1
            if added:
                st.session_state.nodes = combined_nodes
                st.session_state.index = VectorStoreIndex(combined_nodes)
                st.success(f"{added} neue Chunks hinzugefügt! Gesamt: {len(combined_nodes)}.")
            else:
                st.info("Keine neuen Chunks hinzugefügt (möglicherweise waren sie schon im Index).")

    st.markdown("---")

    # 3. Download-Button für den kombinierten Index
    if st.session_state.nodes:
        json_data = index_to_rich_json(st.session_state.nodes)
        st.download_button(
            label="Kombinierten Index als JSON herunterladen",
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
