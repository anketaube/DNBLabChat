import streamlit as st
import requests
import json
import nest_asyncio
from llama_index.readers.web import TrafilaturaWebReader
from llama_index.core import VectorStoreIndex
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.schema import TextNode

nest_asyncio.apply()

GITHUB_JSON_URL = "https://raw.githubusercontent.com/anketaube/DNBLabChat/main/dnblab_index.json"

st.title("DNB Lab Chat – Volltext-Index, Chunking & Download")

# --- Session-State-Initialisierung ---
for key, default in [
    ("index", None),
    ("nodes", None),
    ("index_source", "github"),
    ("chat_history", []),
]:
    if key not in st.session_state:
        st.session_state[key] = default

def fetch_index_from_github():
    """Lade Index als JSON von GitHub und baue einen Index mit allen Chunks."""
    r = requests.get(GITHUB_JSON_URL)
    if r.status_code == 200:
        data = r.json()
        nodes = []
        for entry in data:
            # Nur Felder verwenden, die der TextNode-Konstruktor erwartet!
            node = TextNode(
                text=entry.get("text", ""),
                metadata=entry.get("metadata", {}),
                id_=entry.get("id")  # id_ statt id!
            )
            nodes.append(node)
        index = VectorStoreIndex(nodes)
        return index, nodes
    else:
        st.error("Konnte Index nicht von GitHub laden.")
        return None, None

def create_rich_index(urls):
    documents = TrafilaturaWebReader().load_data(urls)
    parser = SimpleNodeParser()
    nodes = []
    for doc in documents:
        doc.metadata["source"] = doc.metadata.get("url", "")
        doc.metadata["title"] = doc.metadata.get("title", "")
        nodes.extend(parser.get_nodes_from_documents([doc]))
    index = VectorStoreIndex(nodes)
    return index, nodes

def index_to_rich_json(nodes):
    export = []
    for node in nodes:
        export.append({
            "id": node.node_id,        # Speichere als "id"
            "text": node.text,
            "metadata": node.metadata,
        })
    return json.dumps(export, ensure_ascii=False, indent=2)

# URLs-Eingabe
st.subheader("Neue URLs indexieren (optional)")
urls_input = st.text_area("Gib neue URLs ein (eine pro Zeile):")
urls = [u.strip() for u in urls_input.split('\n') if u.strip()]

# Nur wenn URLs eingegeben werden, neuen Index bauen
if urls and st.button("Neuen Index erstellen"):
    with st.spinner("Index wird erstellt..."):
        index, nodes = create_rich_index(urls)
        st.session_state.index = index
        st.session_state.nodes = nodes
        st.session_state.index_source = "custom"
        st.success(f"Neuer Index mit {len(nodes)} Chunks aus {len(urls)} URLs erstellt!")
        # Download-Button für neuen Index
        json_data = index_to_rich_json(nodes)
        st.download_button(
            label="Neuen Index als JSON herunterladen",
            data=json_data,
            file_name="dnblab_index.json",
            mime="application/json"
        )
else:
    # Wenn kein neuer Index, lade aus GitHub (nur einmal pro Session)
    if st.session_state.index is None or st.session_state.index_source != "github":
        with st.spinner("Lade Index aus GitHub..."):
            index, nodes = fetch_index_from_github()
            if index:
                st.session_state.index = index
                st.session_state.nodes = nodes
                st.session_state.index_source = "github"
                st.success(f"Index aus GitHub geladen ({len(nodes)} Chunks).")

# Chat-UI
if st.session_state.index:
    st.subheader("Chat mit dem Index")
    user_input = st.text_input("Frage an den Index stellen:")
    if user_input:
        response = st.session_state.index.as_query_engine(similarity_top_k=3).query(user_input)
        antwort = response.response if hasattr(response, "response") else str(response)
        unique_sources = set(node.metadata.get('source', 'unbekannt') for node in response.source_nodes)
        st.session_state.chat_history.append({
            "frage": user_input,
            "antwort": antwort,
            "quellen": list(unique_sources)
        })
        st.write("Antwort:")
        st.write(antwort)
        st.write("Quellen:")
        for source in unique_sources:
            st.write(f"- {source}")

    # Chatverlauf anzeigen
    st.subheader("Chatverlauf dieser Sitzung")
    for i, entry in enumerate(st.session_state.chat_history):
        st.markdown(f"**Frage {i+1}:** {entry['frage']}")
        st.markdown(f"**Antwort:** {entry['antwort']}")
        st.markdown("**Quellen:**")
        for source in entry["quellen"]:
            st.markdown(f"- {source}")

    # Nachhak-Feld mit Bezug auf den Verlauf
    st.subheader("Nachhaken zur letzten Antwort")
    if st.session_state.chat_history:
        followup = st.text_input("Nachhaken (bezieht sich auf die letzte Antwort):", key="followup")
        if followup:
            last_answer = st.session_state.chat_history[-1]["antwort"]
            followup_prompt = f"Vorherige Antwort: {last_answer}\nNachhaken: {followup}"
            response = st.session_state.index.as_query_engine(similarity_top_k=3).query(followup_prompt)
            antwort = response.response if hasattr(response, "response") else str(response)
            unique_sources = set(node.metadata.get('source', 'unbekannt') for node in response.source_nodes)
            st.write("Antwort auf Nachhaken:")
            st.write(antwort)
            st.write("Quellen:")
            for source in unique_sources:
                st.write(f"- {source}")
            st.session_state.chat_history.append({
                "frage": followup_prompt,
                "antwort": antwort,
                "quellen": list(unique_sources)
            })
