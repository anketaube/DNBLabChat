import streamlit as st
import requests
import json
import nest_asyncio
import uuid
from llama_index.readers.web import TrafilaturaWebReader
from llama_index.core import VectorStoreIndex
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.schema import TextNode

nest_asyncio.apply()

GITHUB_JSON_URL = "https://raw.githubusercontent.com/anketaube/DNBLabChat/main/dnblab_index.json"

st.title("DNB Lab Chat – Kombinierter Index & Download")

# --- Session-State-Initialisierung ---
for key, default in [
    ("index", None),
    ("nodes", []),
    ("index_source", "github"),
    ("chat_history", []),
]:
    if key not in st.session_state:
        st.session_state[key] = default

def is_valid_id(id_value):
    return isinstance(id_value, str) and id_value.strip() != ""

def load_nodes_from_json(data):
    """Hilfsfunktion: Lade Nodes aus JSON-Liste, generiere ggf. neue UUIDs, filtere leere Texte."""
    nodes = []
    for entry in data:
        text = entry.get("text", "")
        id_val = entry.get("id")
        metadata = entry.get("metadata", {})
        if not isinstance(text, str) or not text.strip():
            continue
        if not is_valid_id(id_val):
            id_val = str(uuid.uuid4())
        # Quelle aus metadata übernehmen, falls vorhanden
        if "source" not in metadata:
            metadata["source"] = ""
        node = TextNode(
            text=text,
            metadata=metadata,
            id_=id_val
        )
        nodes.append(node)
    return nodes

def fetch_index_from_github():
    """Lade Index als JSON von GitHub und baue einen Index mit allen Chunks."""
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
    """Erstelle Nodes aus neuen URLs, setze source richtig."""
    documents = TrafilaturaWebReader().load_data(urls)
    parser = SimpleNodeParser()
    nodes = []
    for url, doc in zip(urls, documents):
        doc.metadata["source"] = url  # Quelle setzen!
        doc.metadata["title"] = doc.metadata.get("title", "")
        for node in parser.get_nodes_from_documents([doc]):
            node_id = node.node_id if is_valid_id(node.node_id) else str(uuid.uuid4())
            if node.text and node.text.strip():
                nodes.append(TextNode(
                    text=node.text,
                    metadata=node.metadata,
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

# --- Upload bestehender Index (optional, z.B. aus Datei) ---
st.subheader("Optional: Lade einen bestehenden Index als JSON hoch")
uploaded_file = st.file_uploader("Index-JSON-Datei auswählen", type=["json"])
if uploaded_file is not None:
    data = json.load(uploaded_file)
    nodes = load_nodes_from_json(data)
    if nodes:
        st.session_state.nodes = nodes
        st.session_state.index = VectorStoreIndex(nodes)
        st.session_state.index_source = "uploaded"
        st.success(f"Index aus Datei geladen ({len(nodes)} Chunks).")

# --- Initiales Laden aus GitHub (nur, wenn noch kein Index geladen ist) ---
if not st.session_state.nodes and st.session_state.index_source == "github":
    with st.spinner("Lade Index aus GitHub..."):
        index, nodes = fetch_index_from_github()
        if index:
            st.session_state.index = index
            st.session_state.nodes = nodes
            st.success(f"Index aus GitHub geladen ({len(nodes)} Chunks).")

# --- URLs-Eingabe für neue Quellen ---
st.subheader("Neue URLs indexieren und zum bestehenden Index hinzufügen")
urls_input = st.text_area("Gib neue URLs ein (eine pro Zeile):")
urls = [u.strip() for u in urls_input.split('\n') if u.strip()]

if urls and st.button("Neue URLs indexieren und hinzufügen"):
    with st.spinner("Neue URLs werden indexiert..."):
        new_nodes = create_rich_nodes(urls)
        # Vermeide Duplikate: gleiche id oder gleicher Text
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

# --- Download-Button für den kombinierten Index ---
if st.session_state.nodes:
    json_data = index_to_rich_json(st.session_state.nodes)
    st.download_button(
        label="Kombinierten Index als JSON herunterladen",
        data=json_data,
        file_name="dnblab_index.json",
        mime="application/json"
    )

# --- Chat-UI ---
if st.session_state.index:
    st.subheader("Chat mit dem Index")
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
