import streamlit as st
import json
import uuid
import shutil
import os
import zipfile
import requests
import io

from llama_index.readers.web import TrafilaturaWebReader
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.schema import TextNode
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage

# HuggingFace-Embeddings für die Vektor-Suche explizit setzen (kein API-Key nötig)
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

st.set_page_config(page_title="DNB Lab Index Generator", layout="wide")
st.title("DNB Lab: JSON- und Vektorindex aus URLs erzeugen & Chat mit vorbereitetem Index")

def is_valid_id(id_value):
    return isinstance(id_value, str) and id_value.strip() != ""

def create_rich_nodes(urls):
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

def zip_directory(folder_path, zip_path):
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                abs_path = os.path.join(root, file)
                rel_path = os.path.relpath(abs_path, folder_path)
                zipf.write(abs_path, rel_path)

# Schritt 1: URLs zu Index
st.header("Schritt 1: URLs eingeben und Index erzeugen")

urls_input = st.text_area("Neue URLs (eine pro Zeile):")
urls = [u.strip() for u in urls_input.split('\n') if u.strip() and u.strip().startswith("http")]
invalid_lines = [u for u in urls_input.split('\n') if u.strip() and not u.strip().startswith("http")]
if invalid_lines:
    st.warning(f"Diese Zeilen wurden ignoriert, da sie keine gültigen URLs sind: {invalid_lines}")

if urls and st.button("Index aus URLs erzeugen"):
    with st.spinner("Indexiere URLs..."):
        nodes = create_rich_nodes(urls)
        if not nodes:
            st.error("Keine gültigen Chunks aus den URLs extrahiert.")
        else:
            st.session_state.generated_nodes = nodes
            st.success(f"{len(nodes)} Chunks erzeugt!")

    # Download JSON
    json_data = index_to_rich_json(nodes)
    st.download_button(
        label="Index als JSON herunterladen (dnblab_index.json)",
        data=json_data,
        file_name="dnblab_index.json",
        mime="application/json"
    )

# Schritt 2: Index als ZIP
if "generated_nodes" in st.session_state and st.session_state.generated_nodes:
    st.header("Schritt 2: Vektorindex erzeugen und herunterladen")
    if st.button("Vektorindex aus erzeugtem JSON bauen"):
        with st.spinner("Erzeuge Vektorindex... (kann einige Minuten dauern)"):
            index = VectorStoreIndex(st.session_state.generated_nodes)
            persist_dir = "dnblab_index"
            if os.path.exists(persist_dir):
                shutil.rmtree(persist_dir)
            index.storage_context.persist(persist_dir=persist_dir)
            # Zippe das Verzeichnis für den Download
            zip_path = "dnblab_index.zip"
            zip_directory(persist_dir, zip_path)
            with open(zip_path, "rb") as f:
                st.download_button(
                    label="Vektorindex herunterladen (dnblab_index.zip)",
                    data=f,
                    file_name="dnblab_index.zip",
                    mime="application/zip"
                )
            st.success("Vektorindex wurde erzeugt und steht zum Download bereit!")
    # Optional: Aufräumen
    # shutil.rmtree(persist_dir)
    # os.remove(zip_path)

# Schritt 3: Chat mit vorbereitetem Index aus GitHub
st.header("Schritt 3: Chat mit vorbereitetem Index aus GitHub")

def load_index_from_github_zip():
    ZIP_URL = "https://github.com/anketaube/DNBLabChat/raw/main/dnblab_index.zip"
    extract_dir = "dnblab_index_github"
    # Lade und entpacke ZIP nur, wenn noch nicht vorhanden
    if not os.path.exists(extract_dir):
        response = requests.get(ZIP_URL)
        if response.status_code != 200:
            st.error("Fehler beim Laden des Index.")
            return None
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            z.extractall(extract_dir)
    storage_context = StorageContext.from_defaults(persist_dir=extract_dir)
    index = load_index_from_storage(storage_context)
    return index

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "chat_input" not in st.session_state:
    st.session_state.chat_input = ""

if st.button("Vorbereiteten Index aus GitHub laden"):
    with st.spinner("Lade und initialisiere Index..."):
        st.session_state.index = load_index_from_github_zip()
        if st.session_state.index:
            st.success("Index geladen! Du kannst jetzt Fragen stellen.")

if "index" in st.session_state and st.session_state.index:
    mistral_api_key = st.secrets.get("MISTRAL_API_KEY", "")
    if not mistral_api_key:
        st.warning("Bitte MISTRAL_API_KEY in den Streamlit-Secrets hinterlegen.")
    else:
        from llama_index.llms.mistralai import MistralAI
        llm = MistralAI(api_key=mistral_api_key)
        query_engine = st.session_state.index.as_query_engine(llm=llm, similarity_top_k=3)

        st.subheader("Chat mit dem Index")
        for entry in st.session_state.chat_history:
            st.markdown(f"**Du:** {entry['user']}")
            st.markdown(f"**Bot:** {entry['bot']}")

        # Verarbeitung VOR dem Widget!
        if st.session_state.chat_input:
            user_input = st.session_state.chat_input
            st.session_state.chat_history.append({"user": user_input, "bot": "..."})
            with st.spinner("Antwort wird generiert..."):
                try:
                    response = query_engine.query(user_input)
                    st.session_state.chat_history[-1]["bot"] = response.response
                except Exception as e:
                    if "429" in str(e) or "rate limit" in str(e).lower():
                        st.session_state.chat_history[-1]["bot"] = (
                            "Du hast das Anfragelimit der Mistral-API erreicht. "
                            "Bitte warte einige Minuten und versuche es erneut."
                        )
                    else:
                        st.session_state.chat_history[-1]["bot"] = f"Fehler bei der Anfrage: {e}"
            st.session_state.chat_input = ""
            st.rerun()
        # Jetzt das Widget anzeigen
        st.text_input("Deine Frage an den Index:", key="chat_input")
else:
    st.info("Lade den vorbereiteten Index, um den Chat zu starten.")
