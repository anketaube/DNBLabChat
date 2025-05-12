import streamlit as st
import json
import uuid
import shutil
import os
import zipfile

from llama_index.readers.web import TrafilaturaWebReader
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.schema import TextNode
from llama_index.core import VectorStoreIndex

st.set_page_config(page_title="DNB Lab Index Generator", layout="wide")
st.title("DNB Lab: JSON- und Vektorindex aus URLs erzeugen")

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
urls = [u.strip() for u in urls_input.split('\n') if u.strip()]

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
