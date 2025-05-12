import streamlit as st
import json
import uuid
import re
import shutil
import os
import zipfile
import requests
import io

from llama_index.readers.web import TrafilaturaWebReader
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.schema import TextNode
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.llms.mistralai import MistralAI

st.set_page_config(page_title="DNB Lab Chat", layout="wide")
st.title("DNB Lab: Index-Generator & Chat mit Mistral")

# --- Konfiguration ---
MISTRAL_API_KEY = st.secrets.get("MISTRAL_API_KEY", "frei-verfügbar")
GITHUB_INDEX_URL = "https://github.com/anketaube/DNBLabChat/raw/main/dnblab_index.zip"

# --- Hilfsfunktionen Index-Erstellung ---
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

# --- NEU: Chat-Funktionalität mit Mistral ---
@st.cache_resource(show_spinner=False)
def load_index_from_zip():
    response = requests.get(GITHUB_INDEX_URL)
    if response.status_code != 200:
        st.error("Index konnte nicht geladen werden")
        return None
    
    extract_dir = "dnblab_index"
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        z.extractall(extract_dir)
    
    storage_context = StorageContext.from_defaults(persist_dir=extract_dir)
    return VectorStoreIndex(storage_context=storage_context)

def initialize_chat():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "index" not in st.session_state:
        st.session_state.index = None
    if "llm" not in st.session_state:
        st.session_state.llm = MistralAI(api_key=MISTRAL_API_KEY)

# --- Haupt-UI ---
tab1, tab2 = st.tabs(["📥 Index erstellen", "💬 Chat mit Index"])

with tab1:
    st.header("Vektorindex aus URLs generieren")
    urls_input = st.text_area("URLs eingeben (eine pro Zeile):")
    urls = [u.strip() for u in urls_input.split('\n') if u.strip()]

    if st.button("Index generieren"):
        with st.spinner("Verarbeite URLs..."):
            nodes = create_rich_nodes(urls)
        if nodes:
            st.session_state.generated_nodes = nodes
            st.success(f"{len(nodes)} Chunks erstellt")
            
            # JSON Download
            json_data = index_to_rich_json(nodes)
            st.download_button(
                "JSON herunterladen",
                data=json_data,
                file_name="dnblab_index.json"
            )
            
            # Vektorindex erstellen
            with st.spinner("Baue Vektorindex..."):
                index = VectorStoreIndex(nodes)
                persist_dir = "dnblab_index"
                index.storage_context.persist(persist_dir=persist_dir)
                zip_path = "dnblab_index.zip"
                zip_directory(persist_dir, zip_path)
                
                st.download_button(
                    "Vektorindex herunterladen",
                    data=open(zip_path, "rb"),
                    file_name=zip_path
                )

with tab2:
    st.header("Chat mit vorberechnetem Index")
    initialize_chat()
    
    if st.button("Index laden"):
        with st.spinner("Lade Index von GitHub..."):
            st.session_state.index = load_index_from_zip()
    
    if st.session_state.index:
        query_engine = st.session_state.index.as_query_engine(
            llm=st.session_state.llm,
            similarity_top_k=3
        )
        
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
        
        if prompt := st.chat_input("Stelle eine Frage zu den indexierten Daten"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)
            
            with st.chat_message("assistant"):
                response = query_engine.query(prompt)
                st.write(response.response)
                st.session_state.messages.append({"role": "assistant", "content": response.response})
    else:
        st.warning("Bitte zuerst Index laden")
