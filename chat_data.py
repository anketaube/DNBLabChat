import streamlit as st
import os
import json
import zipfile
import requests
from typing import List

from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores import SimpleVectorStore
from llama_index.core.indices.vector_store import VectorStoreIndex
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.llms.mistralai import MistralAI
from llama_index.readers.web import TrafilaturaWebReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.settings import Settings
from llama_index.core import StorageContext, load_index_from_storage
from sentence_transformers import SentenceTransformer

# -------------------- Seiteneinstellungen und Zusammenfassung ---------------------
st.set_page_config(
    page_title="DNBLab Chat",
    layout="wide"
)

# -------------------- Datenschutzhinweis als Overlay/Modal -----------------------
if "datenschutz_akzeptiert" not in st.session_state:
    st.session_state["datenschutz_akzeptiert"] = False

@st.dialog("Datenschutzhinweis")
def datenschutz_dialog():
    st.markdown("""
    **Wichtiger Hinweis zum Datenschutz:**
    Diese Anwendung verarbeitet die von dir eingegebenen URLs sowie die daraus extrahierten Inhalte ausschließlich zum Zweck der Indexierung und Beantwortung deiner Fragen.
    Es werden keine personenbezogenen Daten dauerhaft gespeichert oder an Dritte weitergegeben.
    Beispiel: Wenn du eine URL eingibst, wird deren Inhalt analysiert und in Text-Chunks zerlegt, jedoch nicht dauerhaft gespeichert.
    """)
    if st.button("Verstanden"):
        st.session_state["datenschutz_akzeptiert"] = True
        st.rerun()

if not st.session_state["datenschutz_akzeptiert"]:
    datenschutz_dialog()

# -------------------- Globale Konfiguration für Embedding-Modell ------------------
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def set_global_embed_model():
    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)

set_global_embed_model()

st.title("DNBLab Chat")
st.markdown("""
**Mit DNBLab Chat kannst du Webseiten-Inhalte aus einer Liste von URLs extrahieren, in Text-Chunks aufteilen, als JSON exportieren, einen Vektorindex erzeugen und schließlich über einen Chat mit dem Index interagieren. Die Anwendung nutzt moderne LLM- und Embedding-Technologien, um Fragen zu den gesammelten Inhalten zu beantworten.**
""")

# -------------------- Auswahl: Eigenen Index bauen oder Direktstart ---------------
st.header("Wie möchtest du starten?")
st.markdown("""
Du hast zwei Möglichkeiten:
- **Schritt 1 & 2:** Eigene URLs eingeben, Inhalte extrahieren und einen neuen Index erstellen.
- **Direkter Start:** Lade einen bestehenden IJSON- oder Vektorindex direkt aus GitHub und beginne sofort mit dem Chat.
""")

start_option = st.radio(
    "Bitte wähle, wie du fortfahren möchtest:",
    [
        "Eigene URLs eingeben und Index erstellen (Schritte 1 & 2)",
        "Direkt mit bestehendem Index aus GitHub starten (empfohlen für schnellen Einstieg)"
    ]
)

def load_index_from_github():
    set_global_embed_model()
    url = "https://github.com/anketaube/DNBLabChat/raw/main/dnblab_index.json"
    try:
        response = requests.get(url)
        response.raise_for_status()
        index_json = response.json()
        nodes = []
        for entry in index_json:
            metadata = entry.get("metadata", {})
            if "source" not in metadata:
                metadata["source"] = ""
            node = TextNode(
                text=entry["text"],
                metadata=metadata,
                id_=entry.get("id", None)
            )
            nodes.append(node)
        index = VectorStoreIndex(nodes, embed_model=Settings.embed_model)
        return index
    except Exception as e:
        st.error(f"Fehler beim Laden des Index von GitHub: {e}")
        return None

# -------------------- Schritt 1: URLs eingeben und Inhalte extrahieren ------------
st.header("Schritt 1: URLs eingeben und Inhalte extrahieren")
st.markdown("Gib eine oder mehrere URLs ein (eine pro Zeile), deren Inhalte du analysieren möchtest.")

urls_input = st.text_area("URLs (eine pro Zeile)")

def is_valid_id(id_value):
    return isinstance(id_value, str) and len(id_value) > 0

def create_rich_nodes(urls: List[str]) -> List[TextNode]:
    nodes = []
    for url in urls:
        docs = TrafilaturaWebReader().load_data([url])
        parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)
        for doc in docs:
            doc_title = doc.metadata.get("title", "")
            chunks = parser.get_nodes_from_documents([doc])
            for chunk in chunks:
