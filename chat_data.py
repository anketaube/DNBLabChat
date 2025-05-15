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

# -------------------- Globale Konfiguration für Embedding-Modell ------------------
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
def set_global_embed_model():
    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)
set_global_embed_model()

# -------------------- Seiteneinstellungen und Zusammenfassung ---------------------
st.set_page_config(
    page_title="DNBLab Chat",
    layout="wide"
)
st.title("DNBLab Chat")

st.markdown("""
**Mit DNBLab Chat kannst du Webseiten-Inhalte aus einer Liste von URLs extrahieren, in Text-Chunks aufteilen, als JSON exportieren, einen Vektorindex erzeugen und schließlich über einen Chat mit dem Index interagieren. Die Anwendung nutzt moderne LLM- und Embedding-Technologien, um Fragen zu den gesammelten Inhalten zu beantworten.**
""")

# -------------------- Datenschutzhinweis beim ersten Öffnen ----------------------
if "datenschutz_akzeptiert" not in st.session_state:
    st.session_state["datenschutz_akzeptiert"] = False

if not st.session_state["datenschutz_akzeptiert"]:
    with st.expander("Datenschutzhinweis", expanded=True):
        st.markdown("""
        **Wichtiger Hinweis zum Datenschutz:**  
        Diese Anwendung verarbeitet die von dir eingegebenen URLs sowie die daraus extrahierten Inhalte ausschließlich zum Zweck der Indexierung und Beantwortung deiner Fragen.  
        Es werden keine personenbezogenen Daten dauerhaft gespeichert oder an Dritte weitergegeben.  
        Beispiel: Wenn du eine URL eingibst, wird deren Inhalt analysiert und in Text-Chunks zerlegt, jedoch nicht dauerhaft gespeichert.
        """)
        if st.button("Hinweis schließen"):
            st.session_state["datenschutz_akzeptiert"] = True
    st.stop()

# -------------------- Auswahl: Eigenen Index bauen oder Direktstart ---------------
st.header("Wie möchtest du starten?")
st.markdown("""
Du hast zwei Möglichkeiten:
- **Schritt 1 & 2:** Eigene URLs eingeben, Inhalte extrahieren und einen neuen Index erstellen.
- **Direkter Start:** Lade einen bestehenden IJSON- oder Vektorindex direkt aus GitHub und beginne sofort mit dem Chat.
""")

start_option = st.radio(
    "Bitte wähle, wie du fortfahren möchtest:",
    (
        "Eigene URLs eingeben und Index erstellen (Schritte 1 & 2)",
        "Direkt mit bestehendem Index aus GitHub starten (empfohlen für schnellen Einstieg)"
    )
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
                chunk.metadata["source"] = url
                chunk.metadata["title"] = doc_title
                if not is_valid_id(chunk.node_id):
                    chunk.node_id = f"{url}_{len(nodes)}"
                nodes.append(chunk)
    return nodes

def index_to_rich_json(nodes: List[TextNode]):
    return [
        {
            "id": node.node_id,
            "text": node.text,
            "metadata": node.metadata,
        }
        for node in nodes
    ]

def zip_directory(folder_path, zip_path):
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, folder_path)
                zipf.write(file_path, arcname)

if "generated_nodes" not in st.session_state:
    st.session_state.generated_nodes = []

if st.button("Inhalte extrahieren"):
    urls = [url.strip() for url in urls_input.splitlines() if url.strip()]
    if urls:
        with st.spinner("Inhalte werden extrahiert..."):
            nodes = create_rich_nodes(urls)
            st.session_state.generated_nodes = nodes
        st.success(f"{len(nodes)} Text-Chunks wurden extrahiert.")
        json_data = index_to_rich_json(nodes)
        st.download_button(
            label="Extrahierte Chunks als JSON herunterladen",
            data=json.dumps(json_data, ensure_ascii=False, indent=2),
            file_name="dnblab_chunks.json",
            mime="application/json"
        )
    else:
        st.warning("Bitte gib mindestens eine gültige URL ein.")

# -------------------- Schritt 2: Index erstellen und herunterladen ----------------
st.header("Schritt 2: Index erstellen")
st.markdown("Erstelle aus den extrahierten Inhalten einen Vektorindex und lade ihn als ZIP-Datei herunter.")

if st.session_state.generated_nodes:
    if st.button("Index erstellen"):
        with st.spinner("Index wird erstellt..."):
            set_global_embed_model()
            index = VectorStoreIndex(
                st.session_state.generated_nodes,
                embed_model=Settings.embed_model
            )
            index.storage_context.persist(persist_dir="dnblab_index")
            zip_directory("dnblab_index", "dnblab_index.zip")
        st.success("Index wurde erstellt und steht zum Download bereit.")
        with open("dnblab_index.zip", "rb") as f:
            st.download_button(
                label="Index als ZIP herunterladen",
                data=f,
                file_name="dnblab_index.zip",
                mime="application/zip"
            )
else:
    st.info("Bitte extrahiere zuerst Inhalte aus URLs in Schritt 1.")

# -------------------- Schritt 3: Chat mit Index (GitHub oder lokal) --------------
st.header("Schritt 3: Chat mit Index und Mistral")

def load_local_index():
    set_global_embed_model()
    persist_dir = "dnblab_index"
    if os.path.exists(persist_dir):
        storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
        index = load_index_from_storage(storage_context)
        return index
    return None

chat_index = None
chat_index_source = None

if st.button("Index aus GitHub laden"):
    chat_index = load_index_from_github()
    chat_index_source = "GitHub"
if st.button("Gerade erzeugten lokalen Index verwenden"):
    chat_index = load_local_index()
    chat_index_source = "lokal"

if chat_index:
    api_key = st.secrets["MISTRAL_API_KEY"]
    try:
        llm = MistralAI(api_key=api_key, model="mistral-medium")
    except Exception as e:
        st.error(f"Fehler beim Initialisieren des LLM: {e}")
        st.stop()
    try:
        query_engine = RetrieverQueryEngine.from_args(chat_index.as_retriever(), llm=llm)
    except Exception as e:
        st.error(f"Fehler beim Initialisieren des Query Engines: {e}")
        st.stop()
    if "chat_history" not in st.session_state or st.session_state.get("last_index_source") != chat_index_source:
        st.session_state.chat_history = []
        st.session_state["last_index_source"] = chat_index_source
    user_input = st.text_input("Deine Frage an den Index:")
    if user_input:
        with st.spinner("Antwort wird generiert..."):
            try:
                response = query_engine.query(user_input)
                sources = set()
                if hasattr(response, "source_nodes") and response.source_nodes:
                    for node in response.source_nodes:
                        url = node.node.metadata.get("source")
                        if url:
                            sources.add(url)
                st.session_state.chat_history.append(("Du", user_input))
                if sources:
                    st.session_state.chat_history.append(
                        ("DNBLab Chat", f"{str(response)}\n\n**Quellen:**\n" + "\n".join(sources))
                    )
                else:
                    st.session_state.chat_history.append(("DNBLab Chat", str(response)))
            except Exception as e:
                st.error(f"Fehler bei der Anfrage: {e}")
    for speaker, text in st.session_state.chat_history:
        st.markdown(f"**{speaker}:** {text}")

st.info("Du kannst nach Schritt 2 direkt mit dem Chat starten oder jederzeit einen bestehenden Index aus GitHub laden.")
