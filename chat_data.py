import streamlit as st
import os
import json
import zipfile
import requests
from typing import List
from llama_index.readers.web import TrafilaturaWebReader
from llama_index.node_parser import SentenceSplitter
from llama_index.schema import TextNode
from llama_index.vector_stores import SimpleVectorStore
from llama_index.indices.vector_store import VectorStoreIndex
from llama_index.embeddings import resolve_embed_model
from llama_index.llms.mistralai import MistralAI
from llama_index.query_engine import RetrieverQueryEngine

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
    # URL zur Rohdatei (raw) auf GitHub
    url = "https://github.com/anketaube/DNBLabChat/raw/main/dnblab_index.json"
    try:
        response = requests.get(url)
        response.raise_for_status()
        index_json = response.json()
        # Aus JSON Nodes rekonstruieren
        nodes = []
        for entry in index_json:
            node = TextNode(
                text=entry["text"],
                metadata=entry.get("metadata", {}),
                id_=entry.get("id", None)
            )
            nodes.append(node)
        embed_model = resolve_embed_model("local:sentence-transformers/all-MiniLM-L6-v2")
        index = VectorStoreIndex.from_documents(nodes, embed_model=embed_model)
        return index
    except Exception as e:
        st.error(f"Fehler beim Laden des Index von GitHub: {e}")
        return None

if start_option == "Direkt mit bestehendem Index aus GitHub starten (empfohlen für schnellen Einstieg)":
    st.success("Der bestehende Index wird aus GitHub geladen. Du kannst sofort mit dem Chat beginnen.")
    index = load_index_from_github()
    if index is not None:
        st.header("Schritt 3: Chat mit dem geladenen Index")
        api_key = st.secrets.get("MISTRAL_API_KEY", None)
        if not api_key:
            st.error("Kein Mistral-API-Key gefunden. Bitte in den Streamlit-Secrets hinterlegen.")
        else:
            llm = MistralAI(api_key=api_key, model="mistral-medium")
            query_engine = RetrieverQueryEngine.from_args(index.as_retriever(), llm=llm)
            if "chat_history" not in st.session_state:
                st.session_state.chat_history = []
            user_input = st.text_input("Deine Frage an den Index:")
            if user_input:
                with st.spinner("Antwort wird generiert..."):
                    try:
                        response = query_engine.query(user_input)
                        st.session_state.chat_history.append(("Du", user_input))
                        st.session_state.chat_history.append(("DNBLab Chat", str(response)))
                    except Exception as e:
                        st.error(f"Fehler bei der Anfrage: {e}")
            for speaker, text in st.session_state.chat_history:
                st.markdown(f"**{speaker}:** {text}")
    else:
        st.warning("Index konnte nicht geladen werden. Bitte überprüfe die GitHub-Integration.")
    st.stop()

# -------------------- Schritt 1: URLs eingeben und Inhalte extrahieren ------------
st.header("Schritt 1: URLs eingeben und Inhalte extrahieren")
st.markdown("Gib eine oder mehrere URLs ein (eine pro Zeile), deren Inhalte du analysieren möchtest.")

urls_input = st.text_area("URLs (eine pro Zeile)")

def is_valid_id(id_value):
    return isinstance(id_value, str) and len(id_value) > 0

def create_rich_nodes(urls: List[str]) -> List[TextNode]:
    docs = TrafilaturaWebReader().load_data(urls)
    parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)
    nodes = []
    for doc in docs:
        chunks = parser.get_nodes_from_documents([doc])
        for chunk in chunks:
            chunk.metadata["source"] = doc.metadata.get("source", "")
            chunk.metadata["title"] = doc.metadata.get("title", "")
            if not is_valid_id(chunk.node_id):
                chunk.node_id = f"{chunk.metadata['source']}_{len(nodes)}"
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
            embed_model = resolve_embed_model("local:sentence-transformers/all-MiniLM-L6-v2")
            index = VectorStoreIndex.from_documents(
                st.session_state.generated_nodes,
                embed_model=embed_model
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

# -------------------- Schritt 3: Chat mit lokalem Index und Mistral --------------
st.header("Schritt 3: Chat mit lokalem Index und Mistral")

def load_index():
    if os.path.exists("dnblab_index"):
        return VectorStoreIndex.load_from_disk("dnblab_index")
    return None

if st.button("Lokal gespeicherten Index laden und Chat starten"):
    index = load_index()
    if index is None:
        st.error("Kein lokaler Index gefunden. Bitte erstelle oder lade einen Index.")
    else:
        api_key = st.secrets.get("MISTRAL_API_KEY", None)
        if not api_key:
            st.error("Kein Mistral-API-Key gefunden. Bitte in den Streamlit-Secrets hinterlegen.")
        else:
            llm = MistralAI(api_key=api_key, model="mistral-medium")
            query_engine = RetrieverQueryEngine.from_args(index.as_retriever(), llm=llm)
            if "chat_history" not in st.session_state:
                st.session_state.chat_history = []
            user_input = st.text_input("Deine Frage an den Index:")
            if user_input:
                with st.spinner("Antwort wird generiert..."):
                    try:
                        response = query_engine.query(user_input)
                        st.session_state.chat_history.append(("Du", user_input))
                        st.session_state.chat_history.append(("DNBLab Chat", str(response)))
                    except Exception as e:
                        st.error(f"Fehler bei der Anfrage: {e}")
            for speaker, text in st.session_state.chat_history:
                st.markdown(f"**{speaker}:** {text}")

st.info("Du kannst nach Schritt 2 direkt mit dem Chat starten oder jederzeit einen bestehenden Index aus GitHub laden.")
