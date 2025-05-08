import streamlit as st
from llama_index.readers.web import TrafilaturaWebReader
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
import nest_asyncio
import os
import requests

nest_asyncio.apply()

INDEX_DIR = "dnblab_index"
GITHUB_RAW_URL = "https://raw.githubusercontent.com/anketaube/DNBLabChat/main/dnblab_index.zip"

def download_and_extract_index():
    if not os.path.exists(INDEX_DIR):
        r = requests.get(GITHUB_RAW_URL)
        if r.status_code == 200:
            import zipfile
            import io
            with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                z.extractall(INDEX_DIR)
            return True
    return False

def create_index(urls):
    documents = TrafilaturaWebReader().load_data(urls)
    for url, doc in zip(urls, documents):
        doc.metadata = {"source": url}
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir=INDEX_DIR)  # Speichern im Verzeichnis
    return index

def load_index():
    storage_context = StorageContext.from_defaults(persist_dir=INDEX_DIR)
    return load_index_from_storage(storage_context)

st.title("DNB Lab Chat – Index Builder mit GitHub-Speicherung")

# Beim Start versuchen, Index von GitHub zu laden
if download_and_extract_index():
    st.info("Index aus GitHub geladen.")

# Index laden, falls vorhanden
index = None
if os.path.exists(INDEX_DIR):
    try:
        index = load_index()
        st.session_state.index = index
        st.success("Lokalen Index geladen.")
    except Exception as e:
        st.warning(f"Index konnte nicht geladen werden: {e}")

urls_input = st.text_area("Gib die URLs ein (eine pro Zeile):")
urls = [u.strip() for u in urls_input.split('\n') if u.strip()]

if urls and st.button("Index mit neuen URLs erstellen und speichern"):
    with st.spinner("Index wird erstellt..."):
        try:
            index = create_index(urls)
            st.session_state.index = index
            st.success(f"Index mit {len(urls)} URLs erstellt und als Verzeichnis gespeichert!")
            st.info("Bitte lade das Verzeichnis dnblab_index als ZIP in dein GitHub-Repo hoch, um es zu teilen.")
        except Exception as e:
            st.error(f"Fehler beim Erstellen des Index: {e}")

if "index" in st.session_state:
    query = st.text_input("Frage an den Index stellen:")
    if query:
        response = st.session_state.index.as_query_engine(similarity_top_k=3).query(query)
        st.write("Antwort:")
        st.write(response.response if hasattr(response, "response") else str(response))
        st.write("Quellen:")
        unique_sources = set(node.metadata.get('source', 'unbekannt') for node in response.source_nodes)
        for source in unique_sources:
            st.write(f"- {source}")

st.markdown("""
**Hinweis:**  
Nach dem Erstellen eines neuen Index muss das Verzeichnis `dnblab_index` gezippt und manuell in das GitHub-Repository [anketaube/DNBLabChat](https://github.com/anketaube/DNBLabChat/tree/main) hochgeladen werden.
""")
