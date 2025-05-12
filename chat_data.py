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

from bs4 import BeautifulSoup

st.set_page_config(page_title="DNB Lab Index Generator", layout="wide")
st.title("DNB Lab: JSON- und Vektorindex aus URLs erzeugen & Chat mit vorbereitetem Index")

def is_valid_id(id_value):
    return isinstance(id_value, str) and id_value.strip() != ""

# ----------- NEU: Automatische Extraktion der Datenset-Links von der DNB-Seite -----------
def extract_dnb_lab_dataset_urls():
    MAIN_URL = "https://www.dnb.de/DE/Professionell/Services/WissenschaftundForschung/DNBLab/dnblabFreieDigitaleObjektsammlung.html?nn=849628"
    resp = requests.get(MAIN_URL)
    soup = BeautifulSoup(resp.text, "html.parser")
    dataset_urls = []
    dataset_titles = []
    # Die Tabelle mit den Datensets finden (anpassen, falls sich das HTML ändert)
    for row in soup.select("table tbody tr"):
        cols = row.find_all("td")
        if not cols or len(cols) < 2:
            continue
        title = cols[0].get_text(strip=True)
        # Alle Links in der Zeile extrahieren
        for a in row.find_all("a", href=True):
            url = a["href"]
            if url.startswith("/"):
                url = "https://www.dnb.de" + url
            dataset_urls.append(url)
            dataset_titles.append(title)
    return list(zip(dataset_urls, dataset_titles))

# ----------- Angepasst: URLs und Titel als Metadaten verarbeiten -----------
def create_rich_nodes(urls_and_titles):
    urls, titles = zip(*urls_and_titles)
    documents = TrafilaturaWebReader().load_data(urls)
    parser = SimpleNodeParser()
    nodes = []
    for url, title, doc in zip(urls, titles, documents):
        doc.metadata["source"] = url
        doc.metadata["title"] = title or doc.metadata.get("title", "")
        for node in parser.get_nodes_from_documents([doc]):
            node_id = node.node_id if is_valid_id(node.node_id) else str(uuid.uuid4())
            if node.text and node.text.strip():
                chunk_metadata = dict(node.metadata)
                chunk_metadata["source"] = url
                chunk_metadata["title"] = title
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
st.header("Schritt 1: DNB-Lab-Datensets automatisch laden oder eigene URLs eingeben")

if st.button("Alle DNB-Lab-Datensets automatisch laden"):
    with st.spinner("Extrahiere Datenset-Links von der DNB-Lab-Seite..."):
        urls_and_titles = extract_dnb_lab_dataset_urls()
        urls = [u for u, _ in urls_and_titles]
        st.session_state.urls_and_titles = urls_and_titles
        st.success(f"{len(urls)} Datenset-Links extrahiert!")
        st.write("Beispiel-Links:", urls[:5])

urls_input = st.text_area("Eigene/zusätzliche URLs (eine pro Zeile):")
extra_urls = [u.strip() for u in urls_input.split('\n') if u.strip()]
# Titel für eigene URLs leer lassen
extra_urls_and_titles = [(u, "") for u in extra_urls]

all_urls_and_titles = []
if "urls_and_titles" in st.session_state:
    all_urls_and_titles.extend(st.session_state.urls_and_titles)
if extra_urls_and_titles:
    all_urls_and_titles.extend(extra_urls_and_titles)

if all_urls_and_titles and st.button("Index aus URLs erzeugen"):
    with st.spinner("Indexiere URLs..."):
        nodes = create_rich_nodes(all_urls_and_titles)
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
