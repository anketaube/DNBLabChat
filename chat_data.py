import streamlit as st
from llama_index.readers.web import TrafilaturaWebReader
from llama_index.core import VectorStoreIndex
import nest_asyncio
import json

nest_asyncio.apply()

def create_index(urls):
    documents = TrafilaturaWebReader().load_data(urls)
    for url, doc in zip(urls, documents):
        doc.metadata = {"source": url}
    index = VectorStoreIndex.from_documents(documents)
    return index, documents

def index_to_json(index, documents):
    # Beispiel: Nur die wichtigsten Infos als JSON exportieren
    # (Du kannst das nach Bedarf anpassen, je nachdem, was du speichern möchtest)
    export = []
    for doc in documents:
        export.append({
            "text": doc.text,
            "source": doc.metadata.get("source", "")
        })
    return json.dumps(export, ensure_ascii=False, indent=2)

st.title("DNB Lab Chat – Index Builder mit Download")

urls_input = st.text_area("Gib die URLs ein (eine pro Zeile):")
urls = [u.strip() for u in urls_input.split('\n') if u.strip()]

if urls and st.button("Index erstellen"):
    with st.spinner("Index wird erstellt..."):
        try:
            index, documents = create_index(urls)
            st.session_state.index = index
            st.session_state.documents = documents
            st.success(f"Index mit {len(urls)} URLs erstellt!")
        except Exception as e:
            st.error(f"Fehler beim Erstellen des Index: {e}")

if "index" in st.session_state and "documents" in st.session_state:
    query = st.text_input("Frage an den Index stellen:")
    if query:
        response = st.session_state.index.as_query_engine(similarity_top_k=3).query(query)
        st.write("Antwort:")
        st.write(response.response if hasattr(response, "response") else str(response))
        st.write("Quellen:")
        unique_sources = set(node.metadata.get('source', 'unbekannt') for node in response.source_nodes)
        for source in unique_sources:
            st.write(f"- {source}")

    # Download-Button für den Index als JSON
    json_data = index_to_json(st.session_state.index, st.session_state.documents)
    st.download_button(
        label="Index als JSON herunterladen",
        data=json_data,
        file_name="dnblab_index.json",
        mime="application/json"
    )
