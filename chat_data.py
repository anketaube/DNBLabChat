# requirements.txt sollte enthalten:
# streamlit
# llama-index
# llama-index-readers-web
# nest-asyncio
# trafilatura

import streamlit as st
from llama_index.readers.web import TrafilaturaWebReader
from llama_index.core import VectorStoreIndex
import nest_asyncio

nest_asyncio.apply()

def create_index(urls):
    documents = TrafilaturaWebReader().load_data(urls)
    for doc in documents:
        # Sicheres Auslesen der URL
        url = doc.metadata.get("url", None)
        doc.metadata = {"source": url if url else "unbekannt"}
    return VectorStoreIndex.from_documents(documents)

st.title("DNB Lab Chat – Index Builder")

urls_input = st.text_area("Gib die URLs ein (eine pro Zeile):")
urls = [u.strip() for u in urls_input.split('\n') if u.strip()]

if urls and st.button("Index erstellen"):
    with st.spinner("Index wird erstellt..."):
        try:
            index = create_index(urls)
            st.session_state.index = index
            st.success(f"Index mit {len(urls)} URLs erstellt!")
        except Exception as e:
            st.error(f"Fehler beim Erstellen des Index: {e}")

if "index" in st.session_state:
    query = st.text_input("Frage an den Index stellen:")
    if query:
        response = st.session_state.index.as_query_engine(similarity_top_k=3).query(query)
        st.write("Antwort:")
        st.write(response)
        st.write("Quellen:")
        for node in response.source_nodes:
            st.write(f"- {node.metadata.get('source', 'unbekannt')}")
