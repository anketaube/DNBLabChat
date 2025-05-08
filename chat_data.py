import streamlit as st
from llama_index.readers.web import TrafilaturaWebReader
from llama_index.core import VectorStoreIndex
import nest_asyncio

nest_asyncio.apply()

def create_index(urls):
    documents = TrafilaturaWebReader().load_data(urls)
    for doc in documents:
        doc.metadata = {"source": doc.metadata["url"]}
    return VectorStoreIndex.from_documents(documents)

st.title("DNB Lab Chat")
urls = st.text_area("URLs (pro Zeile eine URL)").split('\n')

if urls and st.button("Index erstellen"):
    with st.spinner("Index wird generiert..."):
        index = create_index(urls)
        st.session_state.index = index
        st.success(f"Index mit {len(urls)} URLs erstellt!")

if "index" in st.session_state:
    query = st.text_input("Frage eingeben")
    if query:
        response = st.session_state.index.as_query_engine().query(query)
        st.write(response)
        st.write("Quellen:")
        for node in response.source_nodes:
            st.write(f"- {node.metadata['source']}")
