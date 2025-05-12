import streamlit as st
import json
import uuid
import os
import requests

from llama_index.readers.web import TrafilaturaWebReader
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.schema import TextNode
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.settings import Settings
from llama_index.llms.mistralai import MistralAI

st.set_page_config(page_title="DNB Lab Index Generator", layout="wide")
st.title("DNB Lab: Chat mit automatisch gebautem Vektorindex aus GitHub-URLs")

# --- 1. URLs aus GitHub laden ---
URLS_RAW_URL = "https://raw.githubusercontent.com/anketaube/DNBLabChat/main/urls.txt"  # Passe ggf. an!
@st.cache_data(show_spinner="Lade URL-Liste von GitHub...")
def load_urls_from_github():
    resp = requests.get(URLS_RAW_URL)
    resp.raise_for_status()
    urls = [line.strip() for line in resp.text.splitlines() if line.strip()]
    return urls

# --- 2. Embedding-Modell setzen ---
@st.cache_resource(show_spinner="Initialisiere Embedding-Modell...")
def get_embed_model():
    return HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

Settings.embed_model = get_embed_model()

# --- 3. Index aus URLs bauen (nur einmal pro Session) ---
@st.cache_resource(show_spinner="Erzeuge Vektorindex aus URLs...")
def build_index(urls):
    documents = TrafilaturaWebReader().load_data(urls)
    parser = SimpleNodeParser()
    nodes = []
    for url, doc in zip(urls, documents):
        doc.metadata["source"] = url
        doc.metadata["title"] = doc.metadata.get("title", "")
        for node in parser.get_nodes_from_documents([doc]):
            node_id = node.node_id if isinstance(node.node_id, str) and node.node_id.strip() else str(uuid.uuid4())
            if node.text and node.text.strip():
                chunk_metadata = dict(node.metadata)
                chunk_metadata["source"] = url
                nodes.append(TextNode(
                    text=node.text,
                    metadata=chunk_metadata,
                    id_=node_id
                ))
    index = VectorStoreIndex(nodes)
    return index

urls = load_urls_from_github()
index = build_index(urls)

# --- 4. Mistral LLM initialisieren ---
mistral_api_key = st.secrets.get("MISTRAL_API_KEY", "")
if not mistral_api_key:
    st.warning("Bitte hinterlege deinen Mistral API Key in den Streamlit-Secrets als 'MISTRAL_API_KEY'.")
    st.stop()
llm = MistralAI(api_key=mistral_api_key)

# --- 5. Query Engine ---
query_engine = index.as_query_engine(llm=llm, similarity_top_k=3)

# --- 6. Chat UI ---
st.header("Chat mit dem Index")
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "chat_input" not in st.session_state:
    st.session_state.chat_input = ""

# Verarbeitung vor dem Widget!
if st.session_state.chat_input:
    user_input = st.session_state.chat_input
    st.session_state.chat_history.append({"user": user_input, "bot": "..."})
    with st.spinner("Antwort wird generiert..."):
        try:
            response = query_engine.query(user_input)
            st.session_state.chat_history[-1]["bot"] = response.response
        except Exception as e:
            if "429" in str(e) or "rate limit" in str(e).lower():
                st.session_state.chat_history[-1]["bot"] = (
                    "Du hast das Anfragelimit der Mistral-API erreicht. "
                    "Bitte warte einige Minuten und versuche es erneut."
                )
            else:
                st.session_state.chat_history[-1]["bot"] = f"Fehler bei der Anfrage: {e}"
    st.session_state.chat_input = ""
    st.rerun()

# Chatverlauf anzeigen
for entry in st.session_state.chat_history:
    st.markdown(f"**Du:** {entry['user']}")
    st.markdown(f"**Bot:** {entry['bot']}")

# Jetzt das Widget anzeigen
st.text_input("Deine Frage an den Index:", key="chat_input")
