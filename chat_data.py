import streamlit as st
import requests
import nest_asyncio
import os
from llama_index.core import VectorStoreIndex

nest_asyncio.apply()

st.set_page_config(page_title="DNB Lab Chat", layout="wide")
st.title("DNB Lab Chat (Vektorindex)")

# URL zu deinem gespeicherten Vektorindex (z.B. als .llama-Datei)
GITHUB_INDEX_URL = "https://raw.githubusercontent.com/anketaube/DNBLabChat/main/dnblab_index.llama"

def download_index_file(url, filename):
    r = requests.get(url)
    if r.status_code == 200:
        with open(filename, "wb") as f:
            f.write(r.content)
        return True
    else:
        return False

@st.cache_resource
def load_index():
    index_file = "dnblab_index.llama"
    if not os.path.exists(index_file):
        ok = download_index_file(GITHUB_INDEX_URL, index_file)
        if not ok:
            st.error("Konnte den Vektorindex nicht von GitHub laden.")
            return None
    try:
        index = VectorStoreIndex.load_from_disk(index_file)
        return index
    except Exception as e:
        st.error(f"Fehler beim Laden des Index: {e}")
        return None

index = load_index()

# --- SIDEBAR: Download des aktuellen Vektorindex ---
with st.sidebar:
    st.header("Vektorindex herunterladen")
    if os.path.exists("dnblab_index.llama"):
        with open("dnblab_index.llama", "rb") as f:
            st.download_button(
                label="Vektorindex herunterladen",
                data=f,
                file_name="dnblab_index.llama",
                mime="application/octet-stream"
            )

# --- HAUPTBEREICH: Chat ---
st.header("Chat mit dem Vektorindex")
st.write("Stelle Fragen zum geladenen Index. Quellen werden bei der Antwort angezeigt. Nachhaken ist möglich.")

if index:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Nur Eingabefeld, solange noch keine Antwort im Verlauf
    if not st.session_state.chat_history:
        user_input = st.text_input("Frage an den Index stellen:")
        if user_input:
            response = index.as_query_engine(similarity_top_k=3).query(user_input)
            antwort = response.response if hasattr(response, "response") else str(response)
            unique_sources = set(
                node.metadata.get('source', 'unbekannt')
                for node in response.source_nodes
                if hasattr(node, "metadata") and node.metadata.get('source', '').strip()
            )
            st.session_state.chat_history.append({
                "frage": user_input,
                "antwort": antwort,
                "quellen": list(unique_sources)
            })
            st.write("Antwort:")
            st.write(antwort)
            st.write("Quellen:")
            if unique_sources:
                for source in unique_sources:
                    st.write(f"- {source}")
            else:
                st.write("Keine Quelle im Chunk-Metadatum gefunden.")
    else:
        # Nach erster Antwort: Chatverlauf und Nachhaken
        user_input = st.text_input("Frage an den Index stellen:", key="maininput")
        if user_input:
            response = index.as_query_engine(similarity_top_k=3).query(user_input)
            antwort = response.response if hasattr(response, "response") else str(response)
            unique_sources = set(
                node.metadata.get('source', 'unbekannt')
                for node in response.source_nodes
                if hasattr(node, "metadata") and node.metadata.get('source', '').strip()
            )
            st.session_state.chat_history.append({
                "frage": user_input,
                "antwort": antwort,
                "quellen": list(unique_sources)
            })
            st.write("Antwort:")
            st.write(antwort)
            st.write("Quellen:")
            if unique_sources:
                for source in unique_sources:
                    st.write(f"- {source}")
            else:
                st.write("Keine Quelle im Chunk-Metadatum gefunden.")

        st.subheader("Chatverlauf dieser Sitzung")
        for i, entry in enumerate(st.session_state.chat_history):
            st.markdown(f"**Frage {i+1}:** {entry['frage']}")
            st.markdown(f"**Antwort:** {entry['antwort']}")
            st.markdown("**Quellen:**")
            for source in entry["quellen"]:
                st.markdown(f"- {source}")

        # Nachhaken-Feld
        st.subheader("Nachhaken zur letzten Antwort")
        followup = st.text_input("Nachhaken (bezieht sich auf die letzte Antwort):", key="followup")
        if followup:
            last_answer = st.session_state.chat_history[-1]["antwort"]
            followup_prompt = f"Vorherige Antwort: {last_answer}\nNachhaken: {followup}"
            response = index.as_query_engine(similarity_top_k=3).query(followup_prompt)
            antwort = response.response if hasattr(response, "response") else str(response)
            unique_sources = set(
                node.metadata.get('source', 'unbekannt')
                for node in response.source_nodes
                if hasattr(node, "metadata") and node.metadata.get('source', '').strip()
            )
            st.write("Antwort auf Nachhaken:")
            st.write(antwort)
            st.write("Quellen:")
            if unique_sources:
                for source in unique_sources:
                    st.write(f"- {source}")
            else:
                st.write("Keine Quelle im Chunk-Metadatum gefunden.")
            st.session_state.chat_history.append({
                "frage": followup_prompt,
                "antwort": antwort,
                "quellen": list(unique_sources)
            })
else:
    st.info("Der Vektorindex wird beim Start automatisch aus GitHub geladen. Sobald er bereit ist, kannst du den Chat nutzen.")
