import streamlit as st
import requests
import json
import nest_asyncio
from llama_index.readers.web import TrafilaturaWebReader
from llama_index.core import VectorStoreIndex

nest_asyncio.apply()

# GitHub-Rohlink zur JSON-Datei (hier ggf. anpassen)
GITHUB_JSON_URL = "https://raw.githubusercontent.com/anketaube/DNBLabChat/main/dnblab_index.json"

st.title("DNB Lab Chat – mit GitHub-Index, Download & Chatverlauf")

# --- Session-State-Initialisierung ---
for key, default in [
    ("index", None),
    ("documents", None),
    ("index_source", "github"),
    ("chat_history", []),
]:
    if key not in st.session_state:
        st.session_state[key] = default

def fetch_index_from_github():
    """Lade Index als JSON von GitHub und baue einen einfachen Index."""
    r = requests.get(GITHUB_JSON_URL)
    if r.status_code == 200:
        data = r.json()
        # Dokumente nachbauen
        from llama_index.core.schema import Document
        documents = [Document(text=entry["text"], metadata={"source": entry["source"]}) for entry in data]
        index = VectorStoreIndex.from_documents(documents)
        return index, documents
    else:
        st.error("Konnte Index nicht von GitHub laden.")
        return None, None

def create_index(urls):
    documents = TrafilaturaWebReader().load_data(urls)
    for url, doc in zip(urls, documents):
        doc.metadata = {"source": url}
    index = VectorStoreIndex.from_documents(documents)
    return index, documents

def index_to_json(documents):
    export = []
    for doc in documents:
        export.append({
            "text": doc.text,
            "source": doc.metadata.get("source", "")
        })
    return json.dumps(export, ensure_ascii=False, indent=2)

# URLs-Eingabe
st.subheader("Neue URLs indexieren (optional)")
urls_input = st.text_area("Gib neue URLs ein (eine pro Zeile):")
urls = [u.strip() for u in urls_input.split('\n') if u.strip()]

# Nur wenn URLs eingegeben werden, neuen Index bauen
if urls and st.button("Neuen Index erstellen"):
    with st.spinner("Index wird erstellt..."):
        index, documents = create_index(urls)
        st.session_state.index = index
        st.session_state.documents = documents
        st.session_state.index_source = "custom"
        st.success(f"Neuer Index mit {len(urls)} URLs erstellt!")
        # Download-Button für neuen Index
        json_data = index_to_json(documents)
        st.download_button(
            label="Neuen Index als JSON herunterladen",
            data=json_data,
            file_name="dnblab_index.json",
            mime="application/json"
        )
else:
    # Wenn kein neuer Index, lade aus GitHub (nur einmal pro Session)
    if st.session_state.index is None or st.session_state.index_source != "github":
        with st.spinner("Lade Index aus GitHub..."):
            index, documents = fetch_index_from_github()
            if index:
                st.session_state.index = index
                st.session_state.documents = documents
                st.session_state.index_source = "github"
                st.success("Index aus GitHub geladen.")

# Chat-UI
if st.session_state.index:
    st.subheader("Chat mit dem Index")
    user_input = st.text_input("Frage an den Index stellen:")
    if user_input:
        # Kontext: Chatverlauf als Prompt (optional, je nach LLM-Backend)
        response = st.session_state.index.as_query_engine(similarity_top_k=3).query(user_input)
        antwort = response.response if hasattr(response, "response") else str(response)
        unique_sources = set(node.metadata.get('source', 'unbekannt') for node in response.source_nodes)
        quellen = "\n".join(f"- {source}" for source in unique_sources)
        # Verlauf speichern
        st.session_state.chat_history.append({
            "frage": user_input,
            "antwort": antwort,
            "quellen": list(unique_sources)
        })
        st.write("Antwort:")
        st.write(antwort)
        st.write("Quellen:")
        for source in unique_sources:
            st.write(f"- {source}")

    # Chatverlauf anzeigen
    st.subheader("Chatverlauf dieser Sitzung")
    for i, entry in enumerate(st.session_state.chat_history):
        st.markdown(f"**Frage {i+1}:** {entry['frage']}")
        st.markdown(f"**Antwort:** {entry['antwort']}")
        st.markdown("**Quellen:**")
        for source in entry["quellen"]:
            st.markdown(f"- {source}")

    # Nachhak-Feld mit Bezug auf den Verlauf
    st.subheader("Nachhaken zur letzten Antwort")
    if st.session_state.chat_history:
        followup = st.text_input("Nachhaken (bezieht sich auf die letzte Antwort):", key="followup")
        if followup:
            # Optional: Kombiniere letzte Antwort als Kontext
            last_answer = st.session_state.chat_history[-1]["antwort"]
            followup_prompt = f"Vorherige Antwort: {last_answer}\nNachhaken: {followup}"
            response = st.session_state.index.as_query_engine(similarity_top_k=3).query(followup_prompt)
            antwort = response.response if hasattr(response, "response") else str(response)
            unique_sources = set(node.metadata.get('source', 'unbekannt') for node in response.source_nodes)
            st.write("Antwort auf Nachhaken:")
            st.write(antwort)
            st.write("Quellen:")
            for source in unique_sources:
                st.write(f"- {source}")
            # Verlauf erweitern
            st.session_state.chat_history.append({
                "frage": followup_prompt,
                "antwort": antwort,
                "quellen": list(unique_sources)
            })
