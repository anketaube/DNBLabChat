import streamlit as st
import pandas as pd
import httpx
from parsel import Selector
import re
from urllib.parse import urljoin

# ------------------- Konfiguration -------------------
BASE_URL = "https://www.dnb.de/DE/Professionell/Services/WissenschaftundForschung/DNBLab"
START_URL = BASE_URL + "/dnblab_node.html"
EXTRA_URLS = [
    "https://www.dnb.de/DE/Professionell/Services/WissenschaftundForschung/DNBLab/dnblabTutorials.html?nn=849628",
    "https://www.dnb.de/DE/Professionell/Services/WissenschaftundForschung/DNBLabPraxis/dnblabPraxis_node.html",
    "https://www.dnb.de/DE/Professionell/Services/WissenschaftundForschung/DNBLab/dnblabSchnittstellen.html?nn=849628",
    "https://www.dnb.de/DE/Professionell/Services/WissenschaftundForschung/DNBLab/dnblabFreieDigitaleObjektsammlung.html?nn=849628"
]

st.sidebar.title("Konfiguration")
data_source = st.sidebar.radio("Datenquelle wählen:", ["Excel-Datei", "DNBLab-Webseite"])
chatgpt_model = st.sidebar.selectbox(
    "ChatGPT Modell wählen",
    options=["gpt-3.5-turbo", "gpt-4-turbo"],
    index=1
)
st.sidebar.markdown(f"Verwendetes Modell: **{chatgpt_model}**")

if "OPENAI_API_KEY" not in st.secrets:
    st.error("API-Key fehlt. Bitte in den Streamlit-Secrets hinterlegen.")
    st.stop()
api_key = st.secrets["OPENAI_API_KEY"]

# ------------------- Web-Crawler -------------------
@st.cache_data(show_spinner=True)
def crawl_dnblab():
    client = httpx.Client(timeout=10, follow_redirects=True)
    visited = set()
    to_visit = set([START_URL] + EXTRA_URLS)
    data = []
    while to_visit:
        url = to_visit.pop()
        if url in visited:
            continue
        visited.add(url)
        try:
            resp = client.get(url)
            if resp.status_code != 200:
                continue
            selector = Selector(resp.text)
            # Extrahiere Datenset-Namen (Überschrift) + Beschreibung
            for section in selector.xpath('//h2 | //h3'):
                title = section.xpath('text()').get()
                if title:
                    title = title.strip()
                else:
                    continue
                # Beschreibung: nächster Absatz oder Liste
                description = ""
                next_el = section.xpath('following-sibling::*[1]')
                if next_el and next_el.get():
                    description = " ".join(next_el.xpath('.//text()').getall()).strip()
                if title and len(title) > 3 and description and len(description) > 10:
                    data.append({
                        'datensetname': title,
                        'beschreibung': description,
                        'quelle': url
                    })
        except Exception as e:
            st.warning(f"Fehler beim Crawlen von {url}: {e}")
    if data:
        df = pd.DataFrame(data)
        df.columns = df.columns.str.strip().str.lower()
        return df
    else:
        return None

# ------------------- Excel-Lader (robust) -------------------
@st.cache_data
def load_excel(file):
    try:
        df = pd.read_excel(file, header=0, na_filter=False)
        cols = [c.lower() for c in df.columns]
        # Datensetname: erste Spalte oder passende Spalte
        if "datensetname" in cols:
            df["datensetname"] = df[[c for c in df.columns if c.lower() == "datensetname"][0]]
        else:
            df["datensetname"] = df.iloc[:, 0].astype(str)
        # Beschreibung: zweite Spalte oder passende Spalte
        if "beschreibung" in cols:
            df["beschreibung"] = df[[c for c in df.columns if c.lower() == "beschreibung"][0]]
        elif df.shape[1] > 1:
            df["beschreibung"] = df.iloc[:, 1].astype(str)
        else:
            df["beschreibung"] = ""
        df["quelle"] = "Excel-Datei"
        df = df[["datensetname", "beschreibung", "quelle"]]
        df.columns = df.columns.str.strip().str.lower()
        return df
    except Exception as e:
        st.error(f"Fehler beim Laden der Excel-Datei: {e}")
        return None

# ------------------- Kontext-Bau -------------------
def build_context(df, frage):
    mask = (
        df['datensetname'].str.lower().str.contains(frage.lower())
        | df['beschreibung'].str.lower().str.contains(frage.lower())
    )
    relevant = df[mask]
    if relevant.empty:
        relevant = df
    context = "\n".join(
        f"Datenset: {row['datensetname']}\nBeschreibung: {row['beschreibung']}\nQuelle: {row['quelle']}\n"
        for _, row in relevant.iterrows()
    )
    return context, relevant

# ------------------- OpenAI-Fragefunktion -------------------
def ask_question(question, context, model):
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        prompt = f"""
Du bist ein Datenexperte für die DNB-Datensätze.

Hier sind relevante Datensets mit Namen, Beschreibung und Quelle:

{context}

Beantworte die folgende Frage basierend auf den Daten oben in ganzen Sätzen.
Nenne immer den Namen des Datensets und gib die Quelle an.
Frage: {question}
"""
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Fehler bei OpenAI API-Abfrage: {e}")
        return "Fehler bei der Anfrage."

# ------------------- Chatverlauf initialisieren -------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ------------------- Haupt-App -------------------
st.title("DNBLab-Chatbot")

df = None
if data_source == "Excel-Datei":
    uploaded_file = st.sidebar.file_uploader("Excel-Datei hochladen", type=["xlsx"])
    if uploaded_file:
        with st.spinner("Excel-Datei wird geladen..."):
            df = load_excel(uploaded_file)
elif data_source == "DNBLab-Webseite":
    st.sidebar.info("Es wird die DNBLab-Webseite inkl. aller Unterseiten indexiert. Das kann einige Sekunden dauern.")
    with st.spinner("DNBLab-Webseite wird indexiert..."):
        df = crawl_dnblab()

if df is None or df.empty:
    st.info("Bitte laden Sie eine Excel-Datei hoch oder wählen Sie die DNBLab-Webseite aus der Sidebar.")
    st.stop()

st.write(f"Geladene Datensätze: {len(df)}")
st.markdown("**Folgende Quellen wurden indexiert:**")
for url in sorted(df['quelle'].unique()):
    if url.startswith("http"):
        st.markdown(f"- [{url}]({url})")
    else:
        st.markdown(f"- {url}")

# ------------------- Chatverlauf anzeigen -------------------
st.markdown("### Chatverlauf")
for entry in st.session_state.chat_history:
    st.markdown(f"**Du:** {entry['frage']}")
    st.markdown(f"**Bot:** {entry['antwort']}")

# ------------------- Fragesystem -------------------
frage = st.text_input("Frage eingeben oder nachhaken:")

if frage:
    with st.spinner("Antwort wird generiert..."):
        context, relevante = build_context(df, frage)
        antwort = ask_question(frage, context, chatgpt_model)
    st.session_state.chat_history.append({"frage": frage, "antwort": antwort})
    st.experimental_rerun()

# ------------------- Optionale Trefferanzeige -------------------
if st.session_state.chat_history:
    letzte_frage = st.session_state.chat_history[-1]["frage"]
    _, relevante = build_context(df, letzte_frage)
    if not relevante.empty:
        st.markdown("**Relevante Datensätze:**")
        st.dataframe(relevante)
