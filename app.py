# ###########################################################
# CHATBOT SENZA IL CARICAMENTO DEL PDF DA PARTE DELL'UTENTE #
# ###########################################################

import streamlit as st
import pdfplumber
import os
from PIL import Image



# Langchain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Elenco di tutte le icone Streamlit:
# https://streamlit-emoji-shortcodes-streamlit-app-gwckff.streamlit.app/
st.set_page_config(page_title= "Consulente virtuale",
                   page_icon=":material/local_laundry_service:")

# Personalizzazione colori:
# Personalizzazione colori:
st.markdown("""
    <style>
    .stApp {
        background: #F99E1C;
    }
    .stApp:before {
        content: "";
        position: fixed;
    }
            
    /* --- BARRA DOMANDA + BOTTONE VERSIONE 3 --- */
    div[data-testid="stTextInput"] > label {
        font-size: 18px !important;
        font-weight: 700 !important;
        color: #3d2314 !important;
    }
    div[data-testid="stTextInput"] input {
        background-color: #eef6ff !important;
        border-radius: 14px !important;
        border: 2px solid transparent !important;
        padding: 18px 22px !important;
        font-size: 16px !important;
        height: 56px !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08) !important;
    }
    h1, h2, h3 { font-size: 19px !important; }
    div[data-testid="stTextInput"] input:focus {
        background-color: white !important;
        border: 2px solid #FF8C42 !important;
        box-shadow: 0 0 0 5px rgba(255,140,66,0.25) !important;
        outline: none !important;
    }
    div[data-testid="stColumn"] button {
        background-color: #FF8C42 !important;
        color: white !important;
        border-radius: 14px !important;
        border: none !important;
        height: 56px !important;
        font-weight: 700 !important;
        box-shadow: 0 4px 12px rgba(255,140,66,0.4) !important;
    }
    div[data-testid="stColumn"] button:hover {
        background-color: #e67a32 !important;
        transform: translateY(-2px) !important;
    }
        div[data-testid="stMarkdownContainer"] p {
        font-size: 15px !important;
    }
    div[data-testid="stMarkdownContainer"] h3 {
        font-size: 19px !important;
    }
    </style>
""", unsafe_allow_html=True)

st.header("Elettrodomestici su misura")

st.image("Chatbot.webp", width=550)
st.markdown("""
<div style="text-align:center; font-size:30px;">
  <span style="animation: sali 2s infinite;">🫧</span>
  <span style="animation: sali 2s infinite 0.3s;">✨</span>
  <span style="animation: sali 2s infinite 0.6s;">🫧</span>
  <span style="animation: sali 2s infinite 0.9s;">✨</span>
</div>
<style>
@keyframes sali { 0% {transform: translateY(20px); opacity:0;} 50% {opacity:1;} 100% {transform: translateY(-20px); opacity:0;} }
</style>
""", unsafe_allow_html=True)
col_centro = st.container()


with col_centro:
     documento = "CATALOGO_ELETTRODOMESTICI_CHATBOT.pdf"

# Estrazione del contenuto e spezzettamento
if documento is not None:
    @st.cache_data(show_spinner="Sto leggendo il PDF...")
    def estrai_testo_pdf(documento: str) -> str:
        with pdfplumber.open(documento) as pdf:
            # st.write(f"Pagine totali: {len(pdf.pages)} - Comincio la scansione...")
            testo = ""
            for pagina in pdf.pages:
                # Se la pagina è null menttiamo ""
                testo_pagina = pagina.extract_text() or ""
                testo = testo + testo_pagina + "\n"
                # testo += pagina.extract_text() + "\n"
        return testo.strip()
    
    testo = estrai_testo_pdf(documento)

    @st.cache_data(show_spinner=False)
    def crea_frammenti(testo: str):
        taglierina = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " "],
        chunk_size=1000,
        chunk_overlap=200)
        return taglierina.split_text(testo)

    frammenti = crea_frammenti(testo)
    # st.write(f"Totale frammenti creati: {len(frammenti)}")
    # st.write(frammenti)

    # Generiamo gli embeddings
    # e li salviamo in un vector store o vector db (es. FAISS, Pinecone, etc.)
    # Puoi cambiare OpenAIEmbeddings e metterne altri
    # https://docs.langchain.com/oss/python/integrations/embeddings
    @st.cache_resource(show_spinner=False)
    def crea_vectorstore(frammenti):
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=st.secrets["OPENAI_API_KEY"])
        return FAISS.from_texts(frammenti, embedding=embeddings)
    
    vettori = crea_vectorstore(frammenti)
    # st.write("Embedding recuperati!")

    # -------------------------------------------------------------------
    # Gestione prompt
    # -------------------------------------------------------------------
    # def invia():
        # st.session_state.domanda_inviata = st.session_state.domanda_utente
        # salva il contenuto di input, cioè domanda_utente, in domanda_inviata
        # st.session_state.domanda_utente = ""
        # reset dopo invio

    # st.text_input("Chiedi al chatbot:", key="domanda_utente", on_change=invia)
    # key="domanda_utente": assegna a st.session_state ciò che scriviamo (domanda_utente)
    # Ogni volta che l’utente modifica il campo e preme Invio,
    # la funzione invia() viene chiamata.

    # domanda_utente = st.session_state.get("domanda_inviata", "")
    # Recupera il valore salvato in "domanda_inviata".
    # Se "domanda_inviata" non è ancora stato definito (es. al primo avvio dell'app),
    # allora il valore predefinito sarà "" (secondo argomento dell'istruzione)
    # --------------------------------------------------

    

    # --------------------------------------------------

    # Generazione della risposta in una chain di eventi
    # domanda -> embedding -> similarity search -> risultati all'LLM -> risposta

def formatta_documento(documenti):
    domanda = st.session_state.get("domanda_inviata", "") or ""
    domanda_lower = str(domanda).lower()
    if "lavatrice" in domanda_lower and "lavastov" not in domanda_lower:
        keyword = "lavatrice"
    elif "lavastov" in domanda_lower:
        keyword = "lavastoviglie"
    elif "asciugat" in domanda_lower:
        keyword = "asciugatrice"
    elif "frigo" in domanda_lower or "frigorif" in domanda_lower:
        keyword = "frigorifero"
    else:
        keyword = None
    if keyword:
        filtrati = [d for d in documenti if keyword in d.page_content.lower()]
        if len(filtrati) > 0:
            documenti = filtrati
    return "\n\n".join([documento.page_content for documento in documenti])
    # Quando userò il prompt, qui dentro dovrà essere inserito qualcosa chiamato "context"
    # e qualcosa chiamata "question"
    # Qui è come nei roles di ChatGPT, ma qui siamo in Langchain
    # e la struttura è più semplice: "system" e "human"
    # Attenzione che nelle stringhe ''' vengono conservati spazi e indentazioni!
prompt = ChatPromptTemplate.from_messages([
    ("system",
     '''Sei un consulente esperto e cordiale del negozio "Elettrodomestici su misura".
TONO: professionale, gentile, ESAUSTIVO e dettagliato.

REGOLE FONDAMENTALI:
- Non dire MAI "in base al contesto" o "nel documento". Tu SEI il negozio.
- REGOLA CATEGORIA: capisci cosa chiede l'utente (frigorifero, lavastoviglie, asciugatrice, lavatrice) e rispondi SOLO con quella categoria. Non mischiare mai categorie diverse tra loro.
- Sii ESAUSTIVO: quando parli di un prodotto devi sempre dire:
1. **NOME MODELLO - €PREZZO**
2. Caratteristiche tecniche principali (potenza, capacità, classe energetica...)
3. Vantaggi e per chi è adatto
4. Se ci sono modelli simili, fai un piccolo confronto
- Usa elenchi puntati e grassetti per rendere la lettura facile.
- Rispondi SOLO in base al catalogo. Se non trovi il prodotto, proponi l'alternativa più simile MA SEMPRE DELLA STESSA CATEGORIA.
- Se non trovi l'info, di': "Al momento non disponiamo di questo modello, ma posso proporle un'alternativa valida della STESSA categoria.""

Contesto dal tuo catalogo:
{context}'''),
    ("human", "{question}")
])

comparatore = vettori.as_retriever(
        # mmr = maximal marginal relevance
        search_type="mmr",
        # Ritorna i 4 frammenti più simili
        search_kwargs={"k": 6})
    
modello_llm = ChatOpenAI(
        model="gpt-5.4-nano",
        temperature=0.5,
        max_tokens=1000,
        openai_api_key=st.secrets["OPENAI_API_KEY"])
    
catena = (
        # All'inizio mettiamo un dizionario che serve a costruire 
        # la struttura che il prompt vuol in input
        # Il comparatore produce i documenti (es. k=4) e li passa alla formattazione
        # RunnablePassthrough() vuol dire:
        # quando arriverà un input → passalo così com’è
        # Dobbiamo fare così perché ancora l'input concreto non c'è!  
        {"context": comparatore | formatta_documento, 
         "question": RunnablePassthrough()}
        | prompt
        | modello_llm
        | StrOutputParser()
        )
        # StrOutputParser() prende l’output del modello 
        # e lo traforma in una stringa semplice (senza aggiunta di info ecc.)
    
# --- INPUT + BOTTONE TONDO VERSIONE 3 ---
col_input, col_btn = st.columns([4.5, 1])

with col_input:
    domanda_utente = st.text_input(
        "Chiedi al chatbot:", 
        placeholder="Es. cerco lavatrice 10kg sotto i 400€...",
        label_visibility="visible"
     )

with col_btn:
    st.markdown("<div style='height: 28px'></div>", unsafe_allow_html=True)
    invia = st.button("Invia ➤", use_container_width=True)

# Quando scrive e preme INVIO, o clicca Invia, parte il chatbot
if domanda_utente:
    with st.spinner("Sto cercando per te..."):
        risposta = catena.invoke(domanda_utente)
        st.write(risposta)

if os.path.exists("immagini"):
        risposta_clean = risposta.lower().replace(" ", "").replace("-", "").replace("_", "")
        trovati = []
        for file in os.listdir("immagini"):
            nome = file.split(".")[0].lower().replace(" ", "").replace("-", "").replace("_", "")
            if len(nome) > 4 and nome in risposta_clean:
                if file not in trovati:
                    trovati.append(file)

if trovati:
            st.write(f"Ho trovato {len(trovati)} modelli:") # per debug, poi lo togli
            cols = st.columns(min(3, len(trovati)))
            for i, f in enumerate(trovati[:3]):
                with cols[i]:
                    st.image(f"immagini/{f}", caption=f.split(".")[0], width=250)
      
               

           
