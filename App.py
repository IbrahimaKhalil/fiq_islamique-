import os
import glob
import warnings
import logging
import streamlit as st
from dotenv import load_dotenv

# --- CONFIGURATION INITIALE ---
# Bloque les avertissements de dépréciation et les logs bavards de Hugging Face
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# LangChain Imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_community.retrievers import BM25Retriever
# MODIFICATION : Import standard et robuste
from langchain_classic.retrievers.ensemble import EnsembleRetriever

from langchain_core.messages import HumanMessage, AIMessage

# Configuration des logs
logging.getLogger("pypdf").setLevel(logging.ERROR)

load_dotenv()

# ============================================================
# ⚙️ MOTEUR RAG
# ============================================================
class AlAkhdariEngine:
    def __init__(self, model_name="llama-3.1-8b-instant"):
        self.model_name = model_name
        self.persist_db = "./chroma_db"

        # Gestion hybride des secrets (Streamlit Cloud vs Local)
        api_key_secret = None
        try:
            api_key_secret = st.secrets.get("GROQ_API_KEY")
        except Exception:
            pass

        self.api_key = api_key_secret or os.getenv("GROQ_API_KEY")

        if not self.api_key:
            raise ValueError(" Clé API GROQ introuvable. Vérifiez vos secrets.")

        # Modèle d'embeddings multilingue
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        self.llm = ChatGroq(model=self.model_name, api_key=self.api_key, temperature=0)
        self.retriever = None
        self.chain = None

    def _format_docs(self, docs):
        """Formate les documents pour le contexte du prompt"""
        formatted = []
        for doc in docs:
            src = doc.metadata.get("source", "Inconnu")
            page = doc.metadata.get("page", 0) + 1
            formatted.append(f"Extrait de [{src}, Page {page}]:\n{doc.page_content}")
        return "\n\n---\n\n".join(formatted)

    def setup_rag(self, data_path="livres/*.pdf"):
        """Initialise la base de données et le retriever hybride"""
        pdf_files = glob.glob(data_path)
        if not pdf_files:
            st.error(f" Aucun PDF trouvé dans le dossier '{data_path}'. Vérifiez votre dépôt GitHub.")
            st.stop()

        docs = []
        for f in pdf_files:
            pages = PyPDFLoader(f).load()
            for p in pages:
                p.metadata["source"] = os.path.basename(f)
            docs.extend(pages)

        chunks = RecursiveCharacterTextSplitter(
            chunk_size=1200, chunk_overlap=150
        ).split_documents(docs)

        # Recréer la base vectorielle à chaque démarrage pour inclure tous les livres
        if os.path.exists(self.persist_db):
            import shutil
            shutil.rmtree(self.persist_db)

        vectorstore = Chroma.from_documents(
            chunks, self.embeddings, persist_directory=self.persist_db
        )

        # Configuration du retriever hybride
        bm25 = BM25Retriever.from_documents(chunks)
        bm25.k = 5
        vector_ret = vectorstore.as_retriever(search_kwargs={"k": 6})
        
        self.retriever = EnsembleRetriever(
            retrievers=[bm25, vector_ret], 
            weights=[0.5, 0.5]
        )
        self._build_chain()

    def _build_chain(self):
        """Construit la chaîne de traitement LangChain"""
        system_instruction = """Tu es un expert rigoureux en jurisprudence (Fiqh). 
Ton but est d'extraire la vérité des textes fournis.

CONSIGNES DE SÉCURITÉ :
1. Si l'information est présente : Réponds de manière concise et cite [Livre, Page] après chaque affirmation importante.
2. Si l'information est floue : Précise que le texte est ambigu et donne l'interprétation la plus proche.
3. Si l'information est ABSENTE : Dis "Je n'ai trouvé aucune mention de cela dans mes livres." NE RÉPONDS PAS avec tes propres connaissances.

CONTEXTE DE RÉFÉRENCE :
{context}"""

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_instruction),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}")
        ])

        self.chain = (
            {
                "context": RunnableLambda(lambda x: self._format_docs(self.retriever.invoke(x["question"]))),
                "question": RunnablePassthrough() | RunnableLambda(lambda x: x["question"]),
                "history": RunnablePassthrough() | RunnableLambda(lambda x: x["history"]),
            }
            | prompt | self.llm | StrOutputParser()
        )

    def ask(self, question, history=[]):
        if not self.chain:
            raise ValueError("Moteur non initialisé.")
        return self.chain.invoke({"question": question, "history": history})


# ============================================================
# INTERFACE STREAMLIT
# ============================================================
st.set_page_config(page_title="Ma Deen", page_icon="🌙", layout="centered")
st.title("🌙 Assistant Islamique")
st.caption("Expert RAG basé qui aide à répondre sur les questions religieuses en se basant sur des textes fiables(Coran etc )")


# MODIFICATION : Cache optimisé pour éviter l'erreur de récursion
@st.cache_resource(show_spinner="Initialisation de l'expert et des textes...")
def load_al_akhdari_bot(_pdf_list):
    bot = AlAkhdariEngine()
    bot.setup_rag()
    return bot

try:
    # Le cache se recrée si la liste de PDFs change
    pdf_list = tuple(sorted(glob.glob("livres/*.pdf")))
    bot = load_al_akhdari_bot(pdf_list)
except Exception as e:
    st.error(f"Erreur fatale : {e}")
    st.stop()

# Initialisation de l'historique
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Affichage
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input
if prompt := st.chat_input("Posez votre question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyse des textes originaux..."):
            try:
                response = bot.ask(prompt, st.session_state.chat_history)
                st.markdown(response)
                
                st.session_state.chat_history.append(HumanMessage(content=prompt))
                st.session_state.chat_history.append(AIMessage(content=response))
                st.session_state.messages.append({"role": "assistant", "content": response})
                
                if len(st.session_state.chat_history) > 10:
                    st.session_state.chat_history = st.session_state.chat_history[10:]
                    
            except Exception as e:
                st.error(f"Erreur : {e}")

# Barre latérale
with st.sidebar:
    st.header("Options")
    if st.button("🗑️ Effacer la discussion"):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.rerun()