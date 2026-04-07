import os
import glob
from dotenv import load_dotenv
import streamlit as st
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
# CORRECTION 1 : L'import classique pour éviter l'erreur
from langchain_classic.retrievers.ensemble import EnsembleRetriever

# Charger le fichier .env en local (ne fera rien sur Streamlit Cloud, ce qui est parfait)
load_dotenv() 

class AlAkhdariEngine:
    def __init__(self, model_name="llama-3.1-8b-instant"):
        self.model_name = model_name
        self.persist_db = "./chroma_db"
        
        # --- CORRECTION DE SÉCURITÉ ANTI-CRASH ---
        api_key_secret = None
        try:
            # Tente de lire le secret Streamlit
            api_key_secret = st.secrets.get("GROQ_API_KEY")
        except Exception:
            # Ignore l'erreur si secrets.toml n'existe pas (ex: test en local)
            pass 

        # Utilise le secret Streamlit, sinon cherche dans les variables d'environnement (.env)
        self.api_key = api_key_secret or os.getenv("GROQ_API_KEY")
        
        if not self.api_key:
            raise ValueError("🚨 La clé API GROQ est introuvable. Ajoutez-la dans un fichier .env ou .streamlit/secrets.toml")
        # ----------------------------------------
        
        # CORRECTION 2 : Modèle multilingue pour le français et l'arabe
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        self.llm = ChatGroq(model=self.model_name, api_key=self.api_key, temperature=0)
        
        self.retriever = None
        self.chain = None

    def _format_docs(self, docs):
        """Méthode privée pour formater les extraits"""
        formatted = []
        for doc in docs:
            src = doc.metadata.get("source", "Inconnu")
            page = doc.metadata.get("page", 0) + 1 # Correction pour l'affichage de la page
            formatted.append(f"📖 Extrait de [{src}, Page {page}]:\n{doc.page_content}")
        return "\n\n---\n\n".join(formatted)

    def setup_rag(self, data_path="livres/*.pdf"):
        """Initialise la mémoire persistante et le retriever hybride"""
        if os.path.exists(self.persist_db):
            print("📦 Chargement de la base existante...")
            vectorstore = Chroma(persist_directory=self.persist_db, embedding_function=self.embeddings)
            pdf_files = glob.glob(data_path)
            docs = []
            for f in pdf_files: 
                pages = PyPDFLoader(f).load()
                # CORRECTION 3 : Ajouter les métadonnées ici aussi
                for p in pages: p.metadata["source"] = os.path.basename(f)
                docs.extend(pages)
            chunks = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150).split_documents(docs)
        else:
            print("🚀 Création d'une nouvelle base vectorielle...")
            pdf_files = glob.glob(data_path)
            docs = []
            for f in pdf_files:
                loader = PyPDFLoader(f)
                pages = loader.load()
                for p in pages: p.metadata["source"] = os.path.basename(f)
                docs.extend(pages)
            chunks = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150).split_documents(docs)
            vectorstore = Chroma.from_documents(chunks, self.embeddings, persist_directory=self.persist_db)

        # Retriever Hybride
        bm25 = BM25Retriever.from_documents(chunks)
        bm25.k = 5 # CORRECTION 4 : Plus de contexte
        vector_ret = vectorstore.as_retriever(search_kwargs={"k": 5})
        self.retriever = EnsembleRetriever(retrievers=[bm25, vector_ret], weights=[0.5, 0.5])
        
        # Construction de la chaîne finale
        self._build_chain()
        print("✅ Moteur RAG prêt.")

    def _build_chain(self):
        """Construit secrètement la logique de l'IA"""
        # CORRECTION 5 : Ton super prompt de sécurité
        system_instruction = """Tu es un expert rigoureux en jurisprudence (Fiqh). 
Ton but est d'extraire la vérité des textes fournis.

CONSIGNES DE SÉCURITÉ :
1. Si l'information est présente : Réponds de manière concise et cite [Livre, Page] après chaque affirmation importante.
2. Si l'information est floue : Précise que le texte est ambigu et donne l'interprétation la plus proche.
3. Si l'information est ABSENTE : Dis "Je n'ai trouvé aucune mention de cela dans mes livres (Akhdari, etc.)." NE RÉPONDS PAS avec tes propres connaissances.

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
        """Méthode publique pour poser une question"""
        if not self.chain:
            raise ValueError("Le moteur n'est pas initialisé. Appelez d'abord setup_rag().")
        return self.chain.invoke({"question": question, "history": history})

# --- TEST TERMINAL ---
if __name__ == "__main__":
    bot = AlAkhdariEngine()
    bot.setup_rag()
    print("\n🤖 Réponse de l'IA :\n", bot.ask("Quelles sont les obligations de l'ablution ?"))