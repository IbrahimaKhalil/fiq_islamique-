# ============================================================
# IMPORTS
# ============================================================
from dotenv import load_dotenv
import os
import sys

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ============================================================
# 0) Charger les variables d'environnement
# ============================================================
load_dotenv()  # Charge .env à la racine du projet
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
from dotenv import load_dotenv
import os

# Chemin absolu vers le .env
load_dotenv(dotenv_path=r"C:\Users\ibrah\cours_rag_langchain\.env")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if  GROQ_API_KEY == None:
    print("❌ Erreur : GROQ_API_KEY introuvable. Vérifie ton fichier .env !")
    sys.exit(1)

# ============================================================
# 1) Charger le PDF
# ============================================================
loader = PyPDFLoader("livres/Akhdari.pdf")
documents = loader.load()
print(f"✅ {len(documents)} pages chargées")

# ============================================================
# 2) Découper en chunks
# ============================================================
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=100
)
chunks = splitter.split_documents(documents)
print(f"✅ {len(chunks)} chunks créés")

# ============================================================
# 3) Embeddings
# ============================================================
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
print("✅ Embeddings chargés")

# ============================================================
# 4) Chroma (base vectorielle)
# ============================================================
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)
print("✅ Base vectorielle créée")

# ============================================================
# 5) Retriever
# ============================================================
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# ============================================================
# 6) Prompt
# ============================================================
template = """
Tu es un assistant islamique expert.
Utilise UNIQUEMENT le contexte ci-dessous pour répondre.
Si tu ne trouves pas la réponse, dis "Je ne sais pas".
Réponds en français.

Contexte :
{context}

Question : {question}
Réponse :
"""

prompt = PromptTemplate(
    template=template,
    input_variables=["context", "question"]
)

# ============================================================
# 7) LLM (ChatGroq)
# ============================================================
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=GROQ_API_KEY,
    temperature=0
)
print("✅ LLM ChatGroq prêt")

# ============================================================
# 8) Chaîne LCEL
# ============================================================
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# ============================================================
# 9) TEST
# ============================================================
# ============================================================
# INTERACTION CHATBOT
# ============================================================
print("\n" + "="*50)
print("🤖 BIENVENUE SUR L'ASSISTANT L'AKHDARI")
print("Posez vos questions sur la propreté et la prière.")
print("(Tapez 'quitter' ou 'exit' pour arrêter)")
print("="*50 + "\n")

while True:
    # 1. Demander la question à l'utilisateur
    user_query = input("👉 Votre question : ")
    
    # 2. Condition de sortie
    if user_query.lower() in ["quitter", "exit", "quit", "stop"]:
        print("\n👋 Au revoir ! Qu'Allah vous bénisse.")
        break
    
    # 3. Vérifier si la question n'est pas vide
    if not user_query.strip():
        continue

    print("\n🔍 Recherche dans les livres en cours...")
    
    # 4. Lancer la chaîne RAG
    try:
        reponse = chain.invoke(user_query)
        
        print("\n" + "-"*30)
        print(f"📝 RÉPONSE :\n{reponse}")
        print("-"*30 + "\n")
        
    except Exception as e:
        print(f"⚠️ Une erreur est survenue : {e}")