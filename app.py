import streamlit as st
import warnings
import logging

# Désactiver les logs inutiles pour une interface propre
warnings.filterwarnings("ignore")
logging.getLogger("pypdf").setLevel(logging.ERROR)

from chatbot import AlAkhdariEngine
from langchain_core.messages import HumanMessage, AIMessage

# ============================================================
# CONFIGURATION DE LA PAGE
# ============================================================
st.set_page_config(page_title="Al-Akhdari AI", page_icon="🌙", layout="centered")

st.title("🌙 Assistant Jurisprudence Islamique")
st.markdown("---")

# ============================================================
# CHARGEMENT DU MOTEUR (Mis en cache)
# ============================================================
@st.cache_resource
def get_bot():
    with st.spinner("🔄 Initialisation de la base de données..."):
        bot = AlAkhdariEngine()
        bot.setup_rag()
        return bot

# Instanciation unique du moteur
bot = get_bot()

# ============================================================
# GESTION DE L'HISTORIQUE (Mémoire)
# ============================================================
if "messages" not in st.session_state:
    st.session_state.messages = [] 
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [] 

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ============================================================
# ZONE DE CHAT
# ============================================================
if prompt := st.chat_input("Posez votre question..."):
    
    # Affichage utilisateur
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Réponse de l'IA
    with st.chat_message("assistant"):
        with st.spinner("L'expert analyse les textes..."):
            try:
                # 👉 L'APPEL ULTRA SIMPLIFIÉ EST ICI 👈
                response = bot.ask(prompt, st.session_state.chat_history)
                
                st.markdown(response)
                
                # Mise à jour de la mémoire LangChain
                st.session_state.chat_history.append(HumanMessage(content=prompt))
                st.session_state.chat_history.append(AIMessage(content=response))
                
                # Sécurité : limiter la mémoire à 20 messages
                if len(st.session_state.chat_history) > 20:
                    st.session_state.chat_history = st.session_state.chat_history[-20:]
                
                # Mise à jour de l'affichage
                st.session_state.messages.append({"role": "assistant", "content": response})

            except Exception as e:
                st.error(f"❌ Une erreur est survenue : {e}")

# Optionnel : Bouton pour effacer la discussion
if st.sidebar.button("🗑️ Effacer la discussion"):
    st.session_state.messages = []
    st.session_state.chat_history = []
    st.rerun()