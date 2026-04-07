# Assistant Jurisprudence Islamique (RAG)

Application d'intelligence artificielle basee sur le RAG (Retrieval-Augmented Generation) permettant de repondre aux questions de jurisprudence islamique (Fiqh) a partir de livres de reference comme Al-Akhdari, Risala et Ibn Ashir.

## Fonctionnalites

- Charge et analyse plusieurs fichiers PDF.
- Combine la recherche par mots-cles (BM25) et la recherche semantique.
- Cite les sources avec le livre et la page.
- Propose une interface de chat avec memoire de conversation.
- Sauvegarde la base vectorielle pour accelerer les prochains lancements.

## Installation

1. Cloner le projet

```bash
git clone https://github.com/IbrahimaKhalil/fiq_islamique-.git
cd jurisprudence_islamique
```

2. Installer les dependances

```bash
pip install -r requirements.txt
```

3. Configurer la cle API

Cree un fichier `.env` a la racine du projet :

```env
GROQ_API_KEY=ta_cle_groq_ici
```

Tu peux obtenir une cle gratuite sur `console.groq.com`.

4. Ajouter tes livres

Place tes fichiers PDF dans le dossier `livres/`.

Exemple :

```text
livres/
Akhdari.pdf
Risala.pdf
```

## Lancement

```bash
streamlit run app.py
```

L'application sera accessible sur `http://localhost:8501`.

## Architecture du projet

```text
cours_rag_langchain/
livres/
chroma_db/
app.py
requirements.txt
.env
.gitignore
```

## Fonctionnement

Le systeme suit les etapes suivantes :

1. Les PDF sont decoupes en morceaux.
2. Les morceaux sont convertis en embeddings.
3. Les donnees sont stockees dans ChromaDB.
4. Le retriever hybride retrouve les passages pertinents.
5. Le modele Groq genere une reponse a partir des extraits trouves.

## Technologies utilisees

- LangChain pour l'orchestration du pipeline RAG.
- ChromaDB pour la base de donnees vectorielle.
- HuggingFace pour les embeddings (`all-MiniLM-L6-v2`).
- Groq avec Llama 3.1 pour la generation des reponses.
- Streamlit pour l'interface utilisateur.
- BM25 pour la recherche par mots-cles.

## Deploiement

L'application peut etre deployee gratuitement sur Streamlit Cloud.

1. Pousse ton code sur GitHub.
2. Connecte ton depot a Streamlit Cloud.
3. Ajoute `GROQ_API_KEY` dans les secrets de l'application.
4. Lance le deploiement.

## Avertissement

Cette application est un outil d'aide a la comprehension. Les reponses sont basees uniquement sur les livres fournis. Pour toute question religieuse importante, consulte un erudit qualifie.

Developpe avec coeur dans le cadre d'un projet d'etude sur le Fiqh Malekite.
