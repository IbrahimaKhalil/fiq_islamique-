# Assistant Jurisprudence Islamique (RAG)

Application d'intelligence artificielle basée sur le RAG (Retrieval-Augmented Generation) permettant de répondre aux questions de jurisprudence islamique (Fiqh) à partir de livres de référence comme Al-Akhdari, Risala et Ibn Ashir.

## Fonctionnalités

- Multi-livres : charge et analyse plusieurs fichiers PDF simultanément.
- Retriever hybride : combine la recherche par mots-clés (BM25) et la recherche sémantique.
- Citations de sources : l'IA précise le livre et la page d'où provient l'information.
- Interface chat : interface utilisateur moderne avec mémoire de conversation.
- Persistance : la base vectorielle est sauvegardée pour un chargement instantané.

## Installation

### 1. Cloner le projet

```bash
git clone https://github.com/IbrahimaKhalil/fiq_islamique-.git
cd fiq islamique
```

### 2. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 3. Configurer la clé API

Crée un fichier `.env` à la racine du projet :

```env
GROQ_API_KEY=ta_cle_groq_ici
```

Obtiens ta clé gratuite sur `console.groq.com`.

### 4. Ajouter tes livres

Place tes fichiers PDF dans le dossier `livres/`.

Exemple :

```text
livres/
Akhdari.pdf
Risala.pdf
...
```

## Lancement

```bash
streamlit run app.py
```

L'application sera accessible sur `http://localhost:8501`.

## Architecture du projet

```text
cours_rag_langchain/
livres/            Tes livres PDF
chroma_db/         Base vectorielle (auto-générée)
app.py             Application principale
requirements.txt   Dépendances Python
.env               Clé API (non commitée)
.gitignore         Fichiers exclus de Git
```

## Comment ça fonctionne

Le fonctionnement suit ces étapes :

1. Les PDF sont découpés en morceaux.
2. Les morceaux sont convertis en embeddings.
3. Les données sont stockées dans ChromaDB.
4. Le retriever hybride retrouve les passages pertinents.
5. Le modèle Groq génère une réponse à partir des extraits trouvés.

## Technologies utilisées

| Technologie | Rôle |
| --- | --- |
| LangChain | Orchestration du pipeline RAG |
| ChromaDB | Base de données vectorielle |
| HuggingFace | Embeddings (`all-MiniLM-L6-v2`) |
| Groq (Llama 3.1) | Génération de réponses rapide |
| Streamlit | Interface utilisateur |
| BM25 | Recherche par mots-clés |

## Déploiement

L'application est déployable gratuitement sur Streamlit Cloud :

1. Pousse ton code sur GitHub.
2. Connecte ton dépôt sur `share.streamlit.io`.
3. Ajoute `GROQ_API_KEY` dans les secrets de l'application.
4. Clique sur Deploy.

## Avertissement

Cette application est un outil d'aide à la compréhension. Les réponses sont basées uniquement sur les livres fournis. Pour toute question religieuse importante, consulte un érudit qualifié.

Développé avec cœur dans le cadre d'un projet d'étude sur le Fiqh Malékite.
