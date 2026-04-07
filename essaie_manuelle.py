
# Notre "livre" islamique simplifié
texte = """
Le wudu commence par l'intention dans le coeur.
Ensuite on lave les mains trois fois.
Puis on se rince la bouche trois fois.
Puis on lave le visage trois fois.
Puis on lave les bras jusqu'aux coudes.
Le ghusl est obligatoire après la janaba.
Le ghusl commence par laver les parties intimes.
La tayammum remplace le wudu quand il n'y a pas d'eau.
"""

texte = "Le lavage des mains est sunna. L'intention est obligatoire. Se laver le visage est fard."
# Découpage manuel par phrase
chunks = texte.split(". ")
# Résultat : ["Le lavage des mains est sunna", "L'intention est obligatoire", ...]
#Embédding mathématique 

base_de_données = [
    {"id":1 ,"texte":"Le lavage des mains est sunna","vecteur":[1.2,0.5]},
    {"id":2 ,"texte":"L'intention est obligatoire","vecteur":[0.1,3.8]}
]

chuncks= [phase.strip() for phase in texte.split(".") if phase.strip()]
#resulat :
for i, chunk in enumerate(chuncks):
    print(f"Chunk {i}: {chunk}")

# Dictionnaire de mots importants et leur "poids"
def embedding_simple(phrase):
    mots_cles = {
        "wudu": [1, 0, 0, 0],
        "ghusl": [0, 1, 0, 0],
        "tayammum": [0, 0, 1, 0],
        "intention": [0, 0, 0, 1],
        "mains": [0.5, 0, 0, 0],
        "visage": [0.5, 0, 0, 0],
        "eau": [0, 0, 0.5, 0],
        "janaba": [0, 1, 0, 0],
    }
    
    vecteur = [0, 0, 0, 0]
    for mot, poids in mots_cles.items():
        if mot in phrase.lower():
            vecteur = [vecteur[i] + poids[i] for i in range(4)]
    return vecteur

# On crée notre "base de données" à la main
base = []
for i, chunk in enumerate(chunks):
    base.append({
        "id": i,
        "texte": chunk,
        "vecteur": embedding_simple(chunk)
    })

print("Base créée :")
for doc in base:
    print(f"  [{doc['vecteur']}] → {doc['texte']}")

import math

# Calcul de la similarité cosinus (mesure la proximité entre 2 vecteurs)
def similarite(v1, v2):
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a**2 for a in v1))
    norm2 = math.sqrt(sum(b**2 for b in v2))
    if norm1 == 0 or norm2 == 0:
        return 0
    return dot / (norm1 * norm2)

# Recherche du chunk le plus proche
def rechercher(question, base, top_k=2):
    vecteur_question = embedding_simple(question)
    scores = []
    for doc in base:
        score = similarite(vecteur_question, doc["vecteur"])
        scores.append((score, doc["texte"]))
    scores.sort(reverse=True)
    return scores[:top_k]

# Test
question = "Comment faire le wudu ?"
resultats = rechercher(question, base)

print(f"\nQuestion : {question}")
print("Passages trouvés :")
for score, texte in resultats:
    print(f"  Score {score:.2f} → {texte}")
# On simule ce que le LLM reçoit
contexte = "\n".join([texte for _, texte in resultats])

prompt_final = f"""
Tu es un assistant islamique.
Utilise UNIQUEMENT ce contexte pour répondre.

Contexte :
{contexte}

Question : {question}
Réponse :
"""

print(prompt_final)

"""def chunk_text(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")

chunks = chunk_text(texte)
vectors = model.encode(chunks)  # shape: (nb_chunks, 768)

import faiss
import numpy as np

# Indexer les vecteurs
index = faiss.IndexFlatL2(768)
index.add(np.array(vectors))

# Rechercher avec une question
question = "Comment faire le wudu ?"
q_vector = model.encode([question])
distances, indices = index.search(q_vector, k=3)  # top 3 passages

# Récupérer les passages
for i in indices[0]:
    print(chunks[i])
"""