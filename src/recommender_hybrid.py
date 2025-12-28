# src/recommender_hybrid.py
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


# ===============================
# BUILD TF-IDF
# ===============================
def build_tfidf(items):
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=8000,
        ngram_range=(1, 2)
    )
    tfidf_matrix = vectorizer.fit_transform(items["text"])
    return vectorizer, tfidf_matrix


# ===============================
# BUILD SBERT EMBEDDINGS
# ===============================
def build_sbert_embeddings(items, model_name="all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    texts = items["text"].astype(str).tolist()
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True
    )
    return embeddings


# ===============================
# HYBRID SEARCH
# ===============================
def search_book(query, vectorizer, tfidf_matrix, embeddings, alpha=0.5):
    # TF-IDF similarity
    q_tfidf = vectorizer.transform([query])
    sim_tfidf = cosine_similarity(q_tfidf, tfidf_matrix)[0]

    # seed index
    idx = int(sim_tfidf.argmax())

    # SBERT similarity
    sim_embed = cosine_similarity(
        [embeddings[idx]], embeddings
    )[0]

    # Hybrid score
    sim = alpha * sim_tfidf + (1 - alpha) * sim_embed
    best_idx = int(sim.argmax())

    return best_idx, float(sim[best_idx])


# ===============================
# HYBRID RECOMMEND
# ===============================
def recommend_books(idx, items, tfidf_matrix, embeddings, k=6, alpha=0.5):
    sim_tfidf = cosine_similarity(
        tfidf_matrix[idx], tfidf_matrix
    )[0]

    sim_embed = cosine_similarity(
        [embeddings[idx]], embeddings
    )[0]

    sim = alpha * sim_tfidf + (1 - alpha) * sim_embed
    order = sim.argsort()[::-1]

    result = []
    for i in order:
        if i != idx:
            result.append(i)
        if len(result) == k:
            break

    return items.iloc[result]
