import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def build_tfidf(items: pd.DataFrame):
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=8000,
        ngram_range=(1, 2)
    )
    matrix = vectorizer.fit_transform(items["text"])
    return vectorizer, matrix

def search_book(query, vectorizer, tfidf_matrix):
    q_vec = vectorizer.transform([query])
    sim = cosine_similarity(q_vec, tfidf_matrix)[0]
    idx = int(sim.argmax())
    return idx, float(sim[idx])

def recommend_books(book_idx, items, tfidf_matrix, is_valid_fantasy, k=6):
    candidates = items[items["genres"].apply(is_valid_fantasy)]
    if len(candidates) < k + 1:
        candidates = items

    sim = cosine_similarity(
        tfidf_matrix[book_idx],
        tfidf_matrix[candidates.index]
    )[0]

    order = sim.argsort()[::-1]
    result = []

    for i in order:
        idx = candidates.index[i]
        if idx != book_idx:
            result.append(idx)
        if len(result) == k:
            break

    return items.loc[result]
