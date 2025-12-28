import numpy as np
from sentence_transformers import SentenceTransformer, util

# Load model 1 lần
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def build_embeddings(items):
    """
    items['text']: title + genres + description (đã clean)
    """
    texts = items["text"].astype(str).tolist()
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        convert_to_tensor=True
    )
    return embeddings

def search_book(query, embeddings):
    q_emb = model.encode(query, convert_to_tensor=True)
    scores = util.cos_sim(q_emb, embeddings)[0]
    idx = int(scores.argmax())
    return idx, float(scores[idx])

def recommend_books(book_idx, items, embeddings, is_valid_genre, k=6):
    candidates = items[items["genres"].apply(is_valid_genre)]

    if len(candidates) < k + 1:
        candidates = items

    scores = util.cos_sim(
        embeddings[book_idx],
        embeddings[candidates.index]
    )[0]

    order = scores.argsort(descending=True)
    result = []

    for i in order:
        idx = candidates.index[int(i)]
        if idx != book_idx:
            result.append(idx)
        if len(result) == k:
            break

    return items.loc[result]
