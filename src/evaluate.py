import math
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ===============================
# CONFIG
# ===============================
K = 10                 # Precision@K, Recall@K
TOPN_RATING = 20        # số láng giềng dùng để dự đoán rating (RMSE/MAE)
MIN_GT = 1             # tối thiểu số ground-truth để tính metric (genre-based)


# ===============================
# PATH
# ===============================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
ITEMS_PATH = DATA_DIR / "items.csv"


# ===============================
# LOAD
# ===============================
items = pd.read_csv(ITEMS_PATH)

required_cols = ["text", "genres", "rating"]
for c in required_cols:
    if c not in items.columns:
        raise ValueError(f"❌ items.csv thiếu cột bắt buộc: '{c}'")

items["text"] = items["text"].fillna("")
items["genres"] = items["genres"].fillna("").astype(str)
items["rating"] = pd.to_numeric(items["rating"], errors="coerce")

# lọc item có rating hợp lệ để tính RMSE/MAE
rating_mask = items["rating"].notna()
if rating_mask.sum() == 0:
    raise ValueError("❌ Không có rating hợp lệ trong cột 'rating' để tính RMSE/MAE")


# ===============================
# TF-IDF + SIMILARITY
# ===============================
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
tfidf = vectorizer.fit_transform(items["text"])
sim = cosine_similarity(tfidf)


# ===============================
# HELPERS
# ===============================
def tokenize_genres(s: str) -> set[str]:
    # genres của bạn đang là chuỗi cách nhau bằng SPACE
    # ví dụ: "Contemporary Young Adult New Adult Fiction ..."
    toks = [t.strip().lower() for t in str(s).split() if t.strip()]
    return set(toks)


def recommend_indices(idx: int, k: int) -> list[int]:
    scores = list(enumerate(sim[idx]))
    scores.sort(key=lambda x: x[1], reverse=True)
    # bỏ chính nó
    return [i for i, _ in scores[1:k + 1]]


def precision_at_k(rec: list[int], gt: set[int], k: int) -> float:
    return len(set(rec[:k]) & gt) / k


def recall_at_k(rec: list[int], gt: set[int]) -> float:
    return len(set(rec) & gt) / len(gt) if gt else 0.0


def predict_item_rating(idx: int, topn: int) -> float:
    """
    Dự đoán rating của item idx dựa trên TOP-N item giống nhất (content similarity).
    Pred = weighted average rating của neighbors (weight = similarity).
    """
    neighbors = list(enumerate(sim[idx]))
    neighbors.sort(key=lambda x: x[1], reverse=True)

    num = 0.0
    den = 0.0
    used = 0

    for j, s in neighbors[1:]:  # bỏ chính nó
        if used >= topn:
            break
        r = items.at[j, "rating"]
        if pd.isna(r):
            continue
        if s <= 0:
            continue
        num += s * float(r)
        den += s
        used += 1

    if den == 0:
        # fallback: dùng mean rating toàn tập (an toàn, không lỗi)
        return float(items.loc[rating_mask, "rating"].mean())

    return num / den


# ===============================
# (A) RANKING EVAL: Precision@K, Recall@K (genre as ground truth)
# ===============================
precisions = []
recalls = []
ndcgs = []
evaluated_items = 0

for idx, row in items.iterrows():
    q_genres = tokenize_genres(row["genres"])
    if not q_genres:
        continue

    rec = recommend_indices(idx, K)

    # ground truth: item có chung ít nhất 1 genre token
    gt = set(
        items.index[
            items["genres"].apply(lambda g: len(q_genres & tokenize_genres(g)) > 0)
        ]
    )
    gt.discard(idx)

    if len(gt) < MIN_GT:
        continue

    p = precision_at_k(rec, gt, K)
    r = recall_at_k(rec, gt)

    # nDCG@K (bonus, nhưng hữu ích)
    dcg = 0.0
    for rank, rec_idx in enumerate(rec[:K], start=1):
        if rec_idx in gt:
            dcg += 1.0 / math.log2(rank + 1)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(gt), K)))
    ndcg = dcg / ideal if ideal > 0 else 0.0

    precisions.append(p)
    recalls.append(r)
    ndcgs.append(ndcg)
    evaluated_items += 1


# ===============================
# (B) RATING PREDICTION EVAL: RMSE, MAE (content-based rating prediction)
# ===============================
y_true = items.loc[rating_mask, "rating"].astype(float).values
y_pred = np.zeros_like(y_true)

rating_indices = items.index[rating_mask].tolist()
for t, idx in enumerate(rating_indices):
    y_pred[t] = predict_item_rating(idx, TOPN_RATING)

rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
mae = float(np.mean(np.abs(y_true - y_pred)))


# ===============================
# PRINT RESULTS (đúng yêu cầu đề)
# ===============================
print("\n===== EVALUATION (REQUIREMENTS) =====")
print(f"RMSE: {rmse:.4f}")
print(f"MAE:  {mae:.4f}")
print(f"Precision@{K}: {float(np.mean(precisions)):.4f}" if precisions else f"Precision@{K}: 0.0000")
print(f"Recall@{K}:    {float(np.mean(recalls)):.4f}" if recalls else f"Recall@{K}:    0.0000")

# thêm info để bạn ghi báo cáo (không bắt buộc)
print("\n===== EXTRA (OPTIONAL) =====")
print(f"nDCG@{K}: {float(np.mean(ndcgs)):.4f}" if ndcgs else f"nDCG@{K}: 0.0000")
print(f"ItemsEvaluated: {evaluated_items}")
print(f"RatingPred_Neighbors: {TOPN_RATING}")
