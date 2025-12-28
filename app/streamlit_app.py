import sys
from pathlib import Path

# ===============================
# PATH FIX
# ===============================
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

import streamlit as st
import pandas as pd
import base64

# HYBRID builders
from src.recommender_hybrid import (
    build_tfidf,
    build_sbert_embeddings
)

from app.recommendation_ui import render_recommendation
from app.analysis_ui import render_analysis

# ===============================
# PATHS
# ===============================
ITEMS_PATH = BASE_DIR / "data/processed/items.csv"
DEFAULT_IMAGE = BASE_DIR / "data/processed/img.png"

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(page_title="📚 Book Recommendation", layout="wide")

# ===============================
# TITLE
# ===============================
st.markdown(
    """
    <h1 style='text-align: center;'>📚 Book Recommendation System</h1>
    <hr>
    """,
    unsafe_allow_html=True
)

# ===============================
# LOAD DATA
# ===============================
@st.cache_data
def load_items():
    df = pd.read_csv(ITEMS_PATH)
    return df.fillna("").reset_index(drop=True)

items = load_items()

# ===============================
# BUILD HYBRID MODELS
# ===============================
@st.cache_resource
def build_models(items):
    vectorizer, tfidf_matrix = build_tfidf(items)
    sbert_embeddings = build_sbert_embeddings(items)
    return vectorizer, tfidf_matrix, sbert_embeddings

vectorizer, tfidf_matrix, sbert_embeddings = build_models(items)

# ===============================
# IMAGE HELPER
# ===============================
def img_to_uri(path: Path):
    data = path.read_bytes()
    b64 = base64.b64encode(data).decode()
    mime = "png" if path.suffix.lower() == ".png" else "jpeg"
    return f"data:image/{mime};base64,{b64}"

default_image_uri = img_to_uri(DEFAULT_IMAGE)

# ===============================
# TABS
# ===============================
tab1, tab2 = st.tabs(["🔍 Recommendation", "📊 Data Analysis"])

# ===============================
# TAB 1: RECOMMENDATION (HYBRID)
# ===============================
with tab1:
    render_recommendation(
        items=items,
        vectorizer=vectorizer,
        tfidf_matrix=tfidf_matrix,
        embeddings=sbert_embeddings,   # ⚠️ TÊN PHẢI LÀ embeddings
        default_image_uri=default_image_uri,
        alpha=0.6                      # có thể chỉnh 0.5–0.7
    )

# ===============================
# TAB 2: DATA ANALYSIS
# ===============================
with tab2:
    render_analysis(items)
