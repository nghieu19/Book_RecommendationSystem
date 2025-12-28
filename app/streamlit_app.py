import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

import streamlit as st
import pandas as pd
from pathlib import Path
from src.recommender import build_tfidf
from app.recommendation_ui import render_recommendation
from app.analysis_ui import render_analysis
import base64

BASE_DIR = Path(__file__).resolve().parent.parent
ITEMS_PATH = BASE_DIR / "data/processed/items.csv"
DEFAULT_IMAGE = BASE_DIR / "data/processed/img.png"


st.set_page_config(page_title="📚 Book Recommendation", layout="wide")

# ====== TITLE ======
st.markdown(
    """
    <h1 style='text-align: center;'>📚 Book Recommendation System</h1>
    <h4 style='text-align: center; color: gray;'>
    </h4>
    <hr>
    """,
    unsafe_allow_html=True
)


@st.cache_data
def load_items():
    df = pd.read_csv(ITEMS_PATH)
    return df.fillna("").reset_index(drop=True)

items = load_items()
vectorizer, tfidf_matrix = build_tfidf(items)

def img_to_uri(path: Path):
    data = path.read_bytes()
    b64 = base64.b64encode(data).decode()
    mime = "png" if path.suffix.lower() == ".png" else "jpeg"
    return f"data:image/{mime};base64,{b64}"

default_image_uri = img_to_uri(DEFAULT_IMAGE)

tab1, tab2 = st.tabs(["🔍 Recommendation", "📊 Data Analysis"])

with tab1:
    render_recommendation(
        items,
        vectorizer,
        tfidf_matrix,
        default_image_uri,
        is_valid_fantasy=lambda g: True  # hoặc import hàm filter của bạn
    )

with tab2:
    render_analysis(items)
