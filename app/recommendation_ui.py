import streamlit as st
from pathlib import Path
import base64

# HYBRID recommender
from src.recommender_hybrid import search_book, recommend_books
def get_seed_book(query, items):
    exact = items[
        items["title"]
        .str.lower()
        .str.contains(query.lower(), na=False)
    ]
    if len(exact) > 0:
        return exact.index[0], 1.0
    return None, None


# ===============================
# IMAGE UTILS
# ===============================
def img_to_uri(path: Path):
    data = path.read_bytes()
    b64 = base64.b64encode(data).decode()
    mime = "png" if path.suffix.lower() == ".png" else "jpeg"
    return f"data:image/{mime};base64,{b64}"


def show_card(image_path, default_uri, height=220):
    try:
        p = Path(image_path)
        uri = img_to_uri(p) if image_path and p.exists() else default_uri
    except:
        uri = default_uri

    st.markdown(
        f"""
        <div style="
            width:100%;
            height:{height}px;
            border-radius:14px;
            overflow:hidden;
            border:1px solid rgba(0,0,0,0.08)">
            <img src="{uri}" style="
                width:100%;
                height:100%;
                object-fit:cover"/>
        </div>
        """,
        unsafe_allow_html=True
    )


def short(text, n=45):
    return text if len(text) <= n else text[: n - 1] + "…"


# ===============================
# MAIN UI
# ===============================
def render_recommendation(
    items,
    vectorizer,
    tfidf_matrix,
    embeddings,
    default_image_uri,
    alpha=0.6
):
    """
    HYBRID Recommendation UI
    alpha: weight between TF-IDF and SBERT (0–1)
    """

    st.subheader("🔍 Recommendation (Hybrid TF-IDF + SBERT)")

    query = st.text_input(
        "🔍 Nhập tên sách",
        placeholder="Harry Potter, Fantasy, Dragon..."
    )

    if not query.strip():
        st.info("👆 Nhập tên sách để bắt đầu")
        return

    # ===============================
    # SEARCH (HYBRID)
    # ===============================
    seed_idx, score = get_seed_book(query, items)

    if seed_idx is None:
        seed_idx, score = search_book(
            query,
            vectorizer,
            tfidf_matrix,
            embeddings,
            alpha
        )

    book = items.iloc[seed_idx]

    c1, c2 = st.columns([1, 3], gap="large")

    with c1:
        show_card(book["image_path"], default_image_uri, height=260)

    with c2:
        st.markdown(f"### {book['title']}")
        st.markdown(f"✍️ {book['author']}")
        st.caption(book.get("genres", ""))
        st.success(f"Hybrid similarity score: {score * 100:.2f}%")

    st.divider()

    # ===============================
    # RECOMMEND (HYBRID)
    # ===============================
    st.subheader("✨ Sách được gợi ý")

    recs = recommend_books(
        seed_idx,
        items,
        tfidf_matrix,
        embeddings,
        k=6,
        alpha=alpha
    )

    cols = st.columns(3, gap="large")

    for i, (_, r) in enumerate(recs.iterrows()):
        with cols[i % 3]:
            show_card(r["image_path"], default_image_uri)
            st.markdown(f"**{short(r['title'])}**")
            st.caption(short(r["author"], 35))
