import streamlit as st
from pathlib import Path
import base64
from src.recommender import search_book, recommend_books

def img_to_uri(path: Path):
    data = path.read_bytes()
    b64 = base64.b64encode(data).decode()
    mime = "png" if path.suffix.lower() == ".png" else "jpeg"
    return f"data:image/{mime};base64,{b64}"

def show_card(image_path, default_uri, height=220):
    try:
        p = Path(image_path)
        uri = img_to_uri(p) if p.exists() else default_uri
    except:
        uri = default_uri

    st.markdown(
        f"""
        <div style="width:100%;height:{height}px;border-radius:14px;overflow:hidden">
            <img src="{uri}" style="width:100%;height:100%;object-fit:cover"/>
        </div>
        """,
        unsafe_allow_html=True
    )

def render_recommendation(items, vectorizer, tfidf_matrix, default_image_uri, is_valid_fantasy):
    st.subheader("🔍 Recommendation")

    query = st.text_input("🔍 Nhập tên sách")

    if not query.strip():
        st.info("👆 Nhập tên sách để bắt đầu")
        return

    idx, score = search_book(query, vectorizer, tfidf_matrix)
    book = items.iloc[idx]

    c1, c2 = st.columns([1, 3])
    with c1:
        show_card(book["image_path"], default_image_uri, height=260)
    with c2:
        st.markdown(f"### {book['title']}")
        st.markdown(f"✍️ {book['author']}")
        st.caption(book["genres"])
        st.success(f"Độ khớp: {score*100:.2f}%")

    st.divider()
    st.subheader("✨ Sách cùng chủ đề")

    recs = recommend_books(idx, items, tfidf_matrix, is_valid_fantasy)
    cols = st.columns(3)

    for i, (_, r) in enumerate(recs.iterrows()):
        with cols[i % 3]:
            show_card(r["image_path"], default_image_uri)
            st.markdown(f"**{r['title']}**")
            st.caption(r["author"])
