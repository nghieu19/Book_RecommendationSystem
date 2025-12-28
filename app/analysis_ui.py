import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from wordcloud import WordCloud


def render_analysis(items: pd.DataFrame):
    """
    Render Data Analysis / EDA Dashboard
    """

    st.header("📊 Phân tích & Trực quan hóa dữ liệu (EDA Dashboard)")

    # ===============================
    # CLEAN & PREPARE DATA
    # ===============================
    df = items.copy()

    df["rating"] = pd.to_numeric(df.get("rating", np.nan), errors="coerce")
    df["ratings_count"] = pd.to_numeric(df.get("ratings_count", np.nan), errors="coerce")
    df["title"] = df.get("title", "").astype(str)
    df["author"] = df.get("author", "").astype(str)
    df["genres"] = df.get("genres", "").astype(str)
    df["text"] = df.get("text", "").astype(str)

    # Pseudo time (dataset không có thời gian thật)
    df = df.reset_index(drop=True)
    df["pseudo_time"] = pd.to_datetime(df.index, unit="D", origin="2020-01-01")
    df["month"] = df["pseudo_time"].dt.to_period("M").astype(str)

    # ===============================
    # DASHBOARD SETTINGS
    # ===============================
    with st.expander("⚙️ Dashboard Settings", expanded=True):
        k_top = st.slider("Top N (items / genres)", 5, 30, 10)
        bins = st.slider("Bins (Histogram)", 10, 60, 20)
        sample_n = st.slider(
            "Sample cho scatter / network",
            200,
            min(2000, len(df)),
            800
        )

    st.info(
        "📌 Dataset không có cột thời gian → biểu đồ Line/Area "
        "sử dụng *pseudo_time* (mốc giả theo thứ tự item)."
    )

    # ===============================
    # 1️⃣ PHÂN BỐ RATING
    # ===============================
    st.subheader("1️⃣ Phân bố Rating (Histogram / Boxplot / Violin)")

    c1, c2, c3 = st.columns(3)

    with c1:
        fig_hist = px.histogram(
            df.dropna(subset=["rating"]),
            x="rating",
            nbins=bins,
            title="Histogram: Rating"
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with c2:
        fig_box = px.box(
            df.dropna(subset=["rating"]),
            y="rating",
            points="outliers",
            title="Boxplot: Rating"
        )
        st.plotly_chart(fig_box, use_container_width=True)

    with c3:
        fig_violin = px.violin(
            df.dropna(subset=["rating"]),
            y="rating",
            box=True,
            points="all",
            title="Violin: Rating"
        )
        st.plotly_chart(fig_violin, use_container_width=True)

    st.divider()

    # ===============================
    # 2️⃣ TẦN SUẤT NHÓM SẢN PHẨM
    # ===============================
    st.subheader("2️⃣ Tần suất nhóm sản phẩm (Genres)")

    genre_tokens = (
        df["genres"]
        .str.lower()
        .str.split()
        .explode()
        .dropna()
    )

    genre_freq = (
        genre_tokens.value_counts()
        .head(k_top)
        .reset_index()
    )
    genre_freq.columns = ["genre", "count"]

    fig_genre_bar = px.bar(
        genre_freq,
        x="genre",
        y="count",
        text="count",
        title=f"Top {k_top} thể loại phổ biến nhất"
    )
    st.plotly_chart(fig_genre_bar, use_container_width=True)

    st.divider()

    # ===============================
    # 3️⃣ TOP ITEMS – PHỔ BIẾN NHẤT
    # ===============================
    st.subheader("3️⃣ Top Items – Sách phổ biến nhất")

    top_popular = (
        df.dropna(subset=["ratings_count"])
        .sort_values("ratings_count", ascending=False)
        .head(k_top)
    )

    fig_top_pop = px.bar(
        top_popular,
        x="ratings_count",
        y="title",
        orientation="h",
        text="ratings_count",
        title=f"Top {k_top} sách có nhiều lượt đánh giá nhất"
    )
    fig_top_pop.update_layout(yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig_top_pop, use_container_width=True)

    with st.expander("📄 Bảng chi tiết"):
        st.dataframe(
            top_popular[["title", "author", "ratings_count", "rating"]],
            use_container_width=True
        )

    st.divider()

    # ===============================
    # 4️⃣ TOP ITEMS – RATING CAO
    # ===============================
    st.subheader("4️⃣ Top Items – Sách có rating cao nhất")

    min_votes = st.slider(
        "Số lượt đánh giá tối thiểu",
        10, 500, 50, step=10
    )

    top_rated = (
        df.dropna(subset=["rating", "ratings_count"])
        .query("ratings_count >= @min_votes")
        .sort_values("rating", ascending=False)
        .head(k_top)
    )

    fig_top_rating = px.bar(
        top_rated,
        x="rating",
        y="title",
        orientation="h",
        text="rating",
        title=f"Top {k_top} sách rating cao nhất (≥ {min_votes} votes)"
    )
    fig_top_rating.update_layout(
        yaxis=dict(autorange="reversed"),
        xaxis=dict(range=[0, 5])
    )
    st.plotly_chart(fig_top_rating, use_container_width=True)

    st.divider()

    # ===============================
    # 5️⃣ SCATTER + HỒI QUY
    # ===============================
    st.subheader("5️⃣ Scatter + Hồi quy: Rating vs Popularity")

    scatter_df = df.dropna(subset=["rating", "ratings_count"]).sample(
        min(sample_n, len(df.dropna(subset=["rating", "ratings_count"]))),
        random_state=42
    )

    x = np.log1p(scatter_df["ratings_count"].values)
    y = scatter_df["rating"].values

    a, b = np.polyfit(x, y, 1)
    x_line = np.linspace(x.min(), x.max(), 200)
    y_line = a * x_line + b

    fig_scatter = go.Figure()
    fig_scatter.add_trace(go.Scatter(
        x=x, y=y, mode="markers",
        text=scatter_df["title"],
        name="Books"
    ))
    fig_scatter.add_trace(go.Scatter(
        x=x_line, y=y_line, mode="lines",
        name="Regression"
    ))
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.divider()

    # ===============================
    # 6️⃣ HEATMAP TƯƠNG QUAN
    # ===============================
    st.subheader("6️⃣ Heatmap tương quan")

    corr = df[["rating", "ratings_count"]].dropna().corr()
    fig_corr = px.imshow(corr, text_auto=True)
    st.plotly_chart(fig_corr, use_container_width=True)

    st.divider()

    # ===============================
    # 7️⃣ WORDCLOUD
    # ===============================
    st.subheader("7️⃣ WordCloud từ nội dung sách")

    text_all = " ".join(df["text"].dropna())
    wc = WordCloud(width=900, height=400, background_color="white").generate(text_all)

    fig_wc, ax_wc = plt.subplots(figsize=(10, 4))
    ax_wc.imshow(wc, interpolation="bilinear")
    ax_wc.axis("off")
    st.pyplot(fig_wc)

    st.divider()

    # ===============================
    # 8️⃣ NETWORK GRAPH
    # ===============================
    st.subheader("8️⃣ Network graph: Co-occurrence giữa genre")

    top_tokens = genre_tokens.value_counts().head(k_top).index.tolist()
    G = nx.Graph()
    G.add_nodes_from(top_tokens)

    for toks in df["genres"].str.lower().str.split().dropna().head(sample_n):
        toks = [t for t in toks if t in top_tokens]
        for i in range(len(toks)):
            for j in range(i + 1, len(toks)):
                if G.has_edge(toks[i], toks[j]):
                    G[toks[i]][toks[j]]["weight"] += 1
                else:
                    G.add_edge(toks[i], toks[j], weight=1)

    pos = nx.spring_layout(G, seed=42)
    fig_net, ax_net = plt.subplots(figsize=(10, 6))
    nx.draw_networkx(G, pos, ax=ax_net)
    ax_net.axis("off")
    st.pyplot(fig_net)
