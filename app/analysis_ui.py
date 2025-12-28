
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
    # DASHBOARD SETTINGS (CHỈ TRONG DATA ANALYSIS)
    # ===============================
    with st.expander("⚙️ Dashboard Settings", expanded=True):
        k_top = st.slider(
            "Top N (Top items / genres)",
            min_value=5,
            max_value=30,
            value=10
        )

        bins = st.slider(
            "Bins (Histogram)",
            min_value=10,
            max_value=60,
            value=20
        )

        sample_n = st.slider(
            "Sample cho scatter / network",
            min_value=200,
            max_value=min(2000, len(df)),
            value=800
        )

    st.info(
        "📌 Dataset không có cột thời gian → biểu đồ Line/Area sử dụng *pseudo_time* "
        "(mốc giả dựa trên thứ tự item)."
    )

    # ===============================
    # 1️⃣ HISTOGRAM / BOXPLOT / VIOLIN
    # ===============================
    st.subheader("1️⃣ Phân bố Rating (Histogram / Boxplot / Violin)")

    c1, c2, c3 = st.columns(3, gap="large")

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
    # 2️⃣ LINE / AREA (PSEUDO TIME)
    # ===============================
    st.subheader("2️⃣ Line / Area theo thời gian (Pseudo time)")

    agg_time = (
        df.dropna(subset=["rating"])
        .groupby("month", as_index=False)
        .agg(
            avg_rating=("rating", "mean"),
            count=("rating", "size")
        )
    )

    c1, c2 = st.columns(2, gap="large")

    with c1:
        fig_line = px.line(
            agg_time,
            x="month",
            y="avg_rating",
            title="Line: Average Rating theo tháng (pseudo)"
        )
        st.plotly_chart(fig_line, use_container_width=True)

    with c2:
        fig_area = px.area(
            agg_time,
            x="month",
            y="count",
            title="Area: Số lượng sách theo tháng (pseudo)"
        )
        st.plotly_chart(fig_area, use_container_width=True)

    st.divider()

    # ===============================
    # 3️⃣ SCATTER + REGRESSION
    # ===============================
    st.subheader("3️⃣ Scatter + Hồi quy: Rating vs Popularity")

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
        x=x,
        y=y,
        mode="markers",
        text=scatter_df["title"],
        name="Books",
        hovertemplate="log1p(ratings_count)=%{x:.2f}<br>rating=%{y:.2f}<br>%{text}<extra></extra>"
    ))
    fig_scatter.add_trace(go.Scatter(
        x=x_line,
        y=y_line,
        mode="lines",
        name="Regression line"
    ))

    fig_scatter.update_layout(
        title="Scatter: Rating vs log(1 + ratings_count)",
        xaxis_title="log(1 + ratings_count)",
        yaxis_title="Rating"
    )

    st.plotly_chart(fig_scatter, use_container_width=True)

    st.divider()

    # ===============================
    # 4️⃣ CORRELATION HEATMAP
    # ===============================
    st.subheader("4️⃣ Heatmap tương quan")

    corr = df[["rating", "ratings_count"]].dropna().corr()
    fig_corr = px.imshow(
        corr,
        text_auto=True,
        title="Correlation Heatmap"
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    st.divider()

    # ===============================
    # 5️⃣ TREEMAP / SUNBURST
    # ===============================
    st.subheader("5️⃣ Treemap / Sunburst (Genre → Author)")

    tmp = df.copy()
    tmp["main_genre"] = tmp["genres"].str.lower().str.split().str[0].fillna("unknown")
    tmp["main_genre"] = tmp["main_genre"].replace("", "unknown")

    ga = (
        tmp.groupby(["main_genre", "author"], as_index=False)
        .agg(count=("title", "size"))
        .sort_values("count", ascending=False)
    )

    ga_top = ga.groupby("main_genre").head(k_top)

    c1, c2 = st.columns(2, gap="large")

    with c1:
        fig_tree = px.treemap(
            ga_top,
            path=["main_genre", "author"],
            values="count",
            title="Treemap: Genre → Author"
        )
        st.plotly_chart(fig_tree, use_container_width=True)

    with c2:
        fig_sun = px.sunburst(
            ga_top,
            path=["main_genre", "author"],
            values="count",
            title="Sunburst: Genre → Author"
        )
        st.plotly_chart(fig_sun, use_container_width=True)

    st.divider()

    # ===============================
    # 6️⃣ WORDCLOUD
    # ===============================
    st.subheader("6️⃣ WordCloud từ nội dung sách")

    text_all = " ".join(df["text"].dropna().astype(str).tolist())

    if len(text_all.strip()) < 50:
        st.warning("Không đủ dữ liệu để tạo WordCloud.")
    else:
        wc = WordCloud(
            width=900,
            height=400,
            background_color="white"
        ).generate(text_all)

        fig_wc, ax_wc = plt.subplots(figsize=(10, 4))
        ax_wc.imshow(wc, interpolation="bilinear")
        ax_wc.axis("off")
        st.pyplot(fig_wc)

    st.divider()

    # ===============================
    # 7️⃣ NETWORK GRAPH (GENRE CO-OCCURRENCE)
    # ===============================
    st.subheader("7️⃣ Network graph: Co-occurrence giữa genre")

    genre_tokens = df["genres"].str.lower().str.split().explode().dropna()
    top_tokens = genre_tokens.value_counts().head(k_top).index.tolist()

    G = nx.Graph()
    G.add_nodes_from(top_tokens)

    sample_items = df["genres"].str.lower().str.split().dropna().head(sample_n)

    for toks in sample_items:
        toks = [t for t in toks if t in top_tokens]
        for i in range(len(toks)):
            for j in range(i + 1, len(toks)):
                if G.has_edge(toks[i], toks[j]):
                    G[toks[i]][toks[j]]["weight"] += 1
                else:
                    G.add_edge(toks[i], toks[j], weight=1)

    pos = nx.spring_layout(G, seed=42, k=0.7)
    weights = [G[u][v]["weight"] for u, v in G.edges()] if G.edges() else []

    fig_net, ax_net = plt.subplots(figsize=(10, 6))
    nx.draw_networkx_nodes(G, pos, ax=ax_net, node_size=700)
    nx.draw_networkx_labels(G, pos, ax=ax_net, font_size=10)

    if G.edges():
        nx.draw_networkx_edges(
            G,
            pos,
            ax=ax_net,
            width=[0.3 + w * 0.1 for w in weights],
            alpha=0.6
        )

    ax_net.axis("off")
    st.pyplot(fig_net)

    st.caption("Network graph biểu diễn các genre token thường xuất hiện cùng nhau.")
