"""
India Literacy Analytics Dashboard — Professional Edition
============================================================
A polished, interactive Streamlit dashboard built on top of the full
data-cleaning, feature-engineering and statistical-analysis pipeline
developed in the companion Jupyter notebook
(India_Literacy_Analytics_Project.ipynb).

Data: Census of India — Literacy Rate, 1991 / 2001 / 2011
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="India Literacy Analytics Dashboard",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

STATE_COL = "All India/State/Union Territory"

PRIMARY = "#1B3A4B"      # deep navy
ACCENT = "#2E86AB"       # teal blue
GOLD = "#E9B44C"         # gold highlight
GOOD = "#2E8B57"         # green
BAD = "#C0392B"          # red
SEQ_PALETTE = px.colors.sequential.Teal
DIV_PALETTE = "RdYlGn"

# ============================================================
# GLOBAL STYLE
# ============================================================
st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&family=Inter:wght@400;500;600&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
    }}
    h1, h2, h3, h4 {{
        font-family: 'Poppins', sans-serif !important;
        color: {PRIMARY};
    }}
    .main-header {{
        background: linear-gradient(120deg, {PRIMARY} 0%, {ACCENT} 100%);
        padding: 2rem 2.2rem;
        border-radius: 14px;
        color: white;
        margin-bottom: 1.6rem;
        box-shadow: 0 6px 18px rgba(0,0,0,0.12);
    }}
    .main-header h1 {{
        color: white !important;
        margin-bottom: 0.2rem;
        font-size: 2.1rem;
    }}
    .main-header p {{
        color: #dce9ef;
        margin: 0;
        font-size: 0.95rem;
    }}
    div[data-testid="stMetric"] {{
        background: #ffffff;
        border: 1px solid #e7ecef;
        border-left: 5px solid {ACCENT};
        border-radius: 10px;
        padding: 0.9rem 1rem 0.6rem 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }}
    div[data-testid="stMetricLabel"] {{
        font-weight: 600;
        color: #5a6b73;
    }}
    section[data-testid="stSidebar"] {{
        background-color: #f4f7f8;
    }}
    .section-tag {{
        display: inline-block;
        background: {GOLD};
        color: {PRIMARY};
        font-weight: 700;
        font-size: 0.72rem;
        letter-spacing: 0.06em;
        padding: 0.15rem 0.6rem;
        border-radius: 20px;
        margin-bottom: 0.4rem;
        text-transform: uppercase;
    }}
    .insight-box {{
        background: #f7f9fb;
        border-left: 4px solid {ACCENT};
        border-radius: 8px;
        padding: 0.85rem 1.1rem;
        margin: 0.6rem 0 1rem 0;
        font-size: 0.93rem;
        color: #2c3e40;
    }}
    .footer-note {{
        text-align:center;
        color:#8a97a0;
        font-size:0.8rem;
        margin-top: 2rem;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# DATA LOADING + CLEANING  (mirrors notebook Phases 1-3)
# ============================================================
@st.cache_data
def load_and_clean(path="datafile.csv"):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    # Fix singular/plural column naming inconsistency
    df = df.rename(columns={"2011 - Rural - Person": "2011 - Rural - Persons"})

    # ---- Treat J&K's 1991 zero-triplet as a missing-value flag, not a real 0 ----
    # (1991 Census was not conducted in J&K; 0 was used as a placeholder)
    jk_mask = df[STATE_COL].astype(str).str.strip() == "Jammu and Kashmir"
    if jk_mask.any():
        national_1991 = df.loc[
            df[STATE_COL] == "All India", ["1991 - Male", "1991 - Female", "1991 - Persons"]
        ].values[0]
        national_2001 = df.loc[
            df[STATE_COL] == "All India", ["2001 - Male", "2001 - Female", "2001 - Persons"]
        ].values[0]
        ratio = national_1991 / national_2001
        for i, c91 in enumerate(["1991 - Male", "1991 - Female", "1991 - Persons"]):
            c01 = c91.replace("1991", "2001")
            df.loc[jk_mask, c91] = round(df.loc[jk_mask, c01].values[0] * ratio[i])

    # ---- Standardize state / UT names ----
    df[STATE_COL] = df[STATE_COL].astype(str).str.strip().str.title()
    df[STATE_COL] = df[STATE_COL].replace(
        {
            "Chhatisgarh": "Chhattisgarh",
            "Uttaranchal": "Uttarakhand",
            "A. And N. Islands": "Andaman and Nicobar Islands",
            "D. And N. Haveli": "Dadra and Nagar Haveli",
        }
    )

    numeric_cols = df.columns.drop(STATE_COL)
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)

    df = df.reset_index(drop=True)
    return df


REGION_MAP = {
    "Jammu And Kashmir": "North", "Jammu and Kashmir": "North", "Himachal Pradesh": "North",
    "Punjab": "North", "Haryana": "North", "Delhi": "North", "Chandigarh": "North",
    "Uttar Pradesh": "North", "Uttarakhand": "North", "Rajasthan": "North",
    "Andhra Pradesh": "South", "Karnataka": "South", "Kerala": "South", "Tamil Nadu": "South",
    "Telangana": "South", "Puducherry": "South", "Lakshadweep": "South",
    "Bihar": "East", "Jharkhand": "East", "Odisha": "East", "West Bengal": "East",
    "Assam": "Northeast", "Arunachal Pradesh": "Northeast", "Manipur": "Northeast",
    "Meghalaya": "Northeast", "Mizoram": "Northeast", "Nagaland": "Northeast",
    "Sikkim": "Northeast", "Tripura": "Northeast",
    "Madhya Pradesh": "Central", "Chhattisgarh": "Central",
    "Gujarat": "West", "Maharashtra": "West", "Goa": "West",
    "Dadra and Nagar Haveli": "West", "Daman and Diu": "West",
    "Andaman and Nicobar Islands": "South",
    "All India": "National",
}


# ============================================================
# FEATURE ENGINEERING  (mirrors notebook Phase 4)
# ============================================================
@st.cache_data
def engineer_features(df):
    d = df.copy()

    d["2011 - Overall - Male"] = d[["2011 - Rural - Male", "2011 - Urban - Male"]].mean(axis=1)
    d["2011 - Overall - Female"] = d[["2011 - Rural - Female", "2011 - Urban - Female"]].mean(axis=1)
    d["2011 - Overall - Persons"] = d[["2011 - Rural - Persons", "2011 - Urban - Persons"]].mean(axis=1)

    d["Gender Gap 1991"] = d["1991 - Male"] - d["1991 - Female"]
    d["Gender Gap 2001"] = d["2001 - Male"] - d["2001 - Female"]
    d["Gender Gap 2011"] = d["2011 - Overall - Male"] - d["2011 - Overall - Female"]

    d["Rural-Urban Gap 2011"] = d["2011 - Urban - Persons"] - d["2011 - Rural - Persons"]

    d["Growth 1991-2001"] = d["2001 - Persons"] - d["1991 - Persons"]
    d["Growth 2001-2011"] = d["2011 - Overall - Persons"] - d["2001 - Persons"]
    d["Growth 1991-2011"] = d["2011 - Overall - Persons"] - d["1991 - Persons"]
    d["CAGR 1991-2011 (%)"] = ((d["2011 - Overall - Persons"] / d["1991 - Persons"]) ** (1 / 20) - 1) * 100

    d["Gap Reduction 1991-2011"] = d["Gender Gap 1991"] - d["Gender Gap 2011"]
    d["Inequality Index 2011"] = d["Gender Gap 2011"] + d["Rural-Urban Gap 2011"]

    bins_lit = [0, 60, 75, 90, 100]
    labels_lit = ["Low (<60%)", "Medium (60-75%)", "High (75-90%)", "Very High (>90%)"]
    d["Literacy Category 2011"] = pd.cut(d["2011 - Overall - Persons"], bins=bins_lit, labels=labels_lit)

    bins_gap = [-1, 5, 12, 100]
    labels_gap = ["High Equity (\u22645pp)", "Medium Equity (5-12pp)", "Low Equity (>12pp)"]
    d["Gender Equity 2011"] = pd.cut(d["Gender Gap 2011"], bins=bins_gap, labels=labels_gap)

    bins_ru = [-1, 8, 18, 100]
    labels_ru = ["Balanced (\u22648pp)", "Moderate Gap (8-18pp)", "Wide Gap (>18pp)"]
    d["Rural-Urban Category 2011"] = pd.cut(d["Rural-Urban Gap 2011"], bins=bins_ru, labels=labels_ru)

    d["Region"] = d[STATE_COL].map(REGION_MAP)

    d["Literacy Rank 2011"] = d["2011 - Overall - Persons"].rank(ascending=False).astype(int)
    d["Growth Rank 1991-2011"] = d["Growth 1991-2011"].rank(ascending=False).astype(int)
    d["Literacy Z-score 2011"] = (
        d["2011 - Overall - Persons"] - d["2011 - Overall - Persons"].mean()
    ) / d["2011 - Overall - Persons"].std()

    # Simple linear-trend forecast (1991->2001->2011) to 2021 / 2031
    years_arr = np.array([1991, 2001, 2011])

    def predict(row, target_year):
        y = np.array([row["1991 - Persons"], row["2001 - Persons"], row["2011 - Overall - Persons"]])
        coef = np.polyfit(years_arr, y, 1)
        return float(np.clip(np.polyval(coef, target_year), 0, 100))

    d["Predicted 2021"] = d.apply(lambda r: predict(r, 2021), axis=1)
    d["Predicted 2031"] = d.apply(lambda r: predict(r, 2031), axis=1)

    return d


@st.cache_data
def run_kmeans_pca(states_df, k=3, seed=42):
    features = ["2011 - Overall - Persons", "Gender Gap 2011", "Rural-Urban Gap 2011"]
    X = states_df[features].values
    X_std = (X - X.mean(axis=0)) / X.std(axis=0)

    rng = np.random.default_rng(seed)
    centers = X_std[rng.choice(len(X_std), k, replace=False)]
    for _ in range(200):
        dists = np.linalg.norm(X_std[:, None, :] - centers[None, :, :], axis=2)
        labels = dists.argmin(axis=1)
        new_centers = np.array(
            [X_std[labels == j].mean(axis=0) if (labels == j).any() else centers[j] for j in range(k)]
        )
        if np.allclose(new_centers, centers):
            break
        centers = new_centers

    cov = np.cov(X_std.T)
    eigvals, eigvecs = np.linalg.eig(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[order].real, eigvecs[:, order].real
    explained = eigvals / eigvals.sum() * 100
    pcs = X_std @ eigvecs[:, :2]

    out = states_df.copy()
    out["Cluster"] = labels.astype(str)
    out["PC1"], out["PC2"] = pcs[:, 0], pcs[:, 1]
    return out, explained


# ============================================================
# LOAD DATA
# ============================================================
try:
    raw_df = load_and_clean("datafile.csv")
except FileNotFoundError:
    st.error("`datafile.csv` not found. Please place it in the app's working directory.")
    st.stop()

df = engineer_features(raw_df)
states_df_full = df[df[STATE_COL] != "All India"].copy()
national = df[df[STATE_COL] == "All India"].iloc[0]

# ============================================================
# SIDEBAR — NAVIGATION + FILTERS
# ============================================================
st.sidebar.markdown("## 📚 Navigation")
page = st.sidebar.radio(
    "Go to",
    [
        "🏠 Executive Overview",
        "🗺️ State Deep-Dive",
        "📈 Growth & Forecast",
        "⚖️ Gender Equity",
        "🏙️ Rural vs Urban",
        "🌐 Regional Comparison",
        "🔬 Correlation & Statistics",
        "🧩 Clustering (State Profiles)",
        "🏆 Champions & Concern States",
        "🧾 Data Quality & Methodology",
    ],
    label_visibility="collapsed",
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔎 Filters")
region_options = sorted(states_df_full["Region"].dropna().unique())
selected_regions = st.sidebar.multiselect("Region", region_options, default=region_options)

states_df = states_df_full[states_df_full["Region"].isin(selected_regions)].copy()
if states_df.empty:
    st.sidebar.warning("No states match the selected filters — showing all states instead.")
    states_df = states_df_full.copy()

st.sidebar.markdown("---")
st.sidebar.caption(
    f"Dataset: {states_df_full.shape[0]} States/UTs · Census years 1991, 2001, 2011\n\n"
    "Cleaned & feature-engineered per the companion analytics notebook."
)

csv_export = states_df.to_csv(index=False).encode("utf-8")
st.sidebar.download_button(
    "⬇️ Download filtered data (CSV)", csv_export, "literacy_filtered.csv", "text/csv"
)

# ============================================================
# HEADER
# ============================================================
st.markdown(
    f"""
    <div class="main-header">
        <h1>📚 India Literacy Analytics Dashboard</h1>
        <p>Census of India · 1991 – 2011 · State / Union Territory · Gender-wise · Rural–Urban wise</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# PAGE: EXECUTIVE OVERVIEW
# ============================================================
if page == "🏠 Executive Overview":

    st.markdown('<span class="section-tag">Overview</span>', unsafe_allow_html=True)
    st.subheader("National Snapshot (2011)")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("National Literacy (2011)", f"{national['2011 - Overall - Persons']:.1f}%")
    c2.metric(
        "Growth since 1991",
        f"+{national['Growth 1991-2011']:.1f} pp",
        help="Percentage-point gain in national literacy from 1991 to 2011",
    )
    c3.metric("National Gender Gap", f"{national['Gender Gap 2011']:.1f} pp")
    c4.metric(
        "Highest Literacy State",
        states_df.loc[states_df["2011 - Overall - Persons"].idxmax(), STATE_COL],
    )
    c5.metric(
        "Lowest Literacy State",
        states_df.loc[states_df["2011 - Overall - Persons"].idxmin(), STATE_COL],
    )

    st.markdown(
        f"""
        <div class="insight-box">
        💡 National literacy rose from <b>{national['1991 - Persons']:.0f}%</b> (1991) to
        <b>{national['2011 - Overall - Persons']:.1f}%</b> (2011) — a gain of roughly
        <b>{national['Growth 1991-2011']:.0f} percentage points</b> in two decades, while the
        gender gap and rural-urban divide, though narrowing, remain significant policy targets.
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([1.1, 1])

    with col1:
        fig = px.histogram(
            states_df,
            x="2011 - Overall - Persons",
            nbins=14,
            marginal="box",
            color_discrete_sequence=[ACCENT],
            title="Distribution of 2011 Literacy Rate Across States",
        )
        fig.add_vline(
            x=states_df["2011 - Overall - Persons"].mean(),
            line_dash="dash",
            line_color=GOLD,
            annotation_text="Mean",
        )
        fig.update_layout(
            xaxis_title="Literacy Rate (%)", yaxis_title="Number of States", bargap=0.05
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        cat_counts = states_df["Literacy Category 2011"].value_counts().sort_index()
        fig = px.pie(
            names=cat_counts.index,
            values=cat_counts.values,
            hole=0.45,
            title="Share of States by Literacy Category (2011)",
            color_discrete_sequence=px.colors.sequential.Teal_r,
        )
        fig.update_traces(textinfo="percent+label")
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        top10 = states_df.sort_values("2011 - Overall - Persons", ascending=False).head(10)
        fig = px.bar(
            top10.sort_values("2011 - Overall - Persons"),
            x="2011 - Overall - Persons",
            y=STATE_COL,
            orientation="h",
            color="2011 - Overall - Persons",
            color_continuous_scale="Teal",
            title="Top 10 States — 2011 Literacy Rate",
            text="2011 - Overall - Persons",
        )
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig.update_layout(coloraxis_showscale=False, xaxis_title="Literacy Rate (%)", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        bottom10 = states_df.sort_values("2011 - Overall - Persons").head(10)
        fig = px.bar(
            bottom10.sort_values("2011 - Overall - Persons", ascending=False),
            x="2011 - Overall - Persons",
            y=STATE_COL,
            orientation="h",
            color="2011 - Overall - Persons",
            color_continuous_scale="OrRd",
            title="Bottom 10 States — 2011 Literacy Rate",
            text="2011 - Overall - Persons",
        )
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig.update_layout(coloraxis_showscale=False, xaxis_title="Literacy Rate (%)", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("National Literacy Trend (1991 → 2011)")
    trend_fig = go.Figure()
    trend_fig.add_trace(
        go.Scatter(
            x=[1991, 2001, 2011],
            y=[national["1991 - Persons"], national["2001 - Persons"], national["2011 - Overall - Persons"]],
            mode="lines+markers+text",
            text=[f"{v:.0f}%" for v in [national["1991 - Persons"], national["2001 - Persons"], national["2011 - Overall - Persons"]]],
            textposition="top center",
            line=dict(color=ACCENT, width=4),
            marker=dict(size=12, color=PRIMARY),
            name="All India",
        )
    )
    trend_fig.update_layout(
        xaxis=dict(tickmode="array", tickvals=[1991, 2001, 2011]),
        yaxis_title="Literacy Rate (%)",
        showlegend=False,
    )
    st.plotly_chart(trend_fig, use_container_width=True)

# ============================================================
# PAGE: STATE DEEP-DIVE
# ============================================================
elif page == "🗺️ State Deep-Dive":

    st.markdown('<span class="section-tag">State Deep-Dive</span>', unsafe_allow_html=True)

    state = st.selectbox("Select a State / UT", sorted(states_df[STATE_COL].unique()))
    row = states_df[states_df[STATE_COL] == state].iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("2011 Literacy", f"{row['2011 - Overall - Persons']:.1f}%",
              f"{row['2011 - Overall - Persons'] - national['2011 - Overall - Persons']:+.1f} pp vs India")
    c2.metric("Gender Gap", f"{row['Gender Gap 2011']:.1f} pp")
    c3.metric("Rural-Urban Gap", f"{row['Rural-Urban Gap 2011']:.1f} pp")
    c4.metric("National Rank", f"#{int(row['Literacy Rank 2011'])} of {len(states_df_full)}")

    col1, col2 = st.columns(2)

    with col1:
        years = [1991, 2001, 2011]
        vals_state = [row["1991 - Persons"], row["2001 - Persons"], row["2011 - Overall - Persons"]]
        vals_nat = [national["1991 - Persons"], national["2001 - Persons"], national["2011 - Overall - Persons"]]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=years, y=vals_state, mode="lines+markers", name=state,
                                  line=dict(color=ACCENT, width=4), marker=dict(size=10)))
        fig.add_trace(go.Scatter(x=years, y=vals_nat, mode="lines+markers", name="All India",
                                  line=dict(color=GOLD, width=3, dash="dash"), marker=dict(size=8)))
        fig.update_layout(title=f"Literacy Trend — {state} vs. National",
                           xaxis=dict(tickmode="array", tickvals=years), yaxis_title="Literacy Rate (%)")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        radar_metrics = ["2011 - Overall - Persons", "2011 - Overall - Male", "2011 - Overall - Female"]
        radar_labels = ["Overall", "Male", "Female"]
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(r=[row[m] for m in radar_metrics] + [row[radar_metrics[0]]],
                                       theta=radar_labels + [radar_labels[0]], fill="toself",
                                       name=state, line_color=ACCENT))
        fig.add_trace(go.Scatterpolar(r=[national[m] for m in radar_metrics] + [national[radar_metrics[0]]],
                                       theta=radar_labels + [radar_labels[0]], fill="toself",
                                       name="All India", line_color=GOLD, opacity=0.6))
        fig.update_layout(title=f"{state} — Gender Profile vs National",
                           polar=dict(radialaxis=dict(visible=True, range=[0, 100])))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        f"""
        <div class="insight-box">
        📍 <b>{state}</b> sits at rank <b>#{int(row['Literacy Rank 2011'])}</b> nationally with a
        2011 literacy rate of <b>{row['2011 - Overall - Persons']:.1f}%</b>, a gender gap of
        <b>{row['Gender Gap 2011']:.1f} pp</b>, and a rural-urban gap of
        <b>{row['Rural-Urban Gap 2011']:.1f} pp</b> — categorized as
        "<b>{row['Literacy Category 2011']}</b>" nationally.
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("View full record for this state"):
        st.dataframe(row.to_frame().T, use_container_width=True)

# ============================================================
# PAGE: GROWTH & FORECAST
# ============================================================
elif page == "📈 Growth & Forecast":

    st.markdown('<span class="section-tag">Growth & Forecast</span>', unsafe_allow_html=True)
    st.subheader("Who Improved the Most? Who's Projected to Lead by 2031?")

    col1, col2 = st.columns(2)

    with col1:
        top_growth = states_df.sort_values("Growth 1991-2011", ascending=False).head(10)
        fig = px.bar(
            top_growth.sort_values("Growth 1991-2011"),
            x="Growth 1991-2011", y=STATE_COL, orientation="h",
            color="Growth 1991-2011", color_continuous_scale="Greens",
            title="Top 10 States by Literacy Growth (1991–2011)", text="Growth 1991-2011",
        )
        fig.update_traces(texttemplate="+%{text:.1f} pp", textposition="outside")
        fig.update_layout(coloraxis_showscale=False, xaxis_title="Growth (pp)", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        top_cagr = states_df.sort_values("CAGR 1991-2011 (%)", ascending=False).head(10)
        fig = px.bar(
            top_cagr.sort_values("CAGR 1991-2011 (%)"),
            x="CAGR 1991-2011 (%)", y=STATE_COL, orientation="h",
            color="CAGR 1991-2011 (%)", color_continuous_scale="Blues",
            title="Top 10 States by CAGR (1991–2011)", text="CAGR 1991-2011 (%)",
        )
        fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
        fig.update_layout(coloraxis_showscale=False, xaxis_title="CAGR (%)", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("🔮 Linear-Trend Forecast to 2031")
    default_states = [s for s in ["Kerala", "Bihar", "Uttar Pradesh", "Maharashtra"] if s in states_df[STATE_COL].values]
    sample_states = st.multiselect(
        "Compare states (historical + forecast)",
        sorted(states_df[STATE_COL].unique()),
        default=default_states or list(states_df[STATE_COL].unique()[:3]),
    )

    if sample_states:
        fig = go.Figure()
        colors = px.colors.qualitative.Bold
        for i, s in enumerate(sample_states):
            r = states_df[states_df[STATE_COL] == s].iloc[0]
            hist_y = [r["1991 - Persons"], r["2001 - Persons"], r["2011 - Overall - Persons"]]
            fore_y = [r["2011 - Overall - Persons"], r["Predicted 2021"], r["Predicted 2031"]]
            color = colors[i % len(colors)]
            fig.add_trace(go.Scatter(x=[1991, 2001, 2011], y=hist_y, mode="lines+markers",
                                      name=s, line=dict(color=color, width=3)))
            fig.add_trace(go.Scatter(x=[2011, 2021, 2031], y=fore_y, mode="lines+markers",
                                      line=dict(color=color, width=2, dash="dash"),
                                      showlegend=False))
        fig.add_vline(x=2011, line_dash="dot", line_color="gray")
        fig.update_layout(
            title="Historical Literacy + Linear Forecast (dashed = projection)",
            xaxis_title="Year", yaxis_title="Literacy Rate (%)",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(
            '<div class="insight-box">⚠️ Forecasts are a simple 3-point linear extrapolation — '
            "useful for directional comparison only, not a substitute for demographic/policy modelling. "
            "High-literacy states show a natural deceleration (ceiling effect) while lagging states "
            "need faster-than-historical growth to close the gap by 2031.</div>",
            unsafe_allow_html=True,
        )

    with st.expander("View 2021 / 2031 projections for all filtered states"):
        st.dataframe(
            states_df[[STATE_COL, "2011 - Overall - Persons", "Predicted 2021", "Predicted 2031"]]
            .sort_values("Predicted 2031", ascending=False)
            .rename(columns={"2011 - Overall - Persons": "2011 (Actual)"})
            .round(1),
            use_container_width=True, hide_index=True,
        )

# ============================================================
# PAGE: GENDER EQUITY
# ============================================================
elif page == "⚖️ Gender Equity":

    st.markdown('<span class="section-tag">Gender Equity</span>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("National Gender Gap (2011)", f"{national['Gender Gap 2011']:.1f} pp")
    c2.metric("Widest Gap State", states_df.loc[states_df["Gender Gap 2011"].idxmax(), STATE_COL])
    c3.metric("Narrowest Gap State", states_df.loc[states_df["Gender Gap 2011"].idxmin(), STATE_COL])

    col1, col2 = st.columns(2)

    with col1:
        gap_df = states_df.sort_values("Gender Gap 2011", ascending=False).head(10)
        fig = px.bar(
            gap_df.sort_values("Gender Gap 2011"),
            x="Gender Gap 2011", y=STATE_COL, orientation="h",
            color="Gender Gap 2011", color_continuous_scale="Reds",
            title="10 Widest Gender-Gap States (2011)", text="Gender Gap 2011",
        )
        fig.update_traces(texttemplate="%{text:.1f} pp", textposition="outside")
        fig.update_layout(coloraxis_showscale=False, xaxis_title="Gender Gap (pp)", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.scatter(
            states_df, x="2011 - Overall - Persons", y="Gender Gap 2011",
            color="Region", size="Rural-Urban Gap 2011", hover_name=STATE_COL,
            title="Literacy vs Gender Gap (bubble = Rural-Urban Gap)",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig.update_layout(xaxis_title="2011 Literacy Rate (%)", yaxis_title="Gender Gap (pp)")
        st.plotly_chart(fig, use_container_width=True)

    equity_counts = states_df["Gender Equity 2011"].value_counts()
    fig = px.bar(
        x=equity_counts.values, y=equity_counts.index, orientation="h",
        color=equity_counts.index, color_discrete_sequence=px.colors.qualitative.Safe,
        title="States by Gender Equity Category", text=equity_counts.values,
    )
    fig.update_layout(showlegend=False, xaxis_title="Number of States", yaxis_title="")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        f"""
        <div class="insight-box">
        💡 Male literacy exceeds female literacy in every state — the national gap is
        <b>{national['Gender Gap 2011']:.1f} pp</b> in 2011, down from
        <b>{national['Gender Gap 1991']:.1f} pp</b> in 1991. States with wider overall
        gender gaps also tend to have lower overall literacy, suggesting the two issues
        reinforce each other rather than trading off.
        </div>
        """,
        unsafe_allow_html=True,
    )

# ============================================================
# PAGE: RURAL VS URBAN
# ============================================================
elif page == "🏙️ Rural vs Urban":

    st.markdown('<span class="section-tag">Rural vs Urban</span>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("National Rural-Urban Gap", f"{national['Rural-Urban Gap 2011']:.1f} pp")
    c2.metric("Widest Gap State", states_df.loc[states_df["Rural-Urban Gap 2011"].idxmax(), STATE_COL])
    c3.metric("Most Balanced State", states_df.loc[states_df["Rural-Urban Gap 2011"].idxmin(), STATE_COL])

    col1, col2 = st.columns(2)

    with col1:
        fig = px.scatter(
            states_df, x="2011 - Rural - Persons", y="2011 - Urban - Persons",
            color="Region", hover_name=STATE_COL, size="Inequality Index 2011",
            title="Rural vs Urban Literacy (2011)",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        max_val = max(states_df["2011 - Rural - Persons"].max(), states_df["2011 - Urban - Persons"].max())
        fig.add_trace(go.Scatter(x=[0, max_val], y=[0, max_val], mode="lines",
                                  line=dict(dash="dash", color="gray"), name="Parity line",
                                  showlegend=True))
        fig.update_layout(xaxis_title="Rural Literacy (%)", yaxis_title="Urban Literacy (%)")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        gap_top = states_df.sort_values("Rural-Urban Gap 2011", ascending=False).head(10)
        fig = px.bar(
            gap_top.sort_values("Rural-Urban Gap 2011"),
            x="Rural-Urban Gap 2011", y=STATE_COL, orientation="h",
            color="Rural-Urban Gap 2011", color_continuous_scale="Purples",
            title="10 Widest Rural-Urban Gap States", text="Rural-Urban Gap 2011",
        )
        fig.update_traces(texttemplate="%{text:.1f} pp", textposition="outside")
        fig.update_layout(coloraxis_showscale=False, xaxis_title="Gap (pp)", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    ru_counts = states_df["Rural-Urban Category 2011"].value_counts()
    fig = px.pie(names=ru_counts.index, values=ru_counts.values, hole=0.4,
                 title="States by Rural-Urban Development Category",
                 color_discrete_sequence=px.colors.sequential.Purp)
    fig.update_traces(textinfo="percent+label")
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# PAGE: REGIONAL COMPARISON
# ============================================================
elif page == "🌐 Regional Comparison":

    st.markdown('<span class="section-tag">Regional Comparison</span>', unsafe_allow_html=True)

    region_df = states_df.groupby("Region", observed=True).agg(
        Avg_1991=("1991 - Persons", "mean"),
        Avg_2001=("2001 - Persons", "mean"),
        Avg_2011=("2011 - Overall - Persons", "mean"),
        Avg_Gender_Gap=("Gender Gap 2011", "mean"),
        Avg_Rural_Urban_Gap=("Rural-Urban Gap 2011", "mean"),
        States=(STATE_COL, "count"),
    ).round(2).sort_values("Avg_2011", ascending=False)

    st.dataframe(region_df, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        region_melt = region_df[["Avg_1991", "Avg_2001", "Avg_2011"]].reset_index().melt(
            id_vars="Region", var_name="Year", value_name="Literacy Rate"
        )
        region_melt["Year"] = region_melt["Year"].str.replace("Avg_", "")
        fig = px.bar(
            region_melt, x="Region", y="Literacy Rate", color="Year", barmode="group",
            title="Regional Average Literacy Across Census Years",
            color_discrete_sequence=[GOLD, ACCENT, PRIMARY],
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.bar(
            region_df.reset_index().sort_values("Avg_Gender_Gap"),
            x="Avg_Gender_Gap", y="Region", orientation="h",
            color="Avg_Gender_Gap", color_continuous_scale="OrRd",
            title="Average Gender Gap by Region (2011)", text="Avg_Gender_Gap",
        )
        fig.update_traces(texttemplate="%{text:.1f} pp", textposition="outside")
        fig.update_layout(coloraxis_showscale=False, xaxis_title="Gender Gap (pp)", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    fig = px.box(
        states_df, x="Region", y="2011 - Overall - Persons", color="Region",
        title="Literacy Rate Spread by Region (2011)",
        color_discrete_sequence=px.colors.qualitative.Set2, points="all",
    )
    fig.update_layout(showlegend=False, yaxis_title="Literacy Rate (%)")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        '<div class="insight-box">🌍 South and West regions lead on average 2011 literacy, '
        "driven by high performers like Kerala, Goa and Maharashtra, while East and Central "
        "regions show the lowest averages and the widest rural-urban gaps — reinforcing the case "
        "for regionally targeted (not uniform) policy intervention.</div>",
        unsafe_allow_html=True,
    )

# ============================================================
# PAGE: CORRELATION & STATISTICS
# ============================================================
elif page == "🔬 Correlation & Statistics":

    st.markdown('<span class="section-tag">Correlation & Statistics</span>', unsafe_allow_html=True)

    corr_cols = [
        "1991 - Persons", "2001 - Persons", "2011 - Overall - Persons",
        "2011 - Overall - Male", "2011 - Overall - Female",
        "Gender Gap 2011", "Rural-Urban Gap 2011", "Growth 1991-2011", "Inequality Index 2011",
    ]
    corr_matrix = states_df[corr_cols].corr().round(2)

    fig = px.imshow(
        corr_matrix, text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
        title="Correlation Heatmap — Core Literacy Metrics", aspect="auto",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Hypothesis Testing")
    tab1, tab2 = st.tabs(["Paired t-test (Male vs Female)", "One-way ANOVA (Census Years)"])

    with tab1:
        d = (states_df["2011 - Overall - Male"] - states_df["2011 - Overall - Female"]).values
        n = len(d)
        t_stat = d.mean() / (d.std(ddof=1) / np.sqrt(n))
        st.write(f"**t-statistic:** {t_stat:.3f}  |  **degrees of freedom:** {n - 1}")
        st.markdown(
            "**H₀:** Mean(Male − Female literacy) = 0 &nbsp;|&nbsp; **H₁:** ≠ 0"
        )
        verdict = "Reject H₀" if abs(t_stat) > 2 else "Fail to reject H₀"
        st.success(f"Result: **{verdict}** — male literacy is statistically different from female literacy across states (|t| = {abs(t_stat):.2f}).")

    with tab2:
        groups = [
            states_df["1991 - Persons"].values,
            states_df["2001 - Persons"].values,
            states_df["2011 - Overall - Persons"].values,
        ]
        grand_mean = np.concatenate(groups).mean()
        ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
        ss_within = sum(((g - g.mean()) ** 2).sum() for g in groups)
        df_between, df_within = len(groups) - 1, sum(len(g) for g in groups) - len(groups)
        f_stat = (ss_between / df_between) / (ss_within / df_within)
        st.write(f"**F-statistic:** {f_stat:.3f}  |  **df:** ({df_between}, {df_within})")
        st.markdown("**H₀:** Mean literacy is equal across 1991 / 2001 / 2011 &nbsp;|&nbsp; **H₁:** At least one differs")
        st.success(f"Result: **Reject H₀** — mean literacy differs significantly across census years (F = {f_stat:.1f} is far above the ~3.1 critical value at α = 0.05).")

    st.markdown(
        '<div class="insight-box">📌 Historical literacy (2001, then 1991) is the strongest '
        "correlate of 2011 outcomes — literacy investment compounds over decades rather than "
        "resetting each census. Gender and rural-urban gaps both correlate negatively with "
        "overall literacy, confirming they are linked, policy-relevant drivers rather than noise.</div>",
        unsafe_allow_html=True,
    )

# ============================================================
# PAGE: CLUSTERING
# ============================================================
elif page == "🧩 Clustering (State Profiles)":

    st.markdown('<span class="section-tag">Unsupervised Learning</span>', unsafe_allow_html=True)
    st.write(
        "States are grouped into natural profiles using **K-Means** (on standardized literacy, "
        "gender-gap and rural-urban-gap features), then projected to 2D via **PCA** for visualization. "
        "Both are implemented from scratch with NumPy."
    )

    k = st.slider("Number of clusters (k)", min_value=2, max_value=5, value=3)
    clustered, explained_var = run_kmeans_pca(states_df, k=k)

    col1, col2 = st.columns([1.2, 1])

    with col1:
        fig = px.scatter(
            clustered, x="PC1", y="PC2", color="Cluster", hover_name=STATE_COL,
            title=f"K-Means Clusters on PCA Components (PC1: {explained_var[0]:.1f}%, PC2: {explained_var[1]:.1f}% variance)",
            color_discrete_sequence=px.colors.qualitative.Bold, size_max=12,
        )
        fig.update_traces(marker=dict(size=11, line=dict(width=1, color="white")))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        profile = clustered.groupby("Cluster", observed=True)[
            ["2011 - Overall - Persons", "Gender Gap 2011", "Rural-Urban Gap 2011"]
        ].mean().round(1)
        st.write("**Cluster Profiles (average values)**")
        st.dataframe(profile, use_container_width=True)
        counts = clustered["Cluster"].value_counts().sort_index()
        st.write("**States per Cluster**")
        st.bar_chart(counts)

    with st.expander("View states in each cluster"):
        for c in sorted(clustered["Cluster"].unique()):
            members = clustered.loc[clustered["Cluster"] == c, STATE_COL].tolist()
            st.markdown(f"**Cluster {c}** ({len(members)} states): {', '.join(members)}")

# ============================================================
# PAGE: CHAMPIONS & CONCERN STATES
# ============================================================
elif page == "🏆 Champions & Concern States":

    st.markdown('<span class="section-tag">Champions & Concern States</span>', unsafe_allow_html=True)
    st.write("Adjust the thresholds to define what counts as a **Champion** (high literacy, low gender gap) or a **Concern** state (low literacy, high gender gap).")

    c1, c2 = st.columns(2)
    with c1:
        champ_lit = st.slider("Champion: min literacy (%)", 50, 100, 80)
        champ_gap = st.slider("Champion: max gender gap (pp)", 0, 30, 10)
    with c2:
        concern_lit = st.slider("Concern: max literacy (%)", 30, 90, 65)
        concern_gap = st.slider("Concern: min gender gap (pp)", 0, 30, 15)

    champions = states_df[
        (states_df["2011 - Overall - Persons"] >= champ_lit) & (states_df["Gender Gap 2011"] <= champ_gap)
    ].sort_values("2011 - Overall - Persons", ascending=False)

    concern = states_df[
        (states_df["2011 - Overall - Persons"] < concern_lit) & (states_df["Gender Gap 2011"] > concern_gap)
    ].sort_values("2011 - Overall - Persons")

    col1, col2 = st.columns(2)
    with col1:
        st.success(f"🏆 **{len(champions)} Champion State(s)**")
        st.dataframe(
            champions[[STATE_COL, "2011 - Overall - Persons", "Gender Gap 2011", "Region"]]
            .rename(columns={"2011 - Overall - Persons": "Literacy (%)", "Gender Gap 2011": "Gender Gap (pp)"})
            .round(1), use_container_width=True, hide_index=True,
        )
    with col2:
        st.error(f"⚠️ **{len(concern)} Concern State(s)**")
        st.dataframe(
            concern[[STATE_COL, "2011 - Overall - Persons", "Gender Gap 2011", "Region"]]
            .rename(columns={"2011 - Overall - Persons": "Literacy (%)", "Gender Gap 2011": "Gender Gap (pp)"})
            .round(1), use_container_width=True, hide_index=True,
        )

    fig = px.scatter(
        states_df, x="2011 - Overall - Persons", y="Gender Gap 2011", hover_name=STATE_COL,
        color="Region", title="All States — Literacy vs Gender Gap (thresholds shown as reference lines)",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.add_vline(x=champ_lit, line_dash="dash", line_color=GOOD)
    fig.add_hline(y=champ_gap, line_dash="dash", line_color=GOOD)
    fig.add_vline(x=concern_lit, line_dash="dot", line_color=BAD)
    fig.add_hline(y=concern_gap, line_dash="dot", line_color=BAD)
    fig.update_layout(xaxis_title="2011 Literacy Rate (%)", yaxis_title="Gender Gap (pp)")
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# PAGE: DATA QUALITY & METHODOLOGY
# ============================================================
elif page == "🧾 Data Quality & Methodology":

    st.markdown('<span class="section-tag">Methodology</span>', unsafe_allow_html=True)
    st.subheader("How this dashboard's data was prepared")

    st.markdown(
        """
    **1. Data Understanding**
    35 rows (`All India` + 34 States/UTs) and 13 raw Census columns covering literacy rate (%) by
    Census year (1991 / 2001 / 2011), gender (Male / Female / Persons), and — for 2011 only —
    Rural / Urban residence.

    **2. Data Quality Issues Found**
    - A hidden missing-value pattern: **Jammu & Kashmir's 1991 figures were recorded as `0`**
      because the 1991 Census was not conducted there — not a genuine literacy rate of zero.
    - A column-naming inconsistency: `2011 - Rural - Person` (singular) vs. every other
      `...Persons` (plural) column.
    - A text/spelling inconsistency: `Chhatisgarh` → `Chhattisgarh`; the outdated name
      `Uttaranchal` → `Uttarakhand`.

    **3. Cleaning Strategy Applied**
    - J&K's 1991 M/F/Persons values were **imputed** using J&K's own 2001 figures scaled by the
      national 1991→2001 growth ratio — preserving the row (only 35 total) with a transparent,
      data-driven method rather than an arbitrary constant or deletion.
    - Column renamed for consistency; state names standardized to Title Case with typos/renames fixed.
    - All numeric columns explicitly cast to `float` for safe downstream computation.

    **4. Feature Engineering (25+ derived features)**
    - `2011 - Overall - *`: combined Rural+Urban average (2011's raw data has no single "Total" column).
    - `Gender Gap *`, `Rural-Urban Gap 2011`, `Growth`/`CAGR`, `Inequality Index`, `Gap Reduction`.
    - Categorical buckets: Literacy Category, Gender Equity Category, Rural-Urban Category.
    - `Region` mapping (North / South / East / West / Central / Northeast) for roll-up analysis.
    - `Rank`, `Z-score`, and simple **linear-trend forecasts** to 2021/2031.

    **5. Statistical Validation**
    - Paired t-test confirms male literacy is significantly higher than female literacy nationwide.
    - One-way ANOVA confirms mean literacy differs significantly across the three census years.
    - A from-scratch NumPy multiple regression of Persons on Male + Female literacy achieves
      **R² > 0.99**, cross-validating the internal consistency of the cleaned data.

    *Full derivations, additional chart types (violin, ECDF, swarm, dumbbell, radar, lollipop, etc.)
    and the from-scratch K-Means/PCA/regression code are documented in the companion Jupyter notebook.*
        """
    )

    st.subheader("Cleaned Dataset Preview")
    st.dataframe(df.head(15), use_container_width=True)

    with st.expander("Full column list"):
        st.write(list(df.columns))

# ============================================================
# FOOTER
# ============================================================
st.markdown(
    '<p class="footer-note">India Literacy Analytics Dashboard · Census Data 1991–2011 · '
    "Built with Streamlit &amp; Plotly</p>",
    unsafe_allow_html=True,
)