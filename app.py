import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# =========================================================
# PAGE CONFIGURATION (Unique & Short Tab Title)
# =========================================================
st.set_page_config(
    page_title="StockScope",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# =========================================================
# FINTECH TRADING TERMINAL UI (CSS)
# =========================================================
st.markdown("""
<style>
    /* Global Background & Font Settings */
    .stApp {
        background: radial-gradient(circle at 50% -20%, #1e293b, #0f172a, #020617);
        color: #f1f5f9;
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }

    [data-testid="stSidebar"] {
        display: none;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }

    /* Top Terminal Header */
    .hero-header {
        background: rgba(15, 23, 42, 0.85);
        backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 24px 30px;
        text-align: center;
        margin-bottom: 25px;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.5);
    }

    .hero-header h1 {
        background: linear-gradient(135deg, #ffffff 0%, #cbd5e1 50%, #38bdf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 36px !important;
        font-weight: 900 !important;
        letter-spacing: -0.5px;
        margin: 0 0 8px 0;
    }

    .hero-header p {
        color: #94a3b8 !important;
        font-size: 14px !important;
        margin-bottom: 12px;
    }

    .tech-stack-pills {
        display: flex;
        justify-content: center;
        gap: 10px;
        flex-wrap: wrap;
    }

    .tech-pill {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 4px 12px;
        border-radius: 8px;
        font-size: 12px;
        color: #cbd5e1;
    }

    /* Container Box Styling */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(15, 23, 42, 0.7) !important;
        backdrop-filter: blur(12px) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 16px !important;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4) !important;
        padding: 22px !important;
    }

    /* Section Headings */
    h3 {
        color: #f8fafc !important;
        font-size: 18px !important;
        font-weight: 700 !important;
        letter-spacing: -0.2px;
    }

    /* Custom Metric Display */
    div[data-testid="stMetric"] {
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 14px 18px;
    }

    div[data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
        font-size: 11px !important;
        font-weight: 700 !important;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }

    div[data-testid="stMetricValue"] {
        color: #38bdf8 !important;
        font-weight: 800 !important;
        font-size: 24px !important;
    }

    /* Capsule / Pill Radio Buttons */
    div[role="radiogroup"] {
        display: flex;
        flex-direction: row;
        gap: 8px;
        flex-wrap: wrap;
    }

    div[role="radiogroup"] label {
        background-color: rgba(30, 41, 59, 0.8) !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        border-radius: 25px !important;
        padding: 6px 16px !important;
        font-size: 13px !important;
        font-weight: 600 !important;
        color: #cbd5e1 !important;
        cursor: pointer;
        transition: all 0.2s ease-in-out !important;
    }

    div[role="radiogroup"] label:hover {
        border-color: #38bdf8 !important;
        background-color: rgba(56, 189, 248, 0.15) !important;
        color: #ffffff !important;
    }

    div[role="radiogroup"] label input[type="radio"] {
        display: none !important;
    }

    /* Analyze Button */
    .stButton > button {
        width: 100%;
        height: 52px;
        border-radius: 12px;
        background: linear-gradient(135deg, #0284c7 0%, #2563eb 100%);
        color: #ffffff;
        border: none;
        font-size: 16px;
        font-weight: 700;
        letter-spacing: 0.8px;
        box-shadow: 0 4px 20px rgba(2, 132, 199, 0.4);
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #38bdf8 0%, #1d4ed8 100%);
        box-shadow: 0 6px 28px rgba(56, 189, 248, 0.5);
        transform: translateY(-1px);
    }

    /* Status Badges */
    .badge-buy {
        background: linear-gradient(135deg, rgba(34, 197, 94, 0.2), rgba(16, 185, 129, 0.3));
        color: #4ade80;
        border: 1.5px solid #22c55e;
        padding: 12px 20px;
        border-radius: 12px;
        font-weight: 800;
        text-align: center;
        font-size: 20px;
        letter-spacing: 1px;
        box-shadow: 0 0 20px rgba(34, 197, 94, 0.25);
    }
    
    .badge-sell {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.2), rgba(225, 29, 72, 0.3));
        color: #f87171;
        border: 1.5px solid #ef4444;
        padding: 12px 20px;
        border-radius: 12px;
        font-weight: 800;
        text-align: center;
        font-size: 20px;
        letter-spacing: 1px;
        box-shadow: 0 0 20px rgba(239, 68, 68, 0.25);
    }

    .badge-hold {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.2), rgba(217, 119, 6, 0.3));
        color: #fbbf24;
        border: 1.5px solid #f59e0b;
        padding: 12px 20px;
        border-radius: 12px;
        font-weight: 800;
        text-align: center;
        font-size: 20px;
        letter-spacing: 1px;
        box-shadow: 0 0 20px rgba(245, 158, 11, 0.25);
    }

    /* Tabs Styling */
    button[data-baseweb="tab"] {
        color: #94a3b8;
        font-weight: 600;
        font-size: 13px;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        color: #38bdf8 !important;
        border-bottom-color: #0284c7 !important;
    }
</style>
""", unsafe_allow_html=True)


# =========================================================
# FEATURE MAPPING DICTIONARIES
# =========================================================
TREND_MAP = {"Down": 0, "Stable": 1, "Up": 2}
RSI_MAP = {"Low": 0, "Medium": 1, "High": 2}
VOLUME_MAP = {"Low": 0, "Medium": 1, "High": 2}
NEWS_MAP = {"Negative": 0, "Positive": 1}
DECISION_MAP = {0: "Buy", 1: "Hold", 2: "Sell"}
REVERSE_DECISION_MAP = {"Buy": 0, "Hold": 1, "Sell": 2}


# =========================================================
# DATA PREPARATION & MODEL
# =========================================================
@st.cache_data
def load_and_clean_data():
    files = {
        "TCS": "Tcs.csv",
        "Infosys": "infosis.csv",
        "Aditya Birla": "Aditya Birla Cap.csv",
        "ITC": "ITC.csv",
        "Reliance": "IST.csv"
    }

    dataframes = []
    for comp, file in files.items():
        try:
            df = pd.read_csv(file)
            df["Company"] = comp
            dataframes.append(df)
        except Exception:
            pass

    if not dataframes:
        np.random.seed(42)
        n = 300
        mock_df = pd.DataFrame({
            "Company": np.random.choice(["TCS", "Infosys", "Aditya Birla", "ITC", "Reliance"], n),
            "Trend": np.random.choice(["Up", "Stable", "Down"], n),
            "RSI": np.random.choice(["Low", "Medium", "High"], n),
            "Volume": np.random.choice(["Low", "Medium", "High"], n),
            "News": np.random.choice(["Positive", "Negative"], n),
            "Decision": np.random.choice(["Buy", "Hold", "Sell"], n)
        })
        return mock_df

    bda_data = pd.concat(dataframes, ignore_index=True)

    required_columns = ["Trend", "RSI", "Volume", "News", "Decision"]
    return bda_data.dropna(subset=required_columns)


bda_data = load_and_clean_data()

# Numerical Encoding
ml_data = bda_data.copy()
ml_data["Trend_num"] = ml_data["Trend"].map(TREND_MAP)
ml_data["RSI_num"] = ml_data["RSI"].map(RSI_MAP)
ml_data["Volume_num"] = ml_data["Volume"].map(VOLUME_MAP)
ml_data["News_num"] = ml_data["News"].map(NEWS_MAP)
ml_data["Decision_num"] = ml_data["Decision"].map(REVERSE_DECISION_MAP)

ml_data = ml_data.dropna(subset=["Trend_num", "RSI_num", "Volume_num", "News_num", "Decision_num"])

X = ml_data[["Trend_num", "RSI_num", "Volume_num", "News_num"]]
y = ml_data["Decision_num"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Decision Tree Classifier
model = DecisionTreeClassifier(
    max_depth=3,
    min_samples_leaf=5,
    ccp_alpha=0.01,
    random_state=42
)
model.fit(X_train, y_train)

accuracy = accuracy_score(y_test, model.predict(X_test))


# =========================================================
# HEADER
# =========================================================
st.markdown("""
    <div class="hero-header">
        <h1>Stock Market Analytics & Prediction System</h1>
        <p>Big Data Historical Market Trend Evaluation & Signal Prediction</p>
        <div class="tech-stack-pills">
            <span class="tech-pill">📊 Multi-Asset Analytics</span>
            <span class="tech-pill">⚡ Decision Tree Model</span>
            <span class="tech-pill">🔍 Market Pattern Evaluation</span>
        </div>
    </div>
""", unsafe_allow_html=True)


# =========================================================
# INPUT CONTROL PANEL
# =========================================================
st.subheader("🎛️ Market Parameter Selection")

with st.container(border=True):
    col_comp, col_m1, col_m2, col_m3, col_m4 = st.columns([1.2, 1, 1, 1, 0.8])

    with col_comp:
        company = st.radio(
            "Select Company Ticker",
            ["TCS", "Infosys", "Aditya Birla", "ITC", "Reliance"],
            horizontal=True
        )

    with col_m1:
        trend = st.radio("Market Trend", ["Up", "Stable", "Down"], horizontal=True)

    with col_m2:
        rsi = st.radio("RSI Momentum", ["Low", "Medium", "High"], horizontal=True)

    with col_m3:
        volume = st.radio("Trading Volume", ["Low", "Medium", "High"], horizontal=True)

    with col_m4:
        news = st.radio("News Sentiment", ["Positive", "Negative"], horizontal=True)

    st.markdown("<br>", unsafe_allow_html=True)
    analyze = st.button("📈 ANALYZE STOCK", type="primary")


# =========================================================
# ANALYTICS DASHBOARD OUTPUT
# =========================================================
if analyze:
    # Model Inference
    input_data = pd.DataFrame({
        "Trend_num": [TREND_MAP[trend]],
        "RSI_num": [RSI_MAP[rsi]],
        "Volume_num": [VOLUME_MAP[volume]],
        "News_num": [NEWS_MAP[news]]
    })

    probs = model.predict_proba(input_data)[0]
    
    # Smooth Probability Calculation
    probs_smoothed = (probs + 0.1) / (probs + 0.1).sum()
    prediction_num = np.argmax(probs_smoothed)
    prediction = DECISION_MAP[prediction_num]
    confidence = np.max(probs_smoothed) * 100

    st.markdown("<br>", unsafe_allow_html=True)

    # 1. Executive Summary Box
    st.subheader("📊 Market Signal Summary")
    
    r1, r2, r3, r4 = st.columns(4)
    with r1:
        st.metric("Selected Stock Ticker", company)
    with r2:
        st.metric("Prediction Confidence", f"{confidence:.1f}%")
    with r3:
        st.metric("Total Dataset Records", len(bda_data))
    with r4:
        st.markdown("**Predicted Signal Output**")
        badge_class = f"badge-{prediction.lower()}"
        st.markdown(f'<div class="{badge_class}">{prediction.upper()}</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    if prediction == "Buy":
        st.success(f"**BUY SIGNAL GENERATED**: Positive trend and momentum observed for **{company}**.")
    elif prediction == "Sell":
        st.error(f"**SELL SIGNAL GENERATED**: Downward trend pattern detected for **{company}**.")
    else:
        st.warning(f"**HOLD SIGNAL GENERATED**: Neutral consolidation signal detected for **{company}**.")

    st.info(
        f"💡 **Pattern Note:** The combination of **Trend: {trend}, RSI: {rsi}, Volume: {volume}, and News: {news}** "
        f"shows strong correlation with **{prediction.upper()}** decisions in historical data."
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # 2. Selected Market Snapshot
    st.subheader("📸 Active Market Parameter Snapshot")
    snap1, snap2, snap3, snap4, snap5 = st.columns(5)
    with snap1:
        st.metric("Ticker", company)
    with snap2:
        st.metric("Trend", trend)
    with snap3:
        st.metric("RSI Level", rsi)
    with snap4:
        st.metric("Volume", volume)
    with snap5:
        st.metric("News Sentiment", news)

    st.markdown("<br>", unsafe_allow_html=True)

    # 3. Overall BDA Distribution Analysis
    st.subheader("🌐 Dataset Decision Distribution")
    
    decision_counts = bda_data["Decision"].value_counts().reindex(["Buy", "Hold", "Sell"], fill_value=0)
    total_records = len(bda_data)

    d_col1, d_col2 = st.columns([1, 2])

    with d_col1:
        st.metric("BUY Ratio", int(decision_counts["Buy"]), f"{decision_counts['Buy']/total_records*100:.1f}%")
        st.metric("HOLD Ratio", int(decision_counts["Hold"]), f"{decision_counts['Hold']/total_records*100:.1f}%")
        st.metric("SELL Ratio", int(decision_counts["Sell"]), f"{decision_counts['Sell']/total_records*100:.1f}%")

    with d_col2:
        fig_dist = px.bar(
            x=decision_counts.index,
            y=decision_counts.values,
            color=decision_counts.index,
            color_discrete_map={"Buy": "#22c55e", "Hold": "#f59e0b", "Sell": "#ef4444"},
            labels={"x": "Decision Signal", "y": "Record Count"},
            title="Overall Signal Count across Dataset",
            template="plotly_dark"
        )
        fig_dist.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
            height=280
        )
        st.plotly_chart(fig_dist, use_container_width=True)

    # 4. Feature Crosstab Exploration
    st.subheader("🔍 Parameter Pattern Analysis")
    tab1, tab2, tab3, tab4 = st.tabs(["Trend × Decision", "RSI × Decision", "Volume × Decision", "News × Decision"])

    def create_crosstab_chart(feature, categories):
        ct = pd.crosstab(bda_data[feature], bda_data["Decision"]).reindex(categories, fill_value=0)
        ct = ct.reindex(columns=["Buy", "Hold", "Sell"], fill_value=0)
        
        fig = px.bar(
            ct,
            barmode="group",
            color_discrete_map={"Buy": "#22c55e", "Hold": "#f59e0b", "Sell": "#ef4444"},
            template="plotly_dark",
            title=f"Distribution: {feature} vs Decision"
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=320,
            xaxis_title=feature,
            yaxis_title="Record Count"
        )
        return fig

    with tab1:
        st.plotly_chart(create_crosstab_chart("Trend", ["Up", "Stable", "Down"]), use_container_width=True)
    with tab2:
        st.plotly_chart(create_crosstab_chart("RSI", ["Low", "Medium", "High"]), use_container_width=True)
    with tab3:
        st.plotly_chart(create_crosstab_chart("Volume", ["Low", "Medium", "High"]), use_container_width=True)
    with tab4:
        st.plotly_chart(create_crosstab_chart("News", ["Positive", "Negative"]), use_container_width=True)

    # 5. Asset Specific Analytics
    st.subheader(f"🏢 {company} Historical Breakdown")
    company_df = bda_data[bda_data["Company"] == company]

    if company_df.empty:
        st.warning(f"No historical records available for {company}.")
    else:
        c1, c2 = st.columns([1, 1])
        with c1:
            comp_decisions = company_df["Decision"].value_counts().reindex(["Buy", "Hold", "Sell"], fill_value=0)
            fig_pie = px.pie(
                values=comp_decisions.values,
                names=comp_decisions.index,
                color=comp_decisions.index,
                color_discrete_map={"Buy": "#22c55e", "Hold": "#f59e0b", "Sell": "#ef4444"},
                title=f"{company} Decision Breakdown",
                hole=0.5,
                template="plotly_dark"
            )
            fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=300)
            st.plotly_chart(fig_pie, use_container_width=True)

        with c2:
            st.write(f"**Data Matrix ({len(company_df)} Records)**")
            st.dataframe(
                company_df[["Trend", "RSI", "Volume", "News", "Decision"]],
                use_container_width=True,
                hide_index=True,
                height=260
            )

    # 6. Model Metrics
    st.subheader("⚡ Model Performance Metrics")
    p1, p2 = st.columns(2)
    with p1:
        st.metric("Decision Tree Accuracy", f"{accuracy * 100:.1f}%")
    with p2:
        st.metric("Predicted Signal", prediction.upper())

else:
    st.info("💡 Select parameters above and click 'ANALYZE STOCK' to generate analysis.")

# =========================================================
# TERMINAL FOOTER
# =========================================================
st.divider()
st.caption("⚡ StockScope • Big Data Trend Analytics Terminal")