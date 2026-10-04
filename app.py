import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="StockScope | Pro Terminal",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =========================================================
# PRO TRADING TERMINAL UI (CSS)
# =========================================================
st.markdown("""
<style>
    /* Dark Theme Setup */
    .stApp {
        background-color: #060913 !important;
        color: #d1d5db;
        font-family: 'Inter', -apple-system, sans-serif;
    }

    [data-testid="stSidebar"] {
        display: none;
    }

    .block-container {
        padding: 1rem 1.5rem 2rem 1.5rem !important;
        max-width: 100% !important;
    }

    /* Top Compact Terminal Navbar */
    .terminal-nav {
        background: #0d121f;
        border: 1px solid #1e293b;
        border-radius: 8px;
        padding: 8px 16px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 12px;
    }

    .terminal-title {
        color: #38bdf8;
        font-weight: 800;
        font-size: 18px;
        letter-spacing: 0.5px;
    }

    .status-pill {
        background: rgba(34, 197, 94, 0.15);
        color: #4ade80;
        border: 1px solid #22c55e;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 11px;
        font-weight: 700;
    }

    /* Container Box Styling */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: #0b0f19 !important;
        border: 1px solid #1b2438 !important;
        border-radius: 8px !important;
        padding: 12px !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.6) !important;
    }

    /* Dark Grid Metric Cards */
    .metric-card {
        background: #111827;
        border: 1px solid #1f293d;
        border-radius: 6px;
        padding: 10px;
        text-align: center;
    }

    .metric-title {
        color: #6b7280;
        font-size: 10px;
        font-weight: 700;
        text-transform: uppercase;
    }

    .metric-val {
        font-size: 18px;
        font-weight: 800;
        margin-top: 2px;
    }

    .val-orange { color: #f97316; }
    .val-green { color: #22c55e; }
    .val-blue { color: #38bdf8; }
    .val-red { color: #ef4444; }

    /* Button Customization */
    .stButton > button {
        width: 100%;
        height: 40px;
        border-radius: 6px;
        background: linear-gradient(135deg, #0284c7 0%, #2563eb 100%);
        color: #ffffff;
        border: none;
        font-size: 13px;
        font-weight: 700;
        letter-spacing: 0.5px;
        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #38bdf8 0%, #1d4ed8 100%);
    }

    /* Custom Input Controls Styling */
    div[role="radiogroup"] {
        display: flex;
        flex-direction: row;
        gap: 4px;
    }

    div[role="radiogroup"] label {
        background-color: #111827 !important;
        border: 1px solid #1f293d !important;
        border-radius: 4px !important;
        padding: 4px 10px !important;
        font-size: 11px !important;
        color: #9ca3af !important;
    }

    div[role="radiogroup"] label:hover {
        border-color: #38bdf8 !important;
        color: #ffffff !important;
    }
</style>
""", unsafe_allow_html=True)

# =========================================================
# FEATURE MAPPING & DATA PROCESSING
# =========================================================
TREND_MAP = {"Down": 0, "Stable": 1, "Up": 2}
RSI_MAP = {"Low": 0, "Medium": 1, "High": 2}
VOLUME_MAP = {"Low": 0, "Medium": 1, "High": 2}
NEWS_MAP = {"Negative": 0, "Positive": 1}
DECISION_MAP = {0: "Buy", 1: "Hold", 2: "Sell"}
REVERSE_DECISION_MAP = {"Buy": 0, "Hold": 1, "Sell": 2}

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
    return bda_data.dropna(subset=["Trend", "RSI", "Volume", "News", "Decision"])

bda_data = load_and_clean_data()

ml_data = bda_data.copy()
ml_data["Trend_num"] = ml_data["Trend"].map(TREND_MAP)
ml_data["RSI_num"] = ml_data["RSI"].map(RSI_MAP)
ml_data["Volume_num"] = ml_data["Volume"].map(VOLUME_MAP)
ml_data["News_num"] = ml_data["News"].map(NEWS_MAP)
ml_data["Decision_num"] = ml_data["Decision"].map(REVERSE_DECISION_MAP)

ml_data = ml_data.dropna(subset=["Trend_num", "RSI_num", "Volume_num", "News_num", "Decision_num"])

X = ml_data[["Trend_num", "RSI_num", "Volume_num", "News_num"]]
y = ml_data["Decision_num"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, ccp_alpha=0.01, random_state=42)
model.fit(X_train, y_train)
accuracy = accuracy_score(y_test, model.predict(X_test))

# =========================================================
# TOP NAVIGATION HEADER
# =========================================================
st.markdown("""
<div class="terminal-nav">
    <div class="terminal-title">⚡ STOCKSCOPE // PRO TERMINAL</div>
    <div style="display: flex; gap: 15px; align-items: center;">
        <span class="status-pill">● LIVE FEED</span>
        <span style="color: #6b7280; font-size: 12px;">SYSTEM ACCURACY: <b style="color:#38bdf8;">{:.1f}%</b></span>
    </div>
</div>
""".format(accuracy * 100), unsafe_allow_html=True)

# =========================================================
# TOP METRICS ROW (Image Style KPI Panel)
# =========================================================
m1, m2, m3, m4, m5, m6 = st.columns(6)
with m1:
    st.markdown('<div class="metric-card"><div class="metric-title">TOTAL VOLUME</div><div class="metric-val val-orange">01.39M</div></div>', unsafe_allow_html=True)
with m2:
    st.markdown('<div class="metric-card"><div class="metric-title">RSI INDEX</div><div class="metric-val val-orange">68.77</div></div>', unsafe_allow_html=True)
with m3:
    st.markdown('<div class="metric-card"><div class="metric-title">SIGNAL SPREAD</div><div class="metric-val val-orange">0.7.19</div></div>', unsafe_allow_html=True)
with m4:
    st.markdown('<div class="metric-card"><div class="metric-title">TOTAL RECORDS</div><div class="metric-val val-blue">{}</div></div>'.format(len(bda_data)), unsafe_allow_html=True)
with m5:
    st.markdown('<div class="metric-card"><div class="metric-title">BULL/BEAR RATIO</div><div class="metric-val val-green">26:1</div></div>', unsafe_allow_html=True)
with m6:
    st.markdown('<div class="metric-card"><div class="metric-title">TOTAL ASSETS</div><div class="metric-val val-blue">5 STOCKS</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# =========================================================
# PARAMETER INPUT PANEL
# =========================================================
with st.container(border=True):
    col_comp, col_m1, col_m2, col_m3, col_m4, col_btn = st.columns([1.2, 1, 1, 1, 1, 1])

    with col_comp:
        company = st.radio("Company Ticker", ["TCS", "Infosys", "Aditya Birla", "ITC", "Reliance"], horizontal=True)
    with col_m1:
        trend = st.radio("Trend", ["Up", "Stable", "Down"], horizontal=True)
    with col_m2:
        rsi = st.radio("RSI Momentum", ["Low", "Medium", "High"], horizontal=True)
    with col_m3:
        volume = st.radio("Volume", ["Low", "Medium", "High"], horizontal=True)
    with col_m4:
        news = st.radio("News", ["Positive", "Negative"], horizontal=True)
    with col_btn:
        st.markdown("<div style='height: 25px;'></div>", unsafe_allow_html=True)
        analyze = st.button("⚡ ANALYZE STOCK", type="primary")

st.markdown("<br>", unsafe_allow_html=True)

# =========================================================
# MAIN DASHBOARD GRID (Matching Image UI)
# =========================================================
left_col, center_col, right_col = st.columns([1.2, 3, 1.2])

# Inference Calculation
input_data = pd.DataFrame({
    "Trend_num": [TREND_MAP[trend]],
    "RSI_num": [RSI_MAP[rsi]],
    "Volume_num": [VOLUME_MAP[volume]],
    "News_num": [NEWS_MAP[news]]
})
probs = model.predict_proba(input_data)[0]
probs_smoothed = (probs + 0.1) / (probs + 0.1).sum()
prediction_num = np.argmax(probs_smoothed)
prediction = DECISION_MAP[prediction_num]
confidence = np.max(probs_smoothed) * 100

# 1. LEFT SIDE PANEL
with left_col:
    with st.container(border=True):
        st.markdown("<div style='font-weight:700; color:#9ca3af; font-size:12px;'>SIGNAL PROBABILITY</div>", unsafe_allow_html=True)
        
        # Donut Chart like Top Right in Reference Image
        decision_counts = bda_data["Decision"].value_counts().reindex(["Buy", "Hold", "Sell"], fill_value=0)
        fig_donut = px.pie(
            values=decision_counts.values,
            names=decision_counts.index,
            hole=0.6,
            color=decision_counts.index,
            color_discrete_map={"Buy": "#22c55e", "Hold": "#f59e0b", "Sell": "#ef4444"}
        )
        fig_donut.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
            height=160,
            margin=dict(l=0, r=0, t=0, b=0)
        )
        st.plotly_chart(fig_donut, use_container_width=True)

        st.divider()

        st.markdown("<div style='font-weight:700; color:#9ca3af; font-size:12px;'>SIGNAL OUTPUT</div>", unsafe_allow_html=True)
        color_code = "#22c55e" if prediction == "Buy" else ("#ef4444" if prediction == "Sell" else "#f59e0b")
        st.markdown(f"<div style='font-size:28px; font-weight:900; color:{color_code}; text-align:center; padding:10px;'>{prediction.upper()}</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='text-align:center; color:#9ca3af; font-size:12px;'>CONFIDENCE: <b>{confidence:.1f}%</b></div>", unsafe_allow_html=True)

# 2. CENTER MAIN CHARTS
with center_col:
    with st.container(border=True):
        st.markdown(f"<div style='font-weight:700; color:#f3f4f6; font-size:13px;'>{company} // CANDLESTICK & VOLUME WAVE</div>", unsafe_allow_html=True)
        
        # Synthetic Candlestick Chart (Simulating the Red/Green Chart in Image)
        np.random.seed(10)
        dates = pd.date_range(end=pd.Timestamp.now(), periods=50, freq='D')
        close_prices = 100 + np.cumsum(np.random.randn(50) * 2)
        open_prices = close_prices + np.random.randn(50)
        high_prices = np.maximum(open_prices, close_prices) + np.random.rand(50) * 2
        low_prices = np.minimum(open_prices, close_prices) - np.random.rand(50) * 2

        fig_candle = go.Figure(data=[go.Candlestick(
            x=dates,
            open=open_prices,
            high=high_prices,
            low=low_prices,
            close=close_prices,
            increasing_line_color='#22c55e',
            decreasing_line_color='#ef4444'
        )])

        fig_candle.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis_rangeslider_visible=False,
            height=200,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(showgrid=True, gridcolor='#1e293b'),
            yaxis=dict(showgrid=True, gridcolor='#1e293b')
        )
        st.plotly_chart(fig_candle, use_container_width=True)

    with st.container(border=True):
        # Red Area Chart (Matching the Red Chart on Right Side of Image)
        fig_area = px.area(
            x=dates,
            y=np.abs(close_prices - open_prices) * 10,
            title="MARKET VOLATILITY DENSITY"
        )
        fig_area.update_traces(line_color='#ef4444', fillcolor='rgba(239, 68, 68, 0.3)')
        fig_area.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=130,
            margin=dict(l=10, r=10, t=25, b=10),
            font=dict(color='#6b7280', size=10),
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor='#1e293b')
        )
        st.plotly_chart(fig_area, use_container_width=True)

# 3. RIGHT SIDE PANEL
with right_col:
    with st.container(border=True):
        st.markdown("<div style='font-weight:700; color:#ef4444; font-size:14px; text-align:center;'>ORDER BOOK STREAM</div>", unsafe_allow_html=True)
        st.markdown("<div style='font-size:24px; font-weight:900; color:#ef4444; text-align:center;'>$ 339.20</div>", unsafe_allow_html=True)
        
        # Order Stream Table Mock
        order_data = pd.DataFrame({
            "PRICE": [194.52, 178.10, 162.00, 158.40, 142.10],
            "QTY": [120, 450, 230, 890, 105],
            "TYPE": ["SELL", "SELL", "BUY", "BUY", "SELL"]
        })
        
        st.dataframe(order_data, height=180, use_container_width=True, hide_index=True)

        st.markdown("<div style='font-size:12px; font-weight:800; color:#22c55e; text-align:center;'>+ 2354 (7.50%)</div>", unsafe_allow_html=True)

# =========================================================
# BOTTOM FULL WIDTH VOLUME BAR (Matching Image Bottom Bar Chart)
# =========================================================
with st.container(border=True):
    st.markdown("<div style='font-weight:700; color:#6b7280; font-size:11px;'>HISTORICAL VOLUME DISTRIBUTION SNAPSHOT</div>", unsafe_allow_html=True)
    
    vol_data = np.random.randint(100, 500, size=60)
    fig_vol = px.bar(x=list(range(60)), y=vol_data)
    fig_vol.update_traces(marker_color='#0284c7')
    fig_vol.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=100,
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(showgrid=False, showticklabels=False),
        yaxis=dict(showgrid=False, showticklabels=False)
    )
    st.plotly_chart(fig_vol, use_container_width=True)

st.divider()
st.caption("⚡ StockScope Pro Terminal • Dark Mode BDA Analytics Engine")