from datetime import datetime, timedelta
from collections import Counter
import re
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# Page Configuration
st.set_page_config(page_title="Score Probability Predictor", layout="wide")

# Custom Styling
st.markdown(
    """
    <style>
    .metric-card {
        background-color: #1e1e1e;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #333;
        text-align: center;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# SIDEBAR CONFIGURATION
# ---------------------------------------------------------
st.sidebar.header("Predictor Settings")

symbol = st.sidebar.text_input("Ticker Symbol", value="EURUSD=X", help="Enter a valid Yahoo Finance ticker (e.g., EURUSD=X, GC=F, AAPL, BTC-USD)")

trend_mode = st.sidebar.selectbox(
    "Trend Mode",
    ["Open, High, Low, Close + Midline", "Above/Below Midline"],
    help="Choose between detailed OHLC candle scoring or simple Above/Below Midline scoring.",
)

reversed_flag = st.sidebar.checkbox("Reverse Score Direction", value=False, help="Invert positive and negative scores if needed.")

st.sidebar.subheader("Timeframe & History")
timeframe = st.sidebar.selectbox("Timeframe", ["60m", "1d", "1wk", "1mo", "3mo"], index=1)
history_period = st.sidebar.selectbox("History Range", ["1y", "2y", "5y", "10y", "max"], index=1)

lookback_n = st.sidebar.slider("Pattern Lookback Window (Scores)", min_value=1, max_value=5, value=3, help="Number of past consecutive scores to match historically.")

if st.sidebar.button("🔄 Run Analysis"):
    st.rerun()

# ---------------------------------------------------------
# CORE LOGIC: SCORING FUNCTIONS
# ---------------------------------------------------------
def _compute_single_score(p_open, p_high, p_low, p_close, c_close, trend_mode_val, reversed_flag):
    green_candle = p_close >= p_open
    if green_candle:
        midline = ((p_close - p_open) / 2.0) + p_open
    else:
        midline = ((p_open - p_close) / 2.0) + p_close

    score = 0

    if trend_mode_val == "Open, High, Low, Close + Midline":
        if green_candle:
            if c_close >= midline and c_close < p_close:
                score = -1 if reversed_flag else 1
            elif c_close < midline and c_close > p_open:
                score = 1 if reversed_flag else -1
            elif c_close >= p_close and c_close < p_high:
                score = -2 if reversed_flag else 2
            elif c_close <= p_open and c_close > p_low:
                score = 2 if reversed_flag else -2
            elif c_close >= p_high:
                score = -3 if reversed_flag else 3
            elif c_close <= p_low:
                score = 3 if reversed_flag else -3
        else:  # Red candle
            if c_close >= midline and c_close < p_open:
                score = -1 if reversed_flag else 1
            elif c_close < midline and c_close > p_close:
                score = 1 if reversed_flag else -1
            elif c_close >= p_open and c_close < p_high:
                score = -2 if reversed_flag else 2
            elif c_close <= p_close and c_close > p_low:
                score = 2 if reversed_flag else -2
            elif c_close >= p_high:
                score = -3 if reversed_flag else 3
            elif c_close <= p_low:
                score = 3 if reversed_flag else -3

    elif trend_mode_val == "Above/Below Midline":
        if c_close >= midline:
            score = -3 if reversed_flag else 3
        else:
            score = 3 if reversed_flag else -3

    return score


@st.cache_data(ttl=300)
def fetch_data(ticker, period, interval):
    try:
        data = yf.download(ticker, period=period, interval=interval, progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        return data
    except Exception as e:
        return None


def get_score_time_series(df, trend_mode_val, reversed_flag):
    if df is None or len(df) < 3:
        return pd.DataFrame()
    
    scores = []
    dates = []
    for i in range(2, len(df)):
        p_open = df["Open"].iloc[i-1]
        p_high = df["High"].iloc[i-1]
        p_low = df["Low"].iloc[i-1]
        p_close = df["Close"].iloc[i-1]
        c_close = df["Close"].iloc[i]
        
        score = _compute_single_score(p_open, p_high, p_low, p_close, c_close, trend_mode_val, reversed_flag)
        scores.append(score)
        dates.append(df.index[i])
        
    return pd.DataFrame({'Date': dates, 'Score': scores})


def calculate_next_score_probabilities(score_df, n_back=3):
    if score_df.empty or len(score_df) <= n_back:
        return [], {}, 0
    
    scores = score_df['Score'].tolist()
    current_pattern = scores[-n_back:]
    
    next_scores = []
    for i in range(len(scores) - n_back):
        window = scores[i:i+n_back]
        if window == current_pattern:
            if i + n_back < len(scores):
                next_scores.append(scores[i+n_back])
                
    if not next_scores:
        return current_pattern, {}, 0
        
    total_matches = len(next_scores)
    counts = Counter(next_scores)
    
    # Calculate probabilities as percentages
    probabilities = {score: (count / total_matches) * 100 for score, count in counts.items()}
    probabilities = dict(sorted(probabilities.items()))
    
    return current_pattern, probabilities, total_matches


# ---------------------------------------------------------
# MAIN DASHBOARD UI
# ---------------------------------------------------------
st.title("🎯 Score Probability Predictor Dashboard")
st.markdown(f"Analyzing historical pattern probabilities for **{symbol.upper()}** on timeframe **{timeframe}**.")

# Fetch Data
data_interval = timeframe
df = fetch_data(symbol, history_period, data_interval)

if df is None or df.empty:
    st.error(f"Could not retrieve data for ticker '{symbol}'. Please check the symbol and try again.")
else:
    score_history = get_score_time_series(df, trend_mode, reversed_flag)
    
    if score_history.empty:
        st.warning("Not enough historical data points to generate scores.")
    else:
        current_pattern, probabilities, total_matches = calculate_next_score_probabilities(score_history, lookback_n)
        
        # Display Overview Metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Historical Bars Analyzed", len(score_history))
        with col2:
            st.metric("Current Pattern Sequence", " → ".join(map(str, current_pattern)))
        with col3:
            st.metric("Historical Matches Found", total_matches)
            
        st.markdown("---")
        
        # Results Section
        st.subheader("📊 Next Score Probability Breakdown")
        
        if total_matches > 0:
            # Prepare probability dataframe for visualization
            prob_df = pd.DataFrame(list(probabilities.items()), columns=["Next Score", "Probability (%)"])
            prob_df["Probability (%)"] = prob_df["Probability (%)"].round(2)
            prob_df["Probability (Fraction)"] = prob_df["Probability (%)"].apply(lambda x: f"{x}%")
            
            c1, c2 = st.columns([1, 1])
            
            with c1:
                st.markdown("##### Probability Table")
                st.dataframe(prob_df[["Next Score", "Probability (Fraction)"]], use_container_width=True, hide_index=True)
                
            with c2:
                st.markdown("##### Visual Distribution")
                chart_data = prob_df.set_index("Next Score")["Probability (%)"]
                st.bar_chart(chart_data)
        else:
            st.info("⚠️ No historical matches found for this exact score sequence in the selected timeframe range. Try expanding the history range or choosing a smaller lookback window.")
            
        with st.expander("🔍 View Full Historical Score Series"):
            st.dataframe(score_history.tail(100).sort_values(by="Date", ascending=False), use_container_width=True, hide_index=True)