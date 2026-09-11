from datetime import datetime, timedelta
from collections import Counter, defaultdict
import re
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# Page Configuration
st.set_page_config(page_title="", layout="wide")

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

# 1. Ticker symbol as a dropdown list
ticker_options = [
    "EURUSD=X", "GBPUSD=X", "AUDUSD=X", "NZDUSD=X", "USDCAD=X",
    "USDCHF=X", "USDJPY=X", "USDSGD=X", "GC=F", "BZ=F", "ZB=F",
    "BTC-USD", "EURGBP=X", "EURAUD=X", "EURNZD=X", "EURCAD=X",
    "EURCHF=X", "EURJPY=X", "EURSGD=X", "XAUEUR=X", "GBPAUD=X",
    "GBPNZD=X", "GBPCAD=X", "GBPCHF=X", "GBPJPY=X", "GBPSGD=X",
    "AUDNZD=X", "AUDCAD=X", "AUDCHF=X", "AUDJPY=X", "AUDSGD=X",
    "AAPL", "MSFT", "SPY", "QQQ"
]
symbol = st.sidebar.selectbox("Ticker Symbol", options=ticker_options, index=0, help="Select a ticker symbol from the list.")

trend_mode = st.sidebar.selectbox(
    "Trend Mode",
    ["Open, High, Low, Close + Midline", "Above/Below Midline"],
    help="Choose between detailed OHLC candle scoring or simple Above/Below Midline scoring.",
)

reversed_flag = st.sidebar.checkbox("Reverse Score Direction", value=False, help="Invert positive and negative scores if needed.")

st.sidebar.subheader("Timeframe & History")
timeframe = st.sidebar.selectbox("Timeframe", ["60m", "1d", "1wk", "1mo", "3mo"], index=1)
history_period = st.sidebar.selectbox("History Range", ["1y", "2y", "5y", "10y", "max"], index=2)

lookback_n = st.sidebar.slider("Pattern Lookback Window (Scores)", min_value=1, max_value=5, value=2, help="Number of past consecutive scores to match historically.")

# 3. Option to show all patterns
show_all_patterns = st.sidebar.checkbox("Show All Historical Patterns Summary", value=False)

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


def get_score_time_series(df, trend_mode_val, reversed_flag, timeframe):
    if df is None or len(df) < 4:
        return pd.DataFrame()
    
    # If using intraday timeframes, the last row from yfinance is usually the live/in-progress candle.
    # We drop the last row so all calculations are based strictly on closed candles.
    work_df = df.copy()
    if timeframe in ["60m", "30m", "15m", "5m", "1m"]:
        work_df = work_df.iloc[:-1]
        
    scores = []
    dates = []
    for i in range(2, len(work_df)):
        p_open = work_df["Open"].iloc[i-1]
        p_high = work_df["High"].iloc[i-1]
        p_low = work_df["Low"].iloc[i-1]
        p_close = work_df["Close"].iloc[i-1]
        c_close = work_df["Close"].iloc[i]
        
        score = _compute_single_score(p_open, p_high, p_low, p_close, c_close, trend_mode_val, reversed_flag)
        scores.append(score)
        dates.append(work_df.index[i])
        
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
    
    probabilities = {score: (count / total_matches) * 100 for score, count in counts.items()}
    probabilities = dict(sorted(probabilities.items()))
    
    return current_pattern, probabilities, total_matches


def get_all_patterns_summary(score_df, n_back=3):
    if score_df.empty or len(score_df) <= n_back:
        return pd.DataFrame()
    
    scores = score_df['Score'].tolist()
    pattern_transitions = defaultdict(list)
    
    for i in range(len(scores) - n_back):
        window = tuple(scores[i:i+n_back])
        next_val = scores[i+n_back]
        pattern_transitions[window].append(next_val)
        
    summary_data = []
    for pattern, subsequent in pattern_transitions.items():
        total_occurrences = len(subsequent)
        counts = Counter(subsequent)
        top_3 = counts.most_common(3)
        
        row = {
            "Pattern Sequence": " → ".join(map(str, pattern)),
            "Total Sample Occurrences": total_occurrences,
        }
        
        # Populate up to the top 3 most common next scores
        for idx, (score_val, count) in enumerate(top_3):
            pct = (count / total_occurrences) * 100
            row[f"Top {idx+1} Score"] = score_val
            row[f"Top {idx+1} Prob"] = f"{pct:.1f}%"
        
        # Fill placeholders if a pattern has fewer than 3 unique subsequent outcomes
        for idx in range(len(top_3), 3):
            row[f"Top {idx+1} Score"] = "-"
            row[f"Top {idx+1} Prob"] = "-"
            
        summary_data.append(row)
        
    summary_df = pd.DataFrame(summary_data)
    return summary_df.sort_values(by="Total Sample Occurrences", ascending=False)

# ---------------------------------------------------------
# MAIN DASHBOARD UI
# ---------------------------------------------------------
st.title("🎯 Score Probability Predictor Dashboard")
st.markdown(f"Analyzing historical pattern probabilities for **{symbol}** on timeframe **{timeframe}**.")

# Fetch Data
df = fetch_data(symbol, history_period, timeframe)

if df is None or df.empty:
    st.error(f"Could not retrieve data for ticker '{symbol}'. Please check the symbol and try again.")
else:
    score_history = get_score_time_series(df, trend_mode, reversed_flag, timeframe)
    
    if score_history.empty:
        st.warning("Not enough historical data points to generate scores.")
    else:
        current_pattern, probabilities, total_matches = calculate_next_score_probabilities(score_history, lookback_n)
        
        # Display Overview Metrics with Sample Size Highlight
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Dataset Bars", len(score_history))
        with col2:
            st.metric("Current Pattern Sequence", " → ".join(map(str, current_pattern)))
        with col3:
            st.metric("Sample Size (Matching Occurrences)", total_matches)
            
        st.markdown("---")
        
        # Results Section for Current Pattern
        st.subheader("📊 Next Score Probability Breakdown")
        st.info(f"ℹ️ This statistic is based on **{total_matches}** historical sample occurrences where the sequence `{current_pattern}` appeared.")
        
        if total_matches > 0:
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
            st.warning("⚠️ No historical matches found for this exact score sequence in the selected timeframe range. Try expanding the history range or choosing a smaller lookback window.")
            
        # 3. Show all patterns table option if toggled
        if show_all_patterns:
            st.markdown("---")
            st.subheader("📋 All Historical Patterns Summary")
            st.markdown(f"Overview of all unique pattern sequences of length **{lookback_n}** found in history and their sample sizes:")
            all_summary_df = get_all_patterns_summary(score_history, lookback_n)
            if not all_summary_df.empty:
                st.dataframe(all_summary_df, use_container_width=True, hide_index=True)
            else:
                st.info("Not enough data to generate pattern summaries.")

        with st.expander("🔍 View Full Historical Score Series"):
            st.dataframe(score_history.tail(100).sort_values(by="Date", ascending=False), use_container_width=True, hide_index=True)
