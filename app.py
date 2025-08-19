import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf

from metrics.fundamentals import get_fundamentals
from metrics.technicals import get_technicals

# ---------- Page setup ----------
st.set_page_config(
    page_title="Stocks Decision Support Tool",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- Sidebar ----------
st.sidebar.title("📊 Stock Analyzer")
ticker = st.sidebar.text_input("Enter stock ticker:", "AAPL").upper()
run_analysis = st.sidebar.button("Analyze")

# ---------- Color map for decisions ----------
color_map = {"BUY": "green", "HOLD": "orange", "NOT BUY": "red"}

# ---------- Run analysis ----------
if run_analysis:
    st.title(f"Stock Analysis for {ticker}")

    # ---------- Fundamentals ----------
    f_decision, f_fundamentals, f_reasons, f_scores = get_fundamentals(ticker)
    f_data = [[k, v, f_reasons[i], f_scores[i]] for i, (k, v) in enumerate(f_fundamentals.items())]
    f_df = pd.DataFrame(f_data, columns=['Metric', 'Value', 'Interpretation', 'Buy Signal'])

    # ---------- Technicals ----------
    t_decision, t_technicals, t_reasons, t_scores = get_technicals(ticker)
    t_data = [[k, v, t_reasons[i], t_scores[i]] for i, (k, v) in enumerate(t_technicals.items())]
    t_df = pd.DataFrame(t_data, columns=['Metric', 'Value', 'Interpretation', 'Buy Signal'])

    # ---------- Tabs for organization ----------
    tab1, tab2, tab3 = st.tabs(["Fundamentals", "Technicals", "Price Chart"])

    # ---- Fundamentals Tab ----
    with tab1:
        st.subheader("🔹 Fundamental Analysis")
        st.markdown(f"**Decision:** <span style='color:{color_map[f_decision]}'>{f_decision}</span>", unsafe_allow_html=True)
        st.dataframe(f_df)
        with st.expander("View detailed reasons"):
            st.write(f_reasons)

    # ---- Technicals Tab ----
    with tab2:
        st.subheader("🔹 Technical Analysis")
        st.markdown(f"**Decision:** <span style='color:{color_map[t_decision]}'>{t_decision}</span>", unsafe_allow_html=True)
        st.dataframe(t_df)
        with st.expander("View detailed reasons"):
            st.write(t_reasons)

    # ---- Price Chart Tab ----
    with tab3:
        st.subheader("📈 Price & Indicators")

        # Download price data for chart
        stock_data = yf.download(ticker, period="1y", progress=False)
        if not stock_data.empty:
            close = stock_data['Close']
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=close.index, y=close, mode='lines', name='Close'))
            fig.add_trace(go.Scatter(x=close.index, y=close.rolling(50).mean(), mode='lines', name='SMA50'))
            fig.add_trace(go.Scatter(x=close.index, y=close.rolling(200).mean(), mode='lines', name='SMA200'))

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.write("Price data not available.")
