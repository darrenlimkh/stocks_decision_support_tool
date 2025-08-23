import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
import ta

from metrics.fundamentals import get_fundamentals
from metrics.technicals import get_technicals

from machine_learning.predict import get_prediction

@st.cache_data(show_spinner=False)
def is_valid_ticker(ticker):
    try:
        data = yf.Ticker(ticker).history(period="1d", auto_adjust=True)
        return not data.empty
    except:
        return False
    
# Page Setup
st.set_page_config(
    page_title="Stocks Decision Support Tool",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar
st.sidebar.title("📊 Stock Analyzer")
with st.sidebar.form(key="analyze_form"):
    ticker = st.text_input("Enter Ticker:", "AAPL").upper()
    valid = is_valid_ticker(ticker) if ticker else False
        
    holding_period = st.number_input(
        "Enter Holding Period (Days):",
        min_value=1,
        max_value=365,
        value=30,
        step=1
    )

    submit_button = st.form_submit_button(label="Analyze")

color_map = {"BUY": "green", "HOLD": "orange", "DON'T BUY": "red"}

if submit_button:
    if ticker and not valid:
        st.error(f"Error: {ticker} is invalid. Please enter a valid symbol.")
        st.stop()
    # st.success(f"Analyzing {ticker} for {holding_period} days!")
    st.title(f"Stock Analysis for {ticker}")

    # Convert features to dataframe for display (optional)
    # ml_data = [[k, v] for k, v in ml_features.items()]
    # ml_df = pd.DataFrame(ml_data, columns=['Metric', 'Value'])

    f_decision, f_fundamentals, f_reasons, f_scores = get_fundamentals(ticker)
    f_data = [[k, v, f_reasons[i], f_scores[i]] for i, (k, v) in enumerate(f_fundamentals.items())]
    f_df = pd.DataFrame(f_data, columns=['Metric', 'Value', 'Interpretation', 'Buy Signal'])
    f_df["Buy Signal"] = pd.to_numeric(f_df["Buy Signal"], errors="coerce")

    t_technicals, t_benchmarks, t_reasons, t_scores = get_technicals(ticker)
    t_data = [[k, v, t_reasons[i], t_scores[i]] for i, (k, v) in enumerate(t_technicals.items())]
    t_df = pd.DataFrame(t_data, columns=['Metric', 'Value', 'Interpretation', 'Buy Signal'])
    t_df["Buy Signal"] = pd.to_numeric(t_df["Buy Signal"], errors="coerce")
    t_ml_decision, t_ml_features, t_prob = get_prediction(ticker, lookadhead_days=holding_period, model_type='logistic')

    decision_color = "green" if t_ml_decision.lower() == "buy" else "red"
    prob_color = "green" if t_prob > 0 else "red"

    tab1, tab2, tab3 = st.tabs(["Fundamentals", "Technicals", "Price Chart"])

    with tab1:
        st.subheader("Fundamental Analysis")
        st.markdown(f"**Decision:** <span style='color:{color_map[f_decision]}'>{f_decision}</span>", unsafe_allow_html=True)
        st.dataframe(f_df, use_container_width=True)
        with st.expander("View detailed reasons"):
            st.write(f_reasons)

    with tab2:
        st.subheader("Technical Analysis")
        
        st.markdown(f"**Decision:** "
                    f"<span style='color:{color_map[t_ml_decision]}'>{t_ml_decision}</span> "
                    f"<span style='color:{color_map[t_ml_decision]}'> ({t_prob*100:.1f}% Gain Potential) </span>", 
                    unsafe_allow_html=True
        )
        #
        st.dataframe(t_df, use_container_width=True)
        with st.expander("View Buy Signal Benchmarks"):
            for t_benchmark in t_benchmarks:
                st.markdown(f"• {t_benchmark}")

    with tab3:
        st.subheader("Price & Indicators")
        stock_data = yf.download(ticker, period="1y", progress=False, auto_adjust=True)
        stock_data_close = stock_data['Close'][ticker]
        if not stock_data_close.empty:
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(x=stock_data.index, y=stock_data_close, mode='lines', name='Close'))
            fig1.add_trace(go.Scatter(x=stock_data.index, y=stock_data_close.rolling(50).mean(), mode='lines', name='SMA50'))
            fig1.add_trace(go.Scatter(x=stock_data.index, y=stock_data_close.rolling(200).mean(), mode='lines', name='SMA200'))
            fig1.update_layout(title=f"{ticker} Price Chart with SMA50 & SMA200", xaxis_title="Date", yaxis_title="Price (USD)")
            st.plotly_chart(fig1, use_container_width=True)

            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=stock_data.index, y=ta.momentum.RSIIndicator(stock_data_close).rsi(), mode='lines', name='RSI'))
            fig2.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought 70", annotation_position="top left")
            fig2.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold 30", annotation_position="top left")
            fig2.update_layout(title="RSI Indicator", xaxis_title="Date", yaxis_title="RSI")
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.write("Price data not available.")
