import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
import ta

from metrics.fundamentals import get_fundamentals
from metrics.technicals import get_technicals

from machine_learning.t_predict import get_technicals_prediction
from machine_learning.f_predict import get_fundamentals_prediction

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
    initial_sidebar_state="expanded",
)

if "analyzed" not in st.session_state:
    st.session_state.analyzed = False
if "ticker" not in st.session_state:
    st.session_state.ticker = ""
if "valid" not in st.session_state:
    st.session_state.valid = False

# Sidebar
st.sidebar.title("📊 Stock Analyzer")
with st.sidebar.form(key="analyze_form"):
    ticker = st.text_input("Enter Ticker:", "AAPL").upper()
    valid = is_valid_ticker(ticker) if ticker else False
    analyze_button = st.form_submit_button(label="Analyze")

color_map = {
    "BUY": "green", 
    "DON'T BUY": "red"
}

icon_map = {
    "BUY": "🟢",
    "DON'T BUY": "🔴"
}

bg_color_map = {
    "BUY": "#e6ffe6",   # light green
    "DON'T BUY": "#ffe6e6",  # light red
}


if analyze_button:
    st.session_state.analyzed = True
    st.session_state.ticker = ticker
    st.session_state.valid = valid

if st.session_state.analyzed:
    ticker = st.session_state.ticker    
    valid = st.session_state.valid

    if ticker and not valid:
        st.error(f"Error: {ticker} is invalid. Please enter a valid symbol.")
        st.stop()
  

    st.title(f"Stock Analysis for {ticker}")

    f_fundamentals, f_benchmarks, f_reasons, f_scores= get_fundamentals(ticker)
    f_data = [[k, v, f_reasons[i], f_scores[i]] for i, (k, v) in enumerate(f_fundamentals.items())]
    f_df = pd.DataFrame(f_data, columns=['Metric', 'Value', 'Interpretation', 'Buy Signal'])
    f_df["Buy Signal"] = pd.to_numeric(f_df["Buy Signal"], errors="coerce")
    f_ml_decision, f_prob = get_fundamentals_prediction(ticker)

    t_technicals, t_benchmarks, t_reasons, t_scores = get_technicals(ticker)
    t_data = [[k, v, t_reasons[i], t_scores[i]] for i, (k, v) in enumerate(t_technicals.items())]
    t_df = pd.DataFrame(t_data, columns=['Metric', 'Value', 'Interpretation', 'Buy Signal'])
    t_df["Buy Signal"] = pd.to_numeric(t_df["Buy Signal"], errors="coerce")
    t_ml_decision, t_prob = get_technicals_prediction(ticker, lookahead_days=30, model_type='logistic')

    for df in [f_df, t_df]:
        df['Value'] = df['Value'].astype(str)
        df['Buy Signal'] = pd.to_numeric(df['Buy Signal'], errors='coerce')
        df['Metric'] = df['Metric'].astype(str)
        df['Interpretation'] = df['Interpretation'].astype(str)

    tab1, tab2, tab3 = st.tabs(["Fundamentals", "Technicals", "Price Chart"])

    with tab1:
        st.subheader("Fundamental Analysis")
        
        st.markdown(
            f"""
            <div style='
                background-color:{bg_color_map[f_ml_decision]};
                padding: 1.5rem;
                border-radius: 12px;
                text-align: center;
                font-size: 1.4rem;
                font-weight: bold;
                box-shadow: 0px 3px 7px rgba(0,0,0,0.1);
                margin-bottom: 25px;
            '>
                <span style='color:#000000;'>Recommended Decision:</span> 
                <span style='color:{color_map[f_ml_decision]};'>{icon_map[f_ml_decision]} {f_ml_decision}</span> 
                <span style='color:{color_map[f_ml_decision]};'>({f_prob*100:.1f}% Gain Potential)</span>
            </div>
            """.replace("{", "{{").replace("}", "}}").replace("{{bg_color_map", "{bg_color_map").replace("{{color_map", "{color_map").replace("{{icon_map", "{icon_map").replace("{{f_ml_decision", "{f_ml_decision").replace("{{f_prob", "{f_prob"),
            unsafe_allow_html=True
        )
        st.dataframe(f_df, use_container_width=True)
        with st.expander("View Buy Signal Benchmarks"):
            for f_benchmark in f_benchmarks:
                if f_benchmark == "-":
                    continue
                st.markdown(f"• {f_benchmark}")

    if "tech_result" not in st.session_state:
        st.session_state.tech_result = None

    with tab2:
        st.subheader("Technical Analysis")

        with st.form(key="technical_form"):
            holding_period = st.number_input(
                "Enter Holding Period (Days):",
                min_value=1,
                max_value=365,
                value=30,
                step=1,
                help="Holding duration in days, which is used to estimate probability of upside."
            )
            tech_button = st.form_submit_button("Update Technical Prediction")

        if tech_button:
            t_ml_decision, t_prob = get_technicals_prediction(
            ticker,
            lookahead_days=holding_period,
            model_type='logistic'
        )
            st.session_state.tech_result = (t_ml_decision, t_prob)

        if st.session_state.tech_result:
            t_ml_decision, t_prob = st.session_state.tech_result

        st.markdown(
            f"""
            <div style='
                background-color:{bg_color_map[t_ml_decision]};
                padding: 1.5rem;
                border-radius: 12px;
                text-align: center;
                font-size: 1.4rem;
                font-weight: bold;
                box-shadow: 0px 3px 7px rgba(0,0,0,0.1);
                margin-bottom: 25px;
            '>
                <span style='color:#000000;'>Recommended Decision:</span> 
                <span style='color:{color_map[t_ml_decision]};'>{icon_map[t_ml_decision]} {t_ml_decision}</span> 
                <span style='color:{color_map[t_ml_decision]};'>({t_prob*100:.1f}% Gain Potential)</span>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.dataframe(t_df, use_container_width=True)
        with st.expander("View Buy Signal Benchmarks"):
            for t_benchmark in t_benchmarks:
                st.markdown(f"• {t_benchmark}")

    with tab3:
        st.subheader("Price & Indicators")
        stock_data_3y = yf.download(ticker, period="3y", progress=False, auto_adjust=True)
        stock_data_1y = yf.download(ticker, period="1y", progress=False, auto_adjust=True)
        stock_data_close_3y = stock_data_3y['Close'][ticker]
        stock_data_close_1y = stock_data_1y['Close'][ticker]
        
        sma50_series = stock_data_close_3y.rolling(50).mean()
        sma200_series = stock_data_close_3y.rolling(200).mean()
        
        sma_df = pd.concat([sma50_series, sma200_series], axis=1)
        sma_df.columns = ["sma50", "sma200"]

        sma_df = sma_df.dropna().tail(365)
        sma_df = sma_df.loc[stock_data_1y.index.intersection(sma_df.index)]

        if not stock_data_close_1y.empty:
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(x=stock_data_1y.index, y=stock_data_close_1y, mode='lines', name='Close'))
            fig1.add_trace(go.Scatter(x=sma_df.index, y=sma_df['sma50'], mode='lines', name='SMA50'))
            fig1.add_trace(go.Scatter(x=sma_df.index, y=sma_df['sma200'], mode='lines', name='SMA200'))
            fig1.update_layout(title=f"{ticker} Price Chart with SMA50 & SMA200", xaxis_title="Date", yaxis_title="Price (USD)")
            st.plotly_chart(fig1, use_container_width=True)

            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=stock_data_1y.index, y=ta.momentum.RSIIndicator(stock_data_close_1y).rsi(), mode='lines', name='RSI'))
            fig2.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought 70", annotation_position="top left")
            fig2.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold 30", annotation_position="top left")
            fig2.update_layout(title="RSI Indicator", xaxis_title="Date", yaxis_title="RSI")
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.write("Price data not available.")
