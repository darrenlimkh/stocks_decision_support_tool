import yfinance as yf
import numpy as np
import ta

from datetime import datetime, timedelta

def get_technicals(ticker: str):
    end = datetime.today()
    start = end - timedelta(days=365)
    df = yf.download(ticker, start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"), progress=False, auto_adjust=True)

    if df.empty:
        return {}, "NOT NOW", ["No price data available"]

    # Ensure Series not DataFrame
    close = df["Close"][ticker]
    volume = df["Volume"][ticker]

    techs = {}
    reasons = []
    benchmarks = []
    buy_score = []
    # technical_score = 0

    # RSI
    # Using RSI to identify overbought/overs old conditions
    rsi = ta.momentum.RSIIndicator(close).rsi().iloc[-1]
    techs["RSI"] = f"{rsi:.2f}"
    if rsi < 30:
        benchmarks.append("RSI < 30")
        reasons.append(f"Oversold, Buy Opportunity")   
        buy_score.append(1)
        # technical_score += 1
    elif 30 <= rsi <= 70:
        benchmarks.append("30 ≤ RSI ≤ 70")
        reasons.append(f"Neutral")
        buy_score.append(0)
        # technical_score += 0
    elif rsi > 70:
        benchmarks.append("RSI > 70")
        reasons.append(f"Overbought, Avoid Buying")
        buy_score.append(-1)
        # technical_score -= 1

    # MACD
    ## Using MACD to identify trend direction
    macd = ta.trend.MACD(close)
    macd_signal = macd.macd_signal().iloc[-1]
    macd_line = macd.macd().iloc[-1], 2
    macd_diff = macd.macd_diff().iloc[-1]
    techs["MACD Line / Signal / Difference"] = f"{macd_signal:.2f} / {macd_line[0]:.2f} / {macd_diff:.2f}"
    if macd_diff > 0:
        benchmarks.append("MACD Line > Signal")
        reasons.append("Bullish Momentum")
        buy_score.append(1)
        # technical_score += 1
    else:
        benchmarks.append("MACD Line ≤ Signal")
        reasons.append("Bearish Momentum")
        buy_score.append(-1)
        # technical_score -= 1

    # Moving averages
    ## Using 50-day and 200-day SMAs to identify long-term trend
    sma50 = close.rolling(50).mean().iloc[-1]
    sma200 = close.rolling(200).mean().iloc[-1]
    techs["SMA50 / SMA200"] = f"{sma50:.2f} / {sma200:.2f}"
    if sma50 > sma200:
        benchmarks.append("SMA50 > SMA200")
        reasons.append("Bullish Trend")
        buy_score.append(1)
        # technical_score += 1
    else:
        benchmarks.append("SMA50 ≤ SMA200")
        reasons.append("Bearish Trend")
        buy_score.append(-1)
        # technical_score -= 1

    # Breakout
    ## Using 30-day high/low to identify breakout conditions
    last_price = close.iloc[-1]
    high30 = close.rolling(30).max().iloc[-1]
    low30 = close.rolling(30).min().iloc[-1]
    techs["Last Price / 30-day High / 30-day Low"] = f"{last_price:.2f} / {high30:.2f} / {low30:.2f}"
    if last_price > high30:
        benchmarks.append("Last Price > 30-day High")
        reasons.append("Bullish Breakout")
        buy_score.append(1)
        # technical_score += 1
    elif last_price < low30:
        benchmarks.append("Last Price < 30-day Low")
        reasons.append("Bearish Breakdown")
        buy_score.append(-1)
        # technical_score -= 1
    else:
        benchmarks.append("Last Price within 30-day Range")
        reasons.append("Neutral")
        buy_score.append(0)
        # technical_score += 0

    # Volume
    ## Using 30-day average volume to confirm price moves
    avg_vol = volume.rolling(30).mean().iloc[-1]
    std_vol = volume.rolling(30).std().iloc[-1]  
    last_vol = volume.iloc[-1]
    techs["Last Volume"] = f"{last_vol:,}"
    if last_vol > avg_vol + 2 * std_vol:
        benchmarks.append("Last Volume > (30-day Average + 2 · 30-day Std Dev)")
        reasons.append("High Trading Volume")
        buy_score.append(1)
        # technical_score += 1
    elif last_vol < avg_vol - 2 * std_vol:
        benchmarks.append("Last Volume < (30-day Average - 2 · 30-day Std Dev)")
        reasons.append("Low Trading Volume, Avoid Buying")
        buy_score.append(-1)
        # technical_score -= 1
    else:
        benchmarks.append("Last Volume within 95% Confidence Interval of 30-day Average")
        reasons.append("Normal Volume")
        buy_score.append(0)


    return techs, benchmarks, reasons, buy_score