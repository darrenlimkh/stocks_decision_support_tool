import yfinance as yf
import ta

from datetime import datetime, timedelta

def get_technicals(ticker: str):
    end = datetime.today()
    start = end - timedelta(days=365)
    df = yf.download(ticker, start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"), progress=False)

    if df.empty:
        return {}, "NOT NOW", ["No price data available"]

    # Ensure Series not DataFrame
    close = df["Close"][ticker]
    volume = df["Volume"][ticker]

    techs = {}
    reasons = []
    buy_score = []
    technical_score = 0

    # RSI
    # Using RSI to identify overbought/oversold conditions
    rsi = ta.momentum.RSIIndicator(close).rsi().iloc[-1]
    techs["RSI"] = rsi
    if rsi < 30:
        reasons.append(f"RSI ({rsi:.2f}) < 30 (oversold, buy signal)")   
        buy_score.append(1)
        technical_score += 1
    elif 30 <= rsi <= 70:
        reasons.append(f"RSI ({rsi:.2f}) between 30 and 70 (neutral)")
        buy_score.append(0)
        technical_score += 0
    elif rsi > 70:
        reasons.append(f"RSI ({rsi:.2f}) > 70 (overbought, avoid buying)")
        buy_score.append(-1)
        technical_score -= 1

    # MACD
    ## Using MACD to identify trend direction
    macd = ta.trend.MACD(close)
    macd_diff = macd.macd_diff().iloc[-1]
    techs["MACD_diff"] = macd_diff
    if macd_diff > 0:
        reasons.append("MACD > Signal (bullish)")
        buy_score.append(1)
        technical_score += 1
    else:
        reasons.append("MACD <= Signal (bearish)")
        buy_score.append(-1)
        technical_score -= 1

    # Moving averages
    ## Using 50-day and 200-day SMAs to identify long-term trend
    sma50 = close.rolling(50).mean().iloc[-1]
    sma200 = close.rolling(200).mean().iloc[-1]
    techs["SMA50/200"] = (sma50, sma200)
    if sma50 > sma200:
        reasons.append("50-day SMA > 200-day SMA (uptrend)")
        buy_score.append(1)
        technical_score += 1
    else:
        reasons.append("50-day SMA <= 200-day SMA (downtrend)")
        buy_score.append(-1)
        technical_score -= 1

    # Breakout
    ## Using 20-day high/low to identify breakout conditions
    last_price = close.iloc[-1]
    high20 = close.rolling(20).max().iloc[-1]
    low20 = close.rolling(20).min().iloc[-1]
    techs["LastPrice"] = last_price
    if last_price > high20:
        reasons.append("Price broke above 20-day high (bullish breakout)")
        buy_score.append(1)
        technical_score += 1
    elif last_price < low20:
        reasons.append("Price broke below 20-day low (bearish)")
        buy_score.append(-1)
        technical_score -= 1
    else:
        reasons.append("Price within 20-day range (no breakout)")
        buy_score.append(0)
        technical_score += 0

    # Volume
    ## Using 20-day average volume to confirm price moves
    avg_vol = volume.rolling(20).mean().iloc[-1]
    std_vol = volume.rolling(20).std().iloc[-1]  

    last_vol = volume.iloc[-1]
    techs["LastVolume"] = last_vol
    if last_vol > avg_vol + 2 * std_vol:
        reasons.append("High volume confirms move")
        buy_score.append(1)
        technical_score += 1
    else:
        reasons.append("Not high volume, no confirmation to move")
        buy_score.append(-1)
        technical_score += -1

    decision = "NOT BUY"
    if technical_score > 0:
        decision = "BUY"

    return decision, techs, reasons, buy_score