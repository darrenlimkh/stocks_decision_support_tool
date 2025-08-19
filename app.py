import yfinance as yf
import pandas as pd
import numpy as np
import warnings
import argparse
import re
import ta

from datetime import datetime, timedelta
from IPython.display import display

warnings.simplefilter(action='ignore', category=FutureWarning)
pd.set_option('display.max_rows', None)     # Show all rows
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.width', None)        # Auto-adjust width
pd.set_option('display.max_colwidth', None) # Show full column content

def get_sector_benchmarks(sector: str):
    """
    Returns typical valuation & financial benchmarks for a given sector.
    Values are rough averages and should be refined with real market data.
    """
    sector_defaults = {
        "Technology": {
            "P/E": 25,
            "OperatingMargin": 0.15,
            "RevenueGrowth": 0.05,   # high growth expectation
            "DebtEquity": 0.8,
            "PayoutRatio": 0.40,     # tech often reinvests profits
        },
        "Utilities": {
            "P/E": 15,
            "OperatingMargin": 0.10,
            "RevenueGrowth": 0.02,   # slow but stable growth
            "DebtEquity": 1.5,       # higher leverage acceptable
            "PayoutRatio": 0.70,     # dividends are main appeal
        },
        "Financial Services": {
            "P/E": 12,
            "OperatingMargin": 0.20,
            "RevenueGrowth": 0.03,
            "DebtEquity": 2.0,       # banks run high leverage
            "PayoutRatio": 0.50,
        },
        "Healthcare": {
            "P/E": 18,
            "OperatingMargin": 0.12,
            "RevenueGrowth": 0.04,
            "DebtEquity": 1.0,
            "PayoutRatio": 0.50,
        },
        "Consumer Defensive": {
            "P/E": 18,
            "OperatingMargin": 0.08,
            "RevenueGrowth": 0.02,
            "DebtEquity": 1.2,
            "PayoutRatio": 0.60,
        },
        "Consumer Cyclical": {
            "P/E": 20,
            "OperatingMargin": 0.10,
            "RevenueGrowth": 0.03,
            "DebtEquity": 1.0,
            "PayoutRatio": 0.50,
        },
        "Energy": {
            "P/E": 14,
            "OperatingMargin": 0.12,
            "RevenueGrowth": 0.03,
            "DebtEquity": 1.2,
            "PayoutRatio": 0.50,
        },
        "Industrials": {
            "P/E": 17,
            "OperatingMargin": 0.08,
            "RevenueGrowth": 0.03,
            "DebtEquity": 1.0,
            "PayoutRatio": 0.45,
        }
    }

    # default fallback
    return sector_defaults.get(sector, {
        "P/E": 20,
        "OperatingMargin": 0.10,
        "RevenueGrowth": 0.00,
        "DebtEquity": 1.0,
        "PayoutRatio": 0.70
    })

def get_fundamentals(ticker: str):
    stock = yf.Ticker(ticker)
    info = stock.info
    try:
        sector = info['sector']
    except KeyError:
        sector = "Unknown"  

    sector_benchmarks = get_sector_benchmarks(sector)

    fundamentals = {}
    decision = "NOT BUY"
    reasons = []
    buy_score = []

    '''Valuation'''
    valuation_score = 0
    # --- PEG ratio ---
    ## Used to assess growth relative to P/E.
    peg = info.get("trailingPegRatio")
    fundamentals["PEG"] = peg
    if peg is not None:
        if peg < 1:
            reasons.append(f"PEG ({peg:.2f}) < 1 (undervalued)")
            buy_score.append(1)
            valuation_score += 1
        else:
            reasons.append(f"PEG ({peg:.2f}) >= 1 (fair/overvalued)")
            buy_score.append(-1)
            valuation_score -= 1
    else:
        buy_score.append('-')
        reasons.append("PEG data not available")

    # --- P/E ratio ---
    ## Used to assess valuation relative to earnings.
    industry_pe = sector_benchmarks["P/E"]
    pe = info.get("trailingPE")
    fundamentals["P/E"] = pe
    if pe is not None:
        if pe < industry_pe:
            reasons.append(f"P/E ({pe:.2f}) < {industry_pe} (undervalued)")
            buy_score.append(1)
            valuation_score += 1
        else:
            reasons.append(f"P/E ({pe:.2f}) >= {industry_pe} (fair/overvalued)")
            buy_score.append(-1)
            valuation_score -= 1
    else:
        buy_score.append('-')
        reasons.append("P/E data not available")

    # --- P/B ratio ---
    ## Used only when assets drive value. Tech companies with high intangibles may not use P/B.
    if sector != 'Technology':
        pb = info.get("priceToBook")
        fundamentals["P/B"] = pb
        if pb is not None:
            if pb < 1:
                reasons.append(f"P/B ({pb:.2f}) < 1 (undervalued)")
                buy_score.append(1)
                valuation_score += 1
            else:
                reasons.append(f"P/B ({pb:.2f}) >= 1 (fair/overvalued)")
                buy_score.append(-1)
                valuation_score -= 1
        else:
            pb = None
            fundamentals["P/B"] = None
            buy_score.append('-')
            reasons.append("P/B data not available")
    else:
        pb = None
        fundamentals["P/B"] = None
        buy_score.append('-')
        reasons.append("P/B not applicable for tech sector")

    # --- Free Cash Flow ---
    ## Free cash flow is the cash a company generates after accounting for capital expenditures.
    fcf = info.get("freeCashflow")
    market_cap = info.get("marketCap") 

    if fcf is not None and market_cap is not None and market_cap != 0:
        fcf_yield = fcf / market_cap  # decimal form
        fcf_yield_pct = fcf_yield * 100
        fundamentals["FCF_Yield"] = fcf_yield
        if fcf_yield_pct > 5:
            reasons.append(f"FCF Yield ({fcf_yield_pct:.2f}%) > 5 (Cash-generating)")
            buy_score.append(1)
            valuation_score += 1
        elif fcf_yield_pct > 0:
            reasons.append(f"FCF Yield ({fcf_yield_pct:.2f}%) > 0 (Neutral cash flow)")
            buy_score.append(0)
            valuation_score += 0
        else:
            reasons.append(f"FCF Yield ({fcf_yield_pct:.2f}%) < 0 (Cash-burn)")
            buy_score.append(-1)
            valuation_score -= 1
    else:
        fcf_yield = None
        fundamentals["FCFYield"] = None
        buy_score.append('-')
        reasons.append("FCF Yield data not available")

    '''Risk & Financial Health'''
    risk_score = 0
    # --- Debt/Equity ---
    ## Used to assess financial risk.
    industry_debt_equity = sector_benchmarks["DebtEquity"]
    de = info.get("debtToEquity")

    if de is not None:
        de = de / 100
        fundamentals["D/E"] = de    
        if de < 1:
            reasons.append(f"D/E ({de:.2f}) < {industry_debt_equity} (low financial risk)")
            buy_score.append(1)
            risk_score += 1
        else:
            reasons.append(f"D/E ({de:.2f}) >= {industry_debt_equity} (high financial risk)")
            buy_score.append(-1)
            risk_score -= 1
    else:
        de = None
        fundamentals["D/E"] = None
        buy_score.append('-')
        reasons.append("Debt/Equity data not available")

    # --- Operating Margin ---
    ## Operating margin measures the percentage of revenue left after covering operating expenses.
    industry_op_margin = sector_benchmarks["OperatingMargin"]
    op_margin = info.get("operatingMargins")
    fundamentals["OperatingMargin"] = op_margin
    if op_margin is not None:
        if op_margin > industry_op_margin:
            reasons.append(f"Operating margin ({op_margin:.2f}) > {industry_op_margin} (healthy)")
            buy_score.append(1)
            risk_score += 1
        else:
            reasons.append(f"Operating margin ({op_margin:.2f}) <= {industry_op_margin} (unhealthy)")
            buy_score.append(-1)
            risk_score -= 1
    else:
        op_margin = None
        fundamentals["OperatingMargin"] = None
        buy_score.append('-')
        reasons.append("Operating Margin data not available")

    # --- Revenue Growth ---
    ## Revenue growth indicates the company's ability to increase sales over time.
    growth = info.get("revenueGrowth")
    industry_revenue_growth = sector_benchmarks["RevenueGrowth"]
    fundamentals["RevenueGrowth"] = growth
    if growth is not None:
        if growth > 0:
            reasons.append(f"Growth ({growth:.2f}) > {industry_revenue_growth} (growing revenue)")
            buy_score.append(1)
            risk_score += 1
        else:
            reasons.append(f"Growth ({growth:.2f}) <= {industry_revenue_growth} (declining revenue)")
            buy_score.append(-1)
            risk_score -= 1
    else:
        growth = None
        fundamentals["RevenueGrowth"] = None
        buy_score.append('-')
        reasons.append("Revenue Growth data not available")

    # --- Dividend sustainability ---
    ## Dividend payout ratio indicates how much of earnings are paid out as dividends.
    payout = info.get("payoutRatio")
    industry_payout_ratio = sector_benchmarks["PayoutRatio"]
    fundamentals["PayoutRatio"] = payout
    if payout is not None:
        if payout < 0.7:
            reasons.append(f"Dividend payout ({payout:.2f}) < {industry_payout_ratio} (sustainable)")
            buy_score.append(1)
            risk_score += 1
        else:
            reasons.append(f"Dividend payout ({payout:.2f}) >= {industry_payout_ratio} (unsustainable)")
            buy_score.append(-1)
            risk_score -= 1
    else:
        payout = None
        fundamentals["PayoutRatio"] = None
        buy_score.append('-')
        reasons.append("Payout Ratio data not available")

    '''Analyst Recommendations'''
    analyst_score = 0
    # --- Analyst recommendations ---
    ## Analyst recommendations indicate market sentiment.
    recommendations = stock.recommendations
    print(recommendations)
    if not recommendations.empty:
        # Recommendation weights
        rec_weights = {"strongBuy": 2, "buy": 1, "hold": 0, "sell": -1, "strongSell": -2}

        # Horizontal score for each period
        recommendations["score"] = (
            recommendations["strongBuy"]  * rec_weights["strongBuy"] +
            recommendations["buy"]        * rec_weights["buy"] +
            recommendations["hold"]       * rec_weights["hold"] +
            recommendations["sell"]       * rec_weights["sell"] +
            recommendations["strongSell"] * rec_weights["strongSell"]
        )

        # --- Auto-generate vertical weights ---
        # Extract months as integers (e.g., "0m" → 0, "-3m" → 3)
        recommendations["months_ago"] = recommendations["period"].apply(lambda x: int(re.sub("[^0-9]", "", x)))

        # Exponential decay: weight = decay^months_ago
        decay = 0.75  # tune this: closer to 1 = slower decay, smaller = faster decay
        recommendations["period_weight"] = np.power(decay, recommendations["months_ago"])

        # Weighted score per period
        recommendations["weighted_score"] = recommendations["score"] * recommendations["period_weight"]

        # Overall weighted score
        recommendations_score = recommendations["weighted_score"].sum() / recommendations["period_weight"].sum()

        if recommendations_score > 0:
            reasons.append(f"Analyst score: {recommendations_score:.2f} (positive sentiment)")
            buy_score.append(1)
            analyst_score += 1
        else:
            reasons.append(f"Analyst score: {recommendations_score:.2f} (negative sentiment)")
            buy_score.append(-1)
            analyst_score -= 1

        fundamentals["AnalystScore"] = recommendations_score

    else:
        fundamentals["AnalystScore"] = None
        buy_score.append('-')
        reasons.append("No analyst recommendations available")

    # --- Analyst target price ---
    ## Analyst target price indicates expected future price.
    if "targetMeanPrice" in info:
        target_price = info["targetMeanPrice"]
        current_price = info['regularMarketPrice']
        fundamentals["TargetPrice"] = target_price
        if target_price > current_price:
            analyst_score += 1
            reasons.append(f"Analyst target price ({target_price:.2f}) > current price ({current_price:.2f}) (upside potential)")
            buy_score.append(1)
        elif target_price < current_price:
            analyst_score -= 1
            reasons.append(f"Analyst target price ({target_price:.2f}) < current price ({current_price:.2f}) (downside risk)")
            buy_score.append(-1)
    else:
        target_price = None
        fundamentals["TargetPrice"] = None
        buy_score.append('-')
        reasons.append("Analyst target price data not available")

    # Final scoring
    valuation_score = 1 if valuation_score > 0 else 0
    risk_score = 1 if risk_score > 0 else 0
    analyst_score = 1 if analyst_score > 0 else 0

    fundamentals_score = valuation_score + risk_score + analyst_score

    if fundamentals_score >= 2:
        decision = "BUY"
    elif fundamentals_score == 1:
        decision = "HOLD"
    else:
        decision = "NOT BUY"
    
    return decision, fundamentals, reasons, buy_score

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", type=str, required=True, help="Stock ticker symbol (e.g. AAPL)")
    args = parser.parse_args()

    ticker = args.ticker.upper()
    print(f"\n=== Evaluating {ticker} ===\n")

    f_decision, f_fundamentals, f_reasons, f_scores = get_fundamentals(ticker)
    f_data = []
    for i, (key, value) in enumerate(f_fundamentals.items(), start=0):
        f_data.append([key, value, f_reasons[i], f_scores[i]])

    # Create DataFrame
    f_df = pd.DataFrame(f_data, columns=['Metric', 'Value', 'Interpretation', 'Buy Signal'])
    print("Fundamental Decision:", f_decision)
    display(f_df, '\n')

    t_decision, t_technicals, t_reasons, t_scores = get_technicals(ticker)
    t_data = []
    for i, (key, value) in enumerate(t_technicals.items(), start=0):
        t_data.append([key, value, t_reasons[i], t_scores[i]])

    # Create DataFrame
    t_df = pd.DataFrame(t_data, columns=['Metric', 'Value', 'Interpretation', 'Buy Signal'])
    print("Technical Decision:", t_decision)
    display(t_df)
