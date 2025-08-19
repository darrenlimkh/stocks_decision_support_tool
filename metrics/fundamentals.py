import yfinance as yf
import numpy as np
import re

from config.config import get_sector_benchmarks

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