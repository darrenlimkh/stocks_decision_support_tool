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
    reasons = []
    benchmarks = []
    buy_score = []

    '''Valuation'''
    # --- PEG ratio ---
    ## Used to assess growth relative to P/E.
    peg = info.get("trailingPegRatio")
    if peg is not None:
        fundamentals["PEG"] = f"{peg:.2f}"
        if peg < 1:
            benchmarks.append("PEG < 1")
            reasons.append("Undervalued")
            buy_score.append(1)
            # valuation_score += 1
        else:
            benchmarks.append("PEG ≥ 1")
            reasons.append("Fairly valued or overvalued")
            buy_score.append(-1)
            # valuation_score -= 1
    else:
        fundamentals["PEG"] = None
        benchmarks.append("-")
        reasons.append("PEG data not available")
        buy_score.append('-')
        
    # --- P/E ratio ---
    ## Used to assess valuation relative to earnings.
    industry_pe = sector_benchmarks["P/E"]
    pe = info.get("trailingPE")
    if pe is not None:
        fundamentals["P/E"] = f"{pe:.2f}"
        if pe < industry_pe:
            benchmarks.append(f"P/E < {industry_pe}")
            reasons.append(f"Undervalued")
            buy_score.append(1)
            # valuation_score += 1
        else:
            benchmarks.append(f"P/E ≥ {industry_pe}")
            reasons.append(f"Fairly valued or overvalued")
            buy_score.append(-1)
            # valuation_score -= 1
    else:
        fundamentals["PE"] = None
        benchmarks.append("-")
        reasons.append("P/E data not available")
        buy_score.append('-')

    # --- P/B ratio ---
    ## Used only when assets drive value. Tech companies with high intangibles may not use P/B.
    if sector != 'Technology':
        pb = info.get("priceToBook")
        
        if pb is not None:
            fundamentals["P/B"] = f"{pb:.2f}"
            if pb < 1:
                benchmarks.append("P/B < 1")
                reasons.append("Undervalued")   
                buy_score.append(1)
            else:
                benchmarks.append("P/B ≥ 1")
                reasons.append("Fairly valued or overvalued")
                buy_score.append(-1)
        else:
            pb = None
            fundamentals["P/B"] = None
            benchmarks.append("-")
            reasons.append("P/B data not available")
            buy_score.append('-')
    else:
        pb = None
        fundamentals["P/B"] = None
        benchmarks.append("-")
        reasons.append("P/B data not available")
        buy_score.append('-')

    # --- Free Cash Flow ---
    ## Free cash flow is the cash a company generates after accounting for capital expenditures.
    fcf = info.get("freeCashflow")
    market_cap = info.get("marketCap") 

    if fcf is not None and market_cap is not None and market_cap != 0:
        fcf_yield = fcf / market_cap  # decimal form
        fcf_yield_pct = fcf_yield * 100
        fundamentals["Free Cash Flow (FCF) Yield"] = f"{fcf_yield:.2f}"
        if fcf_yield_pct > 5:
            benchmarks.append("FCF Yield > 5%")
            reasons.append("Strong cash-generating ability")
            # reasons.append(f"FCF Yield ({fcf_yield_pct:.2f}%) > 5 (Cash-generating)")
            buy_score.append(1)
            # valuation_score += 1
        elif fcf_yield_pct > 0:
            benchmarks.append("0% < FCF Yield ≤ 5%")
            reasons.append("Neutral cash flow")
            buy_score.append(0)
            # valuation_score += 0
        else:
            benchmarks.append("FCF Yield ≤ 0%")
            reasons.append("Negative cash flow")
            buy_score.append(-1)
            # valuation_score -= 1
    else:
        fcf_yield = None
        fundamentals["Free Cash Flow (FCF) Yield"] = None
        benchmarks.append("-")
        reasons.append("Free Cash Flow data not available")
        buy_score.append('-')


    '''Risk & Financial Health'''
    # --- Debt/Equity ---
    ## Used to assess financial risk.
    industry_debt_equity = sector_benchmarks["DebtEquity"]
    de = info.get("debtToEquity")

    if de is not None:
        de = de / 100
        fundamentals["D/E"] = f"{de:.2f}"    
        if de < 1:
            benchmarks.append("D/E < 1")
            reasons.append("Low financial risk")
            buy_score.append(1)
            # risk_score += 1
        else:
            benchmarks.append("D/E ≥ 1")
            reasons.append("High financial risk")
            buy_score.append(-1)
            # risk_score -= 1
    else:
        de = None
        fundamentals["D/E"] = None
        benchmarks.append("-")
        reasons.append("Debt/Equity data not available")
        buy_score.append('-')
        

    # --- Operating Margin ---
    ## Operating margin measures the percentage of revenue left after covering operating expenses.
    industry_op_margin = sector_benchmarks["OperatingMargin"]
    op_margin = info.get("operatingMargins")
    if op_margin is not None:
        fundamentals["Operating Margin"] = f"{op_margin:.2f}" 
        if op_margin > industry_op_margin:
            benchmarks.append(f"Operating Margin > {industry_op_margin}")
            reasons.append("Efficient operations")
            buy_score.append(1)
            # risk_score += 1
        else:
            benchmarks.append(f"Operating Margin ≤ {industry_op_margin}")
            reasons.append("Inefficient operations")
            buy_score.append(-1)
            # risk_score -= 1
    else:
        op_margin = None
        fundamentals["Operating Margin"] = None
        benchmarks.append("-")
        reasons.append("Operating Margin data not available")
        buy_score.append('-')
        

    # --- Revenue Growth ---
    ## Revenue growth indicates the company's ability to increase sales over time.
    growth = info.get("revenueGrowth")
    industry_revenue_growth = sector_benchmarks["RevenueGrowth"]
    if growth is not None:
        fundamentals["Revenue Growth"] = f"{growth:.2f}" 
        if growth > industry_revenue_growth:
            benchmarks.append(f"Revenue Growth > {industry_revenue_growth}")
            reasons.append("Growing revenue")
            buy_score.append(1)
            # risk_score += 1
        else:
            benchmarks.append(f"Revenue Growth ≤ {industry_revenue_growth}")
            reasons.append("Declining revenue")
            buy_score.append(-1)
            # risk_score -= 1
    else:
        growth = None
        fundamentals["Revenue Growth"] = None
        benchmarks.append("-")
        reasons.append("Revenue Growth data not available")
        buy_score.append('-')


    # --- Dividend sustainability ---
    ## Dividend payout ratio indicates how much of earnings are paid out as dividends.
    payout = info.get("payoutRatio")
    industry_payout_ratio = sector_benchmarks["PayoutRatio"]
    if payout is not None:
        fundamentals["Payout Ratio"] = f"{payout:.2f}" 
        if payout < industry_payout_ratio:
            benchmarks.append(f"Payout Ratio < {industry_payout_ratio}")
            reasons.append("Sustainable dividend")
            buy_score.append(1)
            # risk_score += 1
        else:
            benchmarks.append(f"Payout Ratio ≥ {industry_payout_ratio}")
            reasons.append("Unsustainable dividend")
            buy_score.append(-1)
            # risk_score -= 1
    else:
        payout = None
        fundamentals["Payout Ratio"] = None
        benchmarks.append("-")
        reasons.append("Payout Ratio data not available")
        buy_score.append('-')

    '''Analyst Recommendations'''
    # analyst_score = 0
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
            benchmarks.append("Analyst score > 0")
            reasons.append("Positive analyst sentiment")
            buy_score.append(1)
            # analyst_score += 1
        else:
            benchmarks.append("Analyst score ≤ 0")
            reasons.append("Negative analyst sentiment")
            buy_score.append(-1)
            # analyst_score -= 1

        fundamentals["Analyst Score"] = f"{recommendations_score:.2f}"

    else:
        fundamentals["Analyst Score"] = None
        benchmarks.append("-")
        reasons.append("No analyst recommendations available")
        buy_score.append('-')

    # --- Analyst target price ---
    ## Analyst target price indicates expected future price.
    if "targetMeanPrice" in info:
        target_price = info["targetMeanPrice"]
        current_price = info['regularMarketPrice']
        fundamentals["Current Price / Analyst Price"] = f"{current_price:.2f} / {target_price:.2f}" 
        if target_price > current_price:
            benchmarks.append("Analyst Price > Current Price")
            reasons.append("Upside potential")
            # analyst_score += 1
            buy_score.append(1)
        elif target_price < current_price:
            benchmarks.append("Analyst Price ≤ Current Price")
            reasons.append("Downside risk")
            # analyst_score -= 1
            buy_score.append(-1)
    else:
        target_price = None
        fundamentals["Current Price / Analyst Price"] = None
        benchmarks.append("-")
        reasons.append("Analyst target price data not available")
        buy_score.append('-')

    return fundamentals, benchmarks, reasons, buy_score