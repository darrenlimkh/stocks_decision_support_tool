# Stocks Decision Support Tool

This project is a Python-based tool that evaluates whether a stock is a **BUY**, **HOLD**, or **NOT BUY** based on both **fundamental** and **technical** analysis.

It uses:
- [yfinance](https://pypi.org/project/yfinance/) to fetch financial and price data
- [pandas](https://pandas.pydata.org/) for data handling
- [ta](https://technical-analysis-library-in-python.readthedocs.io/en/latest/) for technical indicators
- Basic scoring logic to generate investment decisions

---

## Features

### 🔎 Fundamental Analysis
- PEG ratio
- P/E ratio
- P/B ratio (if applicable)
- Free cash flow yield
- Debt-to-equity ratio
- Operating margin
- Revenue growth
- Dividend payout ratio
- Analyst recommendations
- Analyst target price

### 📈 Technical Analysis
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- 50-day vs 200-day SMA trend
- Breakouts (20-day highs/lows)
- Volume confirmation

### ✅ Decision Logic
- Scores fundamentals and technicals separately
- Combines results into a final **BUY**, **HOLD**, or **NOT BUY** recommendation
- Displays metrics, values, and reasoning in tables

---

## Installation

Clone the repository:
```bash
git clone git@github.com:darrenlimkh/stocks_decision_support_tool.git
cd stocks_decision_support_tool
```
## Usage
```bash
python app.py --ticker AAPL
```