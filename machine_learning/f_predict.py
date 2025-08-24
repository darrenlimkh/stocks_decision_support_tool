import joblib
import yfinance as yf
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report

def safe_div(df, num, den):
    if num in df.columns and den in df.columns:
        return df[num] / df[den]
    else:
        return pd.Series([np.nan] * len(df), index=df.index)
    
def get_fundamentals(ticker, get_train, lookahead_days=90):
    t = yf.Ticker(ticker)
    inc = t.quarterly_financials.T
    bal = t.quarterly_balance_sheet.T
    cf  = t.quarterly_cashflow.T

    df_raw = pd.concat([inc, bal, cf], axis=1)

    # Profitability Ratios
    df_extract = pd.DataFrame()

    # Profitability Ratios
    df_extract['Gross_Margin']      = safe_div(df_raw, 'Gross Profit', 'Total Revenue')
    df_extract['Operating_Margin']  = safe_div(df_raw, 'Operating Income', 'Total Revenue')
    df_extract['Net_Margin']        = safe_div(df_raw, 'Net Income', 'Total Revenue')
    df_extract['EBITDA_Margin']     = safe_div(df_raw, 'EBITDA', 'Total Revenue')

    # Growth Ratios (YoY)
    df_extract['Revenue_Growth']    = df_raw['Total Revenue'].pct_change(fill_method=None) if 'Total Revenue' in df_raw else np.nan
    df_extract['Net_Income_Growth'] = df_raw['Net Income'].pct_change(fill_method=None) if 'Net Income' in df_raw else np.nan
    df_extract['EPS_Growth']        = df_raw['Diluted EPS'].pct_change(fill_method=None) if 'Diluted EPS' in df_raw else np.nan

    # Return Ratios
    df_extract['ROE']   = safe_div(df_raw, 'Net Income', 'Stockholders Equity')
    df_extract['ROA']   = safe_div(df_raw, 'Net Income', 'Total Assets')
    df_extract['ROIC']  = safe_div(df_raw, 'EBIT', 'Total Debt') + safe_div(df_raw, 'EBIT', 'Stockholders Equity') - safe_div(df_raw, 'EBIT', 'Cash And Cash Equivalents')

    # Leverage Ratios
    df_extract['Debt_to_Equity']      = safe_div(df_raw, 'Total Debt', 'Stockholders Equity')
    df_extract['Net_Debt_to_EBITDA']  = safe_div(df_raw, 'Net Debt', 'EBITDA')

    # Liquidity Ratios
    df_extract['Current_Ratio'] = safe_div(df_raw, 'Current Assets', 'Current Liabilities')
    df_extract['Quick_Ratio']   = safe_div(df_raw, 'Current Assets', 'Current Liabilities') - safe_div(df_raw, 'Inventory', 'Current Liabilities')

    # Cash Flow Metrics
    df_extract['FCF_to_Sales']  = safe_div(df_raw, 'Free Cash Flow', 'Total Revenue')
    df_extract['FCF_yield']     = safe_div(df_raw, 'Free Cash Flow', 'Stockholders Equity')

    df_extract['Report_Date'] = df_extract.index

    for col in df_extract.columns:
        if df_extract[col].isna().all():
            df_extract[col] = df_extract[col].fillna(0)
        else:
            df_extract[col] = df_extract[col].fillna(df_extract[col].mean())

    start_date = df_extract['Report_Date'].min()
    end_date = df_extract['Report_Date'].max() + pd.Timedelta(days=lookahead_days+30)
    prices = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)["Close"][ticker]

    if get_train:
        df_extract['Price_Ahead'] = df_extract['Report_Date'].apply(
            lambda x: prices.get(x + pd.Timedelta(days=lookahead_days))
        )
        df_extract['Price_Today'] = df_extract['Report_Date'].apply(
            lambda x: prices.get(x)
        )
        df_extract = df_extract.dropna()
        df_extract['Price_Increase'] = (df_extract['Price_Ahead'] > df_extract['Price_Today']).astype(int)
        df_extract = df_extract.drop(columns=['Report_Date', 'Price_Today', 'Price_Ahead'])
    
    else:
        df_extract = df_extract.drop(columns=['Report_Date'])
        df_extract = df_extract.dropna()
    
    return df_extract

def train_fundamentals_model():

    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url, header=0)
    tickers = tables[0]['Symbol'].tolist()

    df_train = pd.DataFrame()
    for ticker in tickers:
        try:
            df_fund = get_fundamentals(ticker, get_train=True, lookahead_days=90)
            if df_fund:
                df_train = pd.concat([df_train, df_fund], ignore_index=True)
        except Exception as e:
            pass
            # print(f"Error processing {ticker}: {e}")

    X = df_train.drop(columns=['Price_Increase'])

    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(X.median(), inplace=True)

    y = df_train['Price_Increase']

    # Define pipeline: scaler + classifier
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'))
    ])

    # K-Fold Cross Validation
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accuracies = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X, y), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Train pipeline
        pipeline.fit(X_train, y_train)
        
        # Predict
        y_pred = pipeline.predict(X_test)
        
        # Evaluate
        acc = accuracy_score(y_test, y_pred)
        accuracies.append(acc)
        
        print(f"Fold {fold} Accuracy: {acc:.4f}")
        print(classification_report(y_test, y_pred, zero_division=0))

    print(f"\nAverage Accuracy across folds: {sum(accuracies)/len(accuracies):.4f}")
    
    pipeline.fit(X, y)
    joblib.dump((pipeline, list(X.columns)), f"./machine_learning/models/fundamentals_model.pkl")

def get_fundamentals_prediction(ticker):
    try:
        model, feature_cols = joblib.load(f"./machine_learning/models/fundamentals_model.pkl")
    except:
        train_fundamentals_model()
        model, feature_cols = joblib.load(f"./machine_learning/models/fundamentals_model.pkl")

    X_test = get_fundamentals(ticker, get_train=False).iloc[[-1]]
    prob = model.predict_proba(X_test)[0, 1]
    decision = "BUY" if prob > 0.5 else "DON'T BUY"

    return decision, prob