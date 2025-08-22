import ta
# import joblib
import yfinance as yf
import xgboost as xgb
import pandas as pd

from datetime import datetime, timedelta
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report

def build_training_data(ticker: str, lookahead_days: int = 5):
    end = datetime.today()
    start = end - timedelta(days=365*3)  # 3 years
    df = yf.download(
        ticker,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        progress=False,
        auto_adjust=True
    )

    if df.empty:
        return pd.DataFrame()

    close = df["Close"][ticker]
    volume = df["Volume"][ticker]

    features = pd.DataFrame(index=df.index)

    # Indicators
    features["RSI"] = ta.momentum.RSIIndicator(close).rsi()

    macd = ta.trend.MACD(close)
    features["MACD_diff"] = macd.macd_diff()

    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()
    features["SMA50_gt_SMA200"] = (sma50 > sma200).astype(int)

    high20 = close.rolling(20).max()
    features["BreakoutUp"] = (close > high20).astype(int)

    low20 = close.rolling(20).min()
    features["BreakoutDown"] = (close < low20).astype(int)

    avg_vol = volume.rolling(20).mean()
    std_vol = volume.rolling(20).std()
    features["HighVolume"] = (volume > avg_vol + 2 * std_vol).astype(int)

    # Target
    future_price = close.shift(-lookahead_days)
    features["Target"] = (future_price > close).astype(int)

    return features.dropna()


def train_model_cv(ticker, lookahead_days=5, n_splits=5, model_type="logistic", verbose=False):
    data = build_training_data(ticker, lookahead_days)
    if data.empty:
        raise ValueError("No training data available.")

    X = data.drop(columns=["Target"])
    y = data["Target"]

    if model_type == "logistic":
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=500))
        ])
    elif model_type == "xgboost":
        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
        )
    else:
        raise ValueError("model_type must be 'logistic' or 'xgboost'")
    
    # TimeSeries CV
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []

    if verbose:
        print(f"\n {model_type}: TimeSeries Cross-Validation ({n_splits} folds)")
        print("=" * 50)

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        scores.append(acc)

        if verbose:
            print(f"Fold {fold}: Accuracy = {acc:.4f}")
            print(classification_report(y_test, preds, zero_division=0))

    avg_acc = sum(scores) / len(scores)
    if verbose:
        print("=" * 50)
        print(f"Average Accuracy across folds: {avg_acc:.4f}")

    model.fit(X, y)
    # joblib.dump((model, list(X.columns)), f"./machine_learning/models/{model_type}_{ticker}_model.pkl")

    return model, X.columns

def get_prediction(ticker: str, model_type, model_path=None):
    model, feature_names = train_model_cv(ticker, lookahead_days=10, model_type="logistic")
    # Load trained model
    # if model_path and model_type:
    #     try:
    #         model_path = f"{model_type}_{ticker}_model.pkl"
    #         model, feature_names = joblib.load(model_path)
    #     except FileNotFoundError:
    #         raise FileNotFoundError(f"Model file {model_path} not found. Please train the model first.")

    # else:
    #     model, feature_names = train_model_cv(ticker, lookahead_days=10, model_type="logistic")

    # Fetch latest data
    end = datetime.today()
    start = end - timedelta(days=365)
    df = yf.download(ticker, start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"), progress=False, auto_adjust=True)

    if df.empty:
        return "NOT NOW", {}

    close = df["Close"][ticker]
    volume = df["Volume"][ticker]

    # Build feature row (same as training features)
    features = {}

    features["RSI"] = ta.momentum.RSIIndicator(close).rsi().iloc[-1]

    macd = ta.trend.MACD(close)
    features["MACD_diff"] = macd.macd_diff().iloc[-1]

    sma50 = close.rolling(50).mean().iloc[-1]
    sma200 = close.rolling(200).mean().iloc[-1]
    features["SMA50_gt_SMA200"] = int(sma50 > sma200)

    last_price = close.iloc[-1]
    high20 = close.rolling(20).max().iloc[-1]
    features["BreakoutUp"] = int(last_price > high20)

    low20 = close.rolling(20).min().iloc[-1]
    features["BreakoutDown"] = int(last_price < low20)

    last_vol = volume.iloc[-1]
    avg_vol = volume.rolling(20).mean().iloc[-1]
    std_vol = volume.rolling(20).std().iloc[-1]
    features["HighVolume"] = int(last_vol > avg_vol + 2 * std_vol)

    # Convert to DataFrame in correct column order
    X_live = pd.DataFrame([features])[feature_names]

    # Predict
    prob = model.predict_proba(X_live)[0, 1]  # probability of "up"
    decision = "BUY" if prob > 0.5 else "DON'T BUY"

    return decision, features, prob