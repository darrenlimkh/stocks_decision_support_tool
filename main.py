import argparse
import warnings
import pandas as pd

from IPython.display import display
from metrics.fundamentals import get_fundamentals
from metrics.technicals import get_technicals

warnings.simplefilter(action='ignore', category=FutureWarning)
pd.set_option('display.max_rows', None)     # Show all rows
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.width', None)        # Auto-adjust width
pd.set_option('display.max_colwidth', None) # Show full column content

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", type=str, required=True, help="Stock ticker symbol (e.g. AAPL)")
    args = parser.parse_args()

    ticker = args.ticker.upper()
    print(f"\n=== Evaluating {ticker} ===\n")

    # Fundamentals
    f_decision, f_fundamentals, f_reasons, f_scores = get_fundamentals(ticker)
    f_data = []
    for i, (key, value) in enumerate(f_fundamentals.items(), start=0):
        f_data.append([key, value, f_reasons[i], f_scores[i]])

    f_df = pd.DataFrame(f_data, columns=['Metric', 'Value', 'Interpretation', 'Buy Signal'])
    print("Fundamental Decision:", f_decision)
    display(f_df, '\n')

    # Technicals
    t_decision, t_technicals, t_reasons, t_scores = get_technicals(ticker)
    t_data = []
    for i, (key, value) in enumerate(t_technicals.items(), start=0):
        t_data.append([key, value, t_reasons[i], t_scores[i]])

    # Create DataFrame
    t_df = pd.DataFrame(t_data, columns=['Metric', 'Value', 'Interpretation', 'Buy Signal'])
    print("Technical Decision:", t_decision)
    display(t_df)
