class Config:
    def __init__(self):
        """
        Initializes the configuration with default settings.
        """
        self.setting = 'value'

    def get_sector_benchmarks(self, sector: str):
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