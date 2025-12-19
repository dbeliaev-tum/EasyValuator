

import sys
import yfinance as yf
import pandas as pd
import numpy as np
from functools import lru_cache
from typing import Optional

# Target currency for output
TARGET_CURRENCY = "EUR"

# ---------------------------
# Foreign Exchange (FX) Utilities
# ---------------------------

@lru_cache(maxsize=None)
def get_exchange_rate(currency_from: str, currency_to: str) -> Optional[float]:
    """
    Retrieves the current exchange rate between two currencies using Yahoo Finance API.

    This function implements a caching mechanism to avoid redundant API calls for
    frequently requested currency pairs, improving performance and reducing rate limits.

    Args:
        currency_from (str): The source currency code (e.g., 'USD', 'EUR')
        currency_to (str): The target currency code (e.g., 'JPY', 'GBP')

    Returns:
        Optional[float]: The exchange rate as a float if successful,
                        None if the rate cannot be retrieved or in case of errors.

    Notes:
        - Returns 1.0 automatically for identical currency pairs
        - Handles both real-time rates and historical close prices as fallback
        - Implements graceful error handling for network issues or invalid currencies
    """
    # Validate input parameters
    if not currency_from or not currency_to:
        return None

    # Short-circuit for same currency conversion
    if currency_from.upper() == currency_to.upper():
        return 1.0

    # Construct Yahoo Finance ticker symbol for the currency pair
    ticker = f"{currency_from.upper()}{currency_to.upper()}=X"

    try:
        # Attempt to fetch real-time market data
        data = yf.Ticker(ticker)

        # Try to get current regular market price first
        rate = data.info.get('regularMarketPrice')
        if rate:
            return float(rate)

        # Fallback to most recent daily closing price if real-time unavailable
        hist = data.history(period="1d")
        if not hist.empty:
            return float(hist['Close'].iloc[-1])

    except Exception:
        # Logging would be recommended here in production code
        # We silently fail to allow calling code to handle absence of rate
        pass

    return None