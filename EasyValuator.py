

import sys
import yfinance as yf
import pandas as pd
import numpy as np
from functools import lru_cache
from typing import Optional

# Target currency for output
TARGET_CURRENCY = "EUR"

# --------------------------------
# Foreign Exchange (FX) Utilities
# --------------------------------

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

def convert_to_target_currency(amount: float, from_currency: str) -> Optional[float]:
    """
    Converts a monetary amount from source currency to the predefined target currency.

    This function serves as the main interface for currency conversion operations,
    handling edge cases and delegating to the exchange rate service.

    Args:
        amount (float): The monetary amount to convert
        from_currency (str): The source currency code

    Returns:
        Optional[float]: The converted amount in target currency, or original amount
                        if conversion is not possible/necessary.

    Examples:
        convert_to_target_currency(100, 'USD')  # Assuming TARGET_CURRENCY = 'EUR'
        86.74  # Example conversion rate - actual rate varies by market conditions
        convert_to_target_currency(100, 'EUR')  # Same currency
        100.0

    Notes:
        - Returns None if input amount is None
        - Returns original amount if source currency is None
        - Bypasses conversion for same-currency scenarios
        - Falls back to original amount if exchange rate is unavailable
    """
    # Handle null input values
    if amount is None:
        return None

    # Return amount as-is if no source currency specified
    if from_currency is None:
        return amount

    # Skip conversion for same currency to avoid unnecessary API calls
    if from_currency.upper() == TARGET_CURRENCY.upper():
        return amount

    # Retrieve exchange rate and apply conversion
    rate = get_exchange_rate(from_currency, TARGET_CURRENCY)
    if rate:
        return amount * rate

    # Graceful degradation: return original amount if conversion fails
    return amount