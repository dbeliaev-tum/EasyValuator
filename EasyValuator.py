

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

# -------------------------------
# Country and Regional Detection
# -------------------------------

def get_country_from_ticker(t: yf.Ticker) -> str:
    """
    Determines the country or region of a company based on ticker metadata.

    This function analyzes both exchange information and explicit country data
    from Yahoo Finance to classify companies into major economic regions.

    Args:
        t (yf.Ticker): Yahoo Finance ticker object with company metadata

    Returns:
        str: Two-letter region code ('US', 'EU', 'UK', 'CN', 'JP')

    Notes:
        - Prioritizes exchange detection for more reliable classification
        - Falls back to country name matching when exchange data is ambiguous
        - Defaults to 'US' for unclassified cases (most common scenario)
        - Handles missing info gracefully with empty dict fallback
    """
    # Safely extract ticker info with exception handling
    try:
        info = t.info
    except Exception:
        info = {}

    # Normalize string values for case-insensitive comparison
    exchange = (info.get('exchange') or "").upper()
    country_code = (info.get('country') or "").upper()

    # European exchange detection (Frankfurt, Paris, Xetra, etc.)
    if any(x in exchange for x in ['FRA', 'PAR', 'GER', 'XETRA']):
        return 'EU'

    # London Stock Exchange detection
    elif 'LON' in exchange or 'LSE' in exchange:
        return 'UK'

    # Country-based classification for European nations
    elif country_code in ['GERMANY', 'FRANCE', 'ITALY', 'SPAIN', 'NETHERLANDS']:
        return 'EU'

    # United Kingdom classification
    elif country_code in ['UNITED KINGDOM', 'UK']:
        return 'UK'

    # Greater China region classification
    elif country_code in ['CHINA', 'HONG KONG']:
        return 'CN'

    # Japan classification
    elif country_code in ['JAPAN']:
        return 'JP'

    # Default to United States (most common case)
    else:
        return 'US'

# ---------------------------