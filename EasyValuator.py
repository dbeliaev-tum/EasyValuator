

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
# Risk-Free Rate Calculation
# ---------------------------

def get_risk_free_rate(country: str = 'US') -> float:
    """
    Calculates the appropriate risk-free rate for a given country/region.

    Uses US Treasury yields as a baseline and applies regional adjustments
    based on historical interest rate differentials. Implements robust fallback
    mechanisms for reliability.

    Args:
        country (str): Two-letter region code ('US', 'EU', 'UK', 'CN', 'JP')

    Returns:
        float: Annual risk-free rate as decimal (e.g., 0.041 for 4.1%)

    Strategy:
        1. Attempt to fetch current US Treasury yield from Yahoo Finance
        2. Apply regional adjustment based on historical spreads
        3. Validate result within reasonable bounds
        4. Fall back to conservative historical averages if fetch fails

    Sources:
        - US Treasury: ^TNX (10-Year Treasury Note Yield)
        - Adjustments based on OECD and central bank data
    """
    # Yahoo Finance tickers for government bond yields
    # Note: Using US Treasury as proxy with regional adjustments
    bond_tickers = {
        'US': '^TNX',  # US 10-Year Treasury Note
        'EU': '^TNX',
        'UK': '^TNX',
        'CN': '^TNX',
        'JP': '^TNX'
    }

    # Regional adjustments to US Treasury rates (based on historical spreads)
    # These reflect typical interest rate differentials between regions
    regional_adjustments = {
        'US': 0.0,  # Baseline (no adjustment)
        'EU': -0.015,  # EU rates historically ~1.5% lower than US
        'UK': -0.005,  # UK rates slightly lower than US
        'CN': -0.005,  # China similar to UK in recent years
        'JP': -0.035  # Japan significantly lower due to monetary policy
    }

    # Conservative fallback rates based on recent historical averages
    # Used when real-time data is unavailable
    default_rates = {
        'US': 0.041,  # ~4.1% - recent US 10-year average
        'EU': 0.026,  # ~2.6% - European Central Bank targets
        'UK': 0.036,  # ~3.6% - Bank of England benchmarks
        'CN': 0.028,  # ~2.8% - People's Bank of China rates
        'JP': 0.006  # ~0.6% - Bank of Japan ultra-low rates
    }

    # Get appropriate ticker symbol for the region
    ticker_symbol = bond_tickers.get(country, '^TNX')

    try:
        # Fetch recent bond yield data
        treasury = yf.Ticker(ticker_symbol)
        hist = treasury.history(period="5d")

        if not hist.empty:
            # Extract most recent closing yield
            rate = float(hist['Close'].iloc[-1])

            # Convert percentage format to decimal (e.g., 4.14% → 0.0414)
            if rate > 1.0:
                rate = rate / 100.0

            # Apply region-specific adjustment to US baseline
            rate += regional_adjustments.get(country, 0.0)

            # Validate rate is within economically plausible bounds
            if 0.001 < rate < 0.10:
                return rate

    except Exception:
        # Logging recommended here in production environment
        pass

    # Fallback to conservative historical averages if real-time fetch fails
    return default_rates.get(country, 0.04)

# -------------------------------
# Market Risk Premium Calculation
# -------------------------------

def get_market_risk_premium(country: str = 'US') -> float:
    """
    Provides historically-validated market risk premiums by region.

    The equity risk premium represents the excess return investors expect
    from stocks over the risk-free rate. Based on long-term academic research
    and historical market data.

    Args:
        country (str): Two-letter region code ('US', 'EU', 'UK', 'CN', 'JP')

    Returns:
        float: Market risk premium as decimal (e.g., 0.055 for 5.5%)

    Sources:
        - Damodaran datasets on historical equity risk premiums
        - Academic studies on long-term market returns
        - IMF and World Bank economic research

    Rationale:
        - Developed markets (US, EU, UK, JP): 5.5% long-term average
        - Emerging markets (CN): Higher premium for increased risk
        - Conservative estimates to avoid over-optimistic valuations
    """
    # Historically-validated equity risk premiums by economic region
    # These reflect long-term excess returns of equities over government bonds
    mrp_by_region = {
        'US': 0.055,  # 5.5% - US historical equity risk premium (1928-present)
        'EU': 0.055,  # 5.5% - Similar to US for developed European markets
        'UK': 0.055,  # 5.5% - UK long-term premium aligns with US/EU
        'CN': 0.070,  # 7.0% - Higher premium for emerging market volatility
        'JP': 0.055  # 5.5% - Developed market with established risk premium
    }

    return mrp_by_region.get(country, 0.055)

# ------------------------------------
# Historical Free Cash Flow Extraction
# ------------------------------------

def get_historical_fcf(t: yf.Ticker) -> pd.Series:
    """
    Extracts historical Free Cash Flow (FCF) from company cash flow statements.

    Free Cash Flow is a critical valuation metric calculated as:
    FCF = Operating Cash Flow - Capital Expenditures

    This function handles variations in financial statement labeling across different
    companies and reporting standards to ensure robust FCF calculation.

    Args:
        t (yf.Ticker): Yahoo Finance ticker object with financial data

    Returns:
        pd.Series: Historical Free Cash Flow values with datetime indices,
                  sorted chronologically for time series analysis

    Raises:
        ValueError: When no cash flow data is available or required fields are missing

    Notes:
        - Supports multiple common field names for operating cash flow and capex
        - Handles data type conversion and missing value cleaning
        - Returns data in chronological order for time series modeling
    """
    # Retrieve cash flow statement data
    cashflow = t.cashflow
    if cashflow is None or cashflow.empty:
        raise ValueError("No cashflow data available for financial analysis")

    # Transpose for time series analysis (dates as index)
    cf = cashflow.T

    # Define common field names for operating cash flow across reporting standards
    # Different companies may use varying terminology in their financial statements
    op_cash_fields = [
        'Total Cash From Operating Activities',  # Most common US GAAP
        'Operating Cash Flow',  # Alternative labeling
        'Cash From Operating Activities'  # International standards
    ]

    # Define common field names for capital expenditures
    capex_fields = [
        'Capital Expenditures',  # Standard term
        'Capital Expenditure',  # Singular form
        'Purchase Of Property Plant And Equipment'  # Detailed description
    ]

    # Initialize variables for field detection
    op_cash = None
    capex = None

    # Find operating cash flow field using priority order
    for field in op_cash_fields:
        if field in cf.columns:
            op_cash = cf[field]
            break

    # Find capital expenditures field using priority order
    for field in capex_fields:
        if field in cf.columns:
            capex = cf[field]
            break

    # Validate that both required financial components are found
    if op_cash is None or capex is None:
        raise ValueError(
            "Could not locate required cash flow statement fields. "
            "Available columns: " + ", ".join(cf.columns.tolist())
        )

    # Calculate Free Cash Flow: Operating Cash Flow minus Capital Expenditures
    # This represents cash available to investors after reinvestment needs
    fcf = op_cash - capex

    # Ensure numerical data types and remove any invalid entries
    fcf = pd.to_numeric(fcf, errors='coerce').dropna()

    # Return chronologically sorted time series for analysis
    return fcf.sort_index()

# ---------------------------
# Free Cash Flow Forecasting
# ---------------------------

def forecast_fcf(historical_fcf: pd.Series, years: int = 5) -> tuple:
    """
    Forecasts future Free Cash Flows using a decaying growth rate model.

    Implements a realistic growth transition from historical CAGR towards
    a sustainable terminal growth rate, reflecting typical business maturity patterns.

    Args:
        historical_fcf (pd.Series): Historical Free Cash Flow values (chronologically ordered)
        years (int): Forecast horizon in years (default: 5)

    Returns:
        tuple: Contains two elements:
            - list: Forecasted FCF values for each year
            - float: Calculated historical Compound Annual Growth Rate (CAGR)

    Raises:
        ValueError: When insufficient historical data is provided

    Model Description:
        - Uses historical CAGR as starting growth rate
        - Applies linear decay towards terminal growth rate over forecast period
        - Handles edge cases with negative or zero starting FCF values
        - Implements reasonable growth rate boundaries for financial realism
    """
    # Validate sufficient historical data for meaningful analysis
    if len(historical_fcf) < 2:
        raise ValueError("Minimum 2 years of historical FCF data required for growth calculation")

    # Extract key historical reference points
    first_value = historical_fcf.iloc[0]  # Oldest FCF value
    last_value = historical_fcf.iloc[-1]  # Most recent FCF value
    num_years = len(historical_fcf) - 1  # Historical period length

    # Calculate historical growth rate with robust error handling
    if first_value <= 0:
        # Alternative calculation for negative/zero starting FCF
        # Use average periodic growth rates instead of CAGR formula
        growth_rates = historical_fcf.pct_change().dropna()
        cagr = growth_rates.mean() if not growth_rates.empty else 0.05
    else:
        # Standard CAGR calculation: (End/Start)^(1/Periods) - 1
        cagr = (last_value / first_value) ** (1 / num_years) - 1

    # Apply reasonable growth rate constraints for financial modeling
    # Maximum: 15% (avoiding unrealistic perpetual high growth)
    # Minimum: -5% (allowing for decline but not catastrophic collapse)
    cagr = max(min(cagr, 0.15), -0.05)

    # Terminal growth rate: long-term sustainable growth assumption
    # 1.5% approximates nominal GDP growth in developed economies
    terminal_growth = 0.015

    # Initialize forecasting variables
    forecasts = []
    current_fcf = float(last_value)  # Start from most recent actual FCF

    # Generate year-by-year forecasts
    for year in range(1, years + 1):
        # Calculate decaying growth rate: transition from CAGR to terminal growth
        # Linear decay gives smooth transition over forecast period
        decay_factor = (years - year) / years
        growth_rate = cagr * decay_factor + terminal_growth * (1 - decay_factor)

        # Apply growth rate to current FCF and store result
        current_fcf = current_fcf * (1 + growth_rate)
        forecasts.append(current_fcf)

    return forecasts, cagr