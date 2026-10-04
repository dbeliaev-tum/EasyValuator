"""EasyValuator: a Discounted Cash Flow (DCF) stock valuation toolkit.

Multi-currency, region-aware fundamental valuation built on top of
Yahoo Finance data (via yfinance).

    >>> from easyvaluator import price_stock, print_report
    >>> print_report(price_stock("AAPL"))  # doctest: +SKIP
"""

__version__ = "2.0.0"

from .assumptions import DEFAULT_ASSUMPTIONS, Assumptions
from .dcf import discount_cash_flows, sensitivity_table
from .exceptions import (
    DataUnavailableError,
    EasyValuatorError,
    FXRateUnavailableError,
    InvalidAssumptionError,
    ValuationError,
)
from .market_data import MarketDataProvider, YahooFinanceProvider
from .models import CompanySnapshot, DCFBreakdown, SensitivityTable, ValuationResult, WACCBreakdown
from .report import print_report, render_json, render_text
from .valuation import Valuator, price_stock
from .wacc import calculate_wacc

__all__ = [
    "DEFAULT_ASSUMPTIONS",
    "Assumptions",
    "CompanySnapshot",
    "DCFBreakdown",
    "DataUnavailableError",
    "EasyValuatorError",
    "FXRateUnavailableError",
    "InvalidAssumptionError",
    "MarketDataProvider",
    "SensitivityTable",
    "ValuationError",
    "ValuationResult",
    "Valuator",
    "WACCBreakdown",
    "YahooFinanceProvider",
    "__version__",
    "calculate_wacc",
    "discount_cash_flows",
    "price_stock",
    "print_report",
    "render_json",
    "render_text",
    "sensitivity_table",
]
