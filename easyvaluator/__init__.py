"""EasyValuator: a Discounted Cash Flow (DCF) stock valuation toolkit.

Multi-currency, region-aware fundamental valuation built on top of
Yahoo Finance data (via yfinance).
"""

from .assumptions import DEFAULT_ASSUMPTIONS, Assumptions
from .report import print_report
from .valuation import ValuationResult, calculate_dcf, calculate_wacc, price_stock

__version__ = "1.0.0"

__all__ = [
    "Assumptions",
    "DEFAULT_ASSUMPTIONS",
    "ValuationResult",
    "calculate_dcf",
    "calculate_wacc",
    "price_stock",
    "print_report",
]
