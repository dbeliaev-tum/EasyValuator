"""Historical free cash flow extraction and forward projection."""

from typing import List, Optional, Tuple

import pandas as pd
import yfinance as yf

from .assumptions import Assumptions, DEFAULT_ASSUMPTIONS

# yfinance exposes a ready-made "Free Cash Flow" line item on current data;
# prefer it when present since it's the source-of-truth figure.
FCF_FIELDS = ["Free Cash Flow"]

OP_CASH_FIELDS = [
    "Operating Cash Flow",
    "Total Cash From Operating Activities",
    "Cash Flow From Continuing Operating Activities",
    "Cash From Operating Activities",
]

CAPEX_FIELDS = [
    "Capital Expenditure",
    "Capital Expenditures",
    "Purchase Of PPE",
    "Purchase Of Property Plant And Equipment",
]


def _first_available(cf: pd.DataFrame, fields: List[str]) -> Optional[pd.Series]:
    for field in fields:
        if field in cf.columns:
            return cf[field]
    return None


def get_historical_fcf(t: yf.Ticker) -> pd.Series:
    """Extract historical Free Cash Flow, chronologically sorted.

    Prefers yfinance's own `Free Cash Flow` row. Falls back to
    Operating Cash Flow + Capital Expenditure — yfinance reports CapEx as a
    negative cash outflow, so it must be *added*, not subtracted, to arrive
    at FCF.
    """
    cashflow = t.cashflow
    if cashflow is None or cashflow.empty:
        raise ValueError("No cashflow data available for financial analysis")

    cf = cashflow.T

    fcf = _first_available(cf, FCF_FIELDS)
    if fcf is None:
        op_cash = _first_available(cf, OP_CASH_FIELDS)
        capex = _first_available(cf, CAPEX_FIELDS)
        if op_cash is None or capex is None:
            raise ValueError(
                "Could not locate cash flow fields required to compute FCF. "
                "Available columns: " + ", ".join(cf.columns.tolist())
            )
        fcf = op_cash + capex

    fcf = pd.to_numeric(fcf, errors="coerce").dropna()
    return fcf.sort_index()


def forecast_fcf(
    historical_fcf: pd.Series,
    years: int = 5,
    assumptions: Assumptions = DEFAULT_ASSUMPTIONS,
) -> Tuple[List[float], float]:
    """Forecast future FCF using a growth rate that decays linearly from the
    historical CAGR towards `assumptions.terminal_growth`.

    Returns (forecasts, historical_cagr).
    """
    if len(historical_fcf) < 2:
        raise ValueError("Minimum 2 years of historical FCF data required for growth calculation")

    first_value = historical_fcf.iloc[0]
    last_value = historical_fcf.iloc[-1]
    num_years = len(historical_fcf) - 1

    if first_value <= 0:
        growth_rates = historical_fcf.pct_change().dropna()
        cagr = growth_rates.mean() if not growth_rates.empty else 0.05
    else:
        cagr = (last_value / first_value) ** (1 / num_years) - 1

    cagr = max(min(cagr, assumptions.cagr_max), assumptions.cagr_min)

    forecasts: List[float] = []
    current_fcf = float(last_value)

    for year in range(1, years + 1):
        decay_factor = (years - year) / years
        growth_rate = cagr * decay_factor + assumptions.terminal_growth * (1 - decay_factor)
        current_fcf *= (1 + growth_rate)
        forecasts.append(current_fcf)

    return forecasts, cagr
