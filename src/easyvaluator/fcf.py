"""Free cash flow growth estimation and forward projection."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .assumptions import DEFAULT_ASSUMPTIONS, Assumptions
from .exceptions import ValuationError

DAYS_PER_YEAR = 365.25


@dataclass(frozen=True)
class GrowthEstimate:
    cagr: float
    raw_cagr: float | None  # before clamping; None when it couldn't be computed
    note: str | None = None


def _years_between(historical_fcf: pd.Series) -> float:
    """Elapsed years between the first and last observation.

    Uses the actual dates when the index has them, so a missing year in the
    middle of the series doesn't inflate the growth rate.
    """
    index = historical_fcf.index
    if isinstance(index, pd.DatetimeIndex):
        return (index[-1] - index[0]).days / DAYS_PER_YEAR
    return float(len(historical_fcf) - 1)


def estimate_growth(historical_fcf: pd.Series, assumptions: Assumptions = DEFAULT_ASSUMPTIONS) -> GrowthEstimate:
    """Historical FCF CAGR, clamped to [cagr_min, cagr_max].

    A CAGR is undefined when either endpoint is non-positive; in that case
    the forecast conservatively starts at the terminal growth rate.
    """
    if len(historical_fcf) < 2:
        raise ValuationError("At least 2 years of historical FCF are required to estimate growth")

    first, last = float(historical_fcf.iloc[0]), float(historical_fcf.iloc[-1])
    years = _years_between(historical_fcf)

    if first <= 0 or last <= 0 or years <= 0:
        return GrowthEstimate(
            cagr=assumptions.terminal_growth,
            raw_cagr=None,
            note="CAGR undefined (non-positive FCF at an endpoint); using terminal growth instead",
        )

    raw = (last / first) ** (1 / years) - 1
    clamped = min(max(raw, assumptions.cagr_min), assumptions.cagr_max)
    note = None
    if clamped != raw:
        note = f"Historical CAGR {raw:.1%} clamped to {clamped:.1%}"
    return GrowthEstimate(cagr=clamped, raw_cagr=raw, note=note)


def growth_path(cagr: float, terminal_growth: float, years: int) -> list[float]:
    """Annual growth rates decaying linearly from `cagr` to `terminal_growth`.

    The final year grows at exactly `terminal_growth`, so the explicit
    forecast hands over smoothly to the Gordon Growth perpetuity.
    """
    rates = []
    for year in range(1, years + 1):
        weight = (years - year) / years
        rates.append(cagr * weight + terminal_growth * (1 - weight))
    return rates


def forecast_fcf(base_fcf: float, growth_rates: list[float]) -> list[float]:
    """Compound `base_fcf` forward by each year's growth rate."""
    forecasts = []
    current = base_fcf
    for rate in growth_rates:
        current *= 1 + rate
        forecasts.append(current)
    return forecasts
