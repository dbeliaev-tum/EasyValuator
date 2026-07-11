"""Configurable financial assumptions used throughout the valuation pipeline.

Centralizing these here means every module (forecasting, WACC, terminal
value) reads the same numbers instead of hard-coding its own copy.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Assumptions:
    target_currency: str = "EUR"
    forecast_years: int = 5

    # Long-term perpetual growth rate. Used both as the endpoint of the FCF
    # growth-decay model and as "g" in the Gordon Growth terminal value, so
    # the explicit forecast and the terminal value are always consistent.
    terminal_growth: float = 0.025

    tax_rate: float = 0.21
    fallback_cost_of_debt: float = 0.05

    cagr_min: float = -0.05
    cagr_max: float = 0.15

    wacc_min: float = 0.06
    wacc_max: float = 0.20

    beta_min: float = 0.5
    beta_max: float = 2.0


DEFAULT_ASSUMPTIONS = Assumptions()
