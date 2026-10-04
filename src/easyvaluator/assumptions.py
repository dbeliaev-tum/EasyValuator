"""Configurable financial assumptions used throughout the valuation pipeline.

Centralizing these here means every module (forecasting, WACC, terminal
value) reads the same numbers instead of hard-coding its own copy.
"""

from __future__ import annotations

from dataclasses import dataclass

from .exceptions import InvalidAssumptionError


@dataclass(frozen=True)
class Assumptions:
    """Every tunable number in the model, validated on construction."""

    target_currency: str = "EUR"

    # A ten-year horizon lets above-trend growth fade gradually. With five
    # years, a compounder's growth is cut to `terminal_growth` almost at once
    # and most of its value lands in a terminal value that ignores it.
    forecast_years: int = 10

    # The forecast starts from the average FCF of the last N years rather
    # than the latest one, so a single weak or strong year (working-capital
    # swings, one-off capex) doesn't scale the whole valuation.
    base_fcf_years: int = 3

    # Discount each year's cash flow from mid-year (t - 0.5) rather than
    # year-end: cash arrives throughout the year, not on December 31st.
    mid_year_convention: bool = True

    # Long-term perpetual growth rate. Used both as the endpoint of the FCF
    # growth-decay model and as "g" in the Gordon Growth terminal value, so
    # the explicit forecast and the terminal value are always consistent.
    terminal_growth: float = 0.025

    # Marginal tax rate for the debt tax shield. None means "use the
    # company's reported effective tax rate", falling back to
    # `default_tax_rate` when that figure is missing or implausible.
    tax_rate: float | None = None
    default_tax_rate: float = 0.21

    fallback_cost_of_debt: float = 0.05
    cost_of_debt_min: float = 0.01
    cost_of_debt_max: float = 0.15

    cagr_min: float = -0.05
    cagr_max: float = 0.15

    wacc_min: float = 0.06
    wacc_max: float = 0.20

    beta_min: float = 0.5
    beta_max: float = 2.0

    def __post_init__(self) -> None:
        # Normalize the currency code; object.__setattr__ because the class is frozen.
        object.__setattr__(self, "target_currency", self.target_currency.strip().upper())

        if len(self.target_currency) != 3 or not self.target_currency.isalpha():
            raise InvalidAssumptionError(f"target_currency must be an ISO 4217 code, got {self.target_currency!r}")
        if not 1 <= self.forecast_years <= 30:
            raise InvalidAssumptionError(f"forecast_years must be in [1, 30], got {self.forecast_years}")
        if self.base_fcf_years < 1:
            raise InvalidAssumptionError(f"base_fcf_years must be at least 1, got {self.base_fcf_years}")
        if self.tax_rate is not None and not 0 <= self.tax_rate < 1:
            raise InvalidAssumptionError(f"tax_rate must be in [0, 1), got {self.tax_rate}")
        if not 0 <= self.default_tax_rate < 1:
            raise InvalidAssumptionError(f"default_tax_rate must be in [0, 1), got {self.default_tax_rate}")

        for name in ("cost_of_debt", "cagr", "wacc", "beta"):
            low, high = getattr(self, f"{name}_min"), getattr(self, f"{name}_max")
            if low > high:
                raise InvalidAssumptionError(f"{name}_min ({low}) must not exceed {name}_max ({high})")

        # The Gordon Growth model diverges when g >= WACC. Guaranteeing
        # g < wacc_min up front means no WACC the model can produce breaks it.
        if self.terminal_growth >= self.wacc_min:
            raise InvalidAssumptionError(
                f"terminal_growth ({self.terminal_growth}) must be below wacc_min ({self.wacc_min})"
            )


DEFAULT_ASSUMPTIONS = Assumptions()
