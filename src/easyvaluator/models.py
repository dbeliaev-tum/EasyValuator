"""Plain data containers passed between the data layer, the valuation
engine and the presentation layer.

None of these types perform I/O, which is what lets the valuation engine be
tested against hand-built inputs.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd

from .exceptions import DataUnavailableError


@dataclass(frozen=True, eq=False)
class CompanySnapshot:
    """Everything the model needs to know about a company, at one point in time.

    Currency conventions:
      * `price` and `market_cap` are in `price_currency` (major units —
        pence-quoted listings are already normalized to pounds).
      * `total_debt`, `total_cash`, `interest_expense` and `historical_fcf`
        are in `financial_currency`, the currency of the statements.
    The two differ for ADRs and some cross-listings (e.g. TSM: USD / TWD).

    `frozen` only stops attribute reassignment; the FCF series is copied on
    construction so later changes to the caller's series can't leak in.
    `eq=False` because dataclass equality on a `pd.Series` field raises
    instead of returning a bool.
    """

    symbol: str
    name: str
    country: str | None
    exchange: str | None

    price_currency: str
    financial_currency: str

    price: float
    shares_outstanding: float
    market_cap: float

    total_debt: float
    total_cash: float

    historical_fcf: pd.Series
    beta: float | None = None
    interest_expense: float | None = None
    effective_tax_rate: float | None = None

    def __post_init__(self) -> None:
        if not self.price > 0:
            raise DataUnavailableError(f"{self.symbol}: price must be positive, got {self.price}")
        if not self.shares_outstanding > 0:
            raise DataUnavailableError(
                f"{self.symbol}: shares outstanding must be positive, got {self.shares_outstanding}"
            )
        fcf = self.historical_fcf.astype(float).sort_index()
        object.__setattr__(self, "historical_fcf", fcf)


@dataclass(frozen=True)
class WACCBreakdown:
    risk_free_rate: float
    market_risk_premium: float
    beta: float
    cost_of_equity: float
    cost_of_debt: float
    tax_rate: float
    equity_weight: float
    debt_weight: float
    wacc: float


@dataclass(frozen=True)
class DCFBreakdown:
    """Output of the discounting step, in the currency of the cash flows."""

    pv_forecast: float
    terminal_value: float
    pv_terminal: float
    enterprise_value: float
    equity_value: float
    value_per_share: float

    @property
    def terminal_value_share(self) -> float:
        """Fraction of enterprise value that comes from the terminal value."""
        return self.pv_terminal / self.enterprise_value if self.enterprise_value else math.nan


@dataclass(frozen=True)
class SensitivityTable:
    """Fair value per share across a WACC x terminal-growth grid.

    `values[i][j]` corresponds to `wacc_values[i]` and `growth_values[j]`;
    None marks combinations where WACC <= g and the model is undefined.
    """

    wacc_values: list[float]
    growth_values: list[float]
    values: list[list[float | None]]

    def to_frame(self) -> pd.DataFrame:
        frame: pd.DataFrame = pd.DataFrame(self.values, index=self.wacc_values, columns=self.growth_values)
        frame.index.name, frame.columns.name = "wacc", "terminal_growth"
        return frame


@dataclass(frozen=True, eq=False)
class ValuationResult:
    """Full output of a valuation. Monetary fields are in `target_currency`
    unless their name says otherwise.
    """

    symbol: str
    company_name: str
    region: str
    price_currency: str
    financial_currency: str
    target_currency: str

    current_price: float
    historical_fcf: pd.Series
    base_fcf: float  # normalized starting point of the forecast
    forecast_fcf: list[float]
    cagr: float

    wacc: WACCBreakdown
    dcf: DCFBreakdown
    terminal_growth: float

    shares_outstanding: float
    total_debt: float
    total_cash: float

    sensitivity: SensitivityTable | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def fair_value(self) -> float:
        return self.dcf.value_per_share

    @property
    def upside(self) -> float:
        """Fair value relative to the current price (0.25 == 25% upside)."""
        return self.fair_value / self.current_price - 1

    @property
    def net_debt(self) -> float:
        return self.total_debt - self.total_cash

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable representation."""
        data = asdict(self)
        data["historical_fcf"] = {_label(k): float(v) for k, v in self.historical_fcf.items()}
        data["fair_value"] = self.fair_value
        data["upside"] = self.upside
        data["net_debt"] = self.net_debt
        data["dcf"]["terminal_value_share"] = self.dcf.terminal_value_share
        return data


def _label(key: Any) -> str:
    return str(key.year) if hasattr(key, "year") else str(key)
