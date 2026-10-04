from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any

import pandas as pd
import pytest

from easyvaluator.exceptions import DataUnavailableError
from easyvaluator.models import CompanySnapshot


@dataclass
class FakeProvider:
    """In-memory `MarketDataProvider` so the full pipeline runs offline."""

    companies: dict[str, CompanySnapshot] = field(default_factory=dict)
    fx_rates: dict[tuple[str, str], float] = field(default_factory=dict)
    treasury_yield: float | None = 0.04
    fx_calls: list[tuple[str, str]] = field(default_factory=list)

    def get_company(self, symbol: str) -> CompanySnapshot:
        try:
            return self.companies[symbol]
        except KeyError:
            raise DataUnavailableError(f"unknown ticker {symbol}") from None

    def get_fx_rate(self, from_currency: str, to_currency: str) -> float | None:
        self.fx_calls.append((from_currency, to_currency))
        return self.fx_rates.get((from_currency, to_currency))

    def get_us_treasury_yield(self) -> float | None:
        return self.treasury_yield


def _fcf(*values: float, start_year: int = 2021) -> pd.Series:
    index = pd.to_datetime([f"{start_year + i}-12-31" for i in range(len(values))])
    return pd.Series(values, index=index, dtype=float)


@pytest.fixture
def fcf_series() -> Callable[..., pd.Series]:
    return _fcf


@pytest.fixture
def make_snapshot() -> Callable[..., CompanySnapshot]:
    base = CompanySnapshot(
        symbol="TEST",
        name="Test Corp",
        country="United States",
        exchange="NMS",
        price_currency="USD",
        financial_currency="USD",
        price=50.0,
        shares_outstanding=1_000_000.0,
        market_cap=50_000_000.0,
        total_debt=10_000_000.0,
        total_cash=4_000_000.0,
        historical_fcf=_fcf(2_000_000, 2_200_000, 2_420_000, 2_662_000),
        beta=1.1,
        interest_expense=500_000.0,
        effective_tax_rate=0.21,
    )

    def factory(**overrides: Any) -> CompanySnapshot:
        return replace(base, **overrides)

    return factory


@pytest.fixture
def provider() -> FakeProvider:
    return FakeProvider(fx_rates={("USD", "EUR"): 0.9})
