"""Weighted Average Cost of Capital."""

from __future__ import annotations

from .assumptions import DEFAULT_ASSUMPTIONS, Assumptions
from .models import WACCBreakdown


def _clamp(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)


def calculate_wacc(
    *,
    risk_free_rate: float,
    market_risk_premium: float,
    beta: float,
    market_cap: float,
    total_debt: float,
    cost_of_debt: float,
    tax_rate: float,
    assumptions: Assumptions = DEFAULT_ASSUMPTIONS,
) -> WACCBreakdown:
    """CAPM cost of equity blended with after-tax cost of debt using
    market-value capital-structure weights.

    `market_cap` and `total_debt` must be in the same currency — the weights
    are meaningless otherwise. Beta, cost of debt and the resulting WACC are
    clamped to the bounds in `assumptions`; the breakdown reports the clamped
    values that were actually used.
    """
    beta = _clamp(beta, assumptions.beta_min, assumptions.beta_max)
    cost_of_debt = _clamp(cost_of_debt, assumptions.cost_of_debt_min, assumptions.cost_of_debt_max)
    cost_of_equity = risk_free_rate + beta * market_risk_premium

    total_debt = max(total_debt, 0.0)
    total_capital = market_cap + total_debt
    if total_capital > 0:
        equity_weight = market_cap / total_capital
        debt_weight = total_debt / total_capital
    else:
        equity_weight, debt_weight = 1.0, 0.0

    raw_wacc = equity_weight * cost_of_equity + debt_weight * cost_of_debt * (1 - tax_rate)

    return WACCBreakdown(
        risk_free_rate=risk_free_rate,
        market_risk_premium=market_risk_premium,
        beta=beta,
        cost_of_equity=cost_of_equity,
        cost_of_debt=cost_of_debt,
        tax_rate=tax_rate,
        equity_weight=equity_weight,
        debt_weight=debt_weight,
        wacc=_clamp(raw_wacc, assumptions.wacc_min, assumptions.wacc_max),
    )
