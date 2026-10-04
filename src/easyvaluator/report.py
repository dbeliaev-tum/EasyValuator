"""Rendering of a `ValuationResult` as terminal text or JSON."""

from __future__ import annotations

import json
import math
from typing import Any

from .models import SensitivityTable, ValuationResult

WIDTH = 64


def _section(title: str) -> list[str]:
    return ["", title, "-" * WIDTH]


def _row(label: str, value: str) -> str:
    return f"  {label:<34}{value:>{WIDTH - 36}}"


def _money(value: float, currency: str) -> str:
    """Compact money format: 1.23B EUR, 456.7M EUR, 12.34 EUR."""
    magnitude = abs(value)
    for threshold, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M")):
        if magnitude >= threshold:
            return f"{value / threshold:,.2f}{suffix} {currency}"
    return f"{value:,.2f} {currency}"


def _pct(value: float) -> str:
    return f"{value:.2%}"


def render_sensitivity(table: SensitivityTable, currency: str) -> list[str]:
    header = "  WACC \\ g  " + "".join(f"{g:>10.2%}" for g in table.growth_values)
    lines = [header]
    centre = len(table.wacc_values) // 2
    for i, (wacc, row) in enumerate(zip(table.wacc_values, table.values, strict=True)):
        cells = "".join(f"{'n/a':>10}" if v is None else f"{v:>10,.2f}" for v in row)
        marker = "*" if i == centre else " "
        lines.append(f" {marker}{wacc:>8.2%}  {cells}")
    lines.append(f"  (fair value per share, {currency}; * = base case WACC)")
    return lines


def render_text(result: ValuationResult) -> str:
    cur = result.target_currency
    w, d = result.wacc, result.dcf

    lines = ["=" * WIDTH, f"DCF valuation: {result.company_name} ({result.symbol})", "=" * WIDTH]
    lines.append(_row("Region", result.region))
    lines.append(_row("Listing / statement currency", f"{result.price_currency} / {result.financial_currency}"))
    lines.append(_row("Reporting currency", cur))

    lines += _section("Free cash flow")
    for period, value in result.historical_fcf.items():
        label = str(period.year) if hasattr(period, "year") else str(period)
        lines.append(_row(f"{label} (actual)", _money(value, cur)))
    lines.append(_row("Forecast base (normalized)", _money(result.base_fcf, cur)))
    for year, value in enumerate(result.forecast_fcf, start=1):
        lines.append(_row(f"Year {year} (forecast)", _money(value, cur)))
    lines.append(_row("Historical CAGR (clamped)", _pct(result.cagr)))
    lines.append(_row("Terminal growth", _pct(result.terminal_growth)))

    lines += _section("Cost of capital")
    lines.append(_row("Risk-free rate", _pct(w.risk_free_rate)))
    lines.append(_row("Equity risk premium", _pct(w.market_risk_premium)))
    lines.append(_row("Beta", f"{w.beta:.2f}"))
    lines.append(_row("Cost of equity", _pct(w.cost_of_equity)))
    lines.append(_row("Cost of debt (pre-tax)", _pct(w.cost_of_debt)))
    lines.append(_row("Tax rate", _pct(w.tax_rate)))
    lines.append(_row("Weights (equity / debt)", f"{w.equity_weight:.1%} / {w.debt_weight:.1%}"))
    lines.append(_row("WACC", _pct(w.wacc)))

    lines += _section("Valuation")
    lines.append(_row("PV of forecast FCF", _money(d.pv_forecast, cur)))
    lines.append(_row("PV of terminal value", _money(d.pv_terminal, cur)))
    lines.append(_row("Enterprise value", _money(d.enterprise_value, cur)))
    lines.append(_row("Less: net debt", _money(result.net_debt, cur)))
    lines.append(_row("Equity value", _money(d.equity_value, cur)))
    lines.append(_row("Shares outstanding", f"{result.shares_outstanding / 1e6:,.1f}M"))
    if not math.isnan(d.terminal_value_share):
        lines.append(_row("Terminal value share of EV", f"{d.terminal_value_share:.0%}"))

    lines += ["", "=" * WIDTH]
    lines.append(_row("FAIR VALUE PER SHARE", f"{result.fair_value:,.2f} {cur}"))
    lines.append(_row("CURRENT PRICE", f"{result.current_price:,.2f} {cur}"))
    lines.append(_row("UPSIDE / DOWNSIDE", f"{result.upside:+.1%}"))
    lines.append("=" * WIDTH)

    if result.sensitivity is not None:
        lines += _section("Sensitivity: fair value per share")
        lines += render_sensitivity(result.sensitivity, cur)

    if result.warnings:
        lines += _section("Model warnings")
        lines += [f"  ! {message}" for message in result.warnings]

    return "\n".join(lines)


def print_report(result: ValuationResult) -> None:
    print(render_text(result))


def _json_default(value: Any) -> Any:
    if isinstance(value, float) and math.isnan(value):
        return None
    if hasattr(value, "item"):  # numpy scalars
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def render_json(results: ValuationResult | list[ValuationResult], *, indent: int | None = 2) -> str:
    """JSON for one result (an object) or several (an array)."""
    if isinstance(results, ValuationResult):
        payload: Any = results.to_dict()
    else:
        payload = [result.to_dict() for result in results]
    return json.dumps(payload, indent=indent, default=_json_default)
