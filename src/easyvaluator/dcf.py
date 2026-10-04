"""Two-stage DCF discounting and sensitivity analysis."""

from __future__ import annotations

from collections.abc import Callable

from .exceptions import ValuationError
from .models import DCFBreakdown, SensitivityTable


def discount_cash_flows(
    forecasts: list[float],
    wacc: float,
    terminal_growth: float,
    shares_outstanding: float,
    net_debt: float,
    *,
    mid_year: bool = False,
) -> DCFBreakdown:
    """Present value of explicit forecasts + Gordon Growth terminal value,
    bridged to equity value and value per share.

    By default cash flows arrive at the end of each year. With `mid_year`,
    each year's cash flow is discounted from the middle of that year
    (t - 0.5), and so is the terminal value: the perpetuity's cash flows
    arrive mid-year too, so the Gordon value at year n is worth
    (1 + WACC)^0.5 more than the end-year formula implies.
    """
    if not forecasts:
        raise ValuationError("At least one forecast year is required")
    if wacc <= terminal_growth:
        raise ValuationError(f"WACC ({wacc:.2%}) must exceed terminal growth ({terminal_growth:.2%})")
    if shares_outstanding <= 0:
        raise ValuationError("Shares outstanding must be positive")

    shift = 0.5 if mid_year else 0.0
    n = len(forecasts)

    pv_forecast = sum(fcf / (1 + wacc) ** (year - shift) for year, fcf in enumerate(forecasts, start=1))

    terminal_value = forecasts[-1] * (1 + terminal_growth) / (wacc - terminal_growth)
    pv_terminal = terminal_value / (1 + wacc) ** (n - shift)

    enterprise_value = pv_forecast + pv_terminal
    equity_value = enterprise_value - net_debt

    return DCFBreakdown(
        pv_forecast=pv_forecast,
        terminal_value=terminal_value,
        pv_terminal=pv_terminal,
        enterprise_value=enterprise_value,
        equity_value=equity_value,
        value_per_share=equity_value / shares_outstanding,
    )


def sensitivity_table(
    forecast_for: Callable[[float], list[float]],
    wacc: float,
    terminal_growth: float,
    shares_outstanding: float,
    net_debt: float,
    *,
    mid_year: bool = False,
    wacc_step: float = 0.01,
    growth_step: float = 0.005,
    steps: int = 2,
    scale: float = 1.0,
) -> SensitivityTable:
    """Value per share on a (2*steps+1) x (2*steps+1) grid centred on the
    base-case WACC and terminal growth.

    `forecast_for(g)` must return the explicit forecast whose growth path
    ends at `g`. Rebuilding the forecast for each column keeps every cell
    internally consistent: the last forecast year grows at the same rate the
    perpetuity assumes, exactly as in the base case.

    `scale` multiplies every cell, e.g. an FX rate to report in another
    currency.
    """
    offsets = range(-steps, steps + 1)
    waccs = [round(wacc + i * wacc_step, 10) for i in offsets]
    growths = [round(terminal_growth + j * growth_step, 10) for j in offsets]
    forecasts = {g: forecast_for(g) for g in growths}

    values: list[list[float | None]] = []
    for w in waccs:
        row: list[float | None] = []
        for g in growths:
            if w <= g or w <= 0:
                row.append(None)
                continue
            dcf = discount_cash_flows(forecasts[g], w, g, shares_outstanding, net_debt, mid_year=mid_year)
            row.append(dcf.value_per_share * scale)
        values.append(row)

    return SensitivityTable(wacc_values=waccs, growth_values=growths, values=values)
