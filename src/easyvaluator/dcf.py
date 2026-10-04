"""Two-stage DCF discounting and sensitivity analysis."""

from __future__ import annotations

from .exceptions import ValuationError
from .models import DCFBreakdown, SensitivityTable


def discount_cash_flows(
    forecasts: list[float],
    wacc: float,
    terminal_growth: float,
    shares_outstanding: float,
    net_debt: float,
) -> DCFBreakdown:
    """Present value of explicit forecasts + Gordon Growth terminal value,
    bridged to equity value and value per share.

    Cash flows are assumed to arrive at the end of each year.
    """
    if not forecasts:
        raise ValuationError("At least one forecast year is required")
    if wacc <= terminal_growth:
        raise ValuationError(f"WACC ({wacc:.2%}) must exceed terminal growth ({terminal_growth:.2%})")
    if shares_outstanding <= 0:
        raise ValuationError("Shares outstanding must be positive")

    pv_forecast = sum(fcf / (1 + wacc) ** year for year, fcf in enumerate(forecasts, start=1))

    terminal_value = forecasts[-1] * (1 + terminal_growth) / (wacc - terminal_growth)
    pv_terminal = terminal_value / (1 + wacc) ** len(forecasts)

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
    forecasts: list[float],
    wacc: float,
    terminal_growth: float,
    shares_outstanding: float,
    net_debt: float,
    *,
    wacc_step: float = 0.01,
    growth_step: float = 0.005,
    steps: int = 2,
    scale: float = 1.0,
) -> SensitivityTable:
    """Value per share on a (2*steps+1) x (2*steps+1) grid centred on the
    base-case WACC and terminal growth.

    `scale` multiplies every cell, e.g. an FX rate to report in another
    currency. The forecasts themselves are held fixed, so only the discount
    rate and the perpetuity growth vary.
    """
    offsets = range(-steps, steps + 1)
    waccs = [round(wacc + i * wacc_step, 10) for i in offsets]
    growths = [round(terminal_growth + j * growth_step, 10) for j in offsets]

    values: list[list[float | None]] = []
    for w in waccs:
        row: list[float | None] = []
        for g in growths:
            if w <= g or w <= 0:
                row.append(None)
                continue
            dcf = discount_cash_flows(forecasts, w, g, shares_outstanding, net_debt)
            row.append(dcf.value_per_share * scale)
        values.append(row)

    return SensitivityTable(wacc_values=waccs, growth_values=growths, values=values)
