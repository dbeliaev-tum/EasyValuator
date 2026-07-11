# EasyValuator

A Discounted Cash Flow (DCF) stock valuation tool. Given a ticker symbol, it
pulls financials from Yahoo Finance, forecasts free cash flow, computes a
region-aware WACC, and derives a fair value per share — with every figure
normalized to a single output currency.

This is a personal portfolio project combining software engineering and
equity valuation. It is **not investment advice**: it relies on simplified
assumptions (see below) and should be treated as an educational model, not
a source of trading decisions.

## Features

- Two-stage DCF: explicit forecast period + Gordon Growth terminal value
- Multi-currency handling — market price and financial statements are
  converted independently (they aren't always in the same currency, e.g.
  for ADRs), both normalized to one target currency
- Regional risk parameters (US, EU, UK, CN, JP): risk-free rate spread and
  equity risk premium
- CAPM-based WACC with market-implied capital structure weights and a debt
  tax shield
- Growth-decay FCF forecast: transitions from historical CAGR to the
  assumed terminal growth rate
- Configurable assumptions (forecast horizon, terminal growth, tax rate,
  bounds on WACC/beta/CAGR) via a single `Assumptions` object
- Unit-tested core valuation math (no network calls required)

## Installation

```bash
git clone <this-repo>
cd EasyValuator
pip install -r requirements.txt
```

Or install as an editable package (adds the `easyvaluator` command):

```bash
pip install -e .
```

## Usage

### Command line

```bash
python -m easyvaluator AAPL
python -m easyvaluator MSFT --years 7 --currency USD
python -m easyvaluator SIE.DE -v   # -v surfaces fallback warnings
```

If no ticker is given, you'll be prompted for one interactively.

### As a library

```python
from easyvaluator import Assumptions, price_stock, print_report

result = price_stock("AAPL", Assumptions(target_currency="EUR", forecast_years=5))
print_report(result)

print(result.fair_price, result.wacc, result.cagr)
```

`price_stock` returns a `ValuationResult` dataclass with every intermediate
figure (historical/forecast FCF, WACC, beta, risk-free rate, enterprise and
equity value, fair price...); `print_report` is purely a formatter on top
of it, so the valuation logic has no I/O and is easy to test or feed into
other tooling.

## Methodology

```
Enterprise Value = PV(Explicit Forecast FCF) + PV(Terminal Value)
Equity Value      = Enterprise Value - Net Debt
Fair Value/Share  = Equity Value / Shares Outstanding
```

- **Free Cash Flow**: read directly from Yahoo Finance's `Free Cash Flow`
  line item when available; otherwise `Operating Cash Flow + Capital
  Expenditure` (Yahoo reports CapEx as a negative outflow, so it is added,
  not subtracted).
- **FCF forecast**: growth rate decays linearly from the historical CAGR
  (clamped to `[cagr_min, cagr_max]`) to `terminal_growth` over the
  forecast horizon.
- **Terminal value**: Gordon Growth Model, `FCF_n × (1 + g) / (WACC - g)`,
  using the *same* `g` as the forecast decay endpoint — so the explicit
  forecast and the perpetuity are consistent with each other.
- **Cost of equity (CAPM)**: `risk_free_rate + beta × market_risk_premium`,
  beta clamped to `[beta_min, beta_max]`.
- **Cost of debt**: interest expense (from the income statement, falling
  back to `info`) divided by total debt; a flat fallback rate is used if
  neither is available.
- **WACC**: capital-structure-weighted blend of cost of equity and
  after-tax cost of debt, clamped to `[wacc_min, wacc_max]`.

## Regional parameters

| Region | Code | Risk-free proxy | Adjustment vs. US | Equity risk premium |
|--------|------|------------------|--------------------|----------------------|
| United States  | US | 10Y Treasury (`^TNX`) | baseline | 5.5% |
| European Union | EU | US Treasury + spread  | -1.5%    | 5.5% |
| United Kingdom | UK | US Treasury + spread  | -0.5%    | 5.5% |
| China          | CN | US Treasury + spread  | -0.5%    | 7.0% |
| Japan          | JP | US Treasury + spread  | -3.5%    | 5.5% |

Region is inferred from the ticker's exchange and country metadata,
defaulting to US.

## Project layout

```
easyvaluator/
├── assumptions.py   # Assumptions dataclass: every tunable number in one place
├── fx.py            # FX rate lookup + currency conversion
├── market_data.py   # All yfinance access lives here
├── fcf.py           # Historical FCF extraction + growth-decay forecasting
├── valuation.py      # WACC, DCF math, price_stock() pipeline, ValuationResult
├── report.py        # print_report(): formats a ValuationResult for the terminal
└── cli.py           # argparse entry point
tests/
├── test_fcf.py
└── test_valuation.py
```

Valuation logic and I/O are deliberately separate: `price_stock()` only
computes and returns data, `print_report()` only formats it. This keeps
the math independently testable and reusable outside the CLI.

## Testing

```bash
pip install -e ".[dev]"
pytest
```

Tests cover the pure valuation math (FCF forecasting, DCF present-value
calculation, field-name fallback logic) with no network access required.

## Known limitations

- Risk-free rates for non-US regions are a fixed spread over the US 10Y
  Treasury, not each region's own sovereign yield curve.
- Regional equity risk premiums are static long-run estimates, not
  updated from a live source.
- Cost of debt falls back to a flat 5% assumption when interest expense
  isn't available.
- No sensitivity analysis (WACC × terminal growth grid) yet — see
  Roadmap.

## Roadmap

- [ ] Sensitivity table: fair value across a WACC × terminal-growth grid
- [ ] Cross-check DCF output against trading multiples (EV/EBITDA, P/E)
- [ ] Monte Carlo scenario analysis on FCF growth
- [ ] Simple web front end (e.g. Streamlit)

## Disclaimer

For educational and research purposes only. Not financial advice. The
model rests on simplifying assumptions that may not hold in practice —
verify all figures independently before making investment decisions.

## License

MIT — see [LICENSE](LICENSE).
