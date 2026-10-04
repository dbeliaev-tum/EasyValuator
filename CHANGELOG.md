# Changelog

## Unreleased

### Changed (model)
- Forecast horizon defaults to 10 years (was 5).
- The forecast starts from the 3-year average FCF instead of the latest year.
  Average ≤ 0 is rejected; a negative latest year alone is smoothed and flagged.
- Mid-year discounting convention, on by default (`mid_year_convention`).
- Equity risk premium: 4.5% mature-market implied premium + country premium
  (CN +1%), replacing the 5.5% / 7.0% historical-style figures.

### Fixed
- The sensitivity grid now rebuilds the forecast for each terminal growth rate. Before, only the
  perpetuity used the cell's `g` while the last forecast year still grew at the base `g`.
- `CompanySnapshot` copies its FCF series and rejects a non-positive price or share count.
  `eq=False` on Series-holding dataclasses, where the generated `==` raised.

### Internal
- `Valuator.value_snapshot` split into region / projection / cost-of-capital steps.
- The CLI's JSON output no longer round-trips through `json.loads`. `render_json` accepts a list.
- Removed the unneeded `tests/__init__.py`.

## 2.0.0

### Fixed
- **Pence-quoted listings**: LSE prices arrive in `GBp`. They were treated as GBP,
  which overstated the current price 100× (e.g. SHEL.L showed €4,231 instead of €42).
- **ADR capital structure**: the market cap (listing currency) was weighted against
  debt (statement currency) without conversion. For TSM (USD vs. TWD), this inflated
  the debt weight from ~1.4% to ~30%.
- **Discount-rate currency**: the risk-free rate is now keyed by the currency the cash flows
  are in, not by the listing venue.
- **CAGR period**: the CAGR now uses the actual elapsed time between statements instead of the
  number of data points. A missing year no longer inflates growth.
- **Growth from negative FCF**: the old code averaged `pct_change()` over negative values, which
  is meaningless. The CAGR is now treated as undefined and falls back to terminal growth.
- **Silent FX fallback**: a missing exchange rate used to return the unconverted
  amount. It now raises `FXRateUnavailableError`.
- **Invented inputs**: the defaults `price = 100` and `shares = 1` have been removed. Missing
  essentials now raise `DataUnavailableError`.
- `g ≥ WACC` and other invalid assumptions are rejected up front.

### Added
- WACC × terminal-growth sensitivity table
- `--json` output, multiple tickers per run, `--terminal-growth` and `--tax-rate` flags
- The company's effective tax rate is used for the debt tax shield
- `ValuationResult.warnings` lists every clamp and fallback the model applied
- `MarketDataProvider` protocol and an offline `FakeProvider` test double
- CI (ruff, mypy strict, pytest on Python 3.10–3.13), pre-commit, `py.typed`

### Changed
- `src/` layout. Pure modules (`fcf`, `wacc`, `dcf`) are split from I/O (`market_data`).
- `ValuationResult` now nests `WACCBreakdown` and `DCFBreakdown`. `price_stock()` is kept
  as a convenience wrapper around `Valuator`.
- Python ≥ 3.10.

## 1.0.0

- Initial package: split the monolithic script into modules and added unit tests.
