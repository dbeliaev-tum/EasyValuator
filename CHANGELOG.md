# Changelog

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
