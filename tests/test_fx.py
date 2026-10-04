import pytest

from easyvaluator.exceptions import FXRateUnavailableError
from easyvaluator.fx import FXConverter, normalize_currency


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("USD", ("USD", 1.0)),
        ("eur", ("EUR", 1.0)),
        ("GBp", ("GBP", 0.01)),  # LSE quotes in pence
        ("ZAc", ("ZAR", 0.01)),
        ("ILA", ("ILS", 0.01)),
    ],
)
def test_normalize_currency(code, expected):
    assert normalize_currency(code) == expected


def test_same_currency_needs_no_lookup():
    calls = []
    fx = FXConverter(lambda a, b: calls.append((a, b)))
    assert fx.convert(100.0, "usd", "USD") == 100.0
    assert calls == []


def test_rates_are_cached():
    calls = []

    def fetch(a, b):
        calls.append((a, b))
        return 0.9

    fx = FXConverter(fetch)
    fx.convert(1.0, "USD", "EUR")
    fx.convert(2.0, "USD", "EUR")
    assert calls == [("USD", "EUR")]


def test_falls_back_to_inverse_pair():
    fx = FXConverter(lambda a, b: 2.0 if (a, b) == ("EUR", "XYZ") else None)
    assert fx.rate("XYZ", "EUR") == pytest.approx(0.5)


def test_missing_rate_raises_instead_of_passing_amount_through():
    fx = FXConverter(lambda a, b: None)
    with pytest.raises(FXRateUnavailableError):
        fx.convert(100.0, "USD", "EUR")
