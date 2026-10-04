"""Exception hierarchy.

Every error raised deliberately by the package derives from
`EasyValuatorError`, so callers can catch one type without swallowing
unrelated bugs.
"""


class EasyValuatorError(Exception):
    """Base class for all EasyValuator errors."""


class InvalidAssumptionError(EasyValuatorError, ValueError):
    """An `Assumptions` value is out of range or internally inconsistent."""


class DataUnavailableError(EasyValuatorError):
    """Required market or financial data could not be retrieved."""


class FXRateUnavailableError(DataUnavailableError):
    """No exchange rate could be found for a currency pair."""


class ValuationError(EasyValuatorError):
    """The data was retrieved, but a DCF valuation is not meaningful for it."""
