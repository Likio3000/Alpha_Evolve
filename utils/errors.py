class AlphaEvolveError(Exception):
    """Base error for Alpha Evolve project."""


class BacktestError(AlphaEvolveError):
    """Raised for recoverable backtest failures (e.g., bad inputs)."""


class ConfigError(AlphaEvolveError):
    """Raised for invalid configuration values or combinations."""

