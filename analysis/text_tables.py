"""
Formatting functions for the plain text analysis tables.
"""

from __future__ import annotations

from typing import Callable

import pandas as pd


def format_table(
    rows: list[dict],
    formatters: dict[str, Callable] | None = None,
) -> str:
    """Return `rows` as an aligned table with no index."""
    return pd.DataFrame(rows).to_string(index=False, formatters=formatters)


def print_table(
    rows: list[dict],
    formatters: dict[str, Callable] | None = None,
) -> None:
    """Print `rows` as an aligned table with no index."""
    print(format_table(rows, formatters))
    print()


def fixed(places: int = 3, signed: bool = False) -> Callable:
    """Formatter for a float column, leaving non-float cells untouched.

    Missing entries are printed as whatever string the caller put in the row,
    so a table can mix numbers with "missing" without a separate column.
    """
    spec = f"{{:+.{places}f}}" if signed else f"{{:.{places}f}}"

    def _format(value):
        return spec.format(value) if isinstance(value, float) else str(value)

    return _format
