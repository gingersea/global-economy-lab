"""
Pluggable, key-free data backends for the Global Economy Lab.

Each backend exposes a single ``fetch_series(series_id, start_date, end_date,
**kwargs) -> pandas.DataFrame`` function that returns a standardised
DataFrame with the following columns:

- index ``date`` (``DatetimeIndex``)
- ``value``      – numeric value
- ``series_id``  – the requested series id
- ``source``     – short backend identifier (e.g. ``"fred_csv"``)

Backends should be self-contained, raise :class:`BackendError` on failure
and must NOT require any API key.
"""

from __future__ import annotations


class BackendError(RuntimeError):
    """Raised when a backend fails to return usable data."""


__all__ = ["BackendError"]
