"""
Base class for all data fetchers in the Global Economy Lab.

Provides:
- Unified ``fetch(start_date, end_date)`` interface
- Transparent local disk cache (parquet files under ``data/processed/``)
- Automatic retry with exponential back-off (via tenacity)
- Structured logging (via loguru)
"""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from config.settings import get_settings


class BaseFetcher(ABC):
    """Abstract base class for every data fetcher.

    Subclasses must implement :meth:`_fetch_remote` which actually
    calls the external API / library and returns a raw DataFrame.
    The public :meth:`fetch` method wraps that call with caching and
    retry logic so callers never have to worry about those concerns.
    """

    def __init__(self, name: str) -> None:
        """
        Args:
            name: A short human-readable identifier used for logging
                  and as the cache file prefix.
        """
        self.name = name
        self._settings = get_settings()
        self._cache_dir: Path = self._settings.processed_data_dir
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    # ── Public interface ──────────────────────────────────────────────────────

    def fetch(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        use_cache: bool = True,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """Fetch data for the given date range.

        Results are cached locally as Parquet files.  On subsequent
        calls with the same parameters the cache is returned without
        hitting the network.

        Args:
            start_date: ISO-8601 string (``"YYYY-MM-DD"``).  Defaults
                        to ``"2000-01-01"`` when *None*.
            end_date:   ISO-8601 string.  Defaults to today when *None*.
            use_cache:  Whether to check for a local cache file first.
            force_refresh: If *True*, skip the cache and re-fetch even
                           if a cached file exists.

        Returns:
            A :class:`pandas.DataFrame` with a DatetimeIndex.
        """
        start_date = start_date or "2000-01-01"
        end_date = end_date or str(date.today())

        cache_path = self._cache_path(start_date, end_date)

        if use_cache and not force_refresh and cache_path.exists():
            logger.debug(
                f"[{self.name}] Loading from cache: {cache_path}"
            )
            return pd.read_parquet(cache_path)

        logger.info(
            f"[{self.name}] Fetching data "
            f"from {start_date} to {end_date} …"
        )
        df = self._fetch_with_retry(start_date, end_date)

        if df is not None and not df.empty:
            df.to_parquet(cache_path)
            logger.info(
                f"[{self.name}] Saved {len(df)} rows to {cache_path}"
            )
            self._record_manifest(
                cache_path=cache_path,
                start_date=start_date,
                end_date=end_date,
                rows=len(df),
            )

        return df if df is not None else pd.DataFrame()

    # ── Abstract method ───────────────────────────────────────────────────────

    @abstractmethod
    def _fetch_remote(
        self, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """Perform the actual remote data retrieval.

        Args:
            start_date: ISO-8601 start date string.
            end_date:   ISO-8601 end date string.

        Returns:
            A :class:`pandas.DataFrame` with a DatetimeIndex.
        """

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _fetch_with_retry(
        self, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """Call ``_fetch_remote`` with retry logic.

        Uses :func:`tenacity.retry` with exponential back-off.  Returns
        an empty DataFrame if all attempts fail.
        """
        max_retries = self._settings.max_retries

        @retry(
            reraise=False,
            stop=stop_after_attempt(max_retries),
            wait=wait_exponential(multiplier=1, min=2, max=30),
            retry=retry_if_exception_type(Exception),
        )
        def _attempt() -> pd.DataFrame:
            return self._fetch_remote(start_date, end_date)

        try:
            return _attempt()
        except Exception as exc:  # noqa: BLE001
            logger.error(
                f"[{self.name}] All {max_retries} retry attempts failed: {exc}"
            )
            return pd.DataFrame()

    def _cache_path(self, start_date: str, end_date: str) -> Path:
        """Compute a deterministic cache file path.

        The file is named ``<fetcher_name>_<hash>.parquet`` where the
        hash is derived from the date range.
        """
        key = f"{self.name}_{start_date}_{end_date}"
        digest = hashlib.md5(key.encode()).hexdigest()[:8]
        return self._cache_dir / f"{self.name}_{digest}.parquet"

    def clear_cache(self) -> None:
        """Delete all cached files for this fetcher."""
        for p in self._cache_dir.glob(f"{self.name}_*.parquet"):
            p.unlink()
            logger.info(f"[{self.name}] Deleted cache file: {p}")

    # ── Manifest / lineage ───────────────────────────────────────────────────

    def _record_manifest(
        self,
        cache_path: Path,
        start_date: str,
        end_date: str,
        rows: int,
    ) -> None:
        """Append a JSONL record describing this fetch to data/_meta/.

        Each line records the fetcher name, date range, output row count
        and a short MD5 of the cache file's bytes, providing a simple
        audit trail.  Failures are swallowed – manifest logging must
        never break a fetch.
        """
        try:
            meta_dir = self._settings.data_dir / "_meta"
            meta_dir.mkdir(parents=True, exist_ok=True)
            digest = ""
            if cache_path.exists():
                with cache_path.open("rb") as fh:
                    digest = hashlib.md5(fh.read()).hexdigest()
            record = {
                "fetcher": self.name,
                "start_date": start_date,
                "end_date": end_date,
                "rows": int(rows),
                "cache_path": str(cache_path),
                "md5": digest,
                "fetched_at": datetime.now(timezone.utc).isoformat(),
            }
            with (meta_dir / "manifest.jsonl").open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"[{self.name}] manifest record skipped: {exc}")
