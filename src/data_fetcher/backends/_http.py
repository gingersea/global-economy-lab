"""
Shared HTTP utilities for key-free data backends.

Provides a single ``http_get`` function with:

- A configurable User-Agent (some public CSV endpoints reject the default).
- Exponential back-off retry via ``tenacity``.
- Per-host rate limiting (best-effort, in-process).
- Timeouts and clear error messages.
"""

from __future__ import annotations

import threading
import time
from typing import Dict, Optional
from urllib.parse import urlparse

import requests
from loguru import logger
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from config.settings import get_settings
from src.data_fetcher.backends import BackendError


# Reasonable browser-like UA – several public endpoints (e.g. fredgraph.csv)
# reject empty / generic user-agents.
_USER_AGENT = (
    "Mozilla/5.0 (compatible; GlobalEconomyLab/0.1; "
    "+https://github.com/hjiang555-a11y/global-economy-lab)"
)

# Minimum spacing between requests to the same host, seconds.
_HOST_MIN_INTERVAL: Dict[str, float] = {
    "api.bls.gov": 1.5,
    "api.eia.gov": 0.5,
    "api.worldbank.org": 0.3,
    "sdw-wsrest.ecb.europa.eu": 0.3,
    "stats.oecd.org": 0.5,
    "fred.stlouisfed.org": 0.2,
    "home.treasury.gov": 0.5,
    "www.ecb.europa.eu": 0.3,
}

_last_request_at: Dict[str, float] = {}
_lock = threading.Lock()


def _throttle(host: str) -> None:
    """Sleep just long enough to respect per-host minimum interval."""
    min_interval = _HOST_MIN_INTERVAL.get(host, 0.0)
    if min_interval <= 0:
        return
    with _lock:
        prev = _last_request_at.get(host, 0.0)
        wait = min_interval - (time.monotonic() - prev)
        if wait > 0:
            time.sleep(wait)
        _last_request_at[host] = time.monotonic()


def http_get(
    url: str,
    *,
    params: Optional[Dict[str, str]] = None,
    headers: Optional[Dict[str, str]] = None,
    timeout: Optional[int] = None,
    max_retries: Optional[int] = None,
) -> requests.Response:
    """Perform an HTTP GET with retries, throttling and a real User-Agent.

    Args:
        url:         Target URL.
        params:      Optional query string parameters.
        headers:     Extra HTTP headers (User-Agent is set automatically).
        timeout:     Request timeout in seconds (defaults to settings value).
        max_retries: Number of attempts on transient failures.

    Returns:
        The :class:`requests.Response` (status already validated).

    Raises:
        BackendError: If the request fails after all retries.
    """
    settings = get_settings()
    timeout = timeout or settings.request_timeout
    max_retries = max_retries or settings.max_retries

    merged_headers = {"User-Agent": _USER_AGENT, "Accept": "*/*"}
    if headers:
        merged_headers.update(headers)

    host = urlparse(url).netloc

    @retry(
        reraise=True,
        stop=stop_after_attempt(max(1, max_retries)),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(
            (requests.ConnectionError, requests.Timeout, requests.HTTPError)
        ),
    )
    def _do() -> requests.Response:
        _throttle(host)
        resp = requests.get(
            url, params=params, headers=merged_headers, timeout=timeout
        )
        # Public CSV endpoints sometimes return 200 with HTML error pages – we
        # let downstream parsers detect this, but we still raise for >=400.
        resp.raise_for_status()
        return resp

    try:
        return _do()
    except Exception as exc:
        logger.error(f"http_get failed for {url}: {exc}")
        raise BackendError(f"HTTP GET failed for {url}: {exc}") from exc
