"""
Asset correlation analysis.

Computes point-in-time and rolling correlation matrices between
a list of assets.  Output is designed to feed directly into heatmap
visualization functions.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd
from loguru import logger


def build_return_matrix(
    price_data: Dict[str, pd.DataFrame],
    price_col: str = "Close",
    freq: str = "D",
) -> pd.DataFrame:
    """Combine per-asset price DataFrames into a single returns matrix.

    Args:
        price_data: Dict mapping asset name → OHLCV DataFrame with a
                    DatetimeIndex.
        price_col:  Column to use for returns calculation.
        freq:       Resampling frequency (``"D"`` daily, ``"W"`` weekly,
                    ``"ME"`` monthly-end).

    Returns:
        DataFrame where each column is one asset's simple daily return.
        Rows are dates; missing values are NaN.
    """
    frames: Dict[str, pd.Series] = {}
    for name, df in price_data.items():
        if df.empty:
            logger.warning(
                f"build_return_matrix: {name!r} data is empty – skipped."
            )
            continue
        if price_col not in df.columns:
            # Fall back to the first numeric column
            numeric_cols = df.select_dtypes(include="number").columns.tolist()
            if not numeric_cols:
                logger.warning(
                    f"build_return_matrix: {name!r} has no numeric columns – skipped."
                )
                continue
            price_col_actual = numeric_cols[0]
        else:
            price_col_actual = price_col
        price_series = df[price_col_actual].resample(freq).last().dropna()
        frames[name] = price_series.pct_change(fill_method=None).rename(name)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames.values(), axis=1)


def compute_correlation_matrix(
    price_data: Dict[str, pd.DataFrame],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    method: str = "pearson",
    freq: str = "D",
    price_col: str = "Close",
) -> pd.DataFrame:
    """Compute the full-period correlation matrix for a set of assets.

    Args:
        price_data:  Dict mapping asset name → OHLCV DataFrame.
        start_date:  Optional ISO-8601 filter start.
        end_date:    Optional ISO-8601 filter end.
        method:      Correlation method: ``"pearson"`` | ``"spearman"`` |
                     ``"kendall"``.
        freq:        Return resampling frequency.
        price_col:   Price column used for returns.

    Returns:
        Square DataFrame with asset names as both index and columns.
        Values are correlation coefficients in [-1, 1].
    """
    returns = build_return_matrix(price_data, price_col=price_col, freq=freq)
    if returns.empty:
        return pd.DataFrame()

    if start_date:
        returns = returns.loc[start_date:]
    if end_date:
        returns = returns.loc[:end_date]

    return returns.corr(method=method)


def compute_rolling_correlation(
    price_data: Dict[str, pd.DataFrame],
    asset_a: str,
    asset_b: str,
    window: int = 60,
    freq: str = "D",
    price_col: str = "Close",
) -> pd.Series:
    """Compute rolling pairwise correlation between two assets.

    Args:
        price_data: Dict mapping asset name → OHLCV DataFrame.
        asset_a:    Name of the first asset (must be a key in *price_data*).
        asset_b:    Name of the second asset.
        window:     Rolling window size (in trading periods).
        freq:       Return resampling frequency.
        price_col:  Price column for returns computation.

    Returns:
        A :class:`pd.Series` of rolling correlation values indexed by date.
    """
    returns = build_return_matrix(price_data, price_col=price_col, freq=freq)
    if returns.empty:
        return pd.Series(dtype=float)

    if asset_a not in returns.columns:
        raise ValueError(
            f"Asset {asset_a!r} not found. Available: {list(returns.columns)}"
        )
    if asset_b not in returns.columns:
        raise ValueError(
            f"Asset {asset_b!r} not found. Available: {list(returns.columns)}"
        )

    return (
        returns[asset_a]
        .rolling(window=window)
        .corr(returns[asset_b])
        .rename(f"corr_{asset_a}_{asset_b}")
    )


def compute_rolling_correlation_matrix(
    price_data: Dict[str, pd.DataFrame],
    assets: Optional[List[str]] = None,
    window: int = 60,
    freq: str = "D",
    price_col: str = "Close",
) -> pd.DataFrame:
    """Compute rolling pairwise correlation for all asset pairs.

    Args:
        price_data: Dict mapping asset name → OHLCV DataFrame.
        assets:     Subset of asset names to include.  Defaults to all
                    assets in *price_data*.
        window:     Rolling window size (trading periods).
        freq:       Return resampling frequency.
        price_col:  Price column for returns.

    Returns:
        Long-form DataFrame with columns
        ``["date", "asset_a", "asset_b", "rolling_corr"]``.
    """
    returns = build_return_matrix(price_data, price_col=price_col, freq=freq)
    if returns.empty:
        return pd.DataFrame()

    if assets:
        missing = [a for a in assets if a not in returns.columns]
        if missing:
            logger.warning(
                f"compute_rolling_correlation_matrix: assets not found: {missing}"
            )
        assets = [a for a in assets if a in returns.columns]
    else:
        assets = list(returns.columns)

    rows = []
    for i, a in enumerate(assets):
        for b in assets[i + 1 :]:
            rolling = returns[a].rolling(window=window).corr(returns[b])
            for date, val in rolling.items():
                rows.append(
                    {
                        "date": date,
                        "asset_a": a,
                        "asset_b": b,
                        "rolling_corr": val,
                    }
                )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("date").sort_index()
