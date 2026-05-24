"""
Reference weight-schedule builders for the phase-1 backtest.

Each function returns a ``pd.DataFrame`` indexed identically to the
input panel (or returns frame) and with one column per asset.  All rows
sum to 1.0 (or 0.0 when no return data is available for the row, which
the engine treats as a 100 % cash month).

The plan calls for three benchmarks – two are implemented here, the
third (regime rotation) lives in :mod:`src.analysis.regime_labels`
because it is the consumer of the regime labels.
"""

from __future__ import annotations

from typing import Iterable, Optional

import pandas as pd


def equal_weight_target(
    returns: pd.DataFrame,
    assets: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Equal-weight across *assets* for every month they exist.

    A month where some assets have ``NaN`` returns reduces the active
    universe; weights are renormalised so the row sums to 1.0 (or 0.0
    when every asset is NaN for that month).

    Args:
        returns: Monthly returns DataFrame.
        assets:  Subset of columns to include.  Defaults to all.
    """
    if returns is None or returns.empty:
        return pd.DataFrame()

    cols = list(assets) if assets else list(returns.columns)
    cols = [c for c in cols if c in returns.columns]
    if not cols:
        return pd.DataFrame(index=returns.index)

    mask = returns[cols].notna().astype(float)
    row_sum = mask.sum(axis=1).replace(0.0, pd.NA)
    weights = mask.div(row_sum, axis=0).fillna(0.0)
    return weights


def buy_and_hold_target(
    returns: pd.DataFrame,
    asset: str,
) -> pd.DataFrame:
    """100 % in a single asset every month (the canonical buy-and-hold
    proxy used as a benchmark)."""
    if returns is None or returns.empty:
        return pd.DataFrame()
    if asset not in returns.columns:
        raise KeyError(
            f"buy_and_hold_target: asset '{asset}' not in returns columns "
            f"{list(returns.columns)}."
        )
    weights = pd.DataFrame(0.0, index=returns.index, columns=returns.columns)
    # Only allocate on rows where the asset actually has a return.
    has_return = returns[asset].notna()
    weights.loc[has_return, asset] = 1.0
    return weights


__all__ = [
    "equal_weight_target",
    "buy_and_hold_target",
]
