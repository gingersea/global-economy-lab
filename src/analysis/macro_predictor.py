"""
Macro-economic regime predictor.

Predicts the economic regime 1–3 months ahead using:
1. A Markov-style transition probability matrix estimated from historical labels.
2. A logistic-style trend assessment from lagged macro indicators.

Output is a DataFrame with predicted regime probabilities and a
highest-probability regime label for each forecast horizon.

Pure pandas + numpy — no heavy ML dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from src.analysis.regime_labels import REGIME_LABELS, RegimeConfig, label_regimes

ALL_REGIMES = ["recovery", "overheat", "stagflation", "recession", "unknown"]


@dataclass
class MacroPredictorConfig:
    """Configuration for :class:`MacroRegimePredictor`.

    Attributes:
        regime_config:    Regime labelling parameters.
        lookback_months:  How many months of history to use for transition
                          matrix estimation.
        horizons:         Which forecast horizons to produce (months ahead).
        min_observations: Minimum regime transitions before trusting the matrix.
    """

    regime_config: RegimeConfig = field(default_factory=RegimeConfig)
    lookback_months: int = 60
    horizons: Tuple[int, ...] = (1, 2, 3)
    min_observations: int = 12


@dataclass
class MacroPrediction:
    """Container for one macro prediction snapshot.

    Attributes:
        timestamp:          Reference date (last known month-end).
        current_regime:     Regime label for the reference date.
        probabilities:      DataFrame: rows = horizon months, cols = regimes.
        predicted_regimes:  Dict ``{horizon: predicted_regime}``.
        transition_matrix:  The 4×4 transition probability matrix used.
        confidence:         Dict ``{horizon: float}`` — probability of the
                            predicted regime (0–1).
        indicators:         Dict of current macro indicator values.
    """

    timestamp: pd.Timestamp
    current_regime: str
    probabilities: pd.DataFrame
    predicted_regimes: Dict[int, str]
    transition_matrix: pd.DataFrame
    confidence: Dict[int, float]
    indicators: Dict[str, float] = field(default_factory=dict)


def build_transition_matrix(labelled: pd.DataFrame, regime_col: str = "regime") -> pd.DataFrame:
    """Estimate a 4-regime Markov transition matrix from historical labels.

    Args:
        labelled:   Monthly panel with a regime column.
        regime_col: Name of the regime label column.

    Returns:
        4×4 DataFrame indexed by ``from_regime``, columns are ``to_regime``.
        Each row sums to 1.0.  Rows with zero observations are uniform (0.25).
    """
    if labelled is None or labelled.empty or regime_col not in labelled.columns:
        return pd.DataFrame(0.25, index=ALL_REGIMES[:4], columns=ALL_REGIMES[:4])

    seq = labelled[regime_col].dropna()
    core_regimes = [r for r in ALL_REGIMES[:4] if r in seq.values]
    if len(core_regimes) < 2:
        return pd.DataFrame(0.25, index=ALL_REGIMES[:4], columns=ALL_REGIMES[:4])

    counts = pd.DataFrame(0, index=ALL_REGIMES[:4], columns=ALL_REGIMES[:4], dtype=float)
    for i in range(len(seq) - 1):
        frm = seq.iloc[i]
        to = seq.iloc[i + 1]
        if frm in counts.index and to in counts.columns:
            counts.loc[frm, to] += 1.0

    row_sums = counts.sum(axis=1)
    tmat = counts.div(row_sums.where(row_sums > 0, 1.0), axis=0)
    tmat = tmat.fillna(0.25)
    for idx in tmat.index:
        if row_sums[idx] == 0:
            tmat.loc[idx] = 0.25
    tmat.index.name = "from_regime"
    tmat.columns.name = "to_regime"
    return tmat


def predict_regime_probs(
    current_regime: str,
    trans_matrix: pd.DataFrame,
    horizon: int = 1,
) -> pd.Series:
    """Project regime probabilities N steps ahead via matrix exponentiation.

    Args:
        current_regime: The current regime label.
        trans_matrix:   4×4 transition matrix from :func:`build_transition_matrix`.
        horizon:        Number of months to project forward.

    Returns:
        Series indexed by regime name, values are probabilities summing to 1.
    """
    core = ALL_REGIMES[:4]
    if trans_matrix is None or trans_matrix.empty:
        return pd.Series(0.25, index=core)

    tmat = trans_matrix.reindex(index=core, columns=core, fill_value=0.25).values
    tmat_n = np.linalg.matrix_power(tmat, horizon)

    if current_regime in core:
        idx = core.index(current_regime)
        probs = tmat_n[idx]
    else:
        probs = tmat_n.mean(axis=0)

    probs = probs / probs.sum()
    return pd.Series(probs, index=core)


def trend_assessment(
    panel: pd.DataFrame,
    indicator_cols: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """Assess the directional trend of macro indicators to provide a
    naive signal overlay.

    Args:
        panel:         Monthly panel (most recent rows used for trend).
        indicator_cols: Dict ``{column_name: bullish_threshold}``.
                        Defaults use CPI YoY and unemployment diff.

    Returns:
        Dict of indicator name → z-score or trend score.
    """
    if indicator_cols is None:
        indicator_cols = {"us_cpi_yoy": 0.0, "us_unemployment_diff": 0.0}

    if panel is None or panel.empty:
        return {}

    scores: Dict[str, float] = {}
    for col, threshold in indicator_cols.items():
        if col not in panel.columns:
            continue
        recent = panel[col].dropna().tail(6)
        if len(recent) < 3:
            continue
        mean_val = float(recent.mean())
        std_val = float(recent.std(ddof=1)) if len(recent) >= 3 else 1.0
        if std_val == 0:
            std_val = 1.0
        scores[col] = round((mean_val - threshold) / std_val, 3)
    return scores


def indicator_adjustment(
    probs: pd.Series,
    indicators: Dict[str, float],
) -> pd.Series:
    """Adjust transition-matrix probabilities using a simple indicator overlay.

    Rules (heuristic):
    - Rising CPI YoY (positive z-score) → boost overheat / stagflation.
    - Falling CPI YoY (negative z-score) → boost recovery / recession.
    - Rising unemployment → boost recession / stagflation.
    - Falling unemployment → boost recovery / overheat.

    The adjustment is capped at ±0.15 per regime to keep the transition
    matrix as the primary signal.
    """
    adjusted = probs.copy().astype(float)
    cpi_z = indicators.get("us_cpi_yoy", 0.0)
    unemp_z = indicators.get("us_unemployment_diff", 0.0)
    cap = 0.15

    if cpi_z > 0.5:
        delta = min(cpi_z * 0.05, cap)
        adjusted["overheat"] = min(adjusted.get("overheat", 0.0) + delta, 0.6)
        adjusted["stagflation"] = min(adjusted.get("stagflation", 0.0) + delta * 0.5, 0.5)
        adjusted["recovery"] = max(adjusted.get("recovery", 0.0) - delta * 0.5, 0.05)
    elif cpi_z < -0.5:
        delta = min(abs(cpi_z) * 0.05, cap)
        adjusted["recovery"] = min(adjusted.get("recovery", 0.0) + delta, 0.6)
        adjusted["recession"] = min(adjusted.get("recession", 0.0) + delta * 0.3, 0.5)
        adjusted["overheat"] = max(adjusted.get("overheat", 0.0) - delta * 0.5, 0.05)

    if unemp_z > 0.5:
        delta = min(unemp_z * 0.05, cap)
        adjusted["recession"] = min(adjusted.get("recession", 0.0) + delta, 0.5)
        adjusted["stagflation"] = min(adjusted.get("stagflation", 0.0) + delta * 0.3, 0.5)
        adjusted["recovery"] = max(adjusted.get("recovery", 0.0) - delta * 0.3, 0.05)
    elif unemp_z < -0.5:
        delta = min(abs(unemp_z) * 0.05, cap)
        adjusted["recovery"] = min(adjusted.get("recovery", 0.0) + delta, 0.6)
        adjusted["overheat"] = min(adjusted.get("overheat", 0.0) + delta * 0.3, 0.5)

    total = adjusted.sum()
    if total > 0:
        adjusted = adjusted / total
    return adjusted


class MacroRegimePredictor:
    """Predict economic regimes 1–3 months ahead.

    Combines a historical transition matrix with a heuristic indicator
    overlay to produce regime probability forecasts.

    Args:
        config: :class:`MacroPredictorConfig`.
    """

    def __init__(self, config: Optional[MacroPredictorConfig] = None):
        self.config = config or MacroPredictorConfig()
        self.trans_matrix: Optional[pd.DataFrame] = None

    def fit(self, labelled: pd.DataFrame, regime_col: str = "regime") -> MacroRegimePredictor:
        """Estimate the transition matrix from historical regime labels.

        Args:
            labelled:   Monthly panel with a regime column.
            regime_col: Name of the regime column.

        Returns:
            self (for chaining).
        """
        recent = labelled.tail(self.config.lookback_months) if len(labelled) > self.config.lookback_months else labelled
        self.trans_matrix = build_transition_matrix(recent, regime_col=regime_col)
        logger.info(
            f"MacroRegimePredictor: fitted transition matrix "
            f"({len(recent)} obs, {self.config.lookback_months} lookback)."
        )
        return self

    def predict(
        self,
        panel: pd.DataFrame,
        regime_col: str = "regime",
        indicator_cols: Optional[Dict[str, float]] = None,
    ) -> MacroPrediction:
        """Generate regime predictions for all configured horizons.

        Args:
            panel:         Monthly panel with a regime column.  The last row
                           is used as the reference (current) month.
            regime_col:    Name of the regime column.
            indicator_cols: Passed to :func:`trend_assessment`.

        Returns:
            :class:`MacroPrediction` with probabilities and predicted regimes
            for each horizon.

        Raises:
            ValueError: If the panel has no regime column or < 2 rows.
        """
        if panel is None or panel.empty or regime_col not in panel.columns:
            raise ValueError("MacroRegimePredictor.predict: panel must contain a regime column.")

        if self.trans_matrix is None or self.trans_matrix.empty:
            self.fit(panel, regime_col=regime_col)

        ref_row = panel.iloc[-1]
        current_regime = str(ref_row.get(regime_col, "unknown"))
        if current_regime == "unknown" or current_regime not in ALL_REGIMES[:4]:
            current_regime = "recession"

        indicators = trend_assessment(panel.tail(12), indicator_cols=indicator_cols)

        prob_rows = []
        predicted_regimes: Dict[int, str] = {}
        confidence: Dict[int, float] = {}

        for h in self.config.horizons:
            raw = predict_regime_probs(current_regime, self.trans_matrix, horizon=h)
            adjusted = indicator_adjustment(raw, indicators)
            adj_dict = adjusted.to_dict()
            adj_dict["horizon"] = h
            prob_rows.append(adj_dict)

            best = adjusted.idxmax()
            predicted_regimes[h] = best
            confidence[h] = round(float(adjusted[best]), 4)

        probs_df = pd.DataFrame(prob_rows).set_index("horizon")
        probs_df.index.name = "horizon_months"

        return MacroPrediction(
            timestamp=pd.Timestamp(panel.index[-1]),
            current_regime=current_regime,
            probabilities=probs_df,
            predicted_regimes=predicted_regimes,
            transition_matrix=self.trans_matrix,
            confidence=confidence,
            indicators=indicators,
        )

    def predict_rolling(
        self,
        panel: pd.DataFrame,
        regime_col: str = "regime",
        min_train_months: int = 24,
    ) -> pd.DataFrame:
        """Rolling out-of-sample prediction for backtesting the predictor.

        At each month t (starting from min_train_months), fit on data up to
        t-1 and predict for t.  Returns a DataFrame of predicted vs actual.

        Args:
            panel:            Monthly panel with regime labels.
            regime_col:       Regime column name.
            min_train_months: Minimum training window before first prediction.

        Returns:
            DataFrame indexed by prediction date with columns:
            ``predicted_regime``, ``actual_regime``, ``correct``,
            ``confidence``, and per-regime probability columns.
        """
        if panel is None or panel.empty or regime_col not in panel.columns:
            return pd.DataFrame()

        records = []
        for i in range(min_train_months, len(panel)):
            train = panel.iloc[:i]
            pred_row = panel.iloc[i : i + 1]
            predictor = MacroRegimePredictor(self.config)
            predictor.fit(train, regime_col=regime_col)
            try:
                pred = predictor.predict(pred_row, regime_col=regime_col)
                actual = str(pred_row[regime_col].iloc[0]) if regime_col in pred_row.columns else "unknown"
                rec = {
                    "date": pred_row.index[0],
                    "predicted_regime": pred.predicted_regimes.get(1, "unknown"),
                    "actual_regime": actual,
                    "correct": pred.predicted_regimes.get(1, "unknown") == actual,
                    "confidence": pred.confidence.get(1, 0.0),
                }
                for r in ALL_REGIMES[:4]:
                    rec[f"prob_{r}"] = pred.probabilities.loc[1, r] if 1 in pred.probabilities.index else 0.0
                records.append(rec)
            except Exception as exc:
                logger.warning(f"predict_rolling: skipped {pred_row.index[0]} — {exc}")

        if not records:
            return pd.DataFrame()
        result = pd.DataFrame(records).set_index("date")
        accuracy = result["correct"].mean()
        logger.info(f"MacroRegimePredictor rolling accuracy: {accuracy:.2%} ({result['correct'].sum()}/{len(result)})")
        return result


__all__ = [
    "ALL_REGIMES",
    "MacroPredictorConfig",
    "MacroPrediction",
    "MacroRegimePredictor",
    "build_transition_matrix",
    "predict_regime_probs",
    "trend_assessment",
    "indicator_adjustment",
]
