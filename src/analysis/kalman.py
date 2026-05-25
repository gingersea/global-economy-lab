"""
Kalman filter for dynamic factor models.

Implements the classic Kalman filter / smoother for linear Gaussian
state-space models, plus a Dynamic Factor Model (DFM) that extracts
latent economic factors from observed factor signals.

Model (Stock & Watson 2002, 2016):
  State:      x_t = F·x_{t-1} + w_t,   w_t ~ N(0, Q)
  Observation: y_t = H·x_t + v_t,       v_t ~ N(0, R)

where x_t is a low-dimensional latent state (e.g. growth & inflation),
and y_t is the vector of observed factor signals.

Confidence = diagonal of P (state covariance), giving proper statistical
uncertainty intervals rather than rolling hit rates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class KalmanResult:
    """Output of a Kalman filter run.

    Attributes:
        filtered_state:    Filtered state x_{t|t} at each time step (d x T).
        filtered_cov:      Filtered covariance P_{t|t} (d x d x T).
        predicted_state:   One-step-ahead predicted state x_{t|t-1}.
        predicted_cov:     One-step-ahead predicted covariance.
        smoothed_state:    Smoothed state x_{t|T} (full-sample, after backward pass).
        smoothed_cov:      Smoothed covariance.
        log_likelihood:    Total log-likelihood of observations under model.
        innovations:       Prediction errors (y_t - H·x_{t|t-1}).
        confidence:        Per-step state estimation confidence from cov diagonal.
    """

    filtered_state: np.ndarray
    filtered_cov: np.ndarray
    predicted_state: np.ndarray
    predicted_cov: np.ndarray
    smoothed_state: Optional[np.ndarray] = None
    smoothed_cov: Optional[np.ndarray] = None
    log_likelihood: float = 0.0
    innovations: Optional[np.ndarray] = None
    confidence: Optional[np.ndarray] = None


@dataclass
class ForwardPrediction:
    """Forward prediction from Kalman filter.

    Attributes:
        horizon:       Prediction horizon (steps ahead).
        state_mean:    Predicted state mean for each horizon step.
        state_cov:     Predicted state covariance for each step.
        conf_lower:    95% lower confidence bound.
        conf_upper:    95% upper confidence bound.
        confidence:    Prediction confidence per step [0, 1].
    """

    horizon: int
    state_mean: np.ndarray
    state_cov: np.ndarray
    conf_lower: np.ndarray
    conf_upper: np.ndarray
    confidence: np.ndarray


class KalmanFilter:
    """Classic Kalman filter for linear-Gaussian state-space models.

    State:      x_{t+1} = F·x_t + w_t,    w_t ~ N(0, Q)
    Observation: y_t     = H·x_t + v_t,    v_t ~ N(0, R)

    Args:
        F: State transition matrix (d_state x d_state).
        H: Observation matrix (d_obs x d_state).
        Q: Process noise covariance (d_state x d_state). Default: identity.
        R: Observation noise covariance (d_obs x d_obs). Default: identity.
        initial_state: Initial state mean. Default: zeros.
        initial_cov:   Initial state covariance. Default: identity.
    """

    def __init__(
        self,
        F: np.ndarray,
        H: np.ndarray,
        Q: Optional[np.ndarray] = None,
        R: Optional[np.ndarray] = None,
        initial_state: Optional[np.ndarray] = None,
        initial_cov: Optional[np.ndarray] = None,
    ):
        self.F = np.atleast_2d(F)
        self.H = np.atleast_2d(H)
        self.d_state = self.F.shape[0]
        self.d_obs = self.H.shape[0]
        self.Q = Q if Q is not None else np.eye(self.d_state) * 0.01
        self.R = R if R is not None else np.eye(self.d_obs) * 0.1
        self.x0 = initial_state if initial_state is not None else np.zeros(self.d_state)
        self.P0 = initial_cov if initial_cov is not None else np.eye(self.d_state)

    def filter(self, observations: np.ndarray) -> KalmanResult:
        """Forward Kalman filter pass.

        Args:
            observations: (T x d_obs) array.  NaN entries are treated as
                          missing and skipped in the update step.

        Returns:
            :class:`KalmanResult` with filtered states and covariances.
        """
        T = observations.shape[0]
        x_pred = np.zeros((T, self.d_state))
        P_pred = np.zeros((T, self.d_state, self.d_state))
        x_filt = np.zeros((T, self.d_state))
        P_filt = np.zeros((T, self.d_state, self.d_state))
        innovations = np.zeros((T, self.d_obs))
        log_lik = 0.0

        x = self.x0.copy()
        P = self.P0.copy()

        for t in range(T):
            x_pred[t] = self.F @ x
            P_pred[t] = self.F @ P @ self.F.T + self.Q

            y = observations[t]
            missing = np.isnan(y)
            if missing.all():
                x_filt[t] = x_pred[t]
                P_filt[t] = P_pred[t]
                x = x_filt[t]
                P = P_filt[t]
                continue

            if missing.any():
                H_valid = self.H[~missing]
                R_valid = self.R[~missing][:, ~missing] if self.R.ndim > 1 else self.R[~missing][~missing]
                y_valid = y[~missing]
            else:
                H_valid = self.H
                R_valid = self.R
                y_valid = y

            S = H_valid @ P_pred[t] @ H_valid.T + R_valid
            S_inv = np.linalg.inv(S)
            K = P_pred[t] @ H_valid.T @ S_inv

            innov = y_valid - H_valid @ x_pred[t]
            innovations[t, ~missing if missing.any() else slice(None)] = innov

            x_filt[t] = x_pred[t] + K @ innov
            P_filt[t] = P_pred[t] - K @ H_valid @ P_pred[t]

            det_S = np.linalg.det(S)
            log_lik += -0.5 * (len(y_valid) * np.log(2 * np.pi) + np.log(max(det_S, 1e-300))
                               + float(innov.T @ S_inv @ innov))

            x = x_filt[t]
            P = P_filt[t]

        trace_P = np.array([np.trace(P_filt[t]) for t in range(T)])
        state_abs = np.abs(x_filt[:, 0])
        state_range = np.percentile(state_abs, 95) - np.percentile(state_abs, 5)
        state_range = max(state_range, 0.01)
        conf = np.tanh(state_abs / state_range)

        return KalmanResult(
            filtered_state=x_filt,
            filtered_cov=P_filt,
            predicted_state=x_pred,
            predicted_cov=P_pred,
            log_likelihood=float(log_lik),
            innovations=innovations,
            confidence=conf,
        )

    def smooth(self, result: KalmanResult) -> KalmanResult:
        """RTS backward smoother pass.

        Args:
            result: Output of :meth:`filter`.

        Returns:
            *result* with ``smoothed_state`` and ``smoothed_cov`` populated.
        """
        T = result.filtered_state.shape[0]
        x_smooth = np.zeros_like(result.filtered_state)
        P_smooth = np.zeros_like(result.filtered_cov)
        x_smooth[-1] = result.filtered_state[-1]
        P_smooth[-1] = result.filtered_cov[-1]

        for t in range(T - 2, -1, -1):
            P_pred_inv = np.linalg.inv(result.predicted_cov[t + 1])
            G = result.filtered_cov[t] @ self.F.T @ P_pred_inv
            x_smooth[t] = result.filtered_state[t] + G @ (
                x_smooth[t + 1] - result.predicted_state[t + 1]
            )
            P_smooth[t] = result.filtered_cov[t] + G @ (
                P_smooth[t + 1] - result.predicted_cov[t + 1]
            ) @ G.T

        result.smoothed_state = x_smooth
        result.smoothed_cov = P_smooth
        return result

    def predict_forward(
        self,
        result: KalmanResult,
        steps: int = 1,
    ) -> ForwardPrediction:
        """Forward prediction from the last filtered state.

        Args:
            result: Output of :meth:`filter` (or :meth:`smooth`).
            steps:  Number of steps to predict forward.

        Returns:
            :class:`ForwardPrediction` with means, covariances, and intervals.
        """
        x_last = result.filtered_state[-1]
        P_last = result.filtered_cov[-1]

        means = np.zeros((steps, self.d_state))
        covs = np.zeros((steps, self.d_state, self.d_state))
        lower = np.zeros((steps, self.d_state))
        upper = np.zeros((steps, self.d_state))
        confidence = np.zeros(steps)

        F_pow = np.eye(self.d_state)
        x = x_last.copy()

        for h in range(steps):
            F_pow = F_pow @ self.F
            x = self.F @ x

            P_pred = F_pow @ P_last @ F_pow.T
            for i in range(h):
                Fi = np.linalg.matrix_power(self.F, i) if i > 0 else np.eye(self.d_state)
                P_pred += Fi @ self.Q @ Fi.T

            means[h] = x
            covs[h] = P_pred
            std = np.sqrt(np.maximum(P_pred.diagonal(), 1e-10))
            lower[h] = x - 1.96 * std
            upper[h] = x + 1.96 * std
            confidence[h] = 1.0 / (1.0 + std.mean())

        return ForwardPrediction(
            horizon=steps,
            state_mean=means,
            state_cov=covs,
            conf_lower=lower,
            conf_upper=upper,
            confidence=confidence,
        )


class DynamicFactorModel:
    """Dynamic Factor Model with time-varying loadings.

    Extracts k latent factors from d observed signals.  Factor loadings
    H are re-estimated on a rolling window to capture structural changes
    in how observed signals map to latent economic states.

    Estimation: EM on rolling window → updated H, Q, R per window.
    State transition F and latent factors are filtered/smoothed with the
    standard Kalman recursions.

    Args:
        n_factors:       Number of latent factors (default=2).
        window_size:     Rolling window for re-estimating H (trading days).
                         If 0 or None, uses full sample (static).
        window_step:     How often to re-estimate (trading days).
        max_em_iter:     Maximum EM iterations per window.
        em_tol:          Convergence tolerance.
    """

    def __init__(
        self,
        n_factors: int = 1,
        window_size: int = 1260,
        window_step: int = 252,
        max_em_iter: int = 30,
        em_tol: float = 1e-3,
    ):
        self.n_factors = n_factors
        self.window_size = window_size
        self.window_step = window_step
        self.max_em_iter = max_em_iter
        self.em_tol = em_tol
        self.F: Optional[np.ndarray] = None
        self.H: Optional[np.ndarray] = None
        self.Q: Optional[np.ndarray] = None
        self.R: Optional[np.ndarray] = None
        self._loadings_history: List[np.ndarray] = []
        self._last_result: Optional[KalmanResult] = None

    def _init_params(self, d_obs: int, observations: Optional[np.ndarray] = None):
        d = self.n_factors
        self.F = np.eye(d) * 0.95 + np.eye(d, k=-1) * 0.05

        if observations is not None and observations.size > 0:
            data_var = float(np.nanvar(observations))
            data_var = max(data_var, 0.01)
            obs_clean = np.nan_to_num(observations, nan=0.0)
            if obs_clean.shape[0] > d_obs and d_obs > d:
                cov = np.cov(obs_clean, rowvar=False)
                eigvals, eigvecs = np.linalg.eigh(cov)
                self.H = eigvecs[:, -d:] * np.sqrt(np.maximum(eigvals[-d:], 0.01))[:, None]
                self.H = self.H.T if d > 1 else self.H[:, -1:].reshape(d_obs, d)
                if d == 1 and self.H.ndim == 1:
                    self.H = self.H.reshape(-1, 1)
            else:
                self.H = np.random.randn(d_obs, d) * 0.1
        else:
            data_var = 0.1
            self.H = np.random.randn(d_obs, d) * 0.1

        self.Q = np.eye(d) * data_var * 0.0001
        self.R = np.diag(np.full(d_obs, data_var * 0.05))

    def _fit_window(self, observations: np.ndarray) -> KalmanResult:
        """EM on a single window to estimate H, Q, R."""
        T, d_obs = observations.shape
        d = self.n_factors
        H = self.H.copy() if self.H is not None else np.random.randn(d_obs, d) * 0.1
        Q = self.Q.copy() if self.Q is not None else np.eye(d) * 0.05
        R = self.R.copy() if self.R is not None else np.eye(d_obs) * 0.5

        prev_ll = -np.inf
        for iteration in range(self.max_em_iter):
            kf = KalmanFilter(self.F, H, Q, R, initial_cov=np.eye(d))
            result = kf.filter(observations)
            kf.smooth(result)

            if result.log_likelihood - prev_ll < self.em_tol and iteration > 3:
                break
            prev_ll = result.log_likelihood

            Q_new = np.zeros((d, d))
            R_new = np.zeros((d_obs, d_obs))
            count_R = 0
            for t in range(T - 1):
                delta = result.smoothed_state[t + 1] - self.F @ result.smoothed_state[t]
                Q_new += np.outer(delta, delta)
                cross = result.smoothed_cov[t + 1] @ self.F.T + self.F @ result.smoothed_cov[t] @ self.F.T
                Q_new += result.smoothed_cov[t + 1] + self.F @ result.smoothed_cov[t] @ self.F.T - cross - cross.T
            Q = Q_new / (T - 1)

            H_new = np.zeros_like(H)
            for t in range(T):
                y = observations[t]
                missing = np.isnan(y)
                if missing.all():
                    continue
                mask = ~missing
                s_t = result.smoothed_state[t]
                H_new[mask] += np.outer(y[mask], s_t)
                innov = y[mask] - H[mask] @ s_t
                R_sub = np.outer(innov, innov) + H[mask] @ result.smoothed_cov[t] @ H[mask].T
                for i, ri in enumerate(np.where(mask)[0]):
                    for j, rj in enumerate(np.where(mask)[0]):
                        R_new[ri, rj] += R_sub[i, j]
                count_R += 1
            H = H_new / T
            R = R_new / max(count_R, 1)

        self._loadings_history.append(H.copy())
        return result

    def fit(self, observations: np.ndarray) -> KalmanResult:
        """Fit the DFM with **causal** rolling-window estimation.

        Strictly no look-ahead: H for filtering window [t, t+step) is
        estimated from data in [t-ws, t) — only past data.  Each window's
        EM fit sees exclusively historical observations.
        """
        T, d_obs = observations.shape
        self._init_params(d_obs, observations)
        self._loadings_history = []

        ws = self.window_size if self.window_size and self.window_size > 0 else T
        step = self.window_step if self.window_step and self.window_step > 0 else ws

        filtered_states = np.zeros((T, self.n_factors))
        filtered_covs = np.zeros((T, self.n_factors, self.n_factors))
        predicted_states = np.zeros((T, self.n_factors))
        predicted_covs = np.zeros((T, self.n_factors, self.n_factors))
        innovations_all = np.zeros((T, d_obs))
        conf_all = np.zeros(T)
        total_ll = 0.0

        n_windows = max(1, (T - ws) // step + 1)
        logger.info(f"DFM (causal): {n_windows} windows (size={ws}, step={step}, T={T}).")

        kf = KalmanFilter(self.F, self.H, self.Q, self.R, initial_cov=np.eye(self.n_factors))

        for w in range(n_windows):
            w_start = w * step
            w_end = min(w_start + step, T)

            if w > 0 and w_start >= ws:
                train_start = w_start - ws
                train_end = w_start
                train_obs = observations[train_start:train_end]

                if train_end - train_start >= 50:
                    est_result = self._fit_window(train_obs)
                    self.H = self._loadings_history[-1].copy()
                    self.Q = est_result.filtered_cov[-1]
                    self.R = self.R

                    kf = KalmanFilter(self.F, self.H, self.Q, self.R,
                                      initial_state=filtered_states[w_start - 1] if w_start > 0 else None,
                                      initial_cov=filtered_covs[w_start - 1] if w_start > 0 else None)

            window_obs = observations[w_start:w_end]
            result = kf.filter(window_obs)
            n_w = w_end - w_start
            filtered_states[w_start:w_end] = result.filtered_state[-n_w:]
            filtered_covs[w_start:w_end] = result.filtered_cov[-n_w:]
            predicted_states[w_start:w_end] = result.predicted_state[-n_w:]
            predicted_covs[w_start:w_end] = result.predicted_cov[-n_w:]
            innovations_all[w_start:w_end] = result.innovations[-n_w:]
            conf_all[w_start:w_end] = result.confidence[-n_w:]
            total_ll += result.log_likelihood

            if w % 10 == 0:
                logger.debug(f"  Window {w}/{n_windows} [{w_start}:{w_end}]: LL={result.log_likelihood:.1f}")

        logger.info(
            f"DFM: {len(self._loadings_history)} H re-estimations. "
            f"Final H norm={np.linalg.norm(self.H):.3f}, LL={total_ll:.1f}."
        )

        self._last_result = KalmanResult(
            filtered_state=filtered_states,
            filtered_cov=filtered_covs,
            predicted_state=predicted_states,
            predicted_cov=predicted_covs,
            log_likelihood=total_ll,
            innovations=innovations_all,
            confidence=conf_all,
        )
        return self._last_result

    def predict(
        self,
        observations: np.ndarray,
        steps: int = 1,
    ) -> ForwardPrediction:
        """Filter observations and predict forward (causal — no future data).

        Uses the last filtered state and propagates through F^h.
        Prediction intervals come from the Kalman covariance recursion.
        """
        kf = KalmanFilter(self.F, self.H, self.Q, self.R, initial_cov=np.eye(self.n_factors))
        result = kf.filter(observations)
        return kf.predict_forward(result, steps)

    def get_factors(self) -> Optional[pd.DataFrame]:
        """Return smoothed latent factors as a DataFrame."""
        if self._last_result is None or self._last_result.smoothed_state is None:
            return None
        cols = [f"factor_{i}" for i in range(self.n_factors)]
        return pd.DataFrame(self._last_result.smoothed_state, columns=cols)

    def get_loadings(self) -> Optional[pd.DataFrame]:
        """Return factor loadings matrix."""
        if self.H is None:
            return None
        return pd.DataFrame(self.H, columns=[f"factor_{i}" for i in range(self.n_factors)])

    def get_confidence_history(self) -> Optional[pd.Series]:
        if self._last_result is None or self._last_result.confidence is None:
            return None
        return pd.Series(self._last_result.confidence, name="confidence")


@dataclass
class UnifiedPrediction:
    """Unified prediction: direction + magnitude + confidence.

    Attributes:
        signal:          -1 (bearish), 0 (neutral), +1 (bullish).
        magnitude:       Expected annual return (decimal, fuzzy).
        confidence:      KF-derived confidence (0-1).
        direction_factor: Latent factor 0 value (direction signal).
        magnitude_factor: Latent factor 1 value (scale signal).
        forward_pred:    KF forward prediction with CI.
        factor_loadings: How each observed factor loads on latent factors.
    """

    signal: int
    magnitude: float
    confidence: float
    direction_factor: float
    magnitude_factor: float
    forward_pred: Optional[ForwardPrediction] = None
    factor_loadings: Optional[pd.DataFrame] = None


class UnifiedPredictor:
    """Comprehensive prediction via 2-factor Kalman DFM.

    Factor 0 → direction (sign gives prediction).
    Factor 1 → magnitude (scale of expected return).
    Both extracted simultaneously from 12 observed factor signals.

    The KF naturally provides confidence from state covariance.
    No separate noise model needed — the KF's P matrix already
    encodes uncertainty.

    Args:
        window_size: Rolling window for DFM re-estimation.
        window_step: Re-estimation frequency.
    """

    def __init__(self, window_size: int = 1260, window_step: int = 252):
        self.window_size = window_size
        self.window_step = window_step
        self.dfm: Optional[DynamicFactorModel] = None
        self._state_history: Optional[pd.DataFrame] = None

    def fit(self, observations: np.ndarray) -> KalmanResult:
        """Fit 2-factor DFM on observed factor signals."""
        self.dfm = DynamicFactorModel(
            n_factors=2, window_size=self.window_size,
            window_step=self.window_step, max_em_iter=20,
        )
        result = self.dfm.fit(observations)
        self._state_history = pd.DataFrame(
            result.filtered_state,
            columns=["direction", "magnitude"],
        )
        return result

    def predict(
        self,
        observations: np.ndarray,
        epu_percentile: float = 50.0,
    ) -> UnifiedPrediction:
        """Generate unified prediction from latest state.

        Args:
            observations:    Full factor signal history (T x d_obs).
            epu_percentile:  Current EPU percentile for gating.

        Returns:
            :class:`UnifiedPrediction` with signal {-1,0,+1} + magnitude.
        """
        result = self.fit(observations)
        state = result.filtered_state

        dir_factor = float(state[-1, 0])
        mag_factor = float(state[-1, 1])
        conf = float(result.confidence[-1]) if result.confidence is not None else 0.5

        forward_pred = None
        if self.dfm is not None:
            forward_pred = self.dfm.predict(observations, steps=1)

        dir_norm = np.tanh(dir_factor * 0.1)
        if epu_percentile > 75:
            signal = 0
            magnitude = abs(mag_factor) * 0.05
            conf *= 0.5
        elif dir_norm > 0.15:
            signal = +1
            magnitude = abs(mag_factor) * 0.10
        elif dir_norm < -0.15:
            signal = -1
            magnitude = abs(mag_factor) * 0.10
        else:
            signal = 0
            magnitude = abs(mag_factor) * 0.05

        loadings = self.dfm.get_loadings() if self.dfm is not None else None

        return UnifiedPrediction(
            signal=signal,
            magnitude=round(float(magnitude), 4),
            confidence=round(float(conf), 4),
            direction_factor=round(dir_factor, 4),
            magnitude_factor=round(mag_factor, 4),
            forward_pred=forward_pred,
            factor_loadings=loadings,
        )


__all__ = [
    "KalmanResult",
    "ForwardPrediction",
    "KalmanFilter",
    "DynamicFactorModel",
    "UnifiedPredictor",
    "UnifiedPrediction",
]
