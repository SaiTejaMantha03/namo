"""
uncertainty/bypass_model.py
----------------------------
Gaussian Linear Regressor for trajectory cost intervals.

Extracts three geometric features from a planned A* path and predicts
an expected cost with uncertainty bounds [mu - sigma, mu + sigma].

Features
--------
F_l  : path length (number of steps)
F_s  : smoothness (mean heading change per step, in radians)
F_v  : direction variance (std-dev of heading angles along path)

Paper reference: NAMOUnc (Paper 1), Section IV-B — Bypass Time Estimation.
"""

import math
import numpy as np
from typing import Optional


class TrajectoryRegressionModel:
    """
    Analytically-parameterised Gaussian Linear Regressor for A* path cost.

    The weights below are set from the NAMOUnc paper's regression results
    (Table II). They can optionally be trained on recorded trajectories by
    calling `fit(paths, observed_costs)`.

    Parameters
    ----------
    w_length  : mean cost contribution per step.
    w_smooth  : mean cost contribution per unit of heading change (radians).
    w_var     : mean cost contribution per unit of direction variance.
    sig_len   : std-dev contribution per step.
    sig_turn  : std-dev contribution per 90° turn.
    """

    def __init__(
        self,
        w_length: float = 1.0,
        w_smooth: float = 0.5,
        w_var: float = 0.3,
        sig_len: float = 0.05,
        sig_turn: float = 0.20,
    ):
        self.w_length = w_length
        self.w_smooth = w_smooth
        self.w_var = w_var
        self.sig_len = sig_len
        self.sig_turn = sig_turn

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    @staticmethod
    def extract_features(path: list[tuple]) -> dict:
        """
        Extract geometric features from a planned path.

        Returns dict with keys: length, num_turns, smoothness, direction_var.
        """
        if len(path) < 2:
            return {"length": 0, "num_turns": 0, "smoothness": 0.0, "direction_var": 0.0}

        headings = []
        num_turns = 0
        for i in range(1, len(path)):
            dx = path[i][0] - path[i - 1][0]
            dy = path[i][1] - path[i - 1][1]
            headings.append(math.atan2(dy, dx))

        for i in range(1, len(headings)):
            delta = abs(headings[i] - headings[i - 1])
            if delta > 1e-6:
                num_turns += 1

        smoothness = float(np.mean(np.abs(np.diff(headings)))) if len(headings) > 1 else 0.0
        direction_var = float(np.std(headings)) if headings else 0.0

        return {
            "length": len(path) - 1,
            "num_turns": num_turns,
            "smoothness": smoothness,
            "direction_var": direction_var,
        }

    # ------------------------------------------------------------------
    # Interval prediction
    # ------------------------------------------------------------------

    def predict_interval(
        self, path: list[tuple], base_cost: Optional[float] = None
    ) -> tuple[float, float]:
        """
        Predict [mu - sigma, mu + sigma] for the given planned path.

        Parameters
        ----------
        path      : list of (col, row) cells from A*.
        base_cost : A* g-score (includes risk-map penalty). If None, uses
                    the pure geometric estimate from the regressor.

        Returns
        -------
        (cost_lo, cost_hi) — the 1-sigma cost interval.
        """
        if not path:
            return (float("inf"), float("inf"))

        if base_cost is not None and base_cost == float("inf"):
            return (float("inf"), float("inf"))

        feats = self.extract_features(path)
        length = feats["length"]
        num_turns = feats["num_turns"]
        smoothness = feats["smoothness"]
        direction_var = feats["direction_var"]

        # Mean prediction: linear combination of features
        if base_cost is not None:
            # Use the A* cost as the base; regressor adds turn and direction cost
            mu = base_cost + (num_turns * self.w_smooth) + (direction_var * self.w_var)
        else:
            mu = (length * self.w_length) + (num_turns * self.w_smooth) + (direction_var * self.w_var)

        # Standard deviation: grows with path length and number of turns
        sigma = (length * self.sig_len) + (num_turns * self.sig_turn)

        cost_lo = max(0.0, mu - sigma)
        cost_hi = mu + sigma
        return (cost_lo, cost_hi)

    # ------------------------------------------------------------------
    # Optional: fit weights from observations
    # ------------------------------------------------------------------

    def fit(self, paths: list[list[tuple]], observed_costs: list[float]) -> None:
        """
        Update regression weights using ordinary least squares on recorded data.

        Parameters
        ----------
        paths          : list of recorded A* paths.
        observed_costs : list of actually observed navigation times (steps taken).
        """
        if len(paths) != len(observed_costs) or len(paths) < 3:
            return  # not enough data

        X = []
        for path in paths:
            f = self.extract_features(path)
            X.append([f["length"], f["smoothness"], f["direction_var"]])

        X = np.array(X, dtype=float)
        y = np.array(observed_costs, dtype=float)

        # Add bias column
        X_b = np.column_stack([np.ones(len(X)), X])
        try:
            coeffs, _, _, _ = np.linalg.lstsq(X_b, y, rcond=None)
        except np.linalg.LinAlgError:
            return

        # Update weights (skip bias term)
        self.w_length  = float(coeffs[1])
        self.w_smooth  = float(coeffs[2])
        self.w_var     = float(coeffs[3])

    def __repr__(self) -> str:
        return (
            f"TrajectoryRegressionModel("
            f"w_length={self.w_length}, w_smooth={self.w_smooth}, "
            f"w_var={self.w_var}, sig_len={self.sig_len}, sig_turn={self.sig_turn})"
        )
