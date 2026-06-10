"""
uncertainty/action_uncertainty.py
----------------------------------
Beta-distribution model for manipulation success rate (SR).

Models the prior belief that a robot can successfully push/manipulate an obstacle.
Supports online Bayesian updating and cooperative belief sharing across a fleet.

Paper reference: NAMOUnc (Paper 1), Section IV-A — Action Uncertainty.
"""

import math
from typing import Optional


class ManipulationBeliefModel:
    """
    Bayesian Beta-distribution tracker for manipulation success rate.

    Beta(alpha, beta) is the conjugate prior for a Bernoulli success rate.
    The mean SR = alpha / (alpha + beta).

    Default prior Beta(9, 1) encodes a 90% SR based on empirical trials in
    controlled lab settings (NAMOUnc paper Table I).  Use Beta(2, 2) for a
    flat/uncertain prior when deploying to a novel environment.

    Parameters
    ----------
    alpha : float
        Prior successes + 1 (shape parameter).
    beta  : float
        Prior failures  + 1 (shape parameter).
    obstacle_type : str, optional
        Label for the obstacle category (e.g. "light_box", "heavy_crate").
        Used for logging and type-specific fleet broadcasting.
    """

    def __init__(
        self,
        alpha: float = 9.0,
        beta: float = 1.0,
        obstacle_type: str = "generic",
    ):
        if alpha <= 0 or beta <= 0:
            raise ValueError("alpha and beta must be positive.")
        self._alpha0 = alpha          # store initial prior for merge arithmetic
        self._beta0 = beta
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.obstacle_type = obstacle_type
        self._n_obs = 0               # number of observed manipulation attempts

    # ------------------------------------------------------------------
    # Core Bayesian update
    # ------------------------------------------------------------------

    def observe(self, success: bool) -> None:
        """
        Online update after a single manipulation attempt.

        Call this immediately after the robot's push/lift action resolves.
        success=True  → obstacle moved as intended.
        success=False → obstacle did not move (or moved incorrectly).
        """
        if success:
            self.alpha += 1.0
        else:
            self.beta += 1.0
        self._n_obs += 1

    # ------------------------------------------------------------------
    # Interval queries
    # ------------------------------------------------------------------

    def mean(self) -> float:
        """Expected (mean) success rate."""
        return self.alpha / (self.alpha + self.beta)

    def variance(self) -> float:
        """Variance of the Beta distribution."""
        n = self.alpha + self.beta
        return (self.alpha * self.beta) / (n * n * (n + 1.0))

    def std(self) -> float:
        return math.sqrt(self.variance())

    def success_rate_interval(self, n_sigma: float = 2.0) -> tuple[float, float]:
        """
        Returns [p_lo, p_hi] — the manipulation SR confidence interval.

        Uses the normal approximation to the Beta distribution, which is
        accurate when alpha + beta > 10 (always true after a few trials).

        n_sigma=2 gives approximately a 95% confidence interval.
        """
        mu = self.mean()
        sigma = self.std()
        p_lo = max(0.01, mu - n_sigma * sigma)   # clamp to (0,1)
        p_hi = min(1.00, mu + n_sigma * sigma)
        return p_lo, p_hi

    def interval_width(self, n_sigma: float = 2.0) -> float:
        """Width of the SR confidence interval — measures current uncertainty."""
        lo, hi = self.success_rate_interval(n_sigma)
        return hi - lo

    # ------------------------------------------------------------------
    # Cooperative belief sharing
    # ------------------------------------------------------------------

    def merge(self, other: "ManipulationBeliefModel") -> None:
        """
        Incorporate evidence observed by another robot.

        Adds only the *incremental* observations from `other` (i.e. excluding
        the shared prior) to avoid double-counting the prior.

        Phase 5 — Cooperative Belief Sharing.
        """
        if other.obstacle_type != self.obstacle_type:
            # Only merge evidence for the same obstacle category
            return
        incremental_alpha = other.alpha - other._alpha0
        incremental_beta  = other.beta  - other._beta0
        self.alpha += max(0.0, incremental_alpha)
        self.beta  += max(0.0, incremental_beta)
        self._n_obs += other._n_obs

    def to_dict(self) -> dict:
        """Serialise for network broadcasting."""
        return {
            "alpha": self.alpha,
            "beta": self.beta,
            "alpha0": self._alpha0,
            "beta0": self._beta0,
            "obstacle_type": self.obstacle_type,
            "n_obs": self._n_obs,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ManipulationBeliefModel":
        """Deserialise a broadcast message."""
        m = cls(alpha=d["alpha0"], beta=d["beta0"], obstacle_type=d["obstacle_type"])
        m.alpha = d["alpha"]
        m.beta = d["beta"]
        m._n_obs = d["n_obs"]
        return m

    def __repr__(self) -> str:
        lo, hi = self.success_rate_interval()
        return (
            f"ManipulationBeliefModel(type={self.obstacle_type!r}, "
            f"SR={self.mean():.3f}, interval=[{lo:.3f},{hi:.3f}], "
            f"n_obs={self._n_obs})"
        )
