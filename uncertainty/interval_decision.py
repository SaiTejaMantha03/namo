"""
uncertainty/interval_decision.py
----------------------------------
Laplace (optimism under uncertainty) criterion for interval-based decision making.

This is the theoretical heart of NAMOUnc: instead of comparing two scalar costs
directly, we compare the *midpoints* of two cost intervals. This is the Laplace
criterion — equal weighting of the best and worst case — which is equivalent to
computing the expected cost under a uniform distribution over the interval.

Paper reference: NAMOUnc (Paper 1), Section IV-C — Decision Making.
"""


def laplace_criterion(
    interval_a: tuple[float, float],
    interval_b: tuple[float, float],
) -> int:
    """
    Compare two cost intervals using the Laplace criterion: U = (max + min) / 2.

    Parameters
    ----------
    interval_a : (lo_a, hi_a) — cost interval for option A.
    interval_b : (lo_b, hi_b) — cost interval for option B.

    Returns
    -------
    0 if option A is preferred (U_a <= U_b).
    1 if option B is preferred (U_b <  U_a).

    Notes
    -----
    When one interval contains inf, the other is always preferred.
    If both contain inf (both paths truly impossible), returns 0 by convention.
    """
    inf = float("inf")

    lo_a, hi_a = interval_a
    lo_b, hi_b = interval_b

    a_inf = (lo_a == inf or hi_a == inf)
    b_inf = (lo_b == inf or hi_b == inf)

    if a_inf and b_inf:
        return 0   # both impossible — no preference
    if a_inf:
        return 1   # A is impossible, B is preferred
    if b_inf:
        return 0   # B is impossible, A is preferred

    U_a = (lo_a + hi_a) / 2.0
    U_b = (lo_b + hi_b) / 2.0

    return 0 if U_a <= U_b else 1


def choose_action(
    bypass_interval: tuple[float, float],
    removal_interval: tuple[float, float],
) -> tuple[str, float, float]:
    """
    Convenience wrapper that returns the chosen action label and both U-values.

    Parameters
    ----------
    bypass_interval  : (lo, hi) cost interval for bypass action.
    removal_interval : (lo, hi) cost interval for removal action.

    Returns
    -------
    (decision, U_bypass, U_removal) where decision is "BYPASS" or "REMOVE".
    """
    inf = float("inf")

    def U(interval):
        lo, hi = interval
        if lo == inf or hi == inf:
            return inf
        return (lo + hi) / 2.0

    U_bypass  = U(bypass_interval)
    U_removal = U(removal_interval)

    decision = "REMOVE" if U_removal < U_bypass else "BYPASS"
    return decision, U_bypass, U_removal
