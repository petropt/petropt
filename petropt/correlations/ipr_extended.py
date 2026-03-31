"""Extended IPR correlations — Fetkovich, Rawlins-Schellhardt, PI-based.

Correlations:
    - Fetkovich (1973): IPR for solution-gas-drive wells
    - Rawlins and Schellhardt (1935): C&n backpressure equation for gas wells
    - Productivity Index: Straight-line IPR for undersaturated oil

References:
    Fetkovich, M.J., "The Isochronal Testing of Oil Wells," SPE 4529,
        1973.
    Rawlins, E.L. and Schellhardt, M.A., "Back-Pressure Data on Natural
        Gas Wells and Their Application to Production Practices," USBM
        Monograph 7, 1935.
"""

from __future__ import annotations

import math

import numpy as np

from petropt.correlations.pvt import _validate_positive


# ---------------------------------------------------------------------------
# Fetkovich (1973) — Oil well IPR
# ---------------------------------------------------------------------------

def fetkovich_ipr(
    pr: float,
    c: float,
    n: float,
    pwf: float | np.ndarray | None = None,
    num_points: int = 20,
) -> dict:
    """Fetkovich IPR for oil wells (1973).

    q = C * (Pr^2 - Pwf^2)^n

    This is a generalization of the backpressure equation applied to
    oil wells producing below the bubble point.

    Args:
        pr: Reservoir pressure in psi.
        c: Deliverability coefficient (well-specific).
        n: Deliverability exponent (0.5 to 1.0; 1.0 = laminar, 0.5 = turbulent).
        pwf: Flowing bottomhole pressure in psi. If None, generates full curve.
        num_points: Number of points for curve generation.

    Returns:
        Dict with 'qo', 'pwf', 'pr', 'qmax' (AOF).
    """
    _validate_positive(pr, "reservoir pressure")
    _validate_positive(c, "deliverability coefficient")
    if n < 0.5 or n > 1.0:
        raise ValueError(f"n must be between 0.5 and 1.0, got {n}")

    qmax = c * (pr**2) ** n  # AOF at Pwf = 0

    def _rate(p: float) -> float:
        if p < 0:
            raise ValueError(f"pwf must be non-negative, got {p}")
        if p >= pr:
            return 0.0
        return c * (pr**2 - p**2) ** n

    if pwf is not None:
        if isinstance(pwf, np.ndarray):
            qo = np.array([_rate(p) for p in pwf])
            return {"qo": qo, "pwf": pwf, "pr": pr, "qmax": round(qmax, 2)}
        else:
            return {"qo": _rate(float(pwf)), "pwf": float(pwf), "pr": pr, "qmax": round(qmax, 2)}
    else:
        pwf_array = np.linspace(0, pr, num_points + 1)
        qo_array = np.array([_rate(p) for p in pwf_array])
        return {"qo": qo_array, "pwf": pwf_array, "pr": pr, "qmax": round(qmax, 2)}


def fetkovich_from_tests(
    pr: float,
    pwf_tests: list[float],
    q_tests: list[float],
) -> dict:
    """Determine Fetkovich C and n from multi-rate well test data.

    Uses log-log regression on (Pr^2 - Pwf^2) vs q.

    Args:
        pr: Reservoir pressure in psi.
        pwf_tests: List of test flowing pressures in psi.
        q_tests: List of test flow rates.

    Returns:
        Dict with 'c', 'n', 'r_squared'.
    """
    _validate_positive(pr, "reservoir pressure")
    if len(pwf_tests) < 2 or len(q_tests) < 2:
        raise ValueError("Need at least 2 test points")
    if len(pwf_tests) != len(q_tests):
        raise ValueError("pwf_tests and q_tests must have same length")

    # Log-log regression: log(q) = log(C) + n * log(Pr^2 - Pwf^2)
    x = []
    y = []
    for pwf, q in zip(pwf_tests, q_tests):
        dp2 = pr**2 - pwf**2
        if dp2 > 0 and q > 0:
            x.append(math.log(dp2))
            y.append(math.log(q))

    if len(x) < 2:
        raise ValueError("Not enough valid test points for regression")

    x = np.array(x)
    y = np.array(y)

    # Linear regression
    n_pts = len(x)
    sx = np.sum(x)
    sy = np.sum(y)
    sxy = np.sum(x * y)
    sxx = np.sum(x * x)
    syy = np.sum(y * y)

    n = (n_pts * sxy - sx * sy) / (n_pts * sxx - sx**2)
    log_c = (sy - n * sx) / n_pts
    c = math.exp(log_c)

    # R-squared
    ss_res = np.sum((y - (log_c + n * x))**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r_sq = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # Clamp n to physical range
    n = max(0.5, min(n, 1.0))

    return {"c": round(c, 8), "n": round(n, 4), "r_squared": round(r_sq, 6)}


# ---------------------------------------------------------------------------
# Rawlins-Schellhardt (1935) — Gas well backpressure
# ---------------------------------------------------------------------------

def rawlins_schellhardt(
    pr: float,
    c: float,
    n: float,
    pwf: float | np.ndarray | None = None,
    num_points: int = 20,
) -> dict:
    """Gas well deliverability using Rawlins-Schellhardt C&n equation.

    q = C * (Pr^2 - Pwf^2)^n

    Same mathematical form as Fetkovich but traditionally used for gas wells.

    Args:
        pr: Reservoir pressure in psi.
        c: Deliverability coefficient.
        n: Deliverability exponent (0.5 to 1.0).
        pwf: Flowing pressure in psi. If None, generates full curve.
        num_points: Number of curve points.

    Returns:
        Dict with 'qg' (gas rate), 'pwf', 'pr', 'aof'.
    """
    result = fetkovich_ipr(pr=pr, c=c, n=n, pwf=pwf, num_points=num_points)
    # Rename for gas convention
    return {
        "qg": result["qo"],
        "pwf": result["pwf"],
        "pr": result["pr"],
        "aof": result["qmax"],
    }


# ---------------------------------------------------------------------------
# Productivity Index (PI) — Straight-line IPR
# ---------------------------------------------------------------------------

def pi_ipr(
    pr: float,
    pi: float,
    pwf: float | np.ndarray | None = None,
    num_points: int = 20,
) -> dict:
    """Straight-line IPR based on Productivity Index.

    q = PI * (Pr - Pwf)

    Used for undersaturated oil wells (above bubble point).

    Args:
        pr: Reservoir pressure in psi.
        pi: Productivity index in STB/day/psi.
        pwf: Flowing pressure in psi. If None, generates full curve.
        num_points: Number of curve points.

    Returns:
        Dict with 'qo', 'pwf', 'pr', 'qmax'.
    """
    _validate_positive(pr, "reservoir pressure")
    _validate_positive(pi, "productivity index")

    qmax = pi * pr

    def _rate(p: float) -> float:
        if p < 0:
            raise ValueError(f"pwf must be non-negative, got {p}")
        if p >= pr:
            return 0.0
        return pi * (pr - p)

    if pwf is not None:
        if isinstance(pwf, np.ndarray):
            qo = np.array([_rate(p) for p in pwf])
            return {"qo": qo, "pwf": pwf, "pr": pr, "qmax": round(qmax, 2)}
        else:
            return {"qo": _rate(float(pwf)), "pwf": float(pwf), "pr": pr, "qmax": round(qmax, 2)}
    else:
        pwf_array = np.linspace(0, pr, num_points + 1)
        qo_array = np.array([_rate(p) for p in pwf_array])
        return {"qo": qo_array, "pwf": pwf_array, "pr": pr, "qmax": round(qmax, 2)}


# ---------------------------------------------------------------------------
# Composite IPR — Vogel below Pb + PI above Pb
# ---------------------------------------------------------------------------

def composite_ipr(
    pr: float,
    pb: float,
    pi: float,
    pwf: float | np.ndarray | None = None,
    num_points: int = 20,
) -> dict:
    """Composite IPR: PI above bubble point + Vogel below.

    Above Pb: q = PI * (Pr - Pwf)
    Below Pb: q = qb + (qmax - qb) * [1 - 0.2*(Pwf/Pb) - 0.8*(Pwf/Pb)^2]

    Args:
        pr: Reservoir pressure in psi.
        pb: Bubble point pressure in psi.
        pi: Productivity index in STB/day/psi.
        pwf: Flowing pressure. If None, generates full curve.
        num_points: Number of curve points.

    Returns:
        Dict with 'qo', 'pwf', 'pr', 'pb', 'qmax'.
    """
    _validate_positive(pr, "reservoir pressure")
    _validate_positive(pb, "bubble point pressure")
    _validate_positive(pi, "productivity index")

    if pb > pr:
        raise ValueError(f"pb ({pb}) cannot exceed pr ({pr})")

    # Rate at bubble point
    qb = pi * (pr - pb)
    # AOF using Vogel below Pb
    qmax = qb + pi * pb / 1.8

    def _rate(p: float) -> float:
        if p < 0:
            raise ValueError(f"pwf must be non-negative, got {p}")
        if p >= pr:
            return 0.0
        if p >= pb:
            return pi * (pr - p)
        else:
            ratio = p / pb
            return qb + (qmax - qb) * (1.0 - 0.2 * ratio - 0.8 * ratio**2)

    if pwf is not None:
        if isinstance(pwf, np.ndarray):
            qo = np.array([_rate(p) for p in pwf])
            return {"qo": qo, "pwf": pwf, "pr": pr, "pb": pb, "qmax": round(qmax, 2)}
        else:
            return {"qo": _rate(float(pwf)), "pwf": float(pwf), "pr": pr, "pb": pb, "qmax": round(qmax, 2)}
    else:
        pwf_array = np.linspace(0, pr, num_points + 1)
        qo_array = np.array([_rate(p) for p in pwf_array])
        return {"qo": qo_array, "pwf": pwf_array, "pr": pr, "pb": pb, "qmax": round(qmax, 2)}
