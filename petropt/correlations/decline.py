"""Arps decline curve analysis.

Implements Arps' empirical decline models (1945):
    - Exponential decline (b = 0)
    - Hyperbolic decline (0 < b < 1)
    - Harmonic decline (b = 1)

Reference:
    Arps, J.J., "Analysis of Decline Curves," Trans. AIME, 160, 1945,
    pp. 228-247.
"""

from __future__ import annotations

import numpy as np


def arps_decline(
    qi: float,
    di: float,
    b: float,
    t: float | np.ndarray,
) -> float | np.ndarray:
    """Calculate rate(s) using Arps decline model.

    Args:
        qi: Initial rate in STB/day (or any consistent unit).
        di: Initial decline rate in 1/time (nominal, fractional).
        b: Arps b-factor (0 = exponential, 0<b<1 = hyperbolic, 1 = harmonic).
        t: Time value(s) — scalar or numpy array (same units as di).

    Returns:
        Rate(s) at time t. Same shape as input t.

    Raises:
        ValueError: If qi <= 0, di <= 0, or b is out of range [0, 2].
    """
    if qi <= 0:
        raise ValueError(f"qi must be positive, got {qi}")
    if di <= 0:
        raise ValueError(f"di must be positive, got {di}")
    if b < 0 or b > 2:
        raise ValueError(f"b must be in [0, 2], got {b}")

    t = np.asarray(t, dtype=float)

    if b == 0:
        # Exponential: q(t) = qi * exp(-di * t)
        q = qi * np.exp(-di * t)
    elif b == 1:
        # Harmonic: q(t) = qi / (1 + di * t)
        q = qi / (1.0 + di * t)
    else:
        # Hyperbolic: q(t) = qi / (1 + b * di * t)^(1/b)
        q = qi / (1.0 + b * di * t) ** (1.0 / b)

    # Return scalar if input was scalar
    if q.ndim == 0:
        return float(q)
    return q


def arps_cumulative(
    qi: float,
    di: float,
    b: float,
    t: float,
) -> float:
    """Cumulative production using Arps decline model.

    Args:
        qi: Initial rate in STB/day.
        di: Initial decline rate in 1/time.
        b: Arps b-factor.
        t: Time value (same units as di).

    Returns:
        Cumulative production (Np) in same volume units as qi * time.

    Raises:
        ValueError: If inputs are out of physical range.
    """
    if qi <= 0:
        raise ValueError(f"qi must be positive, got {qi}")
    if di <= 0:
        raise ValueError(f"di must be positive, got {di}")
    if b < 0 or b > 2:
        raise ValueError(f"b must be in [0, 2], got {b}")
    if t < 0:
        raise ValueError(f"t must be non-negative, got {t}")

    if t == 0:
        return 0.0

    if b == 0:
        # Exponential: Np = (qi / di) * (1 - exp(-di * t))
        return (qi / di) * (1.0 - np.exp(-di * t))
    elif b == 1:
        # Harmonic: Np = (qi / di) * ln(1 + di * t)
        return (qi / di) * np.log(1.0 + di * t)
    else:
        # Hyperbolic: Np = (qi^b / ((1-b)*di)) * (qi^(1-b) - q(t)^(1-b))
        q_t = qi / (1.0 + b * di * t) ** (1.0 / b)
        return (qi / ((1.0 - b) * di)) * (1.0 - (q_t / qi) ** (1.0 - b))


def arps_eur(
    qi: float,
    di: float,
    b: float,
    economic_limit: float = 5.0,
    max_time: float = 600.0,
) -> dict:
    """Estimated Ultimate Recovery (EUR) using Arps decline.

    Integrates the decline curve until rate reaches the economic limit
    or max_time is reached.

    Args:
        qi: Initial rate in STB/day.
        di: Initial decline rate in 1/month.
        b: Arps b-factor.
        economic_limit: Minimum economic rate in STB/day.
        max_time: Maximum time in months.

    Returns:
        Dict with 'eur', 'time_to_limit', 'final_rate'.
    """
    if qi <= 0:
        raise ValueError(f"qi must be positive, got {qi}")
    if di <= 0:
        raise ValueError(f"di must be positive, got {di}")
    if b < 0 or b > 2:
        raise ValueError(f"b must be in [0, 2], got {b}")
    if economic_limit < 0:
        raise ValueError(f"economic_limit must be non-negative, got {economic_limit}")

    # Find time when rate = economic_limit
    if b == 0:
        if qi <= economic_limit:
            return {"eur": 0.0, "time_to_limit": 0.0, "final_rate": qi}
        t_limit = -np.log(economic_limit / qi) / di
    elif b == 1:
        if qi <= economic_limit:
            return {"eur": 0.0, "time_to_limit": 0.0, "final_rate": qi}
        t_limit = (qi / economic_limit - 1.0) / di
    else:
        if qi <= economic_limit:
            return {"eur": 0.0, "time_to_limit": 0.0, "final_rate": qi}
        t_limit = ((qi / economic_limit) ** b - 1.0) / (b * di)

    t_limit = min(t_limit, max_time)
    final_rate = float(arps_decline(qi, di, b, t_limit))
    eur = arps_cumulative(qi, di, b, t_limit)

    return {
        "eur": round(float(eur), 2),
        "time_to_limit": round(float(t_limit), 2),
        "final_rate": round(final_rate, 2),
    }
