"""Rate-transient coordinate transforms.

Two building blocks used by every modern RTA method:

1. Pressure-normalized rate: q / (Pi - Pwf). Removes the effect of
   variable flowing pressure, letting rate behavior be interpreted as
   if drawdown were constant.

2. Material balance time: tMB = Np / q. Blasingame's variable-rate
   equivalent of shut-in time — shifts a variable-rate problem onto
   a constant-rate type curve.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def _safe_divide(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """a / b where mask is True, 0.0 elsewhere. Avoids RuntimeWarnings."""
    out = np.zeros_like(a, dtype=float)
    np.divide(a, b, out=out, where=mask)
    return out


def pressure_normalized_rate(
    rate: Sequence[float],
    flowing_pressure: Sequence[float],
    initial_pressure: float,
) -> np.ndarray:
    """Rate normalized by drawdown: q / (Pi - Pwf).

    Any sample with Pwf >= Pi (zero or reverse drawdown) returns 0.

    Args:
        rate: Production rates (bbl/d for oil, Mcf/d for gas).
        flowing_pressure: Bottomhole flowing pressures (psi).
        initial_pressure: Initial reservoir pressure (psi).

    Returns:
        1-D numpy array of pressure-normalized rates, same length as
        ``rate``.
    """
    if initial_pressure <= 0:
        raise ValueError("initial_pressure must be positive")
    q = np.asarray(rate, dtype=float)
    pwf = np.asarray(flowing_pressure, dtype=float)
    if q.size == 0:
        raise ValueError("rate must not be empty")
    if q.shape != pwf.shape:
        raise ValueError("rate and flowing_pressure must have the same length")

    dp = initial_pressure - pwf
    return _safe_divide(q, dp, dp > 0)


def material_balance_time(
    cumulative_production: Sequence[float],
    rate: Sequence[float],
) -> np.ndarray:
    """Blasingame material balance time: tMB = Np / q.

    This variable-rate-equivalent "time" stretches early high-rate
    production and compresses late low-rate production. It is the
    x-axis for Blasingame, Agarwal-Gardner, and NPI plots.

    Args:
        cumulative_production: Cumulative production (same units *
            time as rate — e.g., bbl if rate is bbl/d).
        rate: Instantaneous production rate at each time step.

    Returns:
        1-D numpy array of material balance times. Samples with zero
        rate return 0.
    """
    np_arr = np.asarray(cumulative_production, dtype=float)
    q = np.asarray(rate, dtype=float)
    if q.size == 0:
        raise ValueError("rate must not be empty")
    if np_arr.shape != q.shape:
        raise ValueError("cumulative_production and rate must have the same length")
    return _safe_divide(np_arr, q, q > 0)
