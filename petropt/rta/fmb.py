"""Flowing Material Balance (FMB).

Mattar & Anderson (2005) — estimate original hydrocarbon in place from
producing-well data without shut-in tests. Fits a straight line to

    q / (Pi - Pwf)  vs.  Np * B * ct

and extrapolates to abandonment to recover OHIP.
"""

from __future__ import annotations

from typing import Sequence, TypedDict

import numpy as np

from .transforms import _safe_divide


class FMBResult(TypedDict):
    normalized_rate: np.ndarray
    cumulative_normalized: np.ndarray
    slope: float
    intercept: float
    r_squared: float
    ooip_estimate: float | None
    num_valid_points: int


def flowing_material_balance(
    rates: Sequence[float],
    flowing_pressures: Sequence[float],
    initial_pressure: float,
    fluid_fvf: float,
    total_compressibility: float,
) -> FMBResult:
    """Fit FMB line to derive original hydrocarbon in place.

    The method plots ``q/(Pi-Pwf)`` against cumulative production weighted
    by ``B * ct`` and fits a straight line. Where the line crosses zero
    rate is the cumulative at abandonment; that cumulative, divided by
    ``B * ct``, is the contacted OOIP/OGIP.

    Cumulative is computed internally as the trapezoidal integral of the
    supplied rates, assuming unit time steps. If your rates are not on
    uniform time steps, pre-compute cumulative yourself and use
    :mod:`petropt.rta.type_curves` instead.

    Args:
        rates: Production rates (bbl/d for oil, Mcf/d for gas).
        flowing_pressures: Bottomhole flowing pressures (psi).
        initial_pressure: Initial reservoir pressure (psi).
        fluid_fvf: Formation volume factor (rb/STB oil, rcf/scf gas).
        total_compressibility: Total system compressibility (1/psi).

    Returns:
        Dict with the FMB slope/intercept, R-squared, OOIP estimate (or
        ``None`` if the fit is not physical — slope non-negative or
        intercept non-positive), and arrays used in the regression.
    """
    if initial_pressure <= 0:
        raise ValueError("initial_pressure must be positive")
    if fluid_fvf <= 0:
        raise ValueError("fluid_fvf must be positive")
    if total_compressibility <= 0:
        raise ValueError("total_compressibility must be positive")

    q = np.asarray(rates, dtype=float)
    pwf = np.asarray(flowing_pressures, dtype=float)
    if q.size < 3:
        raise ValueError("Need at least 3 data points")
    if q.shape != pwf.shape:
        raise ValueError("rates and flowing_pressures must have the same length")

    dp = initial_pressure - pwf
    qn = _safe_divide(q, dp, dp > 0)

    # Trapezoidal cumulative with unit time steps
    cum = np.zeros_like(q)
    for i in range(1, q.size):
        cum[i] = cum[i - 1] + 0.5 * (q[i] + q[i - 1])

    cum_norm = cum * fluid_fvf * total_compressibility

    valid = (dp > 0) & (q > 0)
    if valid.sum() < 2:
        raise ValueError("Need at least 2 valid (dp>0, q>0) data points")

    x = cum_norm[valid]
    y = qn[valid]
    slope, intercept = np.polyfit(x, y, 1)
    slope = float(slope)
    intercept = float(intercept)

    y_pred = np.polyval([slope, intercept], x)
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    ooip_estimate: float | None = None
    # Guard against near-flat fits where OOIP explodes under tiny slope noise
    slope_is_physical = slope < 0 and (
        abs(slope) > 1e-6 * max(abs(intercept), 1e-12)
    )
    if slope_is_physical and intercept > 0:
        x_intercept = -intercept / slope
        ooip_estimate = float(x_intercept / (fluid_fvf * total_compressibility))

    return {
        "normalized_rate": qn,
        "cumulative_normalized": cum_norm,
        "slope": slope,
        "intercept": intercept,
        "r_squared": r_squared,
        "ooip_estimate": ooip_estimate,
        "num_valid_points": int(valid.sum()),
    }
