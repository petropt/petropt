"""Square-root-of-time linear flow analysis.

During the linear flow regime of a fractured well (the dominant regime
for shale wells during their first months to years), 1/q vs sqrt(t) is
a straight line. Its slope lets you back out sqrt(k) * xf — the product
of the square root of matrix permeability and the fracture half-length.
With an independent estimate of either, the other follows.

Reference:
    Wattenbarger, R.A., El-Banbi, A.H., Villegas, M.E., and Maggard,
    J.B., "Production Analysis of Linear Flow Into Fractured Tight
    Gas Wells," SPE 39931, 1998.
"""

from __future__ import annotations

import math
from typing import Sequence, TypedDict

import numpy as np

from .transforms import _safe_divide


class SqrtTimeResult(TypedDict):
    sqrt_time: np.ndarray
    inverse_normalized_rate: np.ndarray
    slope: float
    intercept: float
    r_squared: float
    end_of_linear_flow_time: float | None
    num_valid_points: int


def sqrt_time_analysis(
    rates: Sequence[float],
    times: Sequence[float],
    flowing_pressures: Sequence[float],
    initial_pressure: float,
    deviation_threshold_sigma: float = 2.0,
) -> SqrtTimeResult:
    """Fit the linear-flow sqrt(t) line to production data.

    Regress ``dp/q`` on ``sqrt(t)`` over all valid samples and report
    the slope, intercept, R-squared, and the estimated end-of-linear-flow
    time (first sample after 30% of the record whose residual exceeds
    ``deviation_threshold_sigma`` standard deviations of the fit).

    Args:
        rates: Production rates.
        times: Time values (days).
        flowing_pressures: Bottomhole flowing pressures (psi).
        initial_pressure: Initial reservoir pressure (psi).
        deviation_threshold_sigma: Residual cutoff (in fit-residual
            standard deviations) to flag end of linear flow. Default 2.0.

    Returns:
        Dict with sqrt_time, inverse_normalized_rate, slope, intercept,
        r_squared, end_of_linear_flow_time (or ``None`` if no deviation
        is found), and num_valid_points.
    """
    if initial_pressure <= 0:
        raise ValueError("initial_pressure must be positive")
    q = np.asarray(rates, dtype=float)
    t = np.asarray(times, dtype=float)
    pwf = np.asarray(flowing_pressures, dtype=float)
    if q.size < 3:
        raise ValueError("Need at least 3 data points")
    if q.shape != t.shape or pwf.shape != t.shape:
        raise ValueError("rates, times, flowing_pressures must share length")

    dp = initial_pressure - pwf
    sqrt_t = np.sqrt(np.maximum(t, 0.0))
    inv_qn = _safe_divide(dp, q, (q > 0) & (dp > 0))

    valid = (q > 0) & (t > 0) & (dp > 0)
    if valid.sum() < 2:
        raise ValueError("Need at least 2 valid (q>0, t>0, dp>0) data points")

    x = sqrt_t[valid]
    y = inv_qn[valid]
    slope, intercept = np.polyfit(x, y, 1)
    slope = float(slope)
    intercept = float(intercept)

    y_pred = np.polyval([slope, intercept], x)
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    residuals = np.abs(y - y_pred)
    sigma = float(residuals.std()) if residuals.size > 0 else 0.0
    threshold = deviation_threshold_sigma * sigma
    end_linear: float | None = None
    t_valid = t[valid]
    for i, r in enumerate(residuals):
        if r > threshold and i > 0.3 * residuals.size:
            end_linear = float(t_valid[i])
            break

    return {
        "sqrt_time": sqrt_t,
        "inverse_normalized_rate": inv_qn,
        "slope": slope,
        "intercept": intercept,
        "r_squared": r_squared,
        "end_of_linear_flow_time": end_linear,
        "num_valid_points": int(valid.sum()),
    }


class LinearFlowPermResult(TypedDict, total=False):
    sqrt_k_times_xf: float
    permeability_md: float
    fracture_half_length_ft: float


def permeability_from_linear_flow(
    slope: float,
    net_pay_ft: float,
    porosity: float,
    viscosity_cp: float,
    total_compressibility: float,
    fluid_fvf: float = 1.0,
    fracture_half_length_ft: float | None = None,
) -> LinearFlowPermResult:
    """Back out sqrt(k)*xf (or k with known xf) from the linear-flow slope.

    For oil, the linear flow slope m from sqrt(t) analysis relates to
    reservoir properties as

        m = 4.064 * B * mu / (h * xf * sqrt(k * phi * mu * ct))

    so that

        sqrt(k) * xf = 4.064 * B * mu / (m * h * sqrt(phi * mu * ct))

    With a known fracture half-length ``xf``, solve for ``k`` directly.
    Without it, only the product ``sqrt(k) * xf`` is recoverable — the
    canonical "fractured-well ambiguity".

    Args:
        slope: Slope from :func:`sqrt_time_analysis`
            (units psi * d / (bbl * d^0.5)).
        net_pay_ft: Net pay thickness (ft).
        porosity: Porosity (fraction, 0-1).
        viscosity_cp: Fluid viscosity (cp).
        total_compressibility: Total system compressibility (1/psi).
        fluid_fvf: Formation volume factor (rb/STB). Default 1.0.
        fracture_half_length_ft: Fracture half-length (ft). If given, the
            result includes ``permeability_md``.

    Returns:
        Dict with ``sqrt_k_times_xf`` (always) and, when ``xf`` is given,
        ``permeability_md`` and ``fracture_half_length_ft``.
    """
    if slope <= 0:
        raise ValueError("slope must be positive")
    if net_pay_ft <= 0:
        raise ValueError("net_pay_ft must be positive")
    if not 0.0 < porosity <= 1.0:
        raise ValueError("porosity must be in (0, 1]")
    if viscosity_cp <= 0:
        raise ValueError("viscosity_cp must be positive")
    if total_compressibility <= 0:
        raise ValueError("total_compressibility must be positive")
    if fluid_fvf <= 0:
        raise ValueError("fluid_fvf must be positive")

    sqrt_phi_mu_ct = math.sqrt(porosity * viscosity_cp * total_compressibility)
    sqrt_k_xf = (
        4.064 * fluid_fvf * viscosity_cp
        / (slope * net_pay_ft * sqrt_phi_mu_ct)
    )

    result: LinearFlowPermResult = {"sqrt_k_times_xf": float(sqrt_k_xf)}

    if fracture_half_length_ft is not None:
        if fracture_half_length_ft <= 0:
            raise ValueError("fracture_half_length_ft must be positive")
        sqrt_k = sqrt_k_xf / fracture_half_length_ft
        result["permeability_md"] = float(sqrt_k**2)
        result["fracture_half_length_ft"] = float(fracture_half_length_ft)

    return result
