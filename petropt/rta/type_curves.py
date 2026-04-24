"""Type-curve coordinates for Blasingame, Agarwal-Gardner, and NPI.

Each method transforms (t, q, Np, Pwf, Pi) history into coordinates that
collapse onto a family of dimensionless type curves. Matching the
observed trace against the family yields reservoir properties.

    Blasingame:       x = tMB,    y = q/(Pi-Pwf) + integral/derivative
    Agarwal-Gardner:  x = tMB,    y = q/(Pi-Pwf), plus inverse for
                      derivative analysis
    NPI:              x = t,      y = <dp/q>_t (time-integrated), with
                      log-derivative for flow regime diagnosis
"""

from __future__ import annotations

import math
from typing import Sequence, TypedDict

import numpy as np

from .transforms import _safe_divide

_TRAPZ = np.trapezoid if hasattr(np, "trapezoid") else np.trapz


class BlasingameResult(TypedDict):
    material_balance_time: np.ndarray
    normalized_rate: np.ndarray
    rate_integral: np.ndarray
    rate_integral_derivative: np.ndarray


class AgarwalGardnerResult(TypedDict):
    material_balance_time: np.ndarray
    normalized_rate: np.ndarray
    inverse_normalized_rate: np.ndarray
    cumulative_normalized: np.ndarray


class NPIResult(TypedDict):
    time: np.ndarray
    inverse_normalized_rate: np.ndarray
    npi: np.ndarray
    npi_derivative: np.ndarray


def _validate_rta_inputs(
    times: Sequence[float],
    rates: Sequence[float],
    cumulative: Sequence[float] | None,
    flowing_pressures: Sequence[float],
    initial_pressure: float,
) -> tuple[np.ndarray, ...]:
    if initial_pressure <= 0:
        raise ValueError("initial_pressure must be positive")
    t = np.asarray(times, dtype=float)
    q = np.asarray(rates, dtype=float)
    pwf = np.asarray(flowing_pressures, dtype=float)
    n = t.size
    if n < 3:
        raise ValueError("Need at least 3 data points")
    if q.shape != t.shape or pwf.shape != t.shape:
        raise ValueError("times, rates, flowing_pressures must share length")
    if cumulative is None:
        return t, q, pwf
    cum = np.asarray(cumulative, dtype=float)
    if cum.shape != t.shape:
        raise ValueError("cumulative must share length with times")
    return t, q, cum, pwf


def blasingame_variables(
    times: Sequence[float],
    rates: Sequence[float],
    cumulative: Sequence[float],
    flowing_pressures: Sequence[float],
    initial_pressure: float,
) -> BlasingameResult:
    """Blasingame rate-normalized integral and log-derivative variables.

    Returns the four series used for type-curve matching:
        qn   = q / (Pi - Pwf)
        tMB  = Np / q
        qni  = (1/tMB) * integral(qn, d tMB)      (integral)
        qnid = -d qni / d ln(tMB)                  (derivative)

    Args:
        times: Time values (days or months).
        rates: Production rates.
        cumulative: Cumulative production at each time.
        flowing_pressures: Bottomhole flowing pressures (psi).
        initial_pressure: Initial reservoir pressure (psi).

    Returns:
        Dict with numpy arrays: material_balance_time, normalized_rate,
        rate_integral, rate_integral_derivative.
    """
    _, q, cum, pwf = _validate_rta_inputs(
        times, rates, cumulative, flowing_pressures, initial_pressure
    )
    n = q.size
    dp = initial_pressure - pwf
    qn = _safe_divide(q, dp, dp > 0)
    tmb = _safe_divide(cum, q, q > 0)

    rate_integral = np.zeros(n)
    for i in range(1, n):
        if tmb[i] > 0:
            rate_integral[i] = _TRAPZ(qn[: i + 1], tmb[: i + 1]) / tmb[i]
        else:
            rate_integral[i] = qn[i]

    rate_integral_deriv = np.zeros(n)
    for i in range(1, n - 1):
        if tmb[i - 1] > 0 and tmb[i + 1] > 0:
            d_ln = math.log(tmb[i + 1]) - math.log(tmb[i - 1])
            if d_ln != 0:
                rate_integral_deriv[i] = (
                    -(rate_integral[i + 1] - rate_integral[i - 1]) / d_ln
                )
    if n >= 2 and tmb[0] > 0 and tmb[1] > 0:
        d_ln = math.log(tmb[1]) - math.log(tmb[0])
        if d_ln != 0:
            rate_integral_deriv[0] = (
                -(rate_integral[1] - rate_integral[0]) / d_ln
            )
    if n >= 2 and tmb[-1] > 0 and tmb[-2] > 0:
        d_ln = math.log(tmb[-1]) - math.log(tmb[-2])
        if d_ln != 0:
            rate_integral_deriv[-1] = (
                -(rate_integral[-1] - rate_integral[-2]) / d_ln
            )

    return {
        "material_balance_time": tmb,
        "normalized_rate": qn,
        "rate_integral": rate_integral,
        "rate_integral_derivative": rate_integral_deriv,
    }


def agarwal_gardner_variables(
    times: Sequence[float],
    rates: Sequence[float],
    cumulative: Sequence[float],
    flowing_pressures: Sequence[float],
    initial_pressure: float,
) -> AgarwalGardnerResult:
    """Agarwal-Gardner rate-normalized variables.

    Agarwal-Gardner (SPE 57916) plots rate-normalized rate q/dp (or the
    inverse dp/q) against material balance time tMB, collapsing variable
    production history onto a single dimensionless type-curve family.

    Args:
        times: Time values.
        rates: Production rates.
        cumulative: Cumulative production.
        flowing_pressures: Bottomhole flowing pressures (psi).
        initial_pressure: Initial reservoir pressure (psi).

    Returns:
        Dict with arrays: material_balance_time, normalized_rate,
        inverse_normalized_rate (dp/q), cumulative_normalized (Np/dp).
    """
    _, q, cum, pwf = _validate_rta_inputs(
        times, rates, cumulative, flowing_pressures, initial_pressure
    )
    dp = initial_pressure - pwf
    tmb = _safe_divide(cum, q, q > 0)
    qn = _safe_divide(q, dp, dp > 0)
    # Clamp inverse rate consistently: require both q>0 AND dp>0 so that
    # qn * inv_qn == 1 on their common support.
    both = (q > 0) & (dp > 0)
    inv_qn = _safe_divide(dp, q, both)
    cum_norm = _safe_divide(cum, dp, dp > 0)

    return {
        "material_balance_time": tmb,
        "normalized_rate": qn,
        "inverse_normalized_rate": inv_qn,
        "cumulative_normalized": cum_norm,
    }


def npi_variables(
    times: Sequence[float],
    rates: Sequence[float],
    flowing_pressures: Sequence[float],
    initial_pressure: float,
) -> NPIResult:
    """Normalized Pressure Integral (NPI) variables.

    NPI(t) = (1 / (t - t0)) * integral_{t0}^{t} dp(tau) / q(tau) dtau

    The time-integrated inverse-rate smooths noisy production data and
    reveals flow-regime signatures via its log-time derivative.

    Args:
        times: Time values (days).
        rates: Production rates.
        flowing_pressures: Bottomhole flowing pressures (psi).
        initial_pressure: Initial reservoir pressure (psi).

    Returns:
        Dict with arrays: time, inverse_normalized_rate, npi, npi_derivative.
    """
    t, q, pwf = _validate_rta_inputs(
        times, rates, None, flowing_pressures, initial_pressure
    )
    n = t.size
    dp = initial_pressure - pwf
    inv_qn = _safe_divide(dp, q, (q > 0) & (dp > 0))

    npi = np.zeros(n)
    for i in range(1, n):
        if t[i] > t[0]:
            npi[i] = _TRAPZ(inv_qn[: i + 1], t[: i + 1]) / (t[i] - t[0])

    npi_deriv = np.zeros(n)
    for i in range(1, n - 1):
        if t[i + 1] > 0 and t[i - 1] > 0:
            d_ln_t = math.log(t[i + 1]) - math.log(t[i - 1])
            if d_ln_t != 0:
                npi_deriv[i] = (npi[i + 1] - npi[i - 1]) / d_ln_t

    return {
        "time": t,
        "inverse_normalized_rate": inv_qn,
        "npi": npi,
        "npi_derivative": npi_deriv,
    }
