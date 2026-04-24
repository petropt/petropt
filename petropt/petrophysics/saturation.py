"""Water saturation from resistivity logs.

Correlations:
    - Archie (1942) — clean sands:
        Sw = (a * Rw / (phi^m * Rt))^(1/n)
    - Simandoux (1963) — shaly sands (modified form):
        1/Rt = (1-Vsh) * phi^m / (a * Rw) * Sw^n + Vsh / Rsh * Sw
    - Poupon-Leveaux / Indonesian (1971) — highly shaly formations:
        1/sqrt(Rt) = [phi^(m/2)/sqrt(a*Rw) + Vsh^(1-Vsh/2)/sqrt(Rsh)] * Sw^(n/2)

References:
    Archie, G.E., "The Electrical Resistivity Log as an Aid in Determining
        Some Reservoir Characteristics," Trans. AIME, 146, 1942, pp. 54-62.
    Simandoux, P., "Dielectric Measurements on Porous Media: Application
        to the Measurement of Water Saturation," Revue de l'Institut
        Francais du Petrole, 18, 1963.
    Poupon, A. and Leveaux, J., "Evaluation of Water Saturation in Shaly
        Formations," SPWLA 12th Annual Logging Symposium, 1971.
"""

from __future__ import annotations

import math

from ._validation import clamp, validate_fraction, validate_positive


def archie_sw(
    rt: float,
    phi: float,
    rw: float,
    a: float = 1.0,
    m: float = 2.0,
    n: float = 2.0,
) -> float:
    """Water saturation via the Archie equation (clean sands).

        Sw = (a * Rw / (phi^m * Rt))^(1/n)

    Archie's law assumes a clean, water-wet formation with no conductive
    clay. For shaly sands use :func:`simandoux_sw` or :func:`indonesian_sw`.

    Typical defaults (a=1, m=2, n=2) follow Humble's "standard" for clean
    sandstones. For carbonates, m is often 2.0-2.5 and a near 1.0. Source
    your own m, n from SCAL where available.

    Args:
        rt: True formation resistivity (ohm-m).
        phi: Porosity (fraction, 0-1).
        rw: Formation water resistivity at formation temperature (ohm-m).
        a: Tortuosity factor. Default 1.0.
        m: Cementation exponent. Default 2.0.
        n: Saturation exponent. Default 2.0.

    Returns:
        Water saturation as a fraction (0-1). Clamped to [0, 1].
    """
    validate_positive(rt, "rt")
    validate_positive(phi, "phi")
    validate_positive(rw, "rw")
    validate_positive(a, "a")
    validate_positive(m, "m")
    validate_positive(n, "n")
    if phi > 1:
        raise ValueError(f"phi must be <= 1, got {phi}")

    sw = (a * rw / (phi**m * rt)) ** (1.0 / n)
    return clamp(sw)


def simandoux_sw(
    rt: float,
    phi: float,
    rw: float,
    vshale: float,
    rsh: float,
    a: float = 1.0,
    m: float = 2.0,
    n: float = 2.0,
) -> float:
    """Water saturation via the modified Simandoux equation (shaly sands).

    The classical Simandoux model adds a parallel shale-conductance term
    to Archie. For n=2 it reduces to a quadratic in Sw; for arbitrary n
    we solve by bisection.

    Args:
        rt: True formation resistivity (ohm-m).
        phi: Porosity (fraction, 0-1).
        rw: Formation water resistivity (ohm-m).
        vshale: Shale volume (fraction, 0-1).
        rsh: Shale resistivity (ohm-m).
        a: Tortuosity factor. Default 1.0.
        m: Cementation exponent. Default 2.0.
        n: Saturation exponent. Default 2.0.

    Returns:
        Water saturation as a fraction (0-1). Clamped to [0, 1].
    """
    validate_positive(rt, "rt")
    validate_positive(phi, "phi")
    validate_positive(rw, "rw")
    validate_positive(rsh, "rsh")
    validate_positive(a, "a")
    validate_positive(m, "m")
    validate_positive(n, "n")
    if phi > 1:
        raise ValueError(f"phi must be <= 1, got {phi}")
    validate_fraction(vshale, "vshale")

    if vshale == 0:
        return archie_sw(rt, phi, rw, a, m, n)

    factor = (1.0 - vshale) * phi**m
    if factor <= 0:
        # vshale = 1 (pure shale) limit: the Archie-like term vanishes, so
        # 1/Rt = (Vsh / Rsh) * Sw  =>  Sw = Rsh / (Vsh * Rt) -> Rsh / Rt
        return clamp(rsh / rt)

    c_val = a * rw / factor

    if n == 2.0:
        b_coeff = vshale / rsh
        disc = b_coeff**2 + 4.0 / (c_val * rt)
        sw = c_val * (math.sqrt(disc) - b_coeff) / 2.0
    else:
        sw = _bisect_simandoux(rt, phi, rw, vshale, rsh, a, m, n)

    return clamp(sw)


def indonesian_sw(
    rt: float,
    phi: float,
    rw: float,
    vshale: float,
    rsh: float,
    a: float = 1.0,
    m: float = 2.0,
    n: float = 2.0,
) -> float:
    """Water saturation via the Poupon-Leveaux (Indonesian) equation.

    Designed for highly shaly / fresh-water formations where Simandoux
    tends to over-predict water. For n=2 we have a closed-form; for
    arbitrary n we bisect.

    Args:
        rt: True formation resistivity (ohm-m).
        phi: Porosity (fraction, 0-1).
        rw: Formation water resistivity (ohm-m).
        vshale: Shale volume (fraction, 0-1).
        rsh: Shale resistivity (ohm-m).
        a: Tortuosity factor. Default 1.0.
        m: Cementation exponent. Default 2.0.
        n: Saturation exponent. Default 2.0.

    Returns:
        Water saturation as a fraction (0-1). Clamped to [0, 1].
    """
    validate_positive(rt, "rt")
    validate_positive(phi, "phi")
    validate_positive(rw, "rw")
    validate_positive(rsh, "rsh")
    validate_positive(a, "a")
    validate_positive(m, "m")
    validate_positive(n, "n")
    if phi > 1:
        raise ValueError(f"phi must be <= 1, got {phi}")
    validate_fraction(vshale, "vshale")

    if vshale == 0:
        return archie_sw(rt, phi, rw, a, m, n)

    if n == 2.0:
        a_term = phi ** (m / 2.0) / math.sqrt(a * rw)
        b_term = vshale ** (1.0 - vshale / 2.0) / math.sqrt(rsh)
        total = a_term + b_term
        sw = 1.0 / (math.sqrt(rt) * total) if total > 0 else 1.0
    else:
        sw = _bisect_indonesian(rt, phi, rw, vshale, rsh, a, m, n)

    return clamp(sw)


def _bracketed_bisect(residual, lo: float, hi: float, iters: int = 80) -> float:
    """Bisect residual(sw) on [lo, hi]. If the endpoints don't bracket
    (residual shares sign at both ends), return the endpoint with the
    smaller absolute residual — the equation has no interior root in
    the physical range, so the best physical answer is the boundary.
    Residual is assumed monotonic in sw for both Simandoux and
    Indonesian; both are positive-monotone in Sw.
    """
    f_lo = residual(lo)
    f_hi = residual(hi)
    if f_lo == 0.0:
        return lo
    if f_hi == 0.0:
        return hi
    if f_lo * f_hi > 0:
        # No sign change — root lies outside the physical interval.
        return lo if abs(f_lo) < abs(f_hi) else hi
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        f_mid = residual(mid)
        if f_mid == 0.0:
            return mid
        if f_lo * f_mid < 0:
            hi = mid
            f_hi = f_mid
        else:
            lo = mid
            f_lo = f_mid
    return 0.5 * (lo + hi)


def _bisect_simandoux(
    rt: float, phi: float, rw: float,
    vshale: float, rsh: float,
    a: float, m: float, n: float,
) -> float:
    factor = (1.0 - vshale) * phi**m

    def residual(sw: float) -> float:
        return (sw**n / (a * rw)) * factor + vshale * sw / rsh - 1.0 / rt

    return _bracketed_bisect(residual, 0.0, 1.0)


def _bisect_indonesian(
    rt: float, phi: float, rw: float,
    vshale: float, rsh: float,
    a: float, m: float, n: float,
) -> float:
    a_const = phi ** (m / 2.0) / math.sqrt(a * rw)
    b_const = vshale ** (1.0 - vshale / 2.0) / math.sqrt(rsh)
    bracket = a_const + b_const

    def residual(sw: float) -> float:
        return (bracket * sw ** (n / 2.0)) ** 2 - 1.0 / rt

    return _bracketed_bisect(residual, 0.0, 1.0)
