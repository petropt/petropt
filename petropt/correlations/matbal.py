"""Material balance equations for reservoir engineering.

Implements:
    - Havlena-Odeh (1963) straight-line material balance
    - Gas reservoir P/Z analysis
    - Drive mechanism indices

References:
    Havlena, D. and Odeh, A.S., "The Material Balance as an Equation of
        a Straight Line," JPT, August 1963, pp. 896-900.
    Dake, L.P., "Fundamentals of Reservoir Engineering," Elsevier, 1978.
    Craft, B.C. and Hawkins, M.F., "Applied Petroleum Reservoir
        Engineering," Prentice-Hall, 1991.
"""

from __future__ import annotations

import numpy as np

from petropt.correlations.pvt import _validate_positive


# ---------------------------------------------------------------------------
# Gas reservoir P/Z
# ---------------------------------------------------------------------------

def gas_pz(
    pi: float,
    gi: float,
    gp: float | np.ndarray,
) -> float | np.ndarray:
    """Gas reservoir pressure from P/Z material balance.

    P/Z = (Pi/Zi) * (1 - Gp/G)

    Simplified form assuming Z is approximately proportional to P.
    For rigorous analysis, use iterative P/Z with actual Z-factor.

    Args:
        pi: Initial reservoir pressure in psi.
        gi: Gas initially in place (GIIP) in scf.
        gp: Cumulative gas production in scf.

    Returns:
        P/Z values (psi). Same shape as gp.
    """
    _validate_positive(pi, "initial pressure")
    _validate_positive(gi, "gas in place")

    gp = np.asarray(gp, dtype=float)
    pz = pi * (1.0 - gp / gi)
    result = np.maximum(pz, 0.0)

    if result.ndim == 0:
        return float(result)
    return result


def gas_pz_ogip(
    p_array: np.ndarray | list[float],
    z_array: np.ndarray | list[float],
    gp_array: np.ndarray | list[float],
) -> dict:
    """Estimate OGIP from P/Z vs Gp plot using linear regression.

    The x-intercept of the P/Z vs Gp line gives OGIP (G).

    Args:
        p_array: Pressure measurements in psi.
        z_array: Z-factor at each pressure.
        gp_array: Cumulative production at each pressure in scf.

    Returns:
        Dict with 'ogip' (scf), 'pi_over_zi', 'r_squared'.
    """
    p = np.asarray(p_array, dtype=float)
    z = np.asarray(z_array, dtype=float)
    gp = np.asarray(gp_array, dtype=float)

    if len(p) < 2:
        raise ValueError("Need at least 2 data points")

    pz = p / z

    # Linear regression: P/Z = a + b * Gp
    n = len(pz)
    sx = np.sum(gp)
    sy = np.sum(pz)
    sxy = np.sum(gp * pz)
    sxx = np.sum(gp**2)
    syy = np.sum(pz**2)

    b = (n * sxy - sx * sy) / (n * sxx - sx**2)
    a = (sy - b * sx) / n

    # OGIP = x-intercept = -a/b
    if abs(b) < 1e-30:
        raise ValueError("Cannot determine OGIP — slope is zero")

    ogip = -a / b

    # R-squared
    ss_res = np.sum((pz - (a + b * gp))**2)
    ss_tot = np.sum((pz - np.mean(pz))**2)
    r_sq = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return {
        "ogip": round(float(ogip), 2),
        "pi_over_zi": round(float(a), 4),
        "slope": round(float(b), 10),
        "r_squared": round(float(r_sq), 6),
    }


# ---------------------------------------------------------------------------
# Oil reservoir material balance — Havlena-Odeh
# ---------------------------------------------------------------------------

def havlena_odeh_terms(
    p: float,
    pi: float,
    np_cum: float,
    rp: float,
    bo: float,
    boi: float,
    bg: float,
    bgi: float,
    rs: float,
    rsi: float,
    wp: float = 0.0,
    bw: float = 1.0,
    we: float = 0.0,
    wi: float = 0.0,
    gi: float = 0.0,
    cf: float = 0.0,
    cw: float = 0.0,
    sw: float = 0.0,
) -> dict:
    """Compute Havlena-Odeh material balance terms.

    F = Np * [Bo + (Rp - Rs) * Bg] + Wp * Bw - Wi * Bw - Gi * Bg
    Eo = (Bo - Boi) + (Rsi - Rs) * Bg
    Eg = Boi * (Bg/Bgi - 1)
    Efw = Boi * (cf + cw * Sw) / (1 - Sw) * (Pi - P)

    For straight-line analysis: F = N * (Eo + m * Eg + Efw) + We

    Args:
        p: Current pressure in psi.
        pi: Initial pressure in psi.
        np_cum: Cumulative oil production in STB.
        rp: Cumulative producing GOR in scf/STB.
        bo: Current oil FVF in bbl/STB.
        boi: Initial oil FVF in bbl/STB.
        bg: Current gas FVF in bbl/scf.
        bgi: Initial gas FVF in bbl/scf.
        rs: Current solution GOR in scf/STB.
        rsi: Initial solution GOR in scf/STB.
        wp: Cumulative water production in STB.
        bw: Water FVF in bbl/STB.
        we: Cumulative water influx in bbl.
        wi: Cumulative water injection in STB.
        gi: Cumulative gas injection in scf.
        cf: Formation compressibility in 1/psi.
        cw: Water compressibility in 1/psi.
        sw: Connate water saturation (fraction).

    Returns:
        Dict with 'F', 'Eo', 'Eg', 'Efw' terms.
    """
    _validate_positive(pi, "initial pressure")
    _validate_positive(bo, "current oil FVF")
    _validate_positive(boi, "initial oil FVF")
    if bg < 0:
        raise ValueError(f"current gas FVF must be non-negative, got {bg}")
    if bgi < 0:
        raise ValueError(f"initial gas FVF must be non-negative, got {bgi}")
    if sw < 0 or sw >= 1:
        raise ValueError(f"water saturation must be in [0, 1), got {sw}")

    # Underground withdrawal (F)
    f = (
        np_cum * (bo + (rp - rs) * bg)
        + wp * bw
        - wi * bw
        - gi * bg
    )

    # Oil expansion (Eo)
    eo = (bo - boi) + (rsi - rs) * bg

    # Gas cap expansion (Eg)
    eg = boi * (bg / bgi - 1.0) if bgi > 0 else 0.0

    # Formation and water expansion (Efw)
    if sw < 1.0 and (cf > 0 or cw > 0):
        efw = boi * (cf + cw * sw) / (1.0 - sw) * (pi - p)
    else:
        efw = 0.0

    return {
        "F": round(float(f), 4),
        "Eo": round(float(eo), 8),
        "Eg": round(float(eg), 8),
        "Efw": round(float(efw), 8),
    }


def estimate_ooip(
    f_array: np.ndarray | list[float],
    eo_array: np.ndarray | list[float],
) -> dict:
    """Estimate OOIP from Havlena-Odeh straight line (no gas cap, no aquifer).

    For a volumetric undersaturated reservoir: F = N * Eo
    Slope of F vs Eo gives N (OOIP).

    Args:
        f_array: Underground withdrawal (F) values.
        eo_array: Oil expansion (Eo) values.

    Returns:
        Dict with 'ooip' (STB), 'r_squared'.
    """
    f = np.asarray(f_array, dtype=float)
    eo = np.asarray(eo_array, dtype=float)

    if len(f) < 2:
        raise ValueError("Need at least 2 data points")

    # Force through origin: N = sum(F*Eo) / sum(Eo^2)
    n = np.sum(f * eo) / np.sum(eo**2)

    # R-squared
    f_pred = n * eo
    ss_res = np.sum((f - f_pred)**2)
    ss_tot = np.sum((f - np.mean(f))**2)
    r_sq = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return {
        "ooip": round(float(n), 2),
        "r_squared": round(float(r_sq), 6),
    }


# ---------------------------------------------------------------------------
# Drive mechanism indices
# ---------------------------------------------------------------------------

def drive_indices(
    f: float,
    n: float,
    eo: float,
    eg: float = 0.0,
    efw: float = 0.0,
    we: float = 0.0,
    m: float = 0.0,
) -> dict:
    """Calculate reservoir drive mechanism indices.

    DDI = N * Eo / F          (depletion drive)
    SDI = N * m * Eg / F      (segregation/gas cap drive)
    WDI = We / F              (water drive)
    EDI = N * Efw / F         (expansion drive)

    Sum of all indices should equal 1.0.

    Args:
        f: Underground withdrawal (F).
        n: OOIP in STB.
        eo: Oil expansion term.
        eg: Gas cap expansion term.
        efw: Formation/water expansion term.
        we: Cumulative water influx in bbl.
        m: Gas cap ratio (m = G*Bgi / N*Boi).

    Returns:
        Dict with drive indices (DDI, SDI, WDI, EDI).
    """
    if f <= 0:
        raise ValueError(f"F must be positive, got {f}")

    ddi = n * eo / f
    sdi = n * m * eg / f if m > 0 else 0.0
    wdi = we / f if we > 0 else 0.0
    edi = n * efw / f if efw > 0 else 0.0

    return {
        "depletion_drive_index": round(ddi, 4),
        "segregation_drive_index": round(sdi, 4),
        "water_drive_index": round(wdi, 4),
        "expansion_drive_index": round(edi, 4),
        "total": round(ddi + sdi + wdi + edi, 4),
    }
