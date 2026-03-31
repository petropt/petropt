"""Relative permeability correlations.

Implements:
    - Corey (1954): Two-phase oil/water/gas relative permeability
    - Brooks and Corey (1964): Capillary-based relative permeability
    - LET (Lomeland, Ebeltoft, Thomas, 2005): Three-parameter model

References:
    Corey, A.T., "The Interrelation Between Gas and Oil Relative
        Permeabilities," Producers Monthly, 19, 1954, pp. 38-41.
    Brooks, R.H. and Corey, A.T., "Hydraulic Properties of Porous Media,"
        Colorado State University Hydrology Papers, No. 3, 1964.
    Lomeland, F., Ebeltoft, E., and Thomas, W.H., "A New Versatile
        Relative Permeability Correlation," SCA2005-32, 2005.
"""

from __future__ import annotations

import numpy as np


def _normalize_sw(
    sw: float | np.ndarray,
    swi: float,
    sor: float,
) -> np.ndarray:
    """Normalize water saturation to effective saturation."""
    sw = np.asarray(sw, dtype=float)
    if swi + sor >= 1.0:
        raise ValueError(
            f"swi + sor must be < 1.0, got {swi} + {sor} = {swi + sor}"
        )
    sw_eff = (sw - swi) / (1.0 - swi - sor)
    return np.clip(sw_eff, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Corey (1954)
# ---------------------------------------------------------------------------

def corey_oil(
    sw: float | np.ndarray,
    swi: float,
    sor: float,
    kro_max: float = 1.0,
    no: float = 2.0,
) -> float | np.ndarray:
    """Oil relative permeability using Corey model.

    kro = kro_max * (1 - Sw*)^no

    Args:
        sw: Water saturation (fraction, 0-1).
        swi: Irreducible water saturation.
        sor: Residual oil saturation.
        kro_max: Maximum oil relative permeability (default 1.0).
        no: Corey exponent for oil (default 2.0).

    Returns:
        Oil relative permeability (dimensionless).
    """
    sw_eff = _normalize_sw(sw, swi, sor)
    kro = kro_max * (1.0 - sw_eff) ** no
    result = np.where(sw_eff >= 1.0, 0.0, kro)
    return float(result) if result.ndim == 0 else result


def corey_water(
    sw: float | np.ndarray,
    swi: float,
    sor: float,
    krw_max: float = 1.0,
    nw: float = 2.0,
) -> float | np.ndarray:
    """Water relative permeability using Corey model.

    krw = krw_max * Sw*^nw

    Args:
        sw: Water saturation (fraction, 0-1).
        swi: Irreducible water saturation.
        sor: Residual oil saturation.
        krw_max: Maximum water relative permeability (default 1.0).
        nw: Corey exponent for water (default 2.0).

    Returns:
        Water relative permeability (dimensionless).
    """
    sw_eff = _normalize_sw(sw, swi, sor)
    krw = krw_max * sw_eff ** nw
    result = np.where(sw_eff <= 0.0, 0.0, krw)
    return float(result) if result.ndim == 0 else result


def corey_gas(
    sg: float | np.ndarray,
    sgc: float,
    swi: float,
    krg_max: float = 1.0,
    ng: float = 2.0,
) -> float | np.ndarray:
    """Gas relative permeability using Corey model.

    krg = krg_max * Sg*^ng

    Args:
        sg: Gas saturation (fraction, 0-1).
        sgc: Critical gas saturation.
        swi: Irreducible water saturation.
        krg_max: Maximum gas relative permeability (default 1.0).
        ng: Corey exponent for gas (default 2.0).

    Returns:
        Gas relative permeability (dimensionless).
    """
    sg = np.asarray(sg, dtype=float)
    if sgc + swi >= 1.0:
        raise ValueError(
            f"sgc + swi must be < 1.0, got {sgc} + {swi} = {sgc + swi}"
        )
    sg_eff = (sg - sgc) / (1.0 - sgc - swi)
    sg_eff = np.clip(sg_eff, 0.0, 1.0)
    krg = krg_max * sg_eff ** ng
    result = np.where(sg_eff <= 0.0, 0.0, krg)
    return float(result) if result.ndim == 0 else result


# ---------------------------------------------------------------------------
# Brooks-Corey (1964)
# ---------------------------------------------------------------------------

def brooks_corey_oil(
    sw: float | np.ndarray,
    swi: float,
    sor: float,
    kro_max: float = 1.0,
    lam: float = 2.0,
) -> float | np.ndarray:
    """Oil relative permeability using Brooks-Corey model.

    kro = kro_max * (1 - Sw*)^(2 + 3*lambda) / lambda

    Args:
        sw: Water saturation (fraction).
        swi: Irreducible water saturation.
        sor: Residual oil saturation.
        kro_max: Maximum oil relative permeability.
        lam: Pore size distribution index (lambda).

    Returns:
        Oil relative permeability.
    """
    sw_eff = _normalize_sw(sw, swi, sor)
    exponent = (2.0 + 3.0 * lam) / lam
    kro = kro_max * (1.0 - sw_eff) ** exponent
    result = np.where(sw_eff >= 1.0, 0.0, kro)
    return float(result) if result.ndim == 0 else result


def brooks_corey_water(
    sw: float | np.ndarray,
    swi: float,
    sor: float,
    krw_max: float = 1.0,
    lam: float = 2.0,
) -> float | np.ndarray:
    """Water relative permeability using Brooks-Corey model.

    krw = krw_max * Sw*^((2 + 3*lambda) / lambda)

    Args:
        sw: Water saturation (fraction).
        swi: Irreducible water saturation.
        sor: Residual oil saturation.
        krw_max: Maximum water relative permeability.
        lam: Pore size distribution index (lambda).

    Returns:
        Water relative permeability.
    """
    sw_eff = _normalize_sw(sw, swi, sor)
    exponent = (2.0 + 3.0 * lam) / lam
    krw = krw_max * sw_eff ** exponent
    result = np.where(sw_eff <= 0.0, 0.0, krw)
    return float(result) if result.ndim == 0 else result


# ---------------------------------------------------------------------------
# LET model (Lomeland, Ebeltoft, Thomas, 2005)
# ---------------------------------------------------------------------------

def let_oil(
    sw: float | np.ndarray,
    swi: float,
    sor: float,
    kro_max: float = 1.0,
    lo: float = 2.0,
    eo: float = 2.0,
    to: float = 2.0,
) -> float | np.ndarray:
    """Oil relative permeability using LET model.

    kro = kro_max * (1-Sw*)^Lo / ((1-Sw*)^Lo + Eo * Sw*^To)

    Args:
        sw: Water saturation (fraction).
        swi: Irreducible water saturation.
        sor: Residual oil saturation.
        kro_max: Maximum oil relative permeability.
        lo: L parameter for oil.
        eo: E parameter for oil.
        to: T parameter for oil.

    Returns:
        Oil relative permeability.
    """
    sw_eff = _normalize_sw(sw, swi, sor)
    num = (1.0 - sw_eff) ** lo
    den = num + eo * sw_eff ** to
    kro = np.where(den > 0, kro_max * num / den, 0.0)
    kro = np.where(sw_eff >= 1.0, 0.0, kro)
    return float(kro) if kro.ndim == 0 else kro


def let_water(
    sw: float | np.ndarray,
    swi: float,
    sor: float,
    krw_max: float = 1.0,
    lw: float = 2.0,
    ew: float = 2.0,
    tw: float = 2.0,
) -> float | np.ndarray:
    """Water relative permeability using LET model.

    krw = krw_max * Sw*^Lw / (Sw*^Lw + Ew * (1-Sw*)^Tw)

    Args:
        sw: Water saturation (fraction).
        swi: Irreducible water saturation.
        sor: Residual oil saturation.
        krw_max: Maximum water relative permeability.
        lw: L parameter for water.
        ew: E parameter for water.
        tw: T parameter for water.

    Returns:
        Water relative permeability.
    """
    sw_eff = _normalize_sw(sw, swi, sor)
    num = sw_eff ** lw
    den = num + ew * (1.0 - sw_eff) ** tw
    krw = np.where(den > 0, krw_max * num / den, 0.0)
    krw = np.where(sw_eff <= 0.0, 0.0, krw)
    return float(krw) if krw.ndim == 0 else krw
