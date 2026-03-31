"""Volumetric calculations for petroleum engineering.

Implements:
    - STOIIP: Stock Tank Oil Initially In Place
    - GIIP: Gas Initially In Place
    - Drainage area and radius

References:
    Craft, B.C. and Hawkins, M.F., "Applied Petroleum Reservoir
    Engineering," Prentice-Hall, 1991.
"""

from __future__ import annotations

import math

from petropt.correlations.pvt import _validate_positive


def stoiip(
    area: float,
    thickness: float,
    porosity: float,
    sw: float,
    bo: float,
) -> float:
    """Stock Tank Oil Initially In Place (STOIIP).

    N = 7758 * A * h * phi * (1 - Sw) / Bo

    Args:
        area: Drainage area in acres.
        thickness: Net pay thickness in feet.
        porosity: Porosity (fraction, 0-1).
        sw: Water saturation (fraction, 0-1).
        bo: Oil formation volume factor in bbl/STB.

    Returns:
        STOIIP in STB.
    """
    _validate_positive(area, "area")
    _validate_positive(thickness, "thickness")
    if porosity <= 0 or porosity >= 1:
        raise ValueError(f"porosity must be between 0 and 1, got {porosity}")
    if sw < 0 or sw >= 1:
        raise ValueError(f"water saturation must be between 0 and 1, got {sw}")
    _validate_positive(bo, "oil FVF")

    return 7758.0 * area * thickness * porosity * (1.0 - sw) / bo


def giip(
    area: float,
    thickness: float,
    porosity: float,
    sw: float,
    bg: float,
) -> float:
    """Gas Initially In Place (GIIP).

    G = 43560 * A * h * phi * (1 - Sw) / Bg

    Args:
        area: Drainage area in acres.
        thickness: Net pay thickness in feet.
        porosity: Porosity (fraction, 0-1).
        sw: Water saturation (fraction, 0-1).
        bg: Gas formation volume factor in ft³/scf.

    Returns:
        GIIP in scf.
    """
    _validate_positive(area, "area")
    _validate_positive(thickness, "thickness")
    if porosity <= 0 or porosity >= 1:
        raise ValueError(f"porosity must be between 0 and 1, got {porosity}")
    if sw < 0 or sw >= 1:
        raise ValueError(f"water saturation must be between 0 and 1, got {sw}")
    _validate_positive(bg, "gas FVF")

    return 43560.0 * area * thickness * porosity * (1.0 - sw) / bg


def drainage_radius(area: float) -> float:
    """Drainage radius from area assuming circular drainage.

    re = sqrt(A * 43560 / pi)

    Args:
        area: Drainage area in acres.

    Returns:
        Drainage radius in feet.
    """
    _validate_positive(area, "area")
    return math.sqrt(area * 43560.0 / math.pi)


def recovery_factor(
    np_cum: float,
    stoiip_val: float,
) -> float:
    """Recovery factor.

    RF = Np / N

    Args:
        np_cum: Cumulative production in STB.
        stoiip_val: STOIIP in STB.

    Returns:
        Recovery factor (fraction, 0-1).
    """
    _validate_positive(stoiip_val, "STOIIP")
    if np_cum < 0:
        raise ValueError(f"cumulative production must be non-negative, got {np_cum}")
    return np_cum / stoiip_val
