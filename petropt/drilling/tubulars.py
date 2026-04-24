"""Casing and tubing strength calculations.

Internal burst and external collapse pressures for API-grade pipe:

    Burst (Barlow, with 12.5% wall tolerance):
        P_burst = 0.875 * 2 * Fy * t / OD

    Collapse (API Bulletin 5C3, 1994): four regimes selected by the
    D/t ratio and the grade-dependent coefficients A, B, C, F, G.

References:
    API Bulletin 5C3, "Formulas and Calculations for Casing, Tubing,
        Drill Pipe, and Line Pipe Properties," 6th ed., 1994.
    ISO 10400, "Petroleum and natural gas industries — Equations and
        calculations for the properties of casing, tubing, drill pipe
        and line pipe used as casing or tubing," 2007.
    Bourgoyne et al., "Applied Drilling Engineering," 1991, Ch. 7.
"""

from __future__ import annotations

import math
from typing import TypedDict

from ._validation import validate_positive


def burst_pressure_barlow(
    yield_strength_psi: float,
    wall_thickness_in: float,
    od_in: float,
    design_factor: float = 0.875,
) -> float:
    """Internal yield (burst) pressure via Barlow's formula with API wall-tolerance factor.

        P_burst = design_factor * 2 * Fy * t / OD

    The 0.875 default accounts for the 12.5% minimum wall thickness
    tolerance in API 5CT. Use ``design_factor=1.0`` for the nominal
    unadjusted burst.

    Args:
        yield_strength_psi: Minimum yield strength of the pipe (psi).
        wall_thickness_in: Nominal wall thickness (in).
        od_in: Outer diameter (in).
        design_factor: Wall-tolerance / safety multiplier. Default 0.875.

    Returns:
        Burst pressure in psi.
    """
    validate_positive(yield_strength_psi, "yield_strength_psi")
    validate_positive(wall_thickness_in, "wall_thickness_in")
    validate_positive(od_in, "od_in")
    validate_positive(design_factor, "design_factor")
    if wall_thickness_in >= od_in / 2.0:
        raise ValueError(
            f"wall_thickness_in ({wall_thickness_in}) must be less than "
            f"od_in / 2 ({od_in / 2.0})"
        )
    return design_factor * 2.0 * yield_strength_psi * wall_thickness_in / od_in


class CollapseResult(TypedDict):
    collapse_pressure_psi: float
    regime: str
    d_over_t: float


class _API5C3Coeffs(TypedDict):
    A: float
    B: float
    C: float
    F: float
    G: float
    dt_yp: float
    dt_pt: float
    dt_te: float


def _api5c3_coefficients(fy: float) -> _API5C3Coeffs:
    """API 5C3 (1994) collapse coefficients and regime boundaries.

    All closed-form; no numerical solves required. See API Bull 5C3
    sections on plastic, transition, and elastic collapse.
    """
    a = 2.8762 + 0.10679e-5 * fy + 0.21301e-10 * fy**2 - 0.53132e-16 * fy**3
    b = 0.026233 + 0.50609e-6 * fy
    c = -465.93 + 0.030867 * fy - 0.10483e-7 * fy**2 + 0.36989e-13 * fy**3

    ba = b / a
    # Per API 5C3 closed form. Let r = 3*(B/A) / (2 + B/A); then
    #   F = 46.95e6 * r^3 / [Fy * (r - B/A) * (1 - r)^2]
    # The (r - B/A) factor is *not* cancellable — dropping it (as some
    # abbreviated write-ups do) collapses dt_pt onto dt_te and makes
    # the transition regime unreachable.
    r = 3.0 * ba / (2.0 + ba)
    f = 46.95e6 * r**3 / (fy * (r - ba) * (1.0 - r) ** 2)
    g = f * ba

    # Yield-plastic boundary
    bc = b + c / fy
    dt_yp = (math.sqrt((a - 2.0) ** 2 + 8.0 * bc) + (a - 2.0)) / (2.0 * bc)

    # Plastic-transition boundary
    dt_pt = fy * (a - f) / (c + fy * (b - g))

    # Transition-elastic boundary (closed form):
    # elastic == transition: 46.95e6 / (dt*(dt-1)^2) = Fy*(F/dt - G)
    # Standard API 5C3 result:
    dt_te = (2.0 + ba) / (3.0 * ba)

    return {
        "A": a, "B": b, "C": c, "F": f, "G": g,
        "dt_yp": dt_yp, "dt_pt": dt_pt, "dt_te": dt_te,
    }


def collapse_pressure_api5c3(
    od_in: float,
    wall_thickness_in: float,
    yield_strength_psi: float,
) -> CollapseResult:
    """Casing collapse pressure per API Bulletin 5C3 (1994).

    Selects one of four regimes based on the ``D/t`` ratio and applies
    the corresponding formula:

    - **Yield** (thick wall):
        P = 2 * Fy * (D/t - 1) / (D/t)^2
    - **Plastic**:
        P = Fy * (A/(D/t) - B) - C
    - **Transition**:
        P = Fy * (F/(D/t) - G)
    - **Elastic** (thin wall):
        P = 46.95e6 / ((D/t) * (D/t - 1)^2)

    Coefficients A, B, C, F, G and the regime boundaries are derived
    from the grade's minimum yield strength.

    Args:
        od_in: Outer diameter (inches).
        wall_thickness_in: Wall thickness (inches).
        yield_strength_psi: Minimum yield strength (psi).

    Returns:
        Dict with ``collapse_pressure_psi``, ``regime``, and ``d_over_t``.
    """
    validate_positive(od_in, "od_in")
    validate_positive(wall_thickness_in, "wall_thickness_in")
    validate_positive(yield_strength_psi, "yield_strength_psi")
    if wall_thickness_in >= od_in / 2.0:
        raise ValueError(
            f"wall_thickness_in ({wall_thickness_in}) must be less than "
            f"od_in / 2 ({od_in / 2.0})"
        )

    dt = od_in / wall_thickness_in
    coeff = _api5c3_coefficients(yield_strength_psi)
    fy = yield_strength_psi

    if dt <= coeff["dt_yp"]:
        regime = "yield"
        p = 2.0 * fy * (dt - 1.0) / dt**2
    elif dt <= coeff["dt_pt"]:
        regime = "plastic"
        p = fy * (coeff["A"] / dt - coeff["B"]) - coeff["C"]
    elif dt <= coeff["dt_te"]:
        regime = "transition"
        p = fy * (coeff["F"] / dt - coeff["G"])
    else:
        regime = "elastic"
        p = 46.95e6 / (dt * (dt - 1.0) ** 2)

    return {
        "collapse_pressure_psi": max(float(p), 0.0),
        "regime": regime,
        "d_over_t": float(dt),
    }
