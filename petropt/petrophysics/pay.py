"""Net pay summation and hydrocarbon pore thickness (HPT).

A depth-indexed log is reduced to a scalar by applying porosity, water
saturation, and shale cutoffs. Hydrocarbon pore thickness folds the
average rock properties into a single thickness-equivalent.

Formulas:
    Pay flag at depth i: (phi_i >= phi_cut) AND (sw_i <= sw_cut) AND (vsh_i <= vsh_cut)
    Net thickness:        sum of per-sample step thickness where pay flag is True
    HPT:                  thickness * phi * (1 - Sw) * NTG
"""

from __future__ import annotations

from typing import Sequence, TypedDict

import numpy as np

from ._validation import validate_fraction, validate_positive

_EPS = 1e-9


class NetPayResult(TypedDict, total=False):
    net_pay_ft: float
    gross_thickness_ft: float
    net_to_gross: float
    avg_porosity_pay: float | None
    avg_sw_pay: float | None
    avg_vshale_pay: float | None
    num_pay_samples: int
    pay_flags: list[bool]


def net_pay(
    depths: Sequence[float],
    phi: Sequence[float],
    sw: Sequence[float],
    vshale: Sequence[float],
    phi_cutoff: float = 0.06,
    sw_cutoff: float = 0.5,
    vsh_cutoff: float = 0.5,
) -> NetPayResult:
    """Summarize net pay from depth-indexed porosity, Sw, and Vshale arrays.

    Applies the three standard reservoir cutoffs and integrates the pay
    thickness using per-sample step sizes. The last sample reuses the
    prior step (the common depth-step convention).

    Args:
        depths: Measured depths (ft), monotonic.
        phi: Porosity at each depth (fraction, 0-1).
        sw: Water saturation at each depth (fraction, 0-1).
        vshale: Shale volume at each depth (fraction, 0-1).
        phi_cutoff: Minimum porosity to count as pay. Default 0.06.
        sw_cutoff: Maximum water saturation to count as pay. Default 0.5.
        vsh_cutoff: Maximum shale volume to count as pay. Default 0.5.

    Returns:
        Dict with net pay thickness, gross thickness, net-to-gross,
        pay-weighted averages of phi/Sw/Vsh, per-sample pay flags, and
        sample count. Averages are ``None`` if no samples pass cutoffs.
    """
    validate_fraction(phi_cutoff, "phi_cutoff")
    validate_fraction(sw_cutoff, "sw_cutoff")
    validate_fraction(vsh_cutoff, "vsh_cutoff")

    d = np.asarray(depths, dtype=float)
    p = np.asarray(phi, dtype=float)
    s = np.asarray(sw, dtype=float)
    v = np.asarray(vshale, dtype=float)

    n = d.size
    if n < 2:
        raise ValueError("At least 2 depth points required")
    if p.size != n or s.size != n or v.size != n:
        raise ValueError("depths, phi, sw, vshale must all have the same length")

    if not (np.isfinite(d).all() and np.isfinite(p).all()
            and np.isfinite(s).all() and np.isfinite(v).all()):
        raise ValueError("all inputs must be finite (no NaN or inf)")

    steps = np.diff(d)
    if (steps > _EPS).all():
        direction = 1
    elif (steps < -_EPS).all():
        direction = -1
    else:
        raise ValueError(
            "depths must be monotonic (strictly increasing or decreasing)"
        )

    if ((p < 0) | (p > 1)).any():
        raise ValueError("all phi values must be in [0, 1]")
    if ((s < 0) | (s > 1)).any():
        raise ValueError("all sw values must be in [0, 1]")
    if ((v < 0) | (v > 1)).any():
        raise ValueError("all vshale values must be in [0, 1]")

    steps = np.abs(steps)
    thicknesses = np.concatenate([steps, steps[-1:]])  # last sample reuses prior step
    gross = float(thicknesses.sum())

    pay = (p >= phi_cutoff) & (s <= sw_cutoff) & (v <= vsh_cutoff)
    pay_thick = float((thicknesses * pay).sum())

    result: NetPayResult = {
        "net_pay_ft": pay_thick,
        "gross_thickness_ft": gross,
        "net_to_gross": (pay_thick / gross) if gross > 0 else 0.0,
        "num_pay_samples": int(pay.sum()),
        "pay_flags": pay.tolist(),
    }

    if pay_thick > 0:
        weights = thicknesses * pay
        result["avg_porosity_pay"] = float((p * weights).sum() / pay_thick)
        result["avg_sw_pay"] = float((s * weights).sum() / pay_thick)
        result["avg_vshale_pay"] = float((v * weights).sum() / pay_thick)
    else:
        result["avg_porosity_pay"] = None
        result["avg_sw_pay"] = None
        result["avg_vshale_pay"] = None

    return result


def hydrocarbon_pore_thickness(
    thickness: float,
    phi: float,
    sw: float,
    ntg: float = 1.0,
) -> float:
    """Hydrocarbon pore thickness (HPT).

        HPT = thickness * phi * (1 - Sw) * NTG

    HPT is the "how thick would it be if it were pure hydrocarbon"
    reduction of a pay interval — useful for quick volumetric comparisons.

    Args:
        thickness: Net or gross thickness (ft).
        phi: Average porosity (fraction, 0-1).
        sw: Average water saturation (fraction, 0-1).
        ntg: Net-to-gross ratio (fraction, 0-1). Default 1.0 (applied
            when ``thickness`` is gross; pass 1.0 if ``thickness`` is
            already net).

    Returns:
        Hydrocarbon pore thickness in feet.
    """
    validate_positive(thickness, "thickness")
    validate_fraction(phi, "phi")
    validate_fraction(sw, "sw")
    validate_fraction(ntg, "ntg")
    return thickness * phi * (1.0 - sw) * ntg
