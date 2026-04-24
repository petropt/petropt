"""Critical (sonic) wellhead choke flow via the Gilbert correlation.

    q_liq [bbl/d] = P_u [psia] * S [64ths]^1.89 / (435 * GLR [scf/bbl]^0.546)

Valid when downstream-to-upstream pressure ratio is below the critical
pressure ratio (~0.55 for oil field wells), at which point flow becomes
sonic at the choke throat and downstream pressure has no effect.

References:
    Gilbert, W.E., "Flowing and Gas-Lift Well Performance," Drilling and
        Production Practice, API, 1954, pp. 126-157.
    Ros, N.C.J., Achong, I., Baxendell, P.B., Poettmann and Beck — all
        proposed alternative constants with different fits; Gilbert is
        the most widely-taught.
"""

from __future__ import annotations

from typing import TypedDict

from ._validation import validate_fraction, validate_positive


class GilbertChokeFlowResult(TypedDict):
    total_liquid_rate_bpd: float
    oil_rate_bopd: float
    water_rate_bwpd: float
    gas_rate_mcfd: float
    effective_glr_scf_bbl: float
    choke_size_inches: float


def gilbert_choke_flow(
    upstream_pressure_psia: float,
    choke_size_64ths: float,
    gor_scf_bbl: float,
    water_cut: float = 0.0,
) -> GilbertChokeFlowResult:
    """Total liquid rate through a critical-flow wellhead choke (Gilbert 1954).

        q_L = P_u * S^1.89 / (435 * GLR^0.546)

    Valid only when the choke is in critical (sonic) flow —
    ``P_downstream / P_upstream < 0.55`` approximately. Confirm before
    applying.

    Note that Gilbert's GOR is reported per barrel of *oil* in the
    original paper; when water is co-produced the effective GLR is
    computed as ``GOR * (1 - water_cut)``, i.e., gas per bbl of total
    liquid.

    Args:
        upstream_pressure_psia: Upstream (tubing head) pressure (psia).
        choke_size_64ths: Choke bean size in 64ths of an inch (e.g.,
            24 for a 24/64" or 3/8" bean).
        gor_scf_bbl: Producing gas-oil ratio (scf/bbl).
        water_cut: Water cut as a fraction (0-1). Default 0.

    Returns:
        Dict with liquid/oil/water/gas rates, effective GLR, and the
        equivalent choke size in inches.
    """
    validate_positive(upstream_pressure_psia, "upstream_pressure_psia")
    validate_positive(choke_size_64ths, "choke_size_64ths")
    validate_positive(gor_scf_bbl, "gor_scf_bbl")
    validate_fraction(water_cut, "water_cut")

    glr = gor_scf_bbl * (1.0 - water_cut)
    if glr <= 0:
        raise ValueError(
            "Effective GLR (GOR * (1 - water_cut)) must be positive for Gilbert"
        )

    q_total = (
        upstream_pressure_psia * choke_size_64ths**1.89 / (435.0 * glr**0.546)
    )
    q_oil = q_total * (1.0 - water_cut)
    q_water = q_total * water_cut
    q_gas_mcfd = q_oil * gor_scf_bbl / 1000.0

    return {
        "total_liquid_rate_bpd": float(q_total),
        "oil_rate_bopd": float(q_oil),
        "water_rate_bwpd": float(q_water),
        "gas_rate_mcfd": float(q_gas_mcfd),
        "effective_glr_scf_bbl": float(glr),
        "choke_size_inches": float(choke_size_64ths / 64.0),
    }
