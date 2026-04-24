"""Drilling hydraulics — annular velocity, nozzle TFA, bit pressure drop.

Oilfield-unit empirical formulas used in bit-hydraulics optimization:

    AV [ft/min] = 24.51 * Q [gpm] / (Dh² - Dp²) [in²]
    TFA [in²]   = sum( pi/4 * (d/32)² )     (d in 32nds of an inch)
    dP_bit      = MW * Q² / (12032 * TFA²)   [psi]

References:
    Bourgoyne et al., "Applied Drilling Engineering," 1991, Ch. 4.
    IADC Drilling Manual, 12th ed.
"""

from __future__ import annotations

import math
from typing import Sequence

from ._validation import validate_positive


def annular_velocity(
    flow_rate_gpm: float,
    hole_diameter_in: float,
    pipe_od_in: float,
) -> float:
    """Annular velocity in ft/min.

        AV = 24.51 * Q / (Dh² - Dp²)

    Args:
        flow_rate_gpm: Flow rate in gpm.
        hole_diameter_in: Hole or casing ID (inches).
        pipe_od_in: Drill pipe / BHA OD (inches).

    Returns:
        Annular velocity in ft/min.
    """
    validate_positive(flow_rate_gpm, "flow_rate_gpm")
    validate_positive(hole_diameter_in, "hole_diameter_in")
    validate_positive(pipe_od_in, "pipe_od_in")
    if pipe_od_in >= hole_diameter_in:
        raise ValueError(
            f"pipe_od_in ({pipe_od_in}) must be less than "
            f"hole_diameter_in ({hole_diameter_in})"
        )
    annular_area = hole_diameter_in**2 - pipe_od_in**2
    return 24.51 * flow_rate_gpm / annular_area


def nozzle_total_flow_area(nozzle_sizes_32nds: Sequence[int]) -> float:
    """Total flow area (TFA) of bit nozzles.

        TFA [in²] = sum( pi / 4 * (d / 32)² )

    Nozzle sizes are reported in 32nds of an inch in oilfield practice
    (e.g., ``[12, 12, 12]`` means three 12/32" nozzles).

    Args:
        nozzle_sizes_32nds: Iterable of nozzle sizes in 32nds of an inch.

    Returns:
        Total flow area in in².
    """
    sizes = list(nozzle_sizes_32nds)
    if not sizes:
        raise ValueError("nozzle_sizes_32nds must not be empty")
    for i, size in enumerate(sizes):
        if size <= 0:
            raise ValueError(f"nozzle_sizes_32nds[{i}] must be positive, got {size}")
    return sum(math.pi / 4.0 * (size / 32.0) ** 2 for size in sizes)


def bit_pressure_drop(
    flow_rate_gpm: float,
    mud_weight_ppg: float,
    tfa_sqin: float,
) -> float:
    """Bit pressure drop from flow rate, mud weight, and TFA.

        dP_bit [psi] = MW * Q² / (12032 * TFA²)

    Args:
        flow_rate_gpm: Flow rate in gpm.
        mud_weight_ppg: Mud weight in ppg.
        tfa_sqin: Total flow area in in².

    Returns:
        Bit pressure drop in psi.
    """
    validate_positive(flow_rate_gpm, "flow_rate_gpm")
    validate_positive(mud_weight_ppg, "mud_weight_ppg")
    validate_positive(tfa_sqin, "tfa_sqin")
    return mud_weight_ppg * flow_rate_gpm**2 / (12032.0 * tfa_sqin**2)
