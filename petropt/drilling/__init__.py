"""Drilling engineering calculations.

Textbook field-unit formulas for well control, hydraulics, and casing
design. All functions return typed scalars or small dicts; no JSON.

    >>> from petropt import drilling as d
    >>> d.hydrostatic_pressure(mud_weight_ppg=10.0, tvd_ft=10000)
    5200.0
    >>> d.kill_mud_weight(sidp_psi=500, original_mud_weight_ppg=10.0, tvd_ft=10000)
    10.96...
    >>> d.burst_pressure_barlow(yield_strength_psi=80_000, wall_thickness_in=0.5, od_in=9.625)
    7272...

Modules:
    wellcontrol  — hydrostatic, ECD, formation pressure gradient, kill
                   MW, ICP/FCP, MAASP
    hydraulics   — annular velocity, nozzle TFA, bit pressure drop
    tubulars     — Barlow burst pressure, API 5C3 collapse pressure
"""

from .hydraulics import (
    annular_velocity,
    bit_pressure_drop,
    nozzle_total_flow_area,
)
from .tubulars import (
    CollapseResult,
    burst_pressure_barlow,
    collapse_pressure_api5c3,
)
from .wellcontrol import (
    PPG_TO_PSI_PER_FT,
    equivalent_circulating_density,
    formation_pressure_gradient,
    hydrostatic_pressure,
    initial_and_final_circulating_pressure,
    kill_mud_weight,
    maasp,
)

__all__ = [
    "CollapseResult",
    "PPG_TO_PSI_PER_FT",
    "annular_velocity",
    "bit_pressure_drop",
    "burst_pressure_barlow",
    "collapse_pressure_api5c3",
    "equivalent_circulating_density",
    "formation_pressure_gradient",
    "hydrostatic_pressure",
    "initial_and_final_circulating_pressure",
    "kill_mud_weight",
    "maasp",
    "nozzle_total_flow_area",
]
