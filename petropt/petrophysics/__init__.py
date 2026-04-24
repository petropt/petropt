"""Petrophysics — log interpretation correlations.

Textbook-standard single-log calculations for shale volume, porosity,
water saturation, permeability, and net pay.

    >>> from petropt import petrophysics as pp
    >>> pp.vshale_from_gr(gr=75, gr_clean=20, gr_shale=120, method="larionov_tertiary")
    0.2571...
    >>> pp.archie_sw(rt=20.0, phi=0.20, rw=0.05)
    0.25

Modules:
    vshale        — shale volume from gamma ray (linear, Larionov, Clavier)
    porosity      — density, sonic (Wyllie / RHG), neutron-density, effective
    saturation    — Archie, Simandoux, Indonesian (Poupon-Leveaux)
    permeability  — Timur, Coates
    pay           — net pay summation, hydrocarbon pore thickness

All functions return typed scalars or dicts. No JSON serialization;
that's the caller's job.
"""

from .pay import NetPayResult, hydrocarbon_pore_thickness, net_pay
from .permeability import coates_permeability, timur_permeability
from .porosity import (
    density_porosity,
    effective_porosity,
    neutron_density_porosity,
    sonic_porosity,
)
from .saturation import archie_sw, indonesian_sw, simandoux_sw
from .vshale import gamma_ray_index, vshale_from_gr

__all__ = [
    "NetPayResult",
    "archie_sw",
    "coates_permeability",
    "density_porosity",
    "effective_porosity",
    "gamma_ray_index",
    "hydrocarbon_pore_thickness",
    "indonesian_sw",
    "net_pay",
    "neutron_density_porosity",
    "simandoux_sw",
    "sonic_porosity",
    "timur_permeability",
    "vshale_from_gr",
]
