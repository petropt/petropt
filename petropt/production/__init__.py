"""Production engineering — liquid loading, hydrates, erosion, choke flow.

Textbook single-formula tools for day-to-day production surveillance.
Beggs-Brill multiphase pressure drop lives separately in
``petropt.correlations.multiphase``.

    >>> from petropt import production as prod
    >>> v_c = prod.turner_critical_velocity(
    ...     surface_tension_dyne_cm=60, rho_liquid_lb_ft3=65, rho_gas_lb_ft3=0.5
    ... )
    >>> t_h = prod.katz_hydrate_temperature(pressure_psi=500, gas_sg=0.7)
    >>> q = prod.gilbert_choke_flow(
    ...     upstream_pressure_psia=2500, choke_size_64ths=32, gor_scf_bbl=800
    ... )

Modules:
    liquid_loading — Turner / Coleman droplet-lift critical velocity
    hydrates       — Katz hydrate temperature, Hammerschmidt inhibitor
    erosion        — API RP 14E erosional velocity
    choke          — Gilbert critical-flow choke correlation
"""

from .choke import GilbertChokeFlowResult, gilbert_choke_flow
from .erosion import api_rp14e_erosional_velocity
from .hydrates import (
    InhibitorDosingResult,
    hammerschmidt_inhibitor_dosing,
    katz_hydrate_temperature,
)
from .liquid_loading import (
    CriticalLoadingResult,
    coleman_critical_velocity,
    critical_gas_rate_mcfd,
    turner_critical_velocity,
)

__all__ = [
    "CriticalLoadingResult",
    "GilbertChokeFlowResult",
    "InhibitorDosingResult",
    "api_rp14e_erosional_velocity",
    "coleman_critical_velocity",
    "critical_gas_rate_mcfd",
    "gilbert_choke_flow",
    "hammerschmidt_inhibitor_dosing",
    "katz_hydrate_temperature",
    "turner_critical_velocity",
]
