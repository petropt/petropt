"""Hydraulics correlations for petroleum engineering.

Implements:
    - Darcy-Weisbach equation for pressure drop in pipes
    - Churchill (1977) friction factor correlation

References:
    Churchill, S.W., "Friction Factor Equation Spans All Fluid Flow
    Regimes," Chemical Engineering, 84(24), 1977, pp. 91-92.
"""

from __future__ import annotations

import math


def darcy_weisbach(
    flow_rate: float,
    diameter: float,
    length: float,
    roughness: float = 0.0006,
    density: float = 62.4,
    viscosity: float = 1.0,
) -> dict:
    """Pressure drop using Darcy-Weisbach equation.

    dP = f * (L/D) * (rho * v^2 / 2)

    Uses the Churchill (1977) correlation for friction factor, which is
    valid for all flow regimes (laminar, transitional, turbulent).

    Args:
        flow_rate: Volumetric flow rate in bbl/day.
        diameter: Pipe inner diameter in inches.
        length: Pipe length in feet.
        roughness: Pipe roughness in inches (default 0.0006 for commercial steel).
        density: Fluid density in lb/ft³ (default 62.4 for water).
        viscosity: Fluid viscosity in cp (default 1.0).

    Returns:
        Dict with keys:
            - 'pressure_drop_psi': pressure drop in psi
            - 'friction_factor': Darcy friction factor
            - 'velocity_ft_s': fluid velocity in ft/s
            - 'reynolds_number': Reynolds number
            - 'flow_regime': 'laminar', 'transitional', or 'turbulent'
    """
    if flow_rate <= 0:
        raise ValueError(f"flow_rate must be positive, got {flow_rate}")
    if diameter <= 0:
        raise ValueError(f"diameter must be positive, got {diameter}")
    if length <= 0:
        raise ValueError(f"length must be positive, got {length}")
    if roughness < 0:
        raise ValueError(f"roughness must be non-negative, got {roughness}")
    if density <= 0:
        raise ValueError(f"density must be positive, got {density}")
    if viscosity <= 0:
        raise ValueError(f"viscosity must be positive, got {viscosity}")

    # Convert units
    # bbl/day -> ft³/s: 1 bbl = 5.615 ft³, 1 day = 86400 s
    q_ft3_s = flow_rate * 5.615 / 86400.0
    d_ft = diameter / 12.0  # inches to feet
    area = math.pi / 4.0 * d_ft**2
    velocity = q_ft3_s / area  # ft/s

    # Reynolds number: Re = rho * v * D / mu
    # viscosity: cp -> lb/(ft*s): 1 cp = 6.7197e-4 lb/(ft*s)
    mu_imperial = viscosity * 6.7197e-4
    re = density * velocity * d_ft / mu_imperial

    # Churchill friction factor (valid for all flow regimes)
    e_d = roughness / diameter  # relative roughness (both in inches)

    if re < 1:
        # Essentially no flow
        return {
            "pressure_drop_psi": 0.0,
            "friction_factor": 0.0,
            "velocity_ft_s": round(velocity, 4),
            "reynolds_number": round(re, 1),
            "flow_regime": "laminar",
        }

    # Churchill correlation
    a_term = (-2.457 * math.log(
        (7.0 / re) ** 0.9 + 0.27 * e_d
    )) ** 16
    b_term = (37530.0 / re) ** 16

    f_churchill = 8.0 * (
        (8.0 / re) ** 12
        + 1.0 / (a_term + b_term) ** 1.5
    ) ** (1.0 / 12.0)

    # Pressure drop: dP = f * (L/D) * (rho * v^2 / 2) / 144
    # Division by 144 converts lb/ft² to psi
    dp = f_churchill * (length / d_ft) * (density * velocity**2 / 2.0) / 144.0

    # Flow regime
    if re < 2100:
        regime = "laminar"
    elif re < 4000:
        regime = "transitional"
    else:
        regime = "turbulent"

    return {
        "pressure_drop_psi": round(dp, 4),
        "friction_factor": round(f_churchill, 6),
        "velocity_ft_s": round(velocity, 4),
        "reynolds_number": round(re, 1),
        "flow_regime": regime,
    }
