"""Erosional velocity per API RP 14E.

    v_e [ft/s] = C / sqrt(rho_mix [lb/ft^3])

The ``C`` factor is dimensionless and depends on service:
    - 100 — continuous service in carbon steel (conservative default)
    - 125 — intermittent service
    - 150-200 — inhibited or corrosion-resistant alloys

API RP 14E is the industry reference but is recognized as
conservative; higher ``C`` values are justified with proper corrosion
and erosion analysis (e.g., CFD of the specific geometry). For a
critical decision use the rigorous model.

Reference:
    API Recommended Practice 14E, "Design and Installation of Offshore
    Production Platform Piping Systems," 5th ed., 1991.
"""

from __future__ import annotations

import math

from ._validation import validate_positive


def api_rp14e_erosional_velocity(
    density_mix_lb_ft3: float,
    c_factor: float = 100.0,
) -> float:
    """API RP 14E erosional velocity from mixture density and C factor.

    Args:
        density_mix_lb_ft3: Mixture (multiphase) density in lb/ft^3.
        c_factor: Erosional C-factor. Default 100 (continuous carbon
            steel service).

    Returns:
        Erosional velocity in ft/s.
    """
    validate_positive(density_mix_lb_ft3, "density_mix_lb_ft3")
    validate_positive(c_factor, "c_factor")
    return c_factor / math.sqrt(density_mix_lb_ft3)
