"""Porosity estimation from density, sonic, and neutron logs.

Correlations:
    - Density porosity: phi_D = (rho_ma - rho_b) / (rho_ma - rho_f)
    - Sonic porosity (Wyllie time-average, 1956):
        phi_S = (dt - dt_ma) / (dt_f - dt_ma)
    - Sonic porosity (Raymer-Hunt-Gardner, 1980):
        phi_S = 0.625 * (dt - dt_ma) / dt
    - Neutron-density RMS combination:
        phi_ND = sqrt((phi_N^2 + phi_D^2) / 2)
    - Effective porosity: phi_e = phi_t * (1 - Vsh)

References:
    Wyllie, M.R.J., Gregory, A.R., and Gardner, L.W., "Elastic Wave
        Velocities in Heterogeneous and Porous Media," Geophysics 21, 1956.
    Raymer, L.L., Hunt, E.R., and Gardner, J.S., "An Improved Sonic Transit
        Time-to-Porosity Transform," SPWLA 21st Logging Symposium, 1980.
    Schlumberger, "Log Interpretation Principles/Applications," 1989.
"""

from __future__ import annotations

import math

from ._validation import clamp, validate_fraction, validate_positive

_SONIC_METHODS = ("wyllie", "raymer")


def density_porosity(
    rhob: float,
    rho_matrix: float = 2.65,
    rho_fluid: float = 1.0,
) -> float:
    """Porosity from the bulk density log.

        phi_D = (rho_ma - rho_b) / (rho_ma - rho_f)

    Default matrix/fluid values are sandstone with fresh water. Use
    ``rho_matrix=2.71`` for limestone, 2.87 for dolomite. Use
    ``rho_fluid`` near 0.85-0.9 for oil-filled mud-filtrate zones or
    the brine density of the filtrate when known.

    Args:
        rhob: Bulk density from log (g/cc).
        rho_matrix: Matrix (grain) density (g/cc). Default 2.65 (quartz).
        rho_fluid: Fluid density in the invaded zone (g/cc). Default 1.0.

    Returns:
        Density porosity as a fraction (0-1). Clamped to [0, 1].
    """
    validate_positive(rhob, "rhob")
    validate_positive(rho_matrix, "rho_matrix")
    validate_positive(rho_fluid, "rho_fluid")
    if rho_matrix <= rho_fluid:
        raise ValueError(
            f"rho_matrix ({rho_matrix}) must exceed rho_fluid ({rho_fluid})"
        )

    return clamp((rho_matrix - rhob) / (rho_matrix - rho_fluid))


def sonic_porosity(
    dt: float,
    dt_matrix: float = 55.5,
    dt_fluid: float = 189.0,
    method: str = "wyllie",
) -> float:
    """Porosity from the compressional sonic log.

    Wyllie (time-average):
        phi_S = (dt - dt_ma) / (dt_f - dt_ma)
    Raymer-Hunt-Gardner (empirical, better for unconsolidated rock):
        phi_S = 0.625 * (dt - dt_ma) / dt

    Default matrix transit time is 55.5 us/ft (sandstone). Use 47.5 for
    limestone or 43.5 for dolomite. Default fluid transit time 189 us/ft
    is the standard Wyllie value for fresh mud filtrate.

    Args:
        dt: Interval transit time from log (us/ft).
        dt_matrix: Matrix transit time (us/ft). Default 55.5.
        dt_fluid: Fluid transit time (us/ft). Default 189.0. Only used
            for the Wyllie method.
        method: "wyllie" or "raymer".

    Returns:
        Sonic porosity as a fraction (0-1). Clamped to [0, 1].
    """
    method = method.lower()
    if method not in _SONIC_METHODS:
        raise ValueError(f"Unknown method '{method}'. Must be one of: {list(_SONIC_METHODS)}")
    validate_positive(dt, "dt")
    validate_positive(dt_matrix, "dt_matrix")
    validate_positive(dt_fluid, "dt_fluid")
    if method == "wyllie" and dt_fluid <= dt_matrix:
        raise ValueError(
            f"dt_fluid ({dt_fluid}) must exceed dt_matrix ({dt_matrix}) for Wyllie"
        )

    if method == "wyllie":
        phi = (dt - dt_matrix) / (dt_fluid - dt_matrix)
    else:
        phi = 0.625 * (dt - dt_matrix) / dt

    return clamp(phi)


def neutron_density_porosity(phi_neutron: float, phi_density: float) -> float:
    """Quick-look porosity from the RMS combination of neutron and density.

        phi_ND = sqrt((phi_N^2 + phi_D^2) / 2)

    This is the common "crossplot" shortcut used in the field. For gas
    zones it under-reads; for shaly zones it over-reads. For rigorous
    analysis apply matrix-specific crossplots; for typical logs this RMS
    combination is the workhorse.

    Args:
        phi_neutron: Neutron porosity (fraction, 0-1).
        phi_density: Density porosity (fraction, 0-1).

    Returns:
        Combined porosity as a fraction (0-1). Clamped to [0, 1].
    """
    # allow slightly negative density porosity (dense rock) by tolerating
    # non-finite inputs explicitly but squaring handles sign
    if not math.isfinite(phi_neutron) or not math.isfinite(phi_density):
        raise ValueError("porosity inputs must be finite numbers")
    return clamp(math.sqrt((phi_neutron**2 + phi_density**2) / 2.0))


def effective_porosity(phi_total: float, vshale: float) -> float:
    """Effective porosity from total porosity and shale volume.

        phi_e = phi_t * (1 - Vsh)

    Args:
        phi_total: Total porosity (fraction, 0-1).
        vshale: Shale volume (fraction, 0-1).

    Returns:
        Effective porosity as a fraction (0-1).
    """
    validate_fraction(phi_total, "phi_total")
    validate_fraction(vshale, "vshale")
    return clamp(phi_total * (1.0 - vshale))
