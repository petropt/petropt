"""Water/brine PVT correlations.

Correlations:
    - McCain (1990): Brine density, viscosity, FVF
    - Osif (1988): Brine compressibility
    - Standard: Water gas solubility

References:
    McCain, W.D., "The Properties of Petroleum Fluids," PennWell, 1990.
    Osif, T.L., "The Effects of Salt, Gas, Temperature, and Pressure on the
        Compressibility of Water," SPE Reservoir Engineering, Feb 1988.
"""

from __future__ import annotations

import math

from petropt.correlations.pvt import _validate_positive, _validate_non_negative


def water_fvf(
    temp: float,
    pressure: float,
    salinity: float = 0.0,
) -> float:
    """Water/brine formation volume factor.

    Bw = (1 + dVwP)(1 + dVwT)

    Args:
        temp: Temperature in °F.
        pressure: Pressure in psi.
        salinity: Salinity in ppm (default 0 = fresh water).

    Returns:
        Water FVF in bbl/STB.
    """
    _validate_positive(temp, "temperature")
    _validate_positive(pressure, "pressure")
    _validate_non_negative(salinity, "salinity")

    # Temperature effect on volume
    dvwt = (
        -1.0001e-2
        + 1.33391e-4 * temp
        + 5.50654e-7 * temp**2
    )

    # Pressure effect on volume
    dvwp = (
        -1.95301e-9 * pressure * temp
        - 1.72834e-13 * pressure**2 * temp
        - 3.58922e-7 * pressure
        - 2.25341e-10 * pressure**2
    )

    bw = (1.0 + dvwp) * (1.0 + dvwt)

    # Salinity correction
    if salinity > 0:
        s = salinity / 1e6  # convert ppm to fraction
        bw *= (1.0 + s * (
            5.1e-8 * pressure
            + (5.47e-6 - 1.95e-10 * pressure) * (temp - 60.0)
            - 3.23e-8 * (temp - 60.0)**2
        ))

    return max(bw, 0.9)


def water_viscosity(
    temp: float,
    pressure: float = 14.7,
    salinity: float = 0.0,
) -> float:
    """Water/brine viscosity using McCain (1990).

    Args:
        temp: Temperature in °F.
        pressure: Pressure in psi.
        salinity: Salinity in ppm (default 0).

    Returns:
        Water viscosity in cp.
    """
    _validate_positive(temp, "temperature")
    _validate_positive(pressure, "pressure")
    _validate_non_negative(salinity, "salinity")

    s = salinity / 1e6  # ppm to fraction

    # Fresh water viscosity at 1 atm
    a = 109.574 - 8.40564 * s + 0.313314 * s**2 + 8.72213e-3 * s**3
    b = -1.12166 + 2.63951e-2 * s - 6.79461e-4 * s**2 - 5.47119e-5 * s**3 + 1.55586e-6 * s**4

    mu_w1 = a * temp**b

    # Pressure correction
    mu_w = mu_w1 * (0.9994 + 4.0295e-5 * pressure + 3.1062e-9 * pressure**2)

    return max(mu_w, 0.1)


def water_compressibility(
    temp: float,
    pressure: float,
    salinity: float = 0.0,
    rsw: float = 0.0,
) -> float:
    """Water compressibility using Osif (1988).

    cw = 1 / (7.033*P + 0.5415*S - 537*T + 403300)

    Args:
        temp: Temperature in °F.
        pressure: Pressure in psi.
        salinity: Salinity in ppm.
        rsw: Dissolved gas in water in scf/STB.

    Returns:
        Water compressibility in 1/psi.
    """
    _validate_positive(temp, "temperature")
    _validate_positive(pressure, "pressure")
    _validate_non_negative(salinity, "salinity")

    s = salinity / 1e6  # ppm to fraction
    s_percent = s * 100  # Osif uses weight percent

    cw = 1.0 / (7.033 * pressure + 0.5415 * s_percent - 537.0 * temp + 403300.0)

    # Gas correction
    if rsw > 0:
        cw *= (1.0 + 8.9e-3 * rsw)

    return max(cw, 1e-7)


def water_gas_solubility(
    temp: float,
    pressure: float,
    salinity: float = 0.0,
) -> float:
    """Gas solubility in water/brine.

    Args:
        temp: Temperature in °F.
        pressure: Pressure in psi.
        salinity: Salinity in ppm.

    Returns:
        Gas solubility Rsw in scf/STB.
    """
    _validate_positive(temp, "temperature")
    _validate_positive(pressure, "pressure")
    _validate_non_negative(salinity, "salinity")

    # Fresh water gas solubility (McCain 1990)
    a = 8.15839 - 6.12265e-2 * temp + 1.91663e-4 * temp**2 - 2.1654e-7 * temp**3
    b = 1.01021e-2 - 7.44241e-5 * temp + 3.05553e-7 * temp**2 - 2.94883e-10 * temp**3
    c = (-9.02505 + 0.130237 * temp - 8.53425e-4 * temp**2
         + 2.34122e-6 * temp**3 - 2.37049e-9 * temp**4) * 1e-7

    rsw_pure = a + b * pressure + c * pressure**2

    # Salinity correction
    s = salinity / 1e6
    rsw = rsw_pure * (1.0 - (0.0840655 * s * temp**0.285854))

    return max(rsw, 0.0)


def water_density(
    temp: float,
    pressure: float,
    salinity: float = 0.0,
) -> float:
    """Water/brine density at reservoir conditions.

    Args:
        temp: Temperature in °F.
        pressure: Pressure in psi.
        salinity: Salinity in ppm.

    Returns:
        Water density in lb/ft³.
    """
    _validate_positive(temp, "temperature")
    _validate_positive(pressure, "pressure")
    _validate_non_negative(salinity, "salinity")

    s = salinity / 1e6

    # Water density at standard conditions (62.4 lb/ft³ for fresh water)
    rho_w_sc = 62.368 + 0.438603 * s + 1.60074e-3 * s**2

    bw = water_fvf(temp, pressure, salinity)
    return rho_w_sc / bw
