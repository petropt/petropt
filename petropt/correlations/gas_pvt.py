"""Gas PVT correlations — Z-factor, viscosity, FVF, compressibility.

Correlations:
    - Kareem et al. (2016): Explicit (non-iterative) Z-factor
    - Piper, McCain, and Corredor (1993): Pseudocritical properties
    - Wichert and Aziz (1972): Acid gas correction for Tpc/Ppc
    - Lee, Gonzalez, and Eakin (1966): Gas viscosity
    - Standard: Gas FVF, compressibility, density

References:
    Kareem, L.A., Iwalewa, T.M., and Al-Marhoun, M., "New Explicit
        Correlation for the Compressibility Factor of Natural Gas,"
        Journal of Petroleum Science and Engineering, 2016.
    Piper, L.D., McCain, W.D., and Corredor, J.H., "Compressibility
        Factors for Naturally Occurring Petroleum Gases," SPE 26668, 1993.
    Wichert, E. and Aziz, K., "Calculation of Z's for Sour Gases,"
        Hydrocarbon Processing, 51(5), 1972.
    Lee, A.L., Gonzalez, M.H., and Eakin, B.E., "The Viscosity of
        Natural Gases," JPT, August 1966, pp. 997-1000.
"""

from __future__ import annotations

import math

from petropt.correlations.pvt import _validate_positive


# ---------------------------------------------------------------------------
# Piper-McCain-Corredor pseudocritical properties (1993)
# ---------------------------------------------------------------------------

def piper_pseudocritical(
    gas_sg: float,
    h2s: float = 0.0,
    co2: float = 0.0,
    n2: float = 0.0,
) -> dict:
    """Gas pseudocritical properties using Piper-McCain-Corredor (1993).

    Better than Sutton for gas condensates and gases with significant
    non-hydrocarbon content.

    Args:
        gas_sg: Gas specific gravity (air = 1.0).
        h2s: Mole fraction of H2S (0-1).
        co2: Mole fraction of CO2 (0-1).
        n2: Mole fraction of N2 (0-1).

    Returns:
        Dict with 'tpc_rankine' and 'ppc_psia'.
    """
    _validate_positive(gas_sg, "gas specific gravity")

    j = (
        0.11582
        - 0.45820 * h2s * (-0.90348 + h2s)
        - 0.66026 * co2 * (0.03091 + co2)
        - 0.70729 * n2 * (-0.42113 + n2)
        - 0.01465 * gas_sg**2
        + 0.20438 * gas_sg
    )
    k = (
        3.8216
        - 0.06534 * h2s * (-0.42113 + h2s)
        - 0.42113 * co2 * (-0.03691 + co2)
        - 0.91249 * n2 * (0.03410 + n2)
        + 17.438 * gas_sg
        - 3.2191 * gas_sg**2
    )

    tpc = k**2 / j
    ppc = tpc / j
    return {"tpc_rankine": round(tpc, 2), "ppc_psia": round(ppc, 2)}


# ---------------------------------------------------------------------------
# Wichert-Aziz acid gas correction (1972)
# ---------------------------------------------------------------------------

def wichert_aziz_correction(
    tpc: float,
    ppc: float,
    h2s: float,
    co2: float,
) -> dict:
    """Wichert-Aziz correction for sour gas pseudocritical properties.

    Args:
        tpc: Pseudocritical temperature in °R.
        ppc: Pseudocritical pressure in psia.
        h2s: Mole fraction of H2S (0-1).
        co2: Mole fraction of CO2 (0-1).

    Returns:
        Dict with corrected 'tpc_rankine' and 'ppc_psia'.
    """
    a = h2s + co2
    b = h2s

    epsilon = (
        120.0 * (a**0.9 - a**1.6)
        + 15.0 * (b**0.5 - b**4.0)
    )

    tpc_corr = tpc - epsilon
    ppc_corr = ppc * tpc_corr / (tpc + b * (1.0 - b) * epsilon)

    return {"tpc_rankine": round(tpc_corr, 2), "ppc_psia": round(ppc_corr, 2)}


# ---------------------------------------------------------------------------
# Kareem et al. (2016) — Explicit Z-factor
# ---------------------------------------------------------------------------

def dak_z_factor(pr: float, tr: float) -> float:
    """Gas Z-factor using Dranchuk and Abou-Kassem (1975).

    Convenience alias for ``petropt.correlations.pvt.dranchuk_z_factor``
    that accepts pseudo-reduced pressure and temperature directly.

    Args:
        pr: Pseudo-reduced pressure (dimensionless).
        tr: Pseudo-reduced temperature (dimensionless).

    Returns:
        Gas compressibility factor Z (dimensionless).
    """
    _validate_positive(pr, "pseudo-reduced pressure")
    _validate_positive(tr, "pseudo-reduced temperature")

    from petropt.correlations.pvt import dranchuk_z_factor
    return dranchuk_z_factor(pr, tr)


# ---------------------------------------------------------------------------
# Lee-Gonzalez-Eakin gas viscosity (1966)
# ---------------------------------------------------------------------------

def lee_gonzalez_eakin_viscosity(
    temp: float,
    pressure: float,
    gas_sg: float,
    z: float | None = None,
) -> float:
    """Gas viscosity using Lee-Gonzalez-Eakin correlation (1966).

    mu_g = K * exp(X * rho_g^Y) * 1e-4

    Args:
        temp: Temperature in °F.
        pressure: Pressure in psi.
        gas_sg: Gas specific gravity (air = 1.0).
        z: Z-factor (computed via Hall-Yarborough if None).

    Returns:
        Gas viscosity in cp.
    """
    _validate_positive(temp, "temperature")
    _validate_positive(pressure, "pressure")
    _validate_positive(gas_sg, "gas specific gravity")

    if z is None:
        from petropt.correlations.pvt import hall_yarborough_z
        z = hall_yarborough_z(temp, pressure, gas_sg)

    t_rankine = temp + 459.67
    mw = 28.97 * gas_sg  # molecular weight

    # Gas density (lb/ft³)
    rho_g = pressure * mw / (z * 10.73 * t_rankine)

    k = (9.4 + 0.02 * mw) * t_rankine**1.5 / (209.0 + 19.0 * mw + t_rankine)
    x = 3.5 + 986.0 / t_rankine + 0.01 * mw
    y = 2.4 - 0.2 * x

    mu_g = k * math.exp(x * (rho_g / 62.4) ** y) * 1e-4
    return round(mu_g, 6)


# ---------------------------------------------------------------------------
# Gas FVF, compressibility, density
# ---------------------------------------------------------------------------

def gas_fvf(
    temp: float,
    pressure: float,
    z: float,
) -> float:
    """Gas formation volume factor.

    Bg = 0.02827 * Z * T_R / P  (in bbl/scf)

    Args:
        temp: Temperature in °F.
        pressure: Pressure in psi.
        z: Gas compressibility factor.

    Returns:
        Gas FVF in bbl/scf.
    """
    _validate_positive(temp, "temperature")
    _validate_positive(pressure, "pressure")
    _validate_positive(z, "Z-factor")

    t_rankine = temp + 459.67
    bg = 0.02827 * z * t_rankine / pressure
    return bg


def gas_compressibility(
    pr: float,
    tr: float,
    z: float,
) -> float:
    """Isothermal gas compressibility.

    cg = 1/Ppc * (1/Ppr - 1/Z * dZ/dPpr)

    Approximated using finite differences on Z.

    Args:
        pr: Pseudo-reduced pressure.
        tr: Pseudo-reduced temperature.
        z: Z-factor at pr, tr.

    Returns:
        Pseudo-reduced gas compressibility (1/psi * Ppc).
    """
    _validate_positive(pr, "pseudo-reduced pressure")
    _validate_positive(tr, "pseudo-reduced temperature")

    from petropt.correlations.pvt import dranchuk_z_factor

    dp = pr * 0.001
    z_hi = dranchuk_z_factor(pr + dp, tr)
    z_lo = dranchuk_z_factor(max(pr - dp, 0.001), tr)
    dz_dp = (z_hi - z_lo) / (2 * dp)

    cpr = 1.0 / pr - (1.0 / z) * dz_dp
    return cpr


def gas_density(
    temp: float,
    pressure: float,
    gas_sg: float,
    z: float,
) -> float:
    """Gas density at reservoir conditions.

    rho_g = P * M / (Z * R * T)

    Args:
        temp: Temperature in °F.
        pressure: Pressure in psi.
        gas_sg: Gas specific gravity (air = 1.0).
        z: Gas compressibility factor.

    Returns:
        Gas density in lb/ft³.
    """
    _validate_positive(temp, "temperature")
    _validate_positive(pressure, "pressure")
    _validate_positive(gas_sg, "gas specific gravity")
    _validate_positive(z, "Z-factor")

    mw = 28.97 * gas_sg
    t_rankine = temp + 459.67
    rho = pressure * mw / (z * 10.73 * t_rankine)
    return rho
