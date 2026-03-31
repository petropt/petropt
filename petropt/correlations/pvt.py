"""PVT (Pressure-Volume-Temperature) black-oil correlations.

Implements standard petroleum engineering correlations for estimating
fluid properties from readily available field data.

Correlations:
    - Standing (1947): Bubble point pressure, solution GOR, oil FVF
    - Beggs and Robinson (1975): Dead and live oil viscosity
    - Sutton (1985): Gas pseudocritical properties
    - Hall and Yarborough (1973): Gas Z-factor
    - Dranchuk and Abou-Kassem (1975): Gas Z-factor

References:
    Standing, M.B., "A Pressure-Volume-Temperature Correlation for Mixtures
        of California Oils and Gases," API Drilling and Production Practice, 1947.
    Beggs, H.D. and Robinson, J.R., "Estimating the Viscosity of Crude Oil
        Systems," JPT, September 1975, pp. 1140-1141.
    Sutton, R.P., "Compressibility Factors for High-Molecular-Weight Reservoir
        Gases," SPE 14265, 1985.
    Hall, K.R. and Yarborough, L., "A New Equation of State for Z-Factor
        Calculations," Oil and Gas Journal, June 1973.
    Dranchuk, P.M. and Abou-Kassem, J.H., "Calculation of Z Factors for
        Natural Gases Using Equations of State," JCPT, July-September 1975.
"""

from __future__ import annotations

import math


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def _validate_positive(value: float, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def _validate_non_negative(value: float, name: str) -> None:
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")


# ---------------------------------------------------------------------------
# Standing (1947) — Bubble point, solution GOR, oil FVF
# ---------------------------------------------------------------------------

def standing_bubble_point(
    api: float,
    gas_sg: float,
    temp: float,
    rs: float = 500.0,
) -> float:
    """Bubble point pressure using Standing's correlation (1947).

    Pb = 18.2 * ((Rs/Sg)^0.83 * 10^(0.00091*T - 0.0125*API) - 1.4)

    Args:
        api: Oil API gravity (dimensionless).
        gas_sg: Gas specific gravity (air = 1.0).
        temp: Temperature in °F.
        rs: Solution gas-oil ratio in scf/STB (default 500).

    Returns:
        Bubble point pressure in psi.

    Raises:
        ValueError: If any input is out of physical range.
    """
    _validate_positive(api, "API gravity")
    _validate_positive(gas_sg, "gas specific gravity")
    _validate_positive(temp, "temperature")
    _validate_non_negative(rs, "solution GOR")

    if rs == 0:
        return 14.7  # atmospheric

    exponent = 0.00091 * temp - 0.0125 * api
    pb = 18.2 * ((rs / gas_sg) ** 0.83 * 10**exponent - 1.4)
    return max(pb, 14.7)


def standing_rs(
    pressure: float,
    temp: float,
    api: float,
    gas_sg: float,
) -> float:
    """Solution gas-oil ratio using Standing's correlation (1947).

    Rs = Sg * ((P/18.2 + 1.4) * 10^(0.0125*API - 0.00091*T))^1.2048

    Args:
        pressure: Pressure in psi.
        temp: Temperature in °F.
        api: Oil API gravity.
        gas_sg: Gas specific gravity (air = 1.0).

    Returns:
        Solution GOR in scf/STB.
    """
    _validate_positive(pressure, "pressure")
    _validate_positive(temp, "temperature")
    _validate_positive(api, "API gravity")
    _validate_positive(gas_sg, "gas specific gravity")

    exponent = 0.0125 * api - 0.00091 * temp
    rs = gas_sg * ((pressure / 18.2 + 1.4) * 10**exponent) ** 1.2048
    return max(rs, 0.0)


def standing_bo(
    rs: float,
    temp: float,
    api: float,
    gas_sg: float,
) -> float:
    """Oil formation volume factor using Standing's correlation (1947).

    Bo = 0.9759 + 0.000120 * (Rs * (Sg/So)^0.5 + 1.25*T)^1.2

    Args:
        rs: Solution gas-oil ratio in scf/STB.
        temp: Temperature in °F.
        api: Oil API gravity.
        gas_sg: Gas specific gravity (air = 1.0).

    Returns:
        Oil FVF in bbl/STB.
    """
    _validate_non_negative(rs, "solution GOR")
    _validate_positive(temp, "temperature")
    _validate_positive(api, "API gravity")
    _validate_positive(gas_sg, "gas specific gravity")

    oil_sg = 141.5 / (api + 131.5)
    f_factor = rs * (gas_sg / oil_sg) ** 0.5 + 1.25 * temp
    bo = 0.9759 + 0.000120 * f_factor**1.2
    return bo


# ---------------------------------------------------------------------------
# Beggs and Robinson (1975) — Oil viscosity
# ---------------------------------------------------------------------------

def beggs_robinson_viscosity(
    temp: float,
    api: float,
    rs: float = 0.0,
) -> dict:
    """Oil viscosity using Beggs and Robinson correlation (1975).

    Args:
        temp: Temperature in °F.
        api: Oil API gravity.
        rs: Solution gas-oil ratio in scf/STB (0 for dead oil).

    Returns:
        Dict with 'dead_oil_viscosity_cp' and 'live_oil_viscosity_cp'.
    """
    _validate_positive(temp, "temperature")
    _validate_positive(api, "API gravity")
    _validate_non_negative(rs, "solution GOR")

    # Dead oil viscosity
    x = 10 ** (3.0324 - 0.02023 * api) * temp ** (-1.163)
    mu_od = 10**x - 1.0

    result = {"dead_oil_viscosity_cp": round(mu_od, 4)}

    # Live oil viscosity
    if rs > 0:
        a = 10.715 * (rs + 100.0) ** (-0.515)
        b = 5.44 * (rs + 150.0) ** (-0.338)
        mu_o = a * mu_od**b
        result["live_oil_viscosity_cp"] = round(mu_o, 4)
    else:
        result["live_oil_viscosity_cp"] = round(mu_od, 4)

    return result


# ---------------------------------------------------------------------------
# Sutton (1985) — Gas pseudocritical properties
# ---------------------------------------------------------------------------

def sutton_pseudocritical(gas_sg: float) -> dict:
    """Gas pseudocritical properties using Sutton's correlation (1985).

    Args:
        gas_sg: Gas specific gravity (air = 1.0).

    Returns:
        Dict with 'tpc_rankine' and 'ppc_psia'.
    """
    _validate_positive(gas_sg, "gas specific gravity")

    tpc = 169.2 + 349.5 * gas_sg - 74.0 * gas_sg**2
    ppc = 756.8 - 131.0 * gas_sg - 3.6 * gas_sg**2
    return {"tpc_rankine": round(tpc, 2), "ppc_psia": round(ppc, 2)}


# ---------------------------------------------------------------------------
# Hall and Yarborough (1973) — Z-factor
# ---------------------------------------------------------------------------

def hall_yarborough_z(
    temp: float,
    pressure: float,
    gas_sg: float,
) -> float:
    """Gas Z-factor using Hall-Yarborough method (1973).

    Uses Newton-Raphson iteration to solve the Hall-Yarborough EOS
    for reduced density, then calculates Z.

    Args:
        temp: Temperature in °F.
        pressure: Pressure in psi.
        gas_sg: Gas specific gravity (air = 1.0).

    Returns:
        Gas compressibility factor Z (dimensionless).
    """
    _validate_positive(temp, "temperature")
    _validate_positive(pressure, "pressure")
    _validate_positive(gas_sg, "gas specific gravity")

    tpc = 169.2 + 349.5 * gas_sg - 74.0 * gas_sg**2
    ppc = 756.8 - 131.0 * gas_sg - 3.6 * gas_sg**2

    t_rankine = temp + 459.67
    t_pr = t_rankine / tpc
    p_pr = pressure / ppc

    t_inv = 1.0 / t_pr

    a1 = -0.06125 * p_pr * t_inv * math.exp(-1.2 * (1.0 - t_inv) ** 2)
    a2 = 14.76 * t_inv - 9.76 * t_inv**2 + 4.58 * t_inv**3
    a3 = 90.7 * t_inv - 242.2 * t_inv**2 + 42.4 * t_inv**3
    a4 = 2.18 + 2.82 * t_inv

    y = 0.001
    for _ in range(100):
        fy = (
            a1
            + (y + y**2 + y**3 - y**4) / (1.0 - y) ** 3
            - a2 * y**2
            + a3 * y**a4
        )
        dfy = (
            (1.0 + 4.0 * y + 4.0 * y**2 - 4.0 * y**3 + y**4)
            / (1.0 - y) ** 4
            - 2.0 * a2 * y
            + a3 * a4 * y ** (a4 - 1.0)
        )
        if abs(dfy) < 1e-30:
            break
        y_new = y - fy / dfy
        y_new = max(y_new, 1e-10)
        y_new = min(y_new, 0.9999)
        if abs(y_new - y) < 1e-12:
            y = y_new
            break
        y = y_new

    z = -a1 / y if y > 1e-15 else 1.0
    return max(z, 0.05)


# ---------------------------------------------------------------------------
# Dranchuk and Abou-Kassem (1975) — Z-factor
# ---------------------------------------------------------------------------

def dranchuk_z_factor(
    pr: float,
    tr: float,
) -> float:
    """Gas Z-factor using Dranchuk and Abou-Kassem correlation (1975).

    11-coefficient equation fitted to Standing-Katz chart data.
    Uses Newton-Raphson iteration on reduced density.

    Valid for: 1.0 <= Ppr <= 30, 1.0 <= Tpr <= 3.0

    Args:
        pr: Pseudo-reduced pressure (P/Ppc), dimensionless.
        tr: Pseudo-reduced temperature (T/Tpc), dimensionless.

    Returns:
        Gas compressibility factor Z (dimensionless).

    Raises:
        ValueError: If inputs are out of valid range.
    """
    _validate_positive(pr, "pseudo-reduced pressure")
    _validate_positive(tr, "pseudo-reduced temperature")

    # DAK coefficients
    A1, A2, A3, A4, A5 = 0.3265, -1.0700, -0.5339, 0.01569, -0.05165
    A6, A7, A8 = 0.5475, -0.7361, 0.6853
    A9, A10, A11 = 0.1056, 0.6134, 0.7210

    # Initial guess: rho_r = 0.27 * Ppr / (Z * Tpr), start with Z=1
    rho_r = 0.27 * pr / tr

    for _ in range(100):
        rho_r2 = rho_r * rho_r
        rho_r5 = rho_r**5

        c1 = A1 + A2 / tr + A3 / tr**3 + A4 / tr**4 + A5 / tr**5
        c2 = A6 + A7 / tr + A8 / tr**2
        c3 = A9 * (A7 / tr + A8 / tr**2)
        c4 = A10 * (1.0 + A11 * rho_r2) * (rho_r2 / tr**3) * math.exp(
            -A11 * rho_r2
        )

        f = (
            0.27 * pr / (rho_r * tr)
            - 1.0
            - c1 * rho_r
            - c2 * rho_r2
            + c3 * rho_r5
            - c4
        )

        dc4_drho = A10 / tr**3 * math.exp(-A11 * rho_r2) * (
            2.0 * rho_r * (1.0 + A11 * rho_r2)
            + rho_r2 * 2.0 * A11 * rho_r
            - (1.0 + A11 * rho_r2) * rho_r2 * 2.0 * A11 * rho_r
        )

        df = (
            -0.27 * pr / (rho_r2 * tr)
            - c1
            - 2.0 * c2 * rho_r
            + 5.0 * c3 * rho_r**4
            - dc4_drho
        )

        if abs(df) < 1e-30:
            break

        rho_new = rho_r - f / df
        rho_new = max(rho_new, 1e-10)
        rho_new = min(rho_new, 5.0)

        if abs(rho_new - rho_r) < 1e-12:
            rho_r = rho_new
            break
        rho_r = rho_new

    z = 0.27 * pr / (rho_r * tr) if rho_r > 1e-15 else 1.0
    return max(z, 0.05)
