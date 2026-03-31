"""Beggs and Brill (1973) multiphase pipe flow correlation.

Calculates pressure gradient for two-phase (gas-liquid) flow in pipes
at any inclination angle.

Reference:
    Beggs, H.D. and Brill, J.P., "A Study of Two-Phase Flow in Inclined
    Pipes," JPT, May 1973, pp. 607-617.
    Payne, G.A., Palmer, C.M., Brill, J.P., and Beggs, H.D., "Evaluation
    of Inclined-Pipe, Two-Phase Liquid Holdup and Pressure-Loss Correlations
    Using Experimental Data," JPT, September 1979.
"""

from __future__ import annotations

import math

import numpy as np


def beggs_brill_pressure_gradient(
    pressure: float,
    temp: float,
    oil_rate: float,
    water_rate: float,
    gor: float,
    oil_sg: float,
    gas_sg: float,
    water_sg: float = 1.07,
    pipe_id: float = 2.441,
    angle: float = 90.0,
    roughness: float = 0.0006,
) -> dict:
    """Pressure gradient using Beggs and Brill (1973) correlation.

    Args:
        pressure: Flowing pressure in psi.
        temp: Temperature in °F.
        oil_rate: Oil rate in STB/day.
        water_rate: Water rate in STB/day.
        gor: Gas-oil ratio in scf/STB.
        oil_sg: Oil specific gravity (water = 1.0).
        gas_sg: Gas specific gravity (air = 1.0).
        water_sg: Water specific gravity (default 1.07).
        pipe_id: Pipe inner diameter in inches.
        angle: Pipe inclination from horizontal in degrees (90 = vertical).
        roughness: Pipe roughness in inches.

    Returns:
        Dict with:
            - 'dp_dz_psi_ft': total pressure gradient (psi/ft)
            - 'dp_dz_gravity': gravity component
            - 'dp_dz_friction': friction component
            - 'flow_pattern': 'segregated', 'intermittent', 'distributed', or 'transition'
            - 'liquid_holdup': liquid holdup fraction
            - 'mixture_velocity_ft_s': superficial mixture velocity
    """
    if oil_rate <= 0 and water_rate <= 0:
        raise ValueError("At least one liquid rate must be positive")
    if pressure <= 0:
        raise ValueError(f"pressure must be positive, got {pressure}")
    if temp <= 0:
        raise ValueError(f"temp must be positive (°F), got {temp}")
    if oil_sg <= 0:
        raise ValueError(f"oil_sg must be positive, got {oil_sg}")
    if gas_sg <= 0:
        raise ValueError(f"gas_sg must be positive, got {gas_sg}")
    if gor < 0:
        raise ValueError(f"gor must be non-negative, got {gor}")
    if pipe_id <= 0:
        raise ValueError(f"pipe_id must be positive, got {pipe_id}")

    # --- Fluid properties (simplified black-oil) ---
    t_rankine = temp + 459.67
    api = 141.5 / oil_sg - 131.5

    # Solution GOR (Standing)
    if pressure > 14.7:
        exp_rs = 0.0125 * api - 0.00091 * temp
        rs = gas_sg * ((pressure / 18.2 + 1.4) * 10**exp_rs) ** 1.2048
    else:
        rs = 0.0
    rs = min(rs, gor)  # Can't exceed total GOR

    # Oil FVF (Standing)
    f_factor = rs * (gas_sg / oil_sg) ** 0.5 + 1.25 * temp
    bo = 0.9759 + 0.000120 * f_factor**1.2

    # Water FVF (simplified)
    bw = 1.0 + 1.21e-4 * (temp - 60.0)

    # Gas Z-factor (simplified Hall-Yarborough)
    tpc = 169.2 + 349.5 * gas_sg - 74.0 * gas_sg**2
    ppc = 756.8 - 131.0 * gas_sg - 3.6 * gas_sg**2
    tpr = t_rankine / tpc
    ppr = pressure / ppc
    # Simple Z approx for moderate conditions
    z = 1.0 - 3.52 * ppr / (10 ** (0.9813 * tpr)) + 0.274 * ppr**2 / (10 ** (0.8157 * tpr))
    z = max(z, 0.3)

    bg = 0.02827 * z * t_rankine / pressure  # bbl/scf

    # Oil viscosity (Beggs-Robinson simplified)
    x = 10 ** (3.0324 - 0.02023 * api) * temp ** (-1.163)
    mu_o = 10**x - 1.0

    # Water viscosity
    mu_w = math.exp(1.003 - 1.479e-2 * temp + 1.982e-5 * temp**2)

    # Gas viscosity (simplified Lee-Gonzalez-Eakin)
    mw_gas = 28.97 * gas_sg
    rho_g = pressure * mw_gas / (z * 10.73 * t_rankine)
    k_vis = (9.4 + 0.02 * mw_gas) * t_rankine**1.5 / (209 + 19 * mw_gas + t_rankine)
    x_vis = 3.5 + 986 / t_rankine + 0.01 * mw_gas
    y_vis = 2.4 - 0.2 * x_vis
    mu_g = k_vis * math.exp(x_vis * (rho_g / 62.4) ** y_vis) * 1e-4

    # Liquid densities
    rho_o = (350.0 * oil_sg + 0.0764 * rs * gas_sg) / (5.615 * bo)
    rho_w = 62.4 * water_sg / bw

    # --- Flow calculations ---
    area = math.pi / 4.0 * (pipe_id / 12.0) ** 2  # ft²

    # Liquid and gas volumetric flow rates at conditions
    q_o = oil_rate * bo / 86400.0 * 5.615  # ft³/s
    q_w = water_rate * bw / 86400.0 * 5.615  # ft³/s
    free_gas = max(gor - rs, 0.0) * oil_rate
    q_g = free_gas * bg / 86400.0 * 5.615  # ft³/s

    q_l = q_o + q_w
    q_t = q_l + q_g

    if q_t <= 0:
        return {
            "dp_dz_psi_ft": 0.0,
            "dp_dz_gravity": 0.0,
            "dp_dz_friction": 0.0,
            "flow_pattern": "no flow",
            "liquid_holdup": 1.0,
            "mixture_velocity_ft_s": 0.0,
        }

    # Superficial velocities
    vsl = q_l / area
    vsg = q_g / area
    vm = vsl + vsg

    # Input liquid fraction
    lam_l = vsl / vm if vm > 0 else 1.0

    # Mixture properties (for no-slip)
    fw = q_w / q_l if q_l > 0 else 0.0
    rho_l = rho_o * (1 - fw) + rho_w * fw
    mu_l = mu_o * (1 - fw) + mu_w * fw
    rho_ns = rho_l * lam_l + rho_g * (1 - lam_l)  # no-slip density

    # --- Froude number and flow pattern ---
    nfr = vm**2 / (32.174 * pipe_id / 12.0) if pipe_id > 0 else 0

    # Flow pattern boundaries
    l1 = 316.0 * lam_l**0.302
    l2 = 0.0009252 * lam_l**(-2.4684)
    l3 = 0.10 * lam_l**(-1.4516)
    l4 = 0.5 * lam_l**(-6.738)

    # Determine flow pattern
    if lam_l < 0.01:
        pattern = "segregated"
    elif lam_l >= 0.01 and lam_l < 0.4:
        if nfr < l2:
            pattern = "segregated"
        elif nfr >= l2 and nfr <= l3:
            pattern = "transition"
        else:
            pattern = "intermittent"
    elif lam_l >= 0.4 and lam_l <= 0.9:
        if nfr <= l1:
            pattern = "segregated"
        elif nfr > l1 and nfr <= l4:
            pattern = "intermittent"
        else:
            pattern = "distributed"
    else:
        if nfr <= l4:
            pattern = "intermittent"
        else:
            pattern = "distributed"

    # --- Liquid holdup ---
    nlv = vsl * (rho_l / (32.174 * 0.0023 * max(mu_l, 0.001))) ** 0.25  # Liquid velocity number

    if pattern == "segregated":
        a, b, c = 0.980, 0.4846, 0.0868
    elif pattern == "intermittent":
        a, b, c = 0.845, 0.5351, 0.0173
    elif pattern == "distributed":
        a, b, c = 1.065, 0.5824, 0.0609
    else:  # transition
        a, b, c = 0.845, 0.5351, 0.0173  # Use intermittent as base

    hl0 = a * lam_l**b / max(nfr**c, 1e-10)
    hl0 = max(hl0, lam_l)  # Holdup can't be less than input fraction

    # Inclination correction
    angle_rad = math.radians(angle)
    sin_angle = math.sin(angle_rad)

    if pattern == "segregated":
        d, e, f, g = 0.011, -3.768, 3.539, -1.614
    elif pattern == "intermittent":
        d, e, f, g = 2.96, 0.305, -0.4473, 0.0978
    else:
        d, e, f, g = 0.0, 0.0, 0.0, 0.0  # No correction for distributed

    if d != 0:
        psi_correction = 1.0 + d * (
            sin_angle * (1.8 * angle_rad / math.pi)
            - 0.333 * sin_angle**3
        )
    else:
        psi_correction = 1.0

    hl = hl0 * psi_correction
    hl = max(min(hl, 1.0), 0.0)

    # --- Pressure gradient ---
    # Gravity component
    rho_s = rho_l * hl + rho_g * (1 - hl)
    dp_gravity = rho_s * sin_angle / 144.0  # psi/ft

    # Friction component
    # Reynolds number
    d_ft = pipe_id / 12.0
    re_ns = rho_ns * vm * d_ft / (mu_l * lam_l + mu_g * (1 - lam_l)) / 6.7197e-4

    if re_ns > 0:
        e_d = roughness / pipe_id
        # Chen friction factor
        fn = (-2.0 * math.log10(
            e_d / 3.7065 - 5.0452 / max(re_ns, 1) * math.log10(
                e_d**1.1098 / 2.8257 + (7.149 / max(re_ns, 1))**0.8981
            )
        )) ** (-2) if re_ns > 2000 else 16.0 / max(re_ns, 1)
    else:
        fn = 0.0

    # Beggs-Brill friction factor ratio
    y_ratio = lam_l / max(hl**2, 1e-10)
    if 1.0 < y_ratio < 1.2:
        s_factor = math.log(2.2 * y_ratio - 1.2)
    elif y_ratio >= 1.2:
        ln_y = math.log(y_ratio)
        s_factor = ln_y / (-0.0523 + 3.182 * ln_y - 0.8725 * ln_y**2 + 0.01853 * ln_y**4)
    else:
        s_factor = 0.0

    ftp = fn * math.exp(s_factor)
    dp_friction = ftp * rho_ns * vm**2 / (2.0 * 32.174 * d_ft * 144.0)  # psi/ft

    dp_total = dp_gravity + dp_friction

    return {
        "dp_dz_psi_ft": round(dp_total, 6),
        "dp_dz_gravity": round(dp_gravity, 6),
        "dp_dz_friction": round(dp_friction, 6),
        "flow_pattern": pattern,
        "liquid_holdup": round(hl, 4),
        "mixture_velocity_ft_s": round(vm, 4),
    }
