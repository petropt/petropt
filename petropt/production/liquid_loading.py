"""Gas-well liquid loading — Turner and Coleman critical rates.

When the gas velocity in a producing well drops below the level needed
to lift the largest liquid droplet, the well begins to "load up" and
will eventually die. Turner's droplet-lift criterion is the industry
baseline; Coleman scales Turner by 0.8 for low-pressure wells where
the droplet model over-predicts.

Critical velocity (Turner, 1969):

    v_c [ft/s] = 1.593 * sigma^0.25 * (rho_L - rho_g)^0.25 / rho_g^0.5

with sigma in dynes/cm and rho in lb/ft^3.

References:
    Turner, R.G., Hubbard, M.G., and Dukler, A.E., "Analysis and
        Prediction of Minimum Flow Rate for the Continuous Removal of
        Liquids from Gas Wells," JPT, November 1969, pp. 1475-1482.
    Coleman, S.B., Clay, H.B., McCurdy, D.G., and Norris, H.L., III,
        "A New Look at Predicting Gas-Well Load-Up," JPT, March 1991,
        pp. 329-333.
    Lea, J.F., Nickens, H.V., and Wells, M.R., "Gas Well Deliquification,"
        2nd ed., Gulf Professional Publishing, 2008.
"""

from __future__ import annotations

import math
from typing import TypedDict

from ._validation import validate_positive


class CriticalLoadingResult(TypedDict):
    critical_velocity_ft_s: float
    critical_rate_mcfd: float


def turner_critical_velocity(
    surface_tension_dyne_cm: float,
    rho_liquid_lb_ft3: float,
    rho_gas_lb_ft3: float,
) -> float:
    """Turner critical velocity for liquid droplet lift (ft/s).

        v_c = 1.593 * sigma^0.25 * (rho_L - rho_g)^0.25 / rho_g^0.5

    Typical surface tensions: 60 dyne/cm for water (80 °F), 20 dyne/cm
    for condensate. Water is the harder fluid to lift, so when both are
    produced together the water calculation governs.

    Args:
        surface_tension_dyne_cm: Liquid-gas surface tension (dynes/cm).
        rho_liquid_lb_ft3: Liquid density (lb/ft^3).
        rho_gas_lb_ft3: Gas density at wellhead conditions (lb/ft^3).

    Returns:
        Critical velocity in ft/s.
    """
    validate_positive(surface_tension_dyne_cm, "surface_tension_dyne_cm")
    validate_positive(rho_liquid_lb_ft3, "rho_liquid_lb_ft3")
    validate_positive(rho_gas_lb_ft3, "rho_gas_lb_ft3")
    if rho_gas_lb_ft3 >= rho_liquid_lb_ft3:
        raise ValueError(
            f"rho_gas ({rho_gas_lb_ft3}) must be less than rho_liquid "
            f"({rho_liquid_lb_ft3}) — otherwise there is no droplet to lift"
        )

    return (
        1.593
        * surface_tension_dyne_cm**0.25
        * (rho_liquid_lb_ft3 - rho_gas_lb_ft3) ** 0.25
        / rho_gas_lb_ft3**0.5
    )


def coleman_critical_velocity(
    surface_tension_dyne_cm: float,
    rho_liquid_lb_ft3: float,
    rho_gas_lb_ft3: float,
) -> float:
    """Coleman critical velocity — 80% of Turner's, for low-pressure wells.

    For wellhead pressures below ~500 psi, Turner over-predicts the
    velocity needed because the droplet assumption breaks down (film
    flow begins to contribute lift). Coleman recommends scaling Turner
    by 0.8. Above ~1000 psi use Turner directly.

    Args:
        surface_tension_dyne_cm: Liquid-gas surface tension (dynes/cm).
        rho_liquid_lb_ft3: Liquid density (lb/ft^3).
        rho_gas_lb_ft3: Gas density (lb/ft^3).

    Returns:
        Critical velocity in ft/s.
    """
    return 0.8 * turner_critical_velocity(
        surface_tension_dyne_cm, rho_liquid_lb_ft3, rho_gas_lb_ft3
    )


_T_STANDARD_R = 520.0  # 60 °F in Rankine (API standard conditions)
_P_STANDARD_PSIA = 14.7


def critical_gas_rate_mcfd(
    critical_velocity_ft_s: float,
    tubing_id_in: float,
    pressure_psia: float,
    temperature_f: float,
    z_factor: float,
) -> float:
    """Convert a critical velocity to the equivalent standard-condition gas rate.

    The actual volumetric flow at wellhead conditions is ``v * A``. To
    convert to Mcf/d at standard conditions (60 °F, 14.7 psia) apply
    the real-gas ratio:

        q_sc = q_wh * (P / P_sc) * (T_sc / T_wh) / Z

    so

        q_sc [Mcf/d] = v * A * 86400 * (P / Z / T_R) * (T_sc / P_sc) / 1000

    with T_sc = 520 °R and P_sc = 14.7 psia.

    Args:
        critical_velocity_ft_s: Wellhead gas velocity (ft/s).
        tubing_id_in: Tubing inner diameter (in).
        pressure_psia: Wellhead absolute pressure (psia).
        temperature_f: Wellhead temperature (°F).
        z_factor: Gas compressibility factor (dimensionless) at P, T.

    Returns:
        Critical gas rate in Mcf/d at standard conditions.
    """
    validate_positive(critical_velocity_ft_s, "critical_velocity_ft_s")
    validate_positive(tubing_id_in, "tubing_id_in")
    validate_positive(pressure_psia, "pressure_psia")
    validate_positive(temperature_f, "temperature_f")
    validate_positive(z_factor, "z_factor")

    area_sqft = math.pi / 4.0 * (tubing_id_in / 12.0) ** 2
    temp_r = temperature_f + 459.67
    q_actual_ft3_per_day = critical_velocity_ft_s * area_sqft * 86400.0
    q_sc_scf_per_day = (
        q_actual_ft3_per_day
        * pressure_psia
        * _T_STANDARD_R
        / (z_factor * temp_r * _P_STANDARD_PSIA)
    )
    return q_sc_scf_per_day / 1000.0
