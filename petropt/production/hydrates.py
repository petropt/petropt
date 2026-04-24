"""Gas hydrate formation temperature and inhibitor dosing.

Two bread-and-butter flow assurance calculations:

1. **Hydrate formation temperature** via the Katz gas-gravity chart
   (empirical log-pressure fit with a gas-specific-gravity correction).
   Use to decide whether a pipeline or wellbore segment is at risk.

2. **Hammerschmidt (1934) inhibitor dosing** — the industry standard
   for estimating methanol / MEG / ethanol injection volumes needed
   to depress the hydrate point to the operating temperature.

References:
    Katz, D.L., "Prediction of the Conditions for Hydrate Formation in
        Natural Gases," Trans. AIME 160, 1945, pp. 140-149.
    Hammerschmidt, E.G., "Formation of Gas Hydrates in Natural Gas
        Transmission Lines," Industrial and Engineering Chemistry,
        26(8), 1934, pp. 851-855.
    Carroll, J., "Natural Gas Hydrates: A Guide for Engineers,"
        3rd ed., Gulf Professional Publishing, 2014.
"""

from __future__ import annotations

import math
from typing import TypedDict

from ._validation import validate_positive


def katz_hydrate_temperature(pressure_psia: float, gas_sg: float) -> float:
    """Hydrate formation temperature via the Motiee (1991) Katz-chart fit.

        T_hyd [°F] = -20.35 + 13.47 * ln(P) + 34.27 * ln(SG) - 1.675 * ln(P) * ln(SG)

    Motiee's two-variable regression fits the Katz (1945) gas-gravity
    chart over 100-10,000 psia and 0.55-1.0 SG to within ~5 °F — good
    enough for screening. For rigorous predictions with inhibitors,
    acid gases, or near the stability boundary, use an equation-of-state
    solver.

    Args:
        pressure_psia: System pressure in *psia* (not psig). Use absolute
            pressure — the correlation was fit on absolute values.
        gas_sg: Gas specific gravity (air = 1.0), in [0.55, 1.0].

    Returns:
        Hydrate formation temperature in °F.

    Reference:
        Motiee, M., "Estimate Possibility of Hydrate Formation," Hydrocarbon
        Processing, July 1991, pp. 98-99.
    """
    validate_positive(pressure_psia, "pressure_psia")
    validate_positive(gas_sg, "gas_sg")
    if not 0.55 <= gas_sg <= 1.0:
        raise ValueError(
            f"gas_sg must be between 0.55 and 1.0 for this correlation, got {gas_sg}"
        )

    ln_p = math.log(pressure_psia)
    ln_sg = math.log(gas_sg)
    return -20.35 + 13.47 * ln_p + 34.27 * ln_sg - 1.675 * ln_p * ln_sg


# Molecular weight (g/mol), Hammerschmidt K constant (°F-lb/mol), pure
# inhibitor density at standard temperature (lb/gal).
_INHIBITOR_PROPS: dict[str, dict[str, float]] = {
    "methanol": {"M": 32.04, "K": 2335.0, "density_lb_gal": 6.63},
    "meg": {"M": 62.07, "K": 2700.0, "density_lb_gal": 9.35},
    "ethanol": {"M": 46.07, "K": 2335.0, "density_lb_gal": 6.58},
}


class InhibitorDosingResult(TypedDict):
    temperature_depression_f: float
    inhibitor: str
    weight_percent: float
    rate_lb_day: float
    rate_gal_day: float


def hammerschmidt_inhibitor_dosing(
    hydrate_temp_f: float,
    operating_temp_f: float,
    water_rate_bwpd: float,
    inhibitor: str = "methanol",
) -> InhibitorDosingResult:
    """Hammerschmidt dosing — estimate inhibitor injection rate.

        ΔT = K * W / (M * (100 - W))

    where W is the equilibrium weight-percent of inhibitor in the
    aqueous phase, M is the inhibitor molecular weight, and K is the
    Hammerschmidt constant (2335 °F-lb/mol for alcohols, 2700 for MEG).

    Solves for W given the required depression, then computes the
    mass/volume injection rate to achieve that concentration in the
    produced water.

    Args:
        hydrate_temp_f: Hydrate formation temperature (°F) at operating
            pressure — from :func:`katz_hydrate_temperature` or a rigorous
            prediction.
        operating_temp_f: Target operating (minimum) temperature (°F).
        water_rate_bwpd: Liquid water production rate (bwpd).
        inhibitor: One of "methanol", "meg", "ethanol". Default methanol.

    Returns:
        Dict with temperature depression, weight-percent, and injection
        rates in lb/day and gal/day. If the operating temperature is
        already at or above the hydrate point, returns zeros.
    """
    inhibitor_key = inhibitor.lower().strip()
    if inhibitor_key not in _INHIBITOR_PROPS:
        raise ValueError(
            f"Unsupported inhibitor '{inhibitor}'. Use one of: {list(_INHIBITOR_PROPS)}"
        )
    validate_positive(water_rate_bwpd, "water_rate_bwpd")

    dt = hydrate_temp_f - operating_temp_f
    if dt <= 0:
        return {
            "temperature_depression_f": 0.0,
            "inhibitor": inhibitor_key,
            "weight_percent": 0.0,
            "rate_lb_day": 0.0,
            "rate_gal_day": 0.0,
        }

    props = _INHIBITOR_PROPS[inhibitor_key]
    mw, k, density = props["M"], props["K"], props["density_lb_gal"]

    # Solve Hammerschmidt for weight-percent W
    w = 100.0 * mw * dt / (k + mw * dt)
    w = min(w, 95.0)  # physical cap

    # Produced water mass rate. 42 gal/bbl, 8.34 lb/gal fresh water,
    # times a 1.07 factor for typical brine SG.
    water_mass_lb_day = water_rate_bwpd * 42.0 * 8.34 * 1.07
    inhibitor_mass_lb_day = w * water_mass_lb_day / (100.0 - w)
    inhibitor_vol_gal_day = inhibitor_mass_lb_day / density

    return {
        "temperature_depression_f": float(dt),
        "inhibitor": inhibitor_key,
        "weight_percent": float(w),
        "rate_lb_day": float(inhibitor_mass_lb_day),
        "rate_gal_day": float(inhibitor_vol_gal_day),
    }
