"""Well-control and mud-weight calculations.

Field-unit pressure conversions used on every rig:

    P [psi]   = 0.052 * MW [ppg] * TVD [ft]
    MW [ppg]  = P / (0.052 * TVD)
    ECD       = MW + APL / (0.052 * TVD)

The 0.052 factor is the ppg → psi/ft gradient (1 ppg ≈ 0.052 psi/ft).

References:
    Bourgoyne, A.T., Millheim, K.K., Chenevert, M.E., and Young, F.S.,
        "Applied Drilling Engineering," SPE Textbook Series Vol. 2, 1991.
    IADC Drilling Manual, 12th ed.
"""

from __future__ import annotations

from ._validation import validate_non_negative, validate_positive

PPG_TO_PSI_PER_FT = 0.052


def hydrostatic_pressure(mud_weight_ppg: float, tvd_ft: float) -> float:
    """Hydrostatic pressure from mud weight and true vertical depth.

        P [psi] = 0.052 * MW [ppg] * TVD [ft]

    Args:
        mud_weight_ppg: Mud weight in pounds per gallon (ppg).
        tvd_ft: True vertical depth (ft).

    Returns:
        Hydrostatic pressure in psi.
    """
    validate_positive(mud_weight_ppg, "mud_weight_ppg")
    validate_positive(tvd_ft, "tvd_ft")
    return PPG_TO_PSI_PER_FT * mud_weight_ppg * tvd_ft


def equivalent_circulating_density(
    mud_weight_ppg: float,
    annular_pressure_loss_psi: float,
    tvd_ft: float,
) -> float:
    """Equivalent circulating density (ECD).

        ECD [ppg] = MW + APL / (0.052 * TVD)

    ECD captures the effective bottomhole pressure while mud is being
    circulated — the sum of the static mud hydrostatic and the annular
    friction loss, expressed as an equivalent mud density.

    Args:
        mud_weight_ppg: Static mud weight (ppg).
        annular_pressure_loss_psi: Annular pressure loss (psi).
        tvd_ft: True vertical depth (ft).

    Returns:
        ECD in ppg.
    """
    validate_positive(mud_weight_ppg, "mud_weight_ppg")
    validate_non_negative(annular_pressure_loss_psi, "annular_pressure_loss_psi")
    validate_positive(tvd_ft, "tvd_ft")
    return mud_weight_ppg + annular_pressure_loss_psi / (PPG_TO_PSI_PER_FT * tvd_ft)


def formation_pressure_gradient(pressure_psi: float, tvd_ft: float) -> float:
    """Formation pressure expressed as an equivalent mud weight.

        FPG [ppg] = P / (0.052 * TVD)

    Args:
        pressure_psi: Formation pressure (psi).
        tvd_ft: True vertical depth (ft).

    Returns:
        Formation pressure gradient in ppg equivalent.
    """
    validate_non_negative(pressure_psi, "pressure_psi")
    validate_positive(tvd_ft, "tvd_ft")
    return pressure_psi / (PPG_TO_PSI_PER_FT * tvd_ft)


def kill_mud_weight(
    sidp_psi: float,
    original_mud_weight_ppg: float,
    tvd_ft: float,
) -> float:
    """Kill mud weight for well control.

        Kill MW = Original MW + SIDP / (0.052 * TVD)

    The kill mud weight is the density needed to exactly balance the
    formation pressure revealed by the shut-in drill pipe pressure
    (SIDP). Overbalance is applied on top of this in practice.

    Args:
        sidp_psi: Shut-in drill pipe pressure (psi).
        original_mud_weight_ppg: Current mud weight (ppg).
        tvd_ft: True vertical depth (ft).

    Returns:
        Kill mud weight in ppg.
    """
    validate_non_negative(sidp_psi, "sidp_psi")
    validate_positive(original_mud_weight_ppg, "original_mud_weight_ppg")
    validate_positive(tvd_ft, "tvd_ft")
    return original_mud_weight_ppg + sidp_psi / (PPG_TO_PSI_PER_FT * tvd_ft)


class ICPFCPResult(dict):
    """Typed result holder — dict with icp_psi and fcp_psi keys."""


def initial_and_final_circulating_pressure(
    sidp_psi: float,
    slow_circulating_pressure_psi: float,
    kill_mud_weight_ppg: float,
    original_mud_weight_ppg: float,
) -> dict:
    """Driller's / Wait-and-Weight circulating pressures.

        ICP = SIDP + SCP
        FCP = SCP * (Kill MW / Original MW)

    ICP is the standpipe pressure at the start of the kill circulation;
    FCP is the standpipe pressure once the heavier kill mud has reached
    the bit (for Wait-and-Weight) or once the original mud has completed
    one full circulation (for Driller's).

    Args:
        sidp_psi: Shut-in drill pipe pressure (psi).
        slow_circulating_pressure_psi: Slow circulating pressure (SCP, psi).
        kill_mud_weight_ppg: Kill mud weight (ppg).
        original_mud_weight_ppg: Original mud weight (ppg).

    Returns:
        Dict with ``icp_psi`` and ``fcp_psi``.
    """
    validate_non_negative(sidp_psi, "sidp_psi")
    validate_positive(slow_circulating_pressure_psi, "slow_circulating_pressure_psi")
    validate_positive(kill_mud_weight_ppg, "kill_mud_weight_ppg")
    validate_positive(original_mud_weight_ppg, "original_mud_weight_ppg")

    icp = sidp_psi + slow_circulating_pressure_psi
    fcp = slow_circulating_pressure_psi * (
        kill_mud_weight_ppg / original_mud_weight_ppg
    )
    return {"icp_psi": float(icp), "fcp_psi": float(fcp)}


def maasp(
    fracture_gradient_ppg: float,
    mud_weight_ppg: float,
    shoe_tvd_ft: float,
) -> float:
    """Maximum Allowable Annular Surface Pressure.

        MAASP [psi] = (FG - MW) * 0.052 * shoe_TVD

    The surface pressure ceiling before exceeding fracture gradient at
    the last casing shoe. Negative values indicate the current mud is
    already above fracture gradient at the shoe — a physical
    impossibility that is a flag to raise.

    Args:
        fracture_gradient_ppg: Fracture gradient at the shoe (ppg).
        mud_weight_ppg: Current mud weight (ppg).
        shoe_tvd_ft: Casing shoe TVD (ft).

    Returns:
        MAASP in psi. Can be negative (caller should treat as warning).
    """
    validate_positive(fracture_gradient_ppg, "fracture_gradient_ppg")
    validate_positive(mud_weight_ppg, "mud_weight_ppg")
    validate_positive(shoe_tvd_ft, "shoe_tvd_ft")
    return (fracture_gradient_ppg - mud_weight_ppg) * PPG_TO_PSI_PER_FT * shoe_tvd_ft
