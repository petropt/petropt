"""Permeability estimation from porosity and water saturation or NMR.

Correlations:
    - Timur (1968): k [mD] = 0.136 * phi^4.4 / Swirr^2   (phi, Swirr in %)
    - Coates (1991): k [mD] = ((phi/C)^2 * (FFI/BVI))^2  (phi in %, C ~ 10)

Both correlations are log-derived estimators — they use easily available
porosity and irreducible saturation (or NMR bound/free partitions) to
give an order-of-magnitude permeability. Calibration against core is
standard practice in any real field workflow.

References:
    Timur, A., "An Investigation of Permeability, Porosity, and Residual
        Water Saturation Relationships for Sandstone Reservoirs," The Log
        Analyst, 9(4), July-August 1968, pp. 8-17.
    Coates, G.R., Miller, M., Gillen, M., and Henderson, G., "The MRIL in
        Conoco 33-1: An Investigation of a New Magnetic Resonance Imaging
        Log," SPWLA 32nd Annual Logging Symposium, 1991.
"""

from __future__ import annotations

from ._validation import validate_non_negative, validate_positive


def timur_permeability(phi: float, swirr: float) -> float:
    """Permeability estimate via Timur's equation (1968).

        k [mD] = 0.136 * (phi * 100)^4.4 / (Swirr * 100)^2

    Pass ``phi`` and ``swirr`` as fractions (0-1). The function converts
    to percent internally because Timur's original constant 0.136 was
    fitted to percent inputs — this is the source of most confusion
    about this correlation in practice.

    Args:
        phi: Porosity (fraction, 0-1).
        swirr: Irreducible water saturation (fraction, 0-1).

    Returns:
        Permeability in millidarcies (mD).
    """
    validate_positive(phi, "phi")
    validate_positive(swirr, "swirr")
    if phi > 1:
        raise ValueError(f"phi must be <= 1, got {phi}")
    if swirr > 1:
        raise ValueError(f"swirr must be <= 1, got {swirr}")

    phi_pct = phi * 100.0
    swirr_pct = swirr * 100.0
    return 0.136 * phi_pct**4.4 / swirr_pct**2


def coates_permeability(
    phi: float,
    bvi: float,
    ffi: float,
    c: float = 10.0,
) -> float:
    """Permeability estimate via Coates' NMR equation (1991).

        k [mD] = ((phi * 100 / C)^2 * (FFI / BVI))^2

    Pass ``phi`` as a fraction (0-1); it is converted to percent
    internally to match the original Coates constant.

    BVI (bound volume irreducible) and FFI (free fluid index) come from
    partitioning NMR T2 distributions at a cutoff (33 ms sandstone,
    92 ms carbonate are typical). Both are reported as fractions of
    bulk volume and must satisfy ``BVI + FFI <= phi``.

    Args:
        phi: Porosity (fraction, 0-1).
        bvi: Bound volume irreducible (fraction, 0-1).
        ffi: Free fluid index (fraction, 0-1). Zero returns zero k.
        c: Coates constant. Default 10.0 (sandstone). Carbonates often
            use ~8; tune per formation from calibration plugs.

    Returns:
        Permeability in millidarcies (mD).
    """
    validate_positive(phi, "phi")
    validate_positive(bvi, "bvi")
    validate_non_negative(ffi, "ffi")
    validate_positive(c, "c")
    if phi > 1:
        raise ValueError(f"phi must be <= 1, got {phi}")
    if bvi > 1:
        raise ValueError(f"bvi must be <= 1, got {bvi}")
    if ffi > 1:
        raise ValueError(f"ffi must be <= 1, got {ffi}")

    if ffi == 0:
        return 0.0

    phi_pct = phi * 100.0
    return ((phi_pct / c) ** 2 * (ffi / bvi)) ** 2
