"""Shale volume (Vshale) estimation from gamma ray logs.

Correlations:
    - Linear: Vshale = IGR
    - Larionov (1969) — Tertiary rocks: Vshale = 0.083 * (2^(3.7*IGR) - 1)
    - Larionov (1969) — Older rocks: Vshale = 0.33 * (2^(2*IGR) - 1)
    - Clavier: Vshale = 1.7 - sqrt(3.38 - (IGR + 0.7)^2)

The non-linear transforms correct for the fact that shale bound water
and dispersed clay produce a non-linear gamma-ray response; the linear
IGR systematically over-reads Vshale except in simple bimodal systems.

References:
    Larionov, V.V., "Borehole Radiometry," Nedra, Moscow, 1969.
    Asquith, G.B. and Krygowski, D., "Basic Well Log Analysis," 2nd ed.,
        AAPG Methods in Exploration Series No. 16, 2004, Ch. 5. Documents
        the Clavier, linear, and Larionov non-linear transforms used
        throughout industry.
    Bateman, R.M., "Openhole Log Analysis and Formation Evaluation,"
        IHRDC, 1985.
"""

from __future__ import annotations

import math

from ._validation import clamp, validate_finite

_METHODS = ("linear", "larionov_tertiary", "larionov_older", "clavier")


def vshale_from_gr(
    gr: float,
    gr_clean: float,
    gr_shale: float,
    method: str = "linear",
) -> float:
    """Shale volume from gamma ray log.

    Computes the gamma ray index (IGR) then converts to Vshale via the
    selected non-linear transform.

        IGR = (GR - GR_clean) / (GR_shale - GR_clean)

    Args:
        gr: Gamma ray reading at the depth of interest (API units).
        gr_clean: Gamma ray baseline in clean sand (API units).
        gr_shale: Gamma ray baseline in pure shale (API units).
        method: One of "linear", "larionov_tertiary", "larionov_older",
            "clavier". Larionov_tertiary is typical for younger (Tertiary)
            sediments; larionov_older for Mesozoic and older. Clavier sits
            between the two Larionov forms.

    Returns:
        Shale volume as a fraction (0-1, v/v).

    Raises:
        ValueError: If ``method`` is unknown or ``gr_clean == gr_shale``.
    """
    method = method.lower()
    if method not in _METHODS:
        raise ValueError(f"Unknown method '{method}'. Must be one of: {list(_METHODS)}")
    validate_finite(gr, "gr")
    validate_finite(gr_clean, "gr_clean")
    validate_finite(gr_shale, "gr_shale")
    if gr_clean == gr_shale:
        raise ValueError("gr_clean and gr_shale must differ")

    igr = clamp((gr - gr_clean) / (gr_shale - gr_clean))

    if method == "linear":
        vsh = igr
    elif method == "larionov_tertiary":
        vsh = 0.083 * (2.0 ** (3.7 * igr) - 1.0)
    elif method == "larionov_older":
        vsh = 0.33 * (2.0 ** (2.0 * igr) - 1.0)
    else:  # clavier
        vsh = 1.7 - math.sqrt(max(3.38 - (igr + 0.7) ** 2, 0.0))

    return clamp(vsh)


def gamma_ray_index(gr: float, gr_clean: float, gr_shale: float) -> float:
    """Linear gamma ray index (IGR), clamped to [0, 1].

    Args:
        gr: Gamma ray reading (API units).
        gr_clean: Clean sand baseline (API units).
        gr_shale: Shale baseline (API units).

    Returns:
        IGR as a fraction (0-1).
    """
    validate_finite(gr, "gr")
    validate_finite(gr_clean, "gr_clean")
    validate_finite(gr_shale, "gr_shale")
    if gr_clean == gr_shale:
        raise ValueError("gr_clean and gr_shale must differ")
    return clamp((gr - gr_clean) / (gr_shale - gr_clean))
