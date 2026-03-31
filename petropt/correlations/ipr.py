"""Inflow Performance Relationship (IPR) correlations.

Implements:
    - Vogel (1968): IPR for solution-gas-drive reservoirs

Reference:
    Vogel, J.V., "Inflow Performance Relationships for Solution-Gas Drive
    Wells," JPT, January 1968, pp. 83-92.
"""

from __future__ import annotations

import numpy as np


def vogel_ipr(
    qmax: float,
    pwf: float | np.ndarray | None = None,
    pb: float | None = None,
    pr: float | None = None,
    num_points: int = 20,
) -> dict:
    """Vogel IPR for solution-gas-drive reservoirs (1968).

    q/qmax = 1 - 0.2*(Pwf/Pr) - 0.8*(Pwf/Pr)^2

    When called with a single pwf, returns the flow rate at that pressure.
    When called without pwf (or pwf=None), returns the full IPR curve.

    Args:
        qmax: Maximum oil rate (AOF) in STB/day.
        pwf: Flowing bottomhole pressure in psi. If None, generates full curve.
        pb: Bubble point pressure in psi (used as reservoir pressure if pr not given).
        pr: Reservoir pressure in psi. Defaults to pb if not provided.
        num_points: Number of points for curve generation.

    Returns:
        Dict with keys:
            - 'qo': flow rate(s) in STB/day
            - 'pwf': pressure(s) in psi
            - 'qmax': maximum rate in STB/day
            - 'pr': reservoir pressure in psi

    Raises:
        ValueError: If qmax <= 0 or pressures are invalid.
    """
    if qmax <= 0:
        raise ValueError(f"qmax must be positive, got {qmax}")

    # Determine reservoir pressure
    if pr is not None:
        if pr <= 0:
            raise ValueError(f"reservoir pressure must be positive, got {pr}")
        p_res = pr
    elif pb is not None:
        if pb <= 0:
            raise ValueError(f"bubble point pressure must be positive, got {pb}")
        p_res = pb
    else:
        raise ValueError("Either pr or pb must be provided")

    def _vogel_rate(pwf_val: float) -> float:
        if pwf_val < 0:
            raise ValueError(f"pwf must be non-negative, got {pwf_val}")
        if pwf_val >= p_res:
            return 0.0
        ratio = pwf_val / p_res
        return qmax * (1.0 - 0.2 * ratio - 0.8 * ratio**2)

    if pwf is not None:
        # Single-point or array calculation
        if isinstance(pwf, np.ndarray):
            qo = np.array([_vogel_rate(p) for p in pwf])
            return {
                "qo": qo,
                "pwf": pwf,
                "qmax": qmax,
                "pr": p_res,
            }
        else:
            qo = _vogel_rate(float(pwf))
            return {
                "qo": qo,
                "pwf": float(pwf),
                "qmax": qmax,
                "pr": p_res,
            }
    else:
        # Generate full curve
        pwf_array = np.linspace(0, p_res, num_points + 1)
        qo_array = np.array([_vogel_rate(p) for p in pwf_array])
        return {
            "qo": qo_array,
            "pwf": pwf_array,
            "qmax": qmax,
            "pr": p_res,
        }
