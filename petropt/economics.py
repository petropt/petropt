"""Petroleum economics — NPV, IRR, cashflow analysis.

Implements standard discounted cash flow analysis for oil & gas projects.

References:
    Society of Petroleum Engineers, "Petroleum Resources Management System
    (PRMS)," SPE/WPC/AAPG/SPEE/SEG/EAGE/AGA, 2018.
"""

from __future__ import annotations

import numpy as np


def npv(
    cash_flows: list[float] | np.ndarray,
    discount_rate: float,
    periods_per_year: int = 12,
) -> float:
    """Net Present Value of a cash flow stream.

    NPV = sum(CF_t / (1 + r)^t)

    Args:
        cash_flows: Array of cash flows (negative = cost, positive = revenue).
            First element is time 0 (initial investment).
        discount_rate: Annual discount rate as decimal (e.g., 0.10 for 10%).
        periods_per_year: Number of periods per year (12 = monthly, 1 = annual).

    Returns:
        NPV in same currency units as cash_flows.
    """
    cf = np.asarray(cash_flows, dtype=float)
    r_period = (1.0 + discount_rate) ** (1.0 / periods_per_year) - 1.0
    t = np.arange(len(cf))
    return float(np.sum(cf / (1.0 + r_period) ** t))


def irr(
    cash_flows: list[float] | np.ndarray,
    guess: float = 0.1,
    tol: float = 1e-6,
    max_iter: int = 100,
) -> float | None:
    """Internal Rate of Return using Newton-Raphson iteration.

    Finds the discount rate that makes NPV = 0.

    Args:
        cash_flows: Array of cash flows (first is typically negative).
        guess: Initial guess for IRR (default 0.10).
        tol: Convergence tolerance.
        max_iter: Maximum iterations.

    Returns:
        IRR as decimal (e.g., 0.15 for 15%), or None if no convergence.
    """
    cf = np.asarray(cash_flows, dtype=float)
    t = np.arange(len(cf))

    r = guess
    for _ in range(max_iter):
        disc = (1.0 + r) ** t
        f = np.sum(cf / disc)
        df = np.sum(-t * cf / ((1.0 + r) ** (t + 1)))
        if abs(df) < 1e-30:
            return None
        r_new = r - f / df
        if abs(r_new - r) < tol:
            return round(r_new, 6)
        r = r_new

    return None


def payback_period(
    cash_flows: list[float] | np.ndarray,
) -> float | None:
    """Undiscounted payback period.

    Finds the time at which cumulative cash flow becomes positive.

    Args:
        cash_flows: Array of cash flows.

    Returns:
        Payback period in number of periods, or None if never recovered.
    """
    cf = np.asarray(cash_flows, dtype=float)
    cumulative = np.cumsum(cf)

    for i in range(len(cumulative)):
        if cumulative[i] >= 0:
            if i == 0:
                return 0.0
            # Linear interpolation
            frac = -cumulative[i - 1] / (cumulative[i] - cumulative[i - 1])
            return float(i - 1 + frac)

    return None


def oil_gas_cashflow(
    oil_rate: np.ndarray | list[float],
    gas_rate: np.ndarray | list[float],
    oil_price: float,
    gas_price: float,
    working_interest: float = 1.0,
    nri: float = 0.80,
    severance_tax: float = 0.05,
    opex_per_month: float = 5000.0,
    capex: float = 0.0,
) -> dict:
    """Monthly cash flow for an oil & gas well.

    Args:
        oil_rate: Monthly oil production in STB/month.
        gas_rate: Monthly gas production in MCF/month.
        oil_price: Oil price in $/STB.
        gas_price: Gas price in $/MCF.
        working_interest: Working interest fraction (default 1.0 = 100%).
        nri: Net revenue interest fraction (default 0.80 = 80%).
        severance_tax: Severance tax rate (default 0.05 = 5%).
        opex_per_month: Monthly operating expense in $.
        capex: Initial capital expenditure in $ (applied at time 0).

    Returns:
        Dict with arrays: 'gross_revenue', 'net_revenue', 'opex',
        'net_cashflow', 'cumulative_cashflow', and scalars:
        'total_revenue', 'total_opex', 'total_net'.
    """
    oil = np.asarray(oil_rate, dtype=float)
    gas = np.asarray(gas_rate, dtype=float)
    n = max(len(oil), len(gas))

    # Pad to same length
    if len(oil) < n:
        oil = np.pad(oil, (0, n - len(oil)))
    if len(gas) < n:
        gas = np.pad(gas, (0, n - len(gas)))

    gross_revenue = oil * oil_price + gas * gas_price
    # Net revenue: gross * NRI * (1 - severance tax) * WI
    net_revenue = gross_revenue * nri * (1.0 - severance_tax) * working_interest
    opex = np.full(n, opex_per_month * working_interest)
    net_cf = net_revenue - opex

    # Apply capex at time 0
    if capex > 0:
        net_cf = np.concatenate([[-capex * working_interest], net_cf])
    else:
        net_cf = np.concatenate([[0.0], net_cf])

    cumulative = np.cumsum(net_cf)

    return {
        "gross_revenue": gross_revenue,
        "net_revenue": net_revenue,
        "opex": opex,
        "net_cashflow": net_cf,
        "cumulative_cashflow": cumulative,
        "total_revenue": round(float(np.sum(net_revenue * working_interest)), 2),
        "total_opex": round(float(np.sum(opex)), 2),
        "total_net": round(float(np.sum(net_cf)), 2),
    }
