"""Tests for petroleum economics."""

import numpy as np
import pytest
from petropt.economics import npv, irr, payback_period, oil_gas_cashflow


class TestNPV:
    def test_simple(self):
        """Known NPV: -1000 initial, then 500/year for 3 years at 10%."""
        cf = [-1000, 500, 500, 500]
        result = npv(cf, discount_rate=0.10, periods_per_year=1)
        # NPV = -1000 + 500/1.1 + 500/1.21 + 500/1.331 ≈ 243.43
        assert 240 < result < 250

    def test_negative_npv(self):
        cf = [-1000, 100, 100, 100]
        result = npv(cf, discount_rate=0.10, periods_per_year=1)
        assert result < 0

    def test_zero_discount(self):
        cf = [-1000, 500, 500, 500]
        result = npv(cf, discount_rate=0.0, periods_per_year=1)
        assert abs(result - 500) < 1  # Sum of all flows


class TestIRR:
    def test_simple(self):
        cf = [-1000, 400, 400, 400]
        result = irr(cf)
        assert result is not None
        assert 0.05 < result < 0.15

    def test_high_return(self):
        cf = [-100, 200]
        result = irr(cf)
        assert result is not None
        assert abs(result - 1.0) < 0.01  # 100% return

    def test_no_solution(self):
        """All negative cash flows — no IRR."""
        cf = [-100, -100, -100]
        result = irr(cf)
        # May return None or a negative value
        assert result is None or result < 0


class TestPaybackPeriod:
    def test_simple(self):
        cf = [-1000, 300, 300, 300, 300]
        result = payback_period(cf)
        assert result is not None
        assert 3 < result < 4  # Payback between period 3 and 4

    def test_immediate(self):
        cf = [100, 200]
        result = payback_period(cf)
        assert result == 0.0

    def test_never_recovered(self):
        cf = [-1000, 10, 10, 10]
        result = payback_period(cf)
        assert result is None


class TestOilGasCashflow:
    def test_basic(self):
        oil = [500, 480, 460, 440, 420]  # STB/month
        gas = [1000, 950, 900, 850, 800]  # MCF/month
        result = oil_gas_cashflow(
            oil_rate=oil,
            gas_rate=gas,
            oil_price=70.0,
            gas_price=3.0,
            capex=100000,
        )
        assert "net_cashflow" in result
        assert "cumulative_cashflow" in result
        assert result["total_revenue"] > 0
        assert result["net_cashflow"][0] < 0  # Capex at time 0

    def test_no_capex(self):
        oil = [500]
        gas = [1000]
        result = oil_gas_cashflow(
            oil_rate=oil, gas_rate=gas,
            oil_price=70, gas_price=3,
        )
        assert result["net_cashflow"][0] == 0.0  # No capex
