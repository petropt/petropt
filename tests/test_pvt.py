"""Tests for PVT correlations.

Reference values from:
    Standing (1947) — bubble point and oil FVF
    McCain, W.D., "The Properties of Petroleum Fluids," PennWell, 1990.
"""

import math
import pytest
from petropt.correlations.pvt import (
    standing_bubble_point,
    standing_rs,
    standing_bo,
    beggs_robinson_viscosity,
    sutton_pseudocritical,
    hall_yarborough_z,
    dranchuk_z_factor,
)


class TestStandingBubblePoint:
    """Standing (1947) bubble point correlation."""

    def test_typical_oil(self):
        """35 API, 0.65 gas SG, 200°F, Rs=500 scf/STB.
        Expected Pb ≈ 2000-2500 psi (Standing chart).
        """
        pb = standing_bubble_point(api=35, gas_sg=0.65, temp=200, rs=500)
        assert 1500 < pb < 3000
        assert isinstance(pb, float)

    def test_heavy_oil(self):
        """15 API heavy oil — lower bubble point."""
        pb = standing_bubble_point(api=15, gas_sg=0.65, temp=150, rs=100)
        assert pb > 14.7
        assert pb < 1500

    def test_zero_rs(self):
        """Zero solution GOR → atmospheric pressure."""
        pb = standing_bubble_point(api=35, gas_sg=0.65, temp=200, rs=0)
        assert pb == 14.7

    def test_minimum_pb(self):
        """Bubble point should never be below atmospheric."""
        pb = standing_bubble_point(api=60, gas_sg=0.65, temp=60, rs=1)
        assert pb >= 14.7

    def test_invalid_api(self):
        with pytest.raises(ValueError, match="API gravity"):
            standing_bubble_point(api=-10, gas_sg=0.65, temp=200)

    def test_invalid_gas_sg(self):
        with pytest.raises(ValueError, match="gas specific gravity"):
            standing_bubble_point(api=35, gas_sg=0, temp=200)

    def test_invalid_temp(self):
        with pytest.raises(ValueError, match="temperature"):
            standing_bubble_point(api=35, gas_sg=0.65, temp=-100)


class TestStandingRs:
    def test_typical(self):
        rs = standing_rs(pressure=2000, temp=200, api=35, gas_sg=0.65)
        assert 200 < rs < 1000
        assert isinstance(rs, float)

    def test_low_pressure(self):
        rs = standing_rs(pressure=100, temp=200, api=35, gas_sg=0.65)
        assert rs >= 0
        assert rs < 100


class TestStandingBo:
    def test_typical(self):
        """Bo should be > 1.0 for oil with dissolved gas."""
        bo = standing_bo(rs=500, temp=200, api=35, gas_sg=0.65)
        assert bo > 1.0
        assert bo < 2.5

    def test_no_gas(self):
        """Dead oil — Bo close to 1.0."""
        bo = standing_bo(rs=0, temp=200, api=35, gas_sg=0.65)
        assert 0.95 < bo < 1.15


class TestBeggsRobinsonViscosity:
    def test_dead_oil(self):
        result = beggs_robinson_viscosity(temp=200, api=35, rs=0)
        assert "dead_oil_viscosity_cp" in result
        assert result["dead_oil_viscosity_cp"] > 0
        # Dead oil of 35 API at 200°F — typically 1-5 cp
        assert 0.1 < result["dead_oil_viscosity_cp"] < 20

    def test_live_oil(self):
        result = beggs_robinson_viscosity(temp=200, api=35, rs=500)
        assert "live_oil_viscosity_cp" in result
        # Live oil viscosity < dead oil viscosity
        assert result["live_oil_viscosity_cp"] < result["dead_oil_viscosity_cp"]

    def test_invalid_temp(self):
        with pytest.raises(ValueError):
            beggs_robinson_viscosity(temp=0, api=35)


class TestSuttonPseudocritical:
    def test_typical_gas(self):
        """0.65 SG gas — Tpc ≈ 370°R, Ppc ≈ 670 psia (Sutton)."""
        result = sutton_pseudocritical(gas_sg=0.65)
        assert "tpc_rankine" in result
        assert "ppc_psia" in result
        assert 350 < result["tpc_rankine"] < 450
        assert 600 < result["ppc_psia"] < 750


class TestHallYarboroughZ:
    def test_typical_conditions(self):
        """Z-factor at moderate conditions — should be 0.7-1.0."""
        z = hall_yarborough_z(temp=200, pressure=2000, gas_sg=0.65)
        assert 0.5 < z < 1.1
        assert isinstance(z, float)

    def test_low_pressure(self):
        """At low pressure, Z → 1.0."""
        z = hall_yarborough_z(temp=200, pressure=50, gas_sg=0.65)
        assert 0.95 < z < 1.05

    def test_invalid_inputs(self):
        with pytest.raises(ValueError):
            hall_yarborough_z(temp=0, pressure=2000, gas_sg=0.65)


class TestDranchukZFactor:
    def test_known_value(self):
        """Pr=2.0, Tr=1.5 — Z ≈ 0.73 from Standing-Katz chart."""
        z = dranchuk_z_factor(pr=2.0, tr=1.5)
        assert 0.6 < z < 0.9
        assert isinstance(z, float)

    def test_low_pressure(self):
        """At very low Pr, Z → 1.0."""
        z = dranchuk_z_factor(pr=0.1, tr=1.5)
        assert 0.95 < z < 1.05

    def test_high_temperature(self):
        """At high Tr, Z increases toward 1.0."""
        z_low = dranchuk_z_factor(pr=3.0, tr=1.2)
        z_high = dranchuk_z_factor(pr=3.0, tr=2.5)
        assert z_high > z_low

    def test_invalid_pr(self):
        with pytest.raises(ValueError):
            dranchuk_z_factor(pr=-1, tr=1.5)

    def test_invalid_tr(self):
        with pytest.raises(ValueError):
            dranchuk_z_factor(pr=2.0, tr=0)
