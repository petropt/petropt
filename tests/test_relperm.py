"""Tests for relative permeability correlations."""

import numpy as np
import pytest
from petropt.correlations.relperm import (
    corey_oil,
    corey_water,
    corey_gas,
    brooks_corey_oil,
    brooks_corey_water,
    let_oil,
    let_water,
)


class TestCoreyOil:
    def test_at_swi(self):
        """At Sw=Swi, kro should be kro_max."""
        kro = corey_oil(sw=0.2, swi=0.2, sor=0.2, kro_max=0.8)
        assert abs(kro - 0.8) < 0.01

    def test_at_1_minus_sor(self):
        """At Sw=1-Sor, kro should be 0."""
        kro = corey_oil(sw=0.8, swi=0.2, sor=0.2, kro_max=0.8)
        assert abs(kro) < 0.01

    def test_monotonic_decrease(self):
        sw = np.linspace(0.2, 0.8, 20)
        kro = corey_oil(sw=sw, swi=0.2, sor=0.2)
        assert all(kro[i] >= kro[i + 1] for i in range(len(kro) - 1))

    def test_array(self):
        sw = np.array([0.2, 0.4, 0.6, 0.8])
        kro = corey_oil(sw=sw, swi=0.2, sor=0.2)
        assert isinstance(kro, np.ndarray)
        assert len(kro) == 4


class TestCoreyWater:
    def test_at_swi(self):
        krw = corey_water(sw=0.2, swi=0.2, sor=0.2)
        assert abs(krw) < 0.01

    def test_at_1_minus_sor(self):
        krw = corey_water(sw=0.8, swi=0.2, sor=0.2, krw_max=0.5)
        assert abs(krw - 0.5) < 0.01

    def test_monotonic_increase(self):
        sw = np.linspace(0.2, 0.8, 20)
        krw = corey_water(sw=sw, swi=0.2, sor=0.2)
        assert all(krw[i] <= krw[i + 1] for i in range(len(krw) - 1))


class TestCoreyGas:
    def test_at_sgc(self):
        krg = corey_gas(sg=0.05, sgc=0.05, swi=0.2, krg_max=0.9)
        assert abs(krg) < 0.01

    def test_above_sgc(self):
        krg = corey_gas(sg=0.5, sgc=0.05, swi=0.2, krg_max=0.9)
        assert krg > 0


class TestBrooksCorey:
    def test_oil_at_endpoints(self):
        kro_swi = brooks_corey_oil(sw=0.2, swi=0.2, sor=0.2, kro_max=1.0)
        kro_sor = brooks_corey_oil(sw=0.8, swi=0.2, sor=0.2, kro_max=1.0)
        assert abs(kro_swi - 1.0) < 0.01
        assert abs(kro_sor) < 0.01

    def test_water_at_endpoints(self):
        krw_swi = brooks_corey_water(sw=0.2, swi=0.2, sor=0.2, krw_max=0.5)
        krw_sor = brooks_corey_water(sw=0.8, swi=0.2, sor=0.2, krw_max=0.5)
        assert abs(krw_swi) < 0.01
        assert abs(krw_sor - 0.5) < 0.01


class TestLET:
    def test_oil_endpoints(self):
        kro_swi = let_oil(sw=0.2, swi=0.2, sor=0.2, kro_max=0.9)
        kro_sor = let_oil(sw=0.8, swi=0.2, sor=0.2, kro_max=0.9)
        assert abs(kro_swi - 0.9) < 0.01
        assert abs(kro_sor) < 0.01

    def test_water_endpoints(self):
        krw_swi = let_water(sw=0.2, swi=0.2, sor=0.2, krw_max=0.4)
        krw_sor = let_water(sw=0.8, swi=0.2, sor=0.2, krw_max=0.4)
        assert abs(krw_swi) < 0.01
        assert abs(krw_sor - 0.4) < 0.01

    def test_invalid_saturations(self):
        with pytest.raises(ValueError, match="swi \\+ sor"):
            corey_oil(sw=0.5, swi=0.6, sor=0.5)
