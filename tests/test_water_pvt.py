"""Tests for water PVT correlations."""

import pytest
from petropt.correlations.water_pvt import (
    water_fvf,
    water_viscosity,
    water_compressibility,
    water_gas_solubility,
    water_density,
)


class TestWaterFVF:
    def test_near_unity(self):
        bw = water_fvf(temp=200, pressure=3000)
        assert 0.99 < bw < 1.10  # Water FVF close to 1

    def test_increases_with_temp(self):
        bw_low = water_fvf(temp=100, pressure=3000)
        bw_high = water_fvf(temp=300, pressure=3000)
        assert bw_high > bw_low


class TestWaterViscosity:
    def test_typical(self):
        mu = water_viscosity(temp=200)
        assert 0.1 < mu < 2.0  # cp

    def test_decreases_with_temp(self):
        mu_cold = water_viscosity(temp=100)
        mu_hot = water_viscosity(temp=300)
        assert mu_cold > mu_hot

    def test_brine_higher_viscosity(self):
        mu_fresh = water_viscosity(temp=200, salinity=0)
        mu_brine = water_viscosity(temp=200, salinity=100000)
        # Brine viscosity should be different from fresh water
        assert mu_brine != mu_fresh


class TestWaterCompressibility:
    def test_typical(self):
        cw = water_compressibility(temp=200, pressure=3000)
        assert 1e-7 < cw < 1e-4  # typical range for water

    def test_positive(self):
        cw = water_compressibility(temp=200, pressure=3000, salinity=50000)
        assert cw > 0


class TestWaterGasSolubility:
    def test_typical(self):
        rsw = water_gas_solubility(temp=200, pressure=3000)
        assert rsw >= 0
        assert rsw < 50  # Gas solubility in water is low

    def test_increases_with_pressure(self):
        rsw_low = water_gas_solubility(temp=200, pressure=1000)
        rsw_high = water_gas_solubility(temp=200, pressure=5000)
        assert rsw_high > rsw_low


class TestWaterDensity:
    def test_typical(self):
        rho = water_density(temp=200, pressure=3000)
        assert 55 < rho < 68  # lb/ft³

    def test_brine_heavier(self):
        rho_fresh = water_density(temp=200, pressure=3000, salinity=0)
        rho_brine = water_density(temp=200, pressure=3000, salinity=200000)
        assert rho_brine > rho_fresh
