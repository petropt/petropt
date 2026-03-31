"""Tests for gas PVT correlations."""

import pytest
from petropt.correlations.gas_pvt import (
    piper_pseudocritical,
    wichert_aziz_correction,
    dak_z_factor,
    lee_gonzalez_eakin_viscosity,
    gas_fvf,
    gas_compressibility,
    gas_density,
)


class TestPiperPseudocritical:
    def test_sweet_gas(self):
        result = piper_pseudocritical(gas_sg=0.65)
        assert result["tpc_rankine"] > 0
        assert result["ppc_psia"] > 0

    def test_sour_gas(self):
        result = piper_pseudocritical(gas_sg=0.70, h2s=0.05, co2=0.10)
        assert result["tpc_rankine"] > 0
        assert result["ppc_psia"] > 0


class TestWichertAziz:
    def test_correction(self):
        result = wichert_aziz_correction(tpc=400, ppc=670, h2s=0.05, co2=0.10)
        assert result["tpc_rankine"] < 400
        assert result["ppc_psia"] != 670

    def test_sweet_gas_no_change(self):
        result = wichert_aziz_correction(tpc=400, ppc=670, h2s=0.0, co2=0.0)
        assert abs(result["tpc_rankine"] - 400) < 0.1
        assert abs(result["ppc_psia"] - 670) < 0.1


class TestDAKZFactor:
    def test_known_value(self):
        """Pr=2.0, Tr=1.5 — Z ≈ 0.73 from Standing-Katz chart."""
        z = dak_z_factor(pr=2.0, tr=1.5)
        assert 0.6 < z < 0.9

    def test_low_pressure(self):
        z = dak_z_factor(pr=0.1, tr=1.5)
        assert 0.95 < z < 1.05


class TestLeeGonzalezEakin:
    def test_typical(self):
        mu = lee_gonzalez_eakin_viscosity(temp=200, pressure=2000, gas_sg=0.65)
        assert 0.01 < mu < 0.05  # typical gas viscosity in cp
        assert isinstance(mu, float)

    def test_increases_with_pressure(self):
        mu_low = lee_gonzalez_eakin_viscosity(temp=200, pressure=500, gas_sg=0.65)
        mu_high = lee_gonzalez_eakin_viscosity(temp=200, pressure=5000, gas_sg=0.65)
        assert mu_high > mu_low


class TestGasFVF:
    def test_typical(self):
        bg = gas_fvf(temp=200, pressure=2000, z=0.85)
        assert bg > 0
        assert bg < 0.1  # Bg is small in bbl/scf

    def test_inversely_proportional_to_pressure(self):
        bg_low = gas_fvf(temp=200, pressure=1000, z=0.9)
        bg_high = gas_fvf(temp=200, pressure=3000, z=0.8)
        assert bg_low > bg_high


class TestGasDensity:
    def test_typical(self):
        rho = gas_density(temp=200, pressure=2000, gas_sg=0.65, z=0.85)
        assert 2 < rho < 20  # typical gas density in lb/ft³
