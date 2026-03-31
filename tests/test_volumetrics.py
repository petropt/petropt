"""Tests for volumetric calculations."""

import math
import pytest
from petropt.correlations.volumetrics import stoiip, giip, drainage_radius, recovery_factor


class TestSTOIIP:
    def test_typical(self):
        """Typical reservoir: 640 acres, 30 ft, 15% porosity, 25% Sw, Bo=1.2."""
        n = stoiip(area=640, thickness=30, porosity=0.15, sw=0.25, bo=1.2)
        # N = 7758 * 640 * 30 * 0.15 * 0.75 / 1.2 ≈ 13.97 MMSTB
        assert 13e6 < n < 15e6

    def test_invalid_porosity(self):
        with pytest.raises(ValueError, match="porosity"):
            stoiip(area=640, thickness=30, porosity=0, sw=0.25, bo=1.2)

    def test_invalid_sw(self):
        with pytest.raises(ValueError, match="water saturation"):
            stoiip(area=640, thickness=30, porosity=0.15, sw=1.0, bo=1.2)


class TestGIIP:
    def test_typical(self):
        g = giip(area=640, thickness=50, porosity=0.12, sw=0.30, bg=0.005)
        assert g > 0

    def test_invalid_area(self):
        with pytest.raises(ValueError, match="area"):
            giip(area=0, thickness=50, porosity=0.12, sw=0.30, bg=0.005)


class TestDrainageRadius:
    def test_640_acres(self):
        """640 acres (1 section) → re ≈ 2980 ft."""
        re = drainage_radius(640)
        assert abs(re - 2980) < 20

    def test_40_acres(self):
        re = drainage_radius(40)
        assert 700 < re < 800


class TestRecoveryFactor:
    def test_typical(self):
        rf = recovery_factor(np_cum=5e6, stoiip_val=20e6)
        assert abs(rf - 0.25) < 0.001

    def test_zero_production(self):
        rf = recovery_factor(np_cum=0, stoiip_val=20e6)
        assert rf == 0.0
