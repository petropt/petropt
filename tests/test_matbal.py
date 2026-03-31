"""Tests for material balance equations."""

import numpy as np
import pytest
from petropt.correlations.matbal import (
    gas_pz,
    gas_pz_ogip,
    havlena_odeh_terms,
    estimate_ooip,
    drive_indices,
)


class TestGasPZ:
    def test_zero_production(self):
        """At Gp=0, P/Z should equal Pi."""
        pz = gas_pz(pi=5000, gi=10e9, gp=0)
        assert abs(pz - 5000) < 1

    def test_full_depletion(self):
        """At Gp=G, P/Z should equal 0."""
        pz = gas_pz(pi=5000, gi=10e9, gp=10e9)
        assert abs(pz) < 1

    def test_linear_decline(self):
        """P/Z should decline linearly with Gp."""
        gp = np.array([0, 2.5e9, 5e9, 7.5e9, 10e9])
        pz = gas_pz(pi=5000, gi=10e9, gp=gp)
        # Check linearity
        expected = 5000 * (1 - gp / 10e9)
        np.testing.assert_allclose(pz, expected, rtol=1e-10)

    def test_array_input(self):
        gp = np.array([0, 1e9, 2e9])
        result = gas_pz(pi=5000, gi=10e9, gp=gp)
        assert isinstance(result, np.ndarray)
        assert len(result) == 3


class TestGasPZOGIP:
    def test_known_ogip(self):
        """Synthetic data with known OGIP."""
        gi_true = 10e9
        pi = 5000
        zi = 0.85

        # Generate P/Z data
        gp = np.array([0, 1e9, 2e9, 3e9, 5e9])
        pz = (pi / zi) * (1 - gp / gi_true)

        # Back-calculate (need p and z separately)
        z = np.full_like(pz, zi)  # Simplified: constant Z
        p = pz * z

        result = gas_pz_ogip(p_array=p, z_array=z, gp_array=gp)
        assert abs(result["ogip"] - gi_true) / gi_true < 0.01
        assert result["r_squared"] > 0.99

    def test_too_few_points(self):
        with pytest.raises(ValueError, match="at least 2"):
            gas_pz_ogip([5000], [0.85], [0])


class TestHavlenaOdehTerms:
    def test_basic(self):
        result = havlena_odeh_terms(
            p=4000, pi=5000,
            np_cum=100000, rp=800,
            bo=1.3, boi=1.25,
            bg=0.001, bgi=0.0008,
            rs=500, rsi=600,
        )
        assert "F" in result
        assert "Eo" in result
        assert "Eg" in result
        assert "Efw" in result
        assert result["F"] > 0  # Underground withdrawal must be positive
        assert result["Eo"] > 0  # Oil must expand with pressure drop


class TestEstimateOOIP:
    def test_known_ooip(self):
        """Synthetic data: F = N * Eo with known N."""
        n_true = 10e6  # 10 MMSTB
        eo = np.array([0.001, 0.005, 0.010, 0.020, 0.030])
        f = n_true * eo

        result = estimate_ooip(f_array=f, eo_array=eo)
        assert abs(result["ooip"] - n_true) / n_true < 0.01
        assert result["r_squared"] > 0.99


class TestDriveIndices:
    def test_depletion_only(self):
        """Pure depletion drive — DDI should be ~1.0."""
        result = drive_indices(f=100000, n=10e6, eo=0.01)
        assert abs(result["depletion_drive_index"] - 1.0) < 0.01
        assert abs(result["total"] - 1.0) < 0.01

    def test_mixed_drive(self):
        """Mixed drive — indices should sum to ~1.0."""
        result = drive_indices(
            f=100000, n=10e6, eo=0.005, eg=0.002, m=0.5, we=25000
        )
        assert result["depletion_drive_index"] > 0
        assert result["segregation_drive_index"] > 0
        assert result["water_drive_index"] > 0

    def test_invalid_f(self):
        with pytest.raises(ValueError, match="F must be positive"):
            drive_indices(f=0, n=10e6, eo=0.01)
