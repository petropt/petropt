"""Tests for Beggs-Brill multiphase flow correlation."""

import pytest
from petropt.correlations.multiphase import beggs_brill_pressure_gradient


class TestBeggsBrill:
    def test_typical_vertical_well(self):
        result = beggs_brill_pressure_gradient(
            pressure=2000,
            temp=180,
            oil_rate=500,
            water_rate=200,
            gor=800,
            oil_sg=0.85,
            gas_sg=0.65,
            pipe_id=2.441,
            angle=90,
        )
        assert "dp_dz_psi_ft" in result
        assert "flow_pattern" in result
        assert "liquid_holdup" in result
        assert result["dp_dz_psi_ft"] > 0
        assert 0 < result["liquid_holdup"] <= 1.0
        assert result["flow_pattern"] in ("segregated", "intermittent", "distributed", "transition")

    def test_horizontal_pipe(self):
        result = beggs_brill_pressure_gradient(
            pressure=2000,
            temp=180,
            oil_rate=1000,
            water_rate=500,
            gor=500,
            oil_sg=0.85,
            gas_sg=0.65,
            pipe_id=4.0,
            angle=0,  # horizontal
        )
        # Gravity component should be near zero for horizontal
        assert abs(result["dp_dz_gravity"]) < 0.01
        assert result["dp_dz_friction"] > 0

    def test_gravity_dominates_vertical(self):
        result = beggs_brill_pressure_gradient(
            pressure=3000,
            temp=180,
            oil_rate=100,
            water_rate=50,
            gor=300,
            oil_sg=0.85,
            gas_sg=0.65,
            pipe_id=2.441,
            angle=90,
        )
        # For low rates in vertical pipe, gravity >> friction
        assert result["dp_dz_gravity"] > result["dp_dz_friction"]

    def test_friction_increases_with_rate(self):
        """Friction component should increase with flow rate."""
        r1 = beggs_brill_pressure_gradient(
            pressure=2000, temp=180, oil_rate=100, water_rate=50,
            gor=500, oil_sg=0.85, gas_sg=0.65, pipe_id=2.441, angle=0,
        )
        r2 = beggs_brill_pressure_gradient(
            pressure=2000, temp=180, oil_rate=1000, water_rate=500,
            gor=500, oil_sg=0.85, gas_sg=0.65, pipe_id=2.441, angle=0,
        )
        assert r2["dp_dz_friction"] >= r1["dp_dz_friction"]

    def test_invalid_rates(self):
        with pytest.raises(ValueError, match="liquid rate"):
            beggs_brill_pressure_gradient(
                pressure=2000, temp=180, oil_rate=0, water_rate=0,
                gor=500, oil_sg=0.85, gas_sg=0.65,
            )
