"""Tests for hydraulics correlations.

Reference:
    Churchill, S.W., Chemical Engineering, 84(24), 1977.
"""

import pytest
from petropt.correlations.hydraulics import darcy_weisbach


class TestDarcyWeisbach:
    def test_typical_pipe_flow(self):
        """Typical oilfield pipe flow — should produce reasonable dP."""
        result = darcy_weisbach(
            flow_rate=5000,
            diameter=4.0,
            length=5000,
            roughness=0.0006,
            density=55.0,
            viscosity=2.0,
        )
        assert "pressure_drop_psi" in result
        assert "friction_factor" in result
        assert "velocity_ft_s" in result
        assert "reynolds_number" in result
        assert "flow_regime" in result
        assert result["pressure_drop_psi"] > 0
        assert result["friction_factor"] > 0
        assert result["velocity_ft_s"] > 0

    def test_laminar_flow(self):
        """Very low flow rate — should be laminar."""
        result = darcy_weisbach(
            flow_rate=1,
            diameter=4.0,
            length=100,
            viscosity=100,  # Very viscous
        )
        assert result["flow_regime"] == "laminar"

    def test_turbulent_flow(self):
        """High flow rate — should be turbulent."""
        result = darcy_weisbach(
            flow_rate=10000,
            diameter=2.0,
            length=5000,
        )
        assert result["flow_regime"] == "turbulent"

    def test_pressure_increases_with_length(self):
        r1 = darcy_weisbach(flow_rate=5000, diameter=4.0, length=1000)
        r2 = darcy_weisbach(flow_rate=5000, diameter=4.0, length=5000)
        assert r2["pressure_drop_psi"] > r1["pressure_drop_psi"]

    def test_pressure_increases_with_flow(self):
        r1 = darcy_weisbach(flow_rate=1000, diameter=4.0, length=5000)
        r2 = darcy_weisbach(flow_rate=5000, diameter=4.0, length=5000)
        assert r2["pressure_drop_psi"] > r1["pressure_drop_psi"]

    def test_invalid_flow_rate(self):
        with pytest.raises(ValueError, match="flow_rate"):
            darcy_weisbach(flow_rate=0, diameter=4.0, length=5000)

    def test_invalid_diameter(self):
        with pytest.raises(ValueError, match="diameter"):
            darcy_weisbach(flow_rate=5000, diameter=-1, length=5000)

    def test_invalid_length(self):
        with pytest.raises(ValueError, match="length"):
            darcy_weisbach(flow_rate=5000, diameter=4.0, length=0)
