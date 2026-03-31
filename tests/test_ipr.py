"""Tests for IPR correlations.

Reference:
    Vogel, J.V., "Inflow Performance Relationships for Solution-Gas Drive
    Wells," JPT, January 1968.
"""

import numpy as np
import pytest
from petropt.correlations.ipr import vogel_ipr


class TestVogelIPR:
    def test_single_point(self):
        """At Pwf=0, q should equal qmax."""
        result = vogel_ipr(qmax=1000, pwf=0, pr=3000)
        assert abs(result["qo"] - 1000) < 1

    def test_at_reservoir_pressure(self):
        """At Pwf=Pr, q should be 0."""
        result = vogel_ipr(qmax=1000, pwf=3000, pr=3000)
        assert result["qo"] == 0.0

    def test_midpoint(self):
        """At Pwf/Pr = 0.5, q/qmax = 1 - 0.2*0.5 - 0.8*0.25 = 0.7."""
        result = vogel_ipr(qmax=1000, pwf=1500, pr=3000)
        assert abs(result["qo"] - 700) < 1

    def test_full_curve(self):
        """Generate full IPR curve."""
        result = vogel_ipr(qmax=1000, pr=3000, num_points=10)
        assert "qo" in result
        assert "pwf" in result
        assert len(result["qo"]) == 11  # num_points + 1
        assert len(result["pwf"]) == 11
        # First point: Pwf=0, q=qmax
        assert abs(result["qo"][0] - 1000) < 1
        # Last point: Pwf=Pr, q=0
        assert abs(result["qo"][-1]) < 1

    def test_array_pwf(self):
        """Pass array of pressures."""
        pwf_array = np.array([0, 1000, 2000, 3000])
        result = vogel_ipr(qmax=1000, pwf=pwf_array, pr=3000)
        assert isinstance(result["qo"], np.ndarray)
        assert len(result["qo"]) == 4

    def test_pb_as_pr(self):
        """pb used as reservoir pressure when pr not given."""
        result = vogel_ipr(qmax=1000, pwf=0, pb=3000)
        assert abs(result["qo"] - 1000) < 1
        assert result["pr"] == 3000

    def test_invalid_qmax(self):
        with pytest.raises(ValueError, match="qmax"):
            vogel_ipr(qmax=-100, pwf=0, pr=3000)

    def test_no_pressure(self):
        with pytest.raises(ValueError, match="Either pr or pb"):
            vogel_ipr(qmax=1000, pwf=0)

    def test_negative_pwf(self):
        with pytest.raises(ValueError, match="pwf"):
            vogel_ipr(qmax=1000, pwf=-100, pr=3000)
