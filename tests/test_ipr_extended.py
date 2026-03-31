"""Tests for extended IPR correlations."""

import numpy as np
import pytest
from petropt.correlations.ipr_extended import (
    fetkovich_ipr,
    fetkovich_from_tests,
    rawlins_schellhardt,
    pi_ipr,
    composite_ipr,
)


class TestFetkovichIPR:
    def test_at_zero_pwf(self):
        """At Pwf=0, q should equal qmax = C * Pr^(2n)."""
        result = fetkovich_ipr(pr=3000, c=0.001, n=0.8, pwf=0)
        assert result["qo"] > 0
        assert abs(result["qo"] - result["qmax"]) < 1

    def test_at_reservoir_pressure(self):
        result = fetkovich_ipr(pr=3000, c=0.001, n=0.8, pwf=3000)
        assert result["qo"] == 0.0

    def test_full_curve(self):
        result = fetkovich_ipr(pr=3000, c=0.001, n=0.8, num_points=10)
        assert len(result["qo"]) == 11
        assert result["qo"][0] > result["qo"][-1]

    def test_invalid_n(self):
        with pytest.raises(ValueError, match="n must be"):
            fetkovich_ipr(pr=3000, c=0.001, n=0.3)


class TestFetkovichFromTests:
    def test_two_points(self):
        # Generate synthetic test data with known C and n
        pr = 3000
        c_true = 0.001
        n_true = 0.8
        pwf_tests = [2500, 2000, 1500]
        q_tests = [c_true * (pr**2 - p**2)**n_true for p in pwf_tests]

        result = fetkovich_from_tests(pr=pr, pwf_tests=pwf_tests, q_tests=q_tests)
        assert "c" in result
        assert "n" in result
        assert 0.5 <= result["n"] <= 1.0
        assert result["r_squared"] > 0.95


class TestRawlinsSchellhardt:
    def test_basic(self):
        result = rawlins_schellhardt(pr=3000, c=0.0005, n=0.7, pwf=1500)
        assert "qg" in result
        assert result["qg"] > 0
        assert "aof" in result


class TestPIIPR:
    def test_basic(self):
        result = pi_ipr(pr=5000, pi=10, pwf=3000)
        assert abs(result["qo"] - 20000) < 1  # 10 * (5000-3000) = 20000

    def test_at_zero(self):
        result = pi_ipr(pr=5000, pi=10, pwf=0)
        assert abs(result["qo"] - 50000) < 1

    def test_full_curve(self):
        result = pi_ipr(pr=5000, pi=10, num_points=20)
        assert len(result["qo"]) == 21
        # PI IPR is a straight line — q should increase linearly
        assert result["qo"][0] > result["qo"][-1]


class TestCompositeIPR:
    def test_above_pb(self):
        """Above bubble point, should follow PI (straight line)."""
        result = composite_ipr(pr=5000, pb=3000, pi=10, pwf=4000)
        expected = 10 * (5000 - 4000)
        assert abs(result["qo"] - expected) < 1

    def test_at_pb(self):
        """At Pb, rate should equal PI * (Pr - Pb)."""
        result = composite_ipr(pr=5000, pb=3000, pi=10, pwf=3000)
        expected = 10 * (5000 - 3000)
        assert abs(result["qo"] - expected) < 1

    def test_below_pb(self):
        """Below Pb, rate should be higher than at Pb (Vogel curve bends)."""
        result_pb = composite_ipr(pr=5000, pb=3000, pi=10, pwf=3000)
        result_below = composite_ipr(pr=5000, pb=3000, pi=10, pwf=1500)
        assert result_below["qo"] > result_pb["qo"]

    def test_pb_exceeds_pr(self):
        with pytest.raises(ValueError, match="cannot exceed"):
            composite_ipr(pr=3000, pb=5000, pi=10)

    def test_full_curve(self):
        result = composite_ipr(pr=5000, pb=3000, pi=10, num_points=20)
        assert len(result["qo"]) == 21
