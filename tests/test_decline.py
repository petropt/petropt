"""Tests for Arps decline curve correlations.

Reference:
    Arps, J.J., "Analysis of Decline Curves," Trans. AIME, 160, 1945.
"""

import math
import numpy as np
import pytest
from petropt.correlations.decline import arps_decline, arps_cumulative, arps_eur


class TestArpsDecline:
    def test_exponential_t0(self):
        """At t=0, rate should equal qi."""
        q = arps_decline(qi=1000, di=0.1, b=0, t=0)
        assert abs(q - 1000) < 0.01

    def test_exponential_decay(self):
        """Exponential: q(t) = qi * exp(-di * t)."""
        q = arps_decline(qi=1000, di=0.1, b=0, t=10)
        expected = 1000 * math.exp(-0.1 * 10)
        assert abs(q - expected) < 0.01

    def test_hyperbolic(self):
        """Hyperbolic: q(t) = qi / (1 + b*di*t)^(1/b)."""
        q = arps_decline(qi=1000, di=0.1, b=0.5, t=10)
        expected = 1000 / (1 + 0.5 * 0.1 * 10) ** (1 / 0.5)
        assert abs(q - expected) < 0.01

    def test_harmonic(self):
        """Harmonic: q(t) = qi / (1 + di * t)."""
        q = arps_decline(qi=1000, di=0.1, b=1, t=10)
        expected = 1000 / (1 + 0.1 * 10)
        assert abs(q - expected) < 0.01

    def test_array_input(self):
        """Array of time values."""
        t = np.array([0, 5, 10, 20])
        q = arps_decline(qi=1000, di=0.1, b=0.5, t=t)
        assert isinstance(q, np.ndarray)
        assert len(q) == 4
        assert q[0] > q[1] > q[2] > q[3]

    def test_decline_is_monotonic(self):
        """Rate should always decrease."""
        t = np.linspace(0, 100, 50)
        for b in [0, 0.5, 1.0]:
            q = arps_decline(qi=1000, di=0.1, b=b, t=t)
            assert all(q[i] >= q[i + 1] for i in range(len(q) - 1))

    def test_invalid_qi(self):
        with pytest.raises(ValueError, match="qi"):
            arps_decline(qi=0, di=0.1, b=0.5, t=10)

    def test_invalid_di(self):
        with pytest.raises(ValueError, match="di"):
            arps_decline(qi=1000, di=-0.1, b=0.5, t=10)

    def test_invalid_b(self):
        with pytest.raises(ValueError, match="b must"):
            arps_decline(qi=1000, di=0.1, b=3, t=10)


class TestArpsCumulative:
    def test_exponential(self):
        """Exponential Np = (qi/di) * (1 - exp(-di*t))."""
        np_val = arps_cumulative(qi=1000, di=0.1, b=0, t=10)
        expected = (1000 / 0.1) * (1 - math.exp(-0.1 * 10))
        assert abs(np_val - expected) < 1

    def test_harmonic(self):
        """Harmonic Np = (qi/di) * ln(1 + di*t)."""
        np_val = arps_cumulative(qi=1000, di=0.1, b=1, t=10)
        expected = (1000 / 0.1) * math.log(1 + 0.1 * 10)
        assert abs(np_val - expected) < 1

    def test_zero_time(self):
        assert arps_cumulative(qi=1000, di=0.1, b=0.5, t=0) == 0.0

    def test_cumulative_increases(self):
        """Cumulative production should always increase with time."""
        for t in [1, 5, 10, 50]:
            np1 = arps_cumulative(qi=1000, di=0.1, b=0.5, t=t)
            np2 = arps_cumulative(qi=1000, di=0.1, b=0.5, t=t + 1)
            assert np2 > np1


class TestArpsEUR:
    def test_basic(self):
        result = arps_eur(qi=1000, di=0.1, b=0.5, economic_limit=10)
        assert "eur" in result
        assert "time_to_limit" in result
        assert "final_rate" in result
        assert result["eur"] > 0
        assert result["final_rate"] >= 10 or result["time_to_limit"] >= 600

    def test_exponential_eur(self):
        result = arps_eur(qi=1000, di=0.1, b=0, economic_limit=10)
        assert result["eur"] > 0

    def test_qi_below_limit(self):
        """If qi <= economic_limit, EUR should be 0."""
        result = arps_eur(qi=5, di=0.1, b=0.5, economic_limit=10)
        assert result["eur"] == 0.0
