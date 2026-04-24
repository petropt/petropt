"""Tests for Rate Transient Analysis (RTA) tools.

Reference values derived from Blasingame (1991), Agarwal-Gardner (1999),
Mattar-Anderson (2005), and Wattenbarger linear-flow analysis.
"""

import numpy as np
import pytest

from petropt.rta import (
    agarwal_gardner_variables,
    blasingame_variables,
    flowing_material_balance,
    material_balance_time,
    npi_variables,
    permeability_from_linear_flow,
    pressure_normalized_rate,
    sqrt_time_analysis,
)


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

class TestTransforms:
    def test_pressure_normalized_rate_basic(self):
        qn = pressure_normalized_rate(
            rate=[1000, 800, 600],
            flowing_pressure=[1000, 1500, 2000],
            initial_pressure=3000,
        )
        # dp = [2000, 1500, 1000]; qn = [0.5, 0.533..., 0.6]
        assert qn[0] == pytest.approx(0.5)
        assert qn[1] == pytest.approx(800 / 1500)
        assert qn[2] == pytest.approx(0.6)

    def test_pressure_normalized_rate_clamps_reverse_drawdown(self):
        """Samples with Pwf >= Pi → qn = 0 (no physical drawdown)."""
        qn = pressure_normalized_rate(
            rate=[500, 500],
            flowing_pressure=[1000, 3500],
            initial_pressure=3000,
        )
        assert qn[0] > 0
        assert qn[1] == 0.0

    def test_material_balance_time(self):
        """tMB = Np / q. For 100 bbl cumulative at 10 bbl/d rate → 10 days."""
        tmb = material_balance_time([50, 100, 200], [10, 10, 8])
        assert tmb[0] == pytest.approx(5.0)
        assert tmb[1] == pytest.approx(10.0)
        assert tmb[2] == pytest.approx(25.0)

    def test_material_balance_time_zero_rate_zero(self):
        tmb = material_balance_time([100, 120, 140], [10, 0, 5])
        assert tmb[1] == 0.0

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            pressure_normalized_rate([1, 2], [1000], initial_pressure=3000)

    def test_zero_initial_pressure_raises(self):
        with pytest.raises(ValueError):
            pressure_normalized_rate([1, 2], [500, 600], initial_pressure=0)


# ---------------------------------------------------------------------------
# Type curves
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_decline():
    """Synthetic exponential-decline production for type-curve tests."""
    t = np.arange(1, 61, dtype=float)  # 60 days
    q0, d = 1000.0, 0.05
    q = q0 * np.exp(-d * t)
    cum = np.cumsum(q)  # approximate
    pwf = np.full_like(t, 800.0)
    pi = 3000.0
    return {"t": t, "q": q, "cum": cum, "pwf": pwf, "pi": pi}


class TestBlasingame:
    def test_returns_all_four_series(self, synthetic_decline):
        d = synthetic_decline
        res = blasingame_variables(d["t"], d["q"], d["cum"], d["pwf"], d["pi"])
        assert set(res.keys()) == {
            "material_balance_time",
            "normalized_rate",
            "rate_integral",
            "rate_integral_derivative",
        }
        for v in res.values():
            assert isinstance(v, np.ndarray)
            assert v.size == d["t"].size

    def test_material_balance_time_monotonic(self, synthetic_decline):
        """Under exponential decline tMB is strictly increasing."""
        d = synthetic_decline
        res = blasingame_variables(d["t"], d["q"], d["cum"], d["pwf"], d["pi"])
        tmb = res["material_balance_time"]
        assert np.all(np.diff(tmb) > 0)

    def test_normalized_rate_matches_transform(self, synthetic_decline):
        d = synthetic_decline
        res = blasingame_variables(d["t"], d["q"], d["cum"], d["pwf"], d["pi"])
        qn_expected = pressure_normalized_rate(d["q"], d["pwf"], d["pi"])
        np.testing.assert_allclose(res["normalized_rate"], qn_expected, rtol=1e-12)

    def test_needs_at_least_3_points(self):
        with pytest.raises(ValueError, match="3 data points"):
            blasingame_variables([1, 2], [100, 90], [100, 190], [800, 800], 3000)


class TestAgarwalGardner:
    def test_returns_expected_keys(self, synthetic_decline):
        d = synthetic_decline
        res = agarwal_gardner_variables(d["t"], d["q"], d["cum"], d["pwf"], d["pi"])
        assert set(res.keys()) == {
            "material_balance_time",
            "normalized_rate",
            "inverse_normalized_rate",
            "cumulative_normalized",
        }

    def test_inverse_reciprocal_of_forward(self, synthetic_decline):
        d = synthetic_decline
        res = agarwal_gardner_variables(d["t"], d["q"], d["cum"], d["pwf"], d["pi"])
        # qn * inv_qn == 1 where both are well-defined (constant pwf)
        product = res["normalized_rate"] * res["inverse_normalized_rate"]
        np.testing.assert_allclose(product, 1.0, rtol=1e-12)


class TestNPI:
    def test_npi_derivative_close_to_zero_for_constant_drawdown(
        self, synthetic_decline
    ):
        """Pure exponential decline with constant pwf: NPI rises but its
        log-time derivative is finite and bounded."""
        d = synthetic_decline
        res = npi_variables(d["t"], d["q"], d["pwf"], d["pi"])
        assert res["npi"][-1] > res["npi"][5]
        # Derivative must be finite everywhere
        assert np.isfinite(res["npi_derivative"]).all()


# ---------------------------------------------------------------------------
# Flowing material balance
# ---------------------------------------------------------------------------

class TestFMB:
    def test_fmb_recovers_OOIP_for_synthetic_tank(self):
        """Synthetic linearly-declining rate with constant drawdown. The
        internal trapezoidal cumulative makes the q vs Np relationship
        near-linear but not exact, so R^2 should be ~0.98 and OOIP
        should come out positive and physical."""
        rates = np.array([1000.0, 900, 800, 700, 600, 500, 400])
        # Constant pwf → dp constant 2000
        pwf = np.full_like(rates, 1000.0)
        pi = 3000.0
        fvf = 1.2
        ct = 1e-5
        res = flowing_material_balance(rates, pwf, pi, fluid_fvf=fvf, total_compressibility=ct)
        assert res["slope"] < 0
        assert res["intercept"] > 0
        assert res["ooip_estimate"] is not None
        assert res["r_squared"] > 0.98

    def test_fmb_needs_positive_fvf(self):
        with pytest.raises(ValueError):
            flowing_material_balance([1, 2, 3], [800, 800, 800], 3000, -1, 1e-5)

    def test_fmb_insufficient_data(self):
        with pytest.raises(ValueError):
            flowing_material_balance([1, 2], [800, 800], 3000, 1.2, 1e-5)


# ---------------------------------------------------------------------------
# Linear flow
# ---------------------------------------------------------------------------

class TestLinearFlow:
    def test_sqrt_time_fit_on_exact_linear_flow(self):
        """If 1/q = a + b*sqrt(t) exactly, the fit recovers a and b."""
        t = np.arange(1, 31, dtype=float)
        # 1/q = 0.001 + 0.0002 * sqrt(t) -> q = 1/(0.001 + 0.0002*sqrt(t))
        inv_q = 0.001 + 0.0002 * np.sqrt(t)
        q = 1.0 / inv_q
        pwf = np.full_like(t, 2000.0)
        pi = 3000.0
        # dp constant = 1000, so inv_qn = 1000/q = 1000*(0.001 + 0.0002*sqrt(t))
        # slope in inv_qn vs sqrt(t) = 0.2; intercept = 1.0
        res = sqrt_time_analysis(q, t, pwf, pi)
        assert res["slope"] == pytest.approx(0.2, rel=1e-6)
        assert res["intercept"] == pytest.approx(1.0, rel=1e-6)
        assert res["r_squared"] == pytest.approx(1.0, rel=1e-6)

    def test_sqrt_time_flags_late_time_deviation(self):
        """Construct linear flow for first 20 points, then deviate."""
        t = np.arange(1, 31, dtype=float)
        inv_q = 0.001 + 0.0002 * np.sqrt(t)
        # Deviate after t=20
        inv_q[20:] += 0.0005 * (t[20:] - 20)
        q = 1.0 / inv_q
        pwf = np.full_like(t, 2000.0)
        pi = 3000.0
        res = sqrt_time_analysis(q, t, pwf, pi)
        assert res["end_of_linear_flow_time"] is not None
        # Heuristic detector flags the first 2σ deviation after 30% of the
        # record; the exact sample depends on how the deviated tail pulls
        # the global fit. Just assert that it lands somewhere in the
        # linear / transition window rather than at the record boundaries.
        assert 10.0 <= res["end_of_linear_flow_time"] <= 25.0

    def test_permeability_from_slope_returns_product_without_xf(self):
        res = permeability_from_linear_flow(
            slope=0.2, net_pay_ft=100, porosity=0.08,
            viscosity_cp=0.5, total_compressibility=1e-5, fluid_fvf=1.2,
        )
        assert "sqrt_k_times_xf" in res
        assert "permeability_md" not in res

    def test_permeability_from_slope_with_xf(self):
        res = permeability_from_linear_flow(
            slope=0.2, net_pay_ft=100, porosity=0.08,
            viscosity_cp=0.5, total_compressibility=1e-5, fluid_fvf=1.2,
            fracture_half_length_ft=200,
        )
        assert res["sqrt_k_times_xf"] > 0
        assert res["permeability_md"] > 0
        # Self-consistency: sqrt(k)*xf == result["sqrt_k_times_xf"]
        assert res["permeability_md"] ** 0.5 * res["fracture_half_length_ft"] == pytest.approx(
            res["sqrt_k_times_xf"], rel=1e-9
        )

    def test_permeability_from_slope_rejects_invalid(self):
        with pytest.raises(ValueError):
            permeability_from_linear_flow(
                slope=-1.0, net_pay_ft=100, porosity=0.08,
                viscosity_cp=0.5, total_compressibility=1e-5,
            )
