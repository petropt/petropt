"""Tests for drilling engineering calculations.

Reference values:
    Bourgoyne et al., "Applied Drilling Engineering," SPE Textbook Vol. 2, 1991.
    API Bulletin 5C3 (1994) published casing rating tables.
    IADC Drilling Manual, 12th ed.
"""

import math

import pytest

from petropt.drilling import (
    annular_velocity,
    bit_pressure_drop,
    burst_pressure_barlow,
    collapse_pressure_api5c3,
    equivalent_circulating_density,
    formation_pressure_gradient,
    hydrostatic_pressure,
    initial_and_final_circulating_pressure,
    kill_mud_weight,
    maasp,
    nozzle_total_flow_area,
)


# ---------------------------------------------------------------------------
# Well control
# ---------------------------------------------------------------------------

class TestHydrostatic:
    def test_hand_calc(self):
        """10 ppg * 10000 ft → 5200 psi."""
        assert hydrostatic_pressure(10.0, 10000) == pytest.approx(5200.0)

    def test_monotonic_in_mw(self):
        assert hydrostatic_pressure(12, 8000) > hydrostatic_pressure(10, 8000)

    def test_rejects_zero_mw(self):
        with pytest.raises(ValueError):
            hydrostatic_pressure(0, 10000)

    def test_rejects_zero_tvd(self):
        with pytest.raises(ValueError):
            hydrostatic_pressure(10, 0)


class TestECD:
    def test_hand_calc(self):
        """10 ppg + 200 psi / (0.052 * 10000) = 10.3846 ppg."""
        ecd = equivalent_circulating_density(10.0, 200.0, 10000.0)
        assert ecd == pytest.approx(10.0 + 200.0 / 520.0, rel=1e-6)

    def test_zero_apl_equals_mud_weight(self):
        assert equivalent_circulating_density(11.5, 0, 12000) == 11.5

    def test_rejects_negative_apl(self):
        with pytest.raises(ValueError):
            equivalent_circulating_density(10, -10, 5000)


class TestFormationPressureGradient:
    def test_hand_calc(self):
        """5200 psi at 10000 ft → 10 ppg."""
        assert formation_pressure_gradient(5200.0, 10000.0) == pytest.approx(10.0)

    def test_zero_pressure_zero_gradient(self):
        assert formation_pressure_gradient(0, 10000) == 0.0


class TestKillMudWeight:
    def test_hand_calc(self):
        """SIDP 500 psi @ 10000 ft, original 10 ppg → kill MW 10.962 ppg."""
        kmw = kill_mud_weight(500.0, 10.0, 10000.0)
        assert kmw == pytest.approx(10.0 + 500.0 / 520.0, rel=1e-6)

    def test_zero_sidp_equals_original(self):
        """Balanced well: SIDP=0 means the current mud weight is already the kill weight."""
        assert kill_mud_weight(0, 12, 8000) == 12.0


class TestICPFCP:
    def test_driller_hand_calc(self):
        """SIDP 400 + SCP 800 → ICP = 1200 psi;
        FCP = 800 * (11/10) = 880 psi."""
        res = initial_and_final_circulating_pressure(
            sidp_psi=400, slow_circulating_pressure_psi=800,
            kill_mud_weight_ppg=11.0, original_mud_weight_ppg=10.0,
        )
        assert res["icp_psi"] == pytest.approx(1200.0)
        assert res["fcp_psi"] == pytest.approx(880.0)

    def test_fcp_equals_scp_when_no_kill_needed(self):
        res = initial_and_final_circulating_pressure(0, 500, 10, 10)
        assert res["fcp_psi"] == pytest.approx(500.0)


class TestMAASP:
    def test_hand_calc(self):
        """FG 14 - MW 10 ppg = 4 ppg margin at 8000 ft → 1664 psi."""
        assert maasp(14.0, 10.0, 8000.0) == pytest.approx(1664.0)

    def test_negative_when_mw_above_fg(self):
        """MW heavier than FG is a flag — function returns negative MAASP."""
        assert maasp(10.0, 11.0, 8000.0) < 0


# ---------------------------------------------------------------------------
# Hydraulics
# ---------------------------------------------------------------------------

class TestAnnularVelocity:
    def test_hand_calc(self):
        """800 gpm in 12.25" hole with 5" DP:
        AV = 24.51*800 / (12.25² - 5²) = 19608 / 125.0625 = 156.79 ft/min."""
        av = annular_velocity(800, 12.25, 5.0)
        expected = 24.51 * 800 / (12.25**2 - 5.0**2)
        assert av == pytest.approx(expected, rel=1e-6)

    def test_pipe_at_or_above_hole_raises(self):
        with pytest.raises(ValueError):
            annular_velocity(500, 8.5, 8.5)
        with pytest.raises(ValueError):
            annular_velocity(500, 8.5, 9.0)


class TestNozzleTFA:
    def test_three_equal_nozzles(self):
        """Three 12/32" nozzles → 3 * pi/4 * (12/32)² = 0.3313 in²."""
        tfa = nozzle_total_flow_area([12, 12, 12])
        assert tfa == pytest.approx(3 * math.pi / 4 * (12.0 / 32.0) ** 2, rel=1e-12)

    def test_mixed_sizes(self):
        """14/32 + 2 * 11/32 — a classic asymmetric jet layout."""
        tfa = nozzle_total_flow_area([14, 11, 11])
        expected = (
            math.pi / 4 * (14.0 / 32.0) ** 2
            + 2 * math.pi / 4 * (11.0 / 32.0) ** 2
        )
        assert tfa == pytest.approx(expected, rel=1e-12)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            nozzle_total_flow_area([])

    def test_non_positive_raises(self):
        with pytest.raises(ValueError):
            nozzle_total_flow_area([12, 0, 12])


class TestBitPressureDrop:
    def test_hand_calc(self):
        """600 gpm, 10 ppg, 0.33 in² TFA:
        dP = 10 * 600² / (12032 * 0.33²) ≈ 2747.5 psi."""
        dp = bit_pressure_drop(600, 10, 0.33)
        expected = 10 * 600**2 / (12032.0 * 0.33**2)
        assert dp == pytest.approx(expected, rel=1e-6)

    def test_quadratic_in_flow(self):
        """Doubling flow rate → 4x bit pressure drop."""
        dp1 = bit_pressure_drop(500, 10, 0.5)
        dp2 = bit_pressure_drop(1000, 10, 0.5)
        assert dp2 / dp1 == pytest.approx(4.0, rel=1e-6)

    def test_inverse_square_tfa(self):
        """Halving TFA → 4x pressure drop."""
        dp_large = bit_pressure_drop(600, 10, 0.4)
        dp_small = bit_pressure_drop(600, 10, 0.2)
        assert dp_small / dp_large == pytest.approx(4.0, rel=1e-6)


# ---------------------------------------------------------------------------
# Tubulars — burst and collapse
# ---------------------------------------------------------------------------

class TestBurstBarlow:
    def test_seven_inch_n80_matches_api_rating(self):
        """7\" 26 lb/ft N-80 (Fy=80k, t=0.362\", OD=7.0\") →
        API published internal yield = 7240 psi."""
        p = burst_pressure_barlow(
            yield_strength_psi=80_000, wall_thickness_in=0.362, od_in=7.0
        )
        assert p == pytest.approx(7240.0, abs=5.0)

    def test_design_factor_one_gives_nominal(self):
        """design_factor=1.0 drops the 0.875 API wall-tolerance multiplier."""
        with_tol = burst_pressure_barlow(80_000, 0.5, 9.625)
        nominal = burst_pressure_barlow(80_000, 0.5, 9.625, design_factor=1.0)
        assert nominal > with_tol
        assert nominal / with_tol == pytest.approx(1.0 / 0.875, rel=1e-6)

    def test_rejects_wall_too_thick(self):
        with pytest.raises(ValueError):
            burst_pressure_barlow(80_000, 4.0, 7.0)  # t >= OD/2


class TestCollapseAPI5C3:
    def test_seven_inch_n80_matches_api_rating(self):
        """7\" 26 lb/ft N-80 → API published collapse = 5410 psi (plastic)."""
        res = collapse_pressure_api5c3(7.0, 0.362, 80_000)
        assert res["regime"] == "plastic"
        assert res["collapse_pressure_psi"] == pytest.approx(5410.0, abs=5.0)
        assert res["d_over_t"] == pytest.approx(7.0 / 0.362, rel=1e-6)

    def test_heavy_wall_is_yield_regime(self):
        """Very thick wall (low D/t) → yield regime."""
        res = collapse_pressure_api5c3(od_in=7.0, wall_thickness_in=1.0, yield_strength_psi=80_000)
        assert res["regime"] == "yield"
        assert res["collapse_pressure_psi"] > 0

    def test_thin_wall_is_elastic_regime(self):
        """Very thin wall (high D/t) → elastic regime with low collapse."""
        res = collapse_pressure_api5c3(od_in=13.375, wall_thickness_in=0.250, yield_strength_psi=55_000)
        # K-55/J-55 13-3/8" 36 lb/ft is a classic elastic example
        assert res["regime"] in {"transition", "elastic"}
        assert 0 < res["collapse_pressure_psi"] < 2000

    def test_collapse_decreases_with_d_over_t(self):
        """For fixed grade, thinner walls collapse at lower pressure."""
        p_thick = collapse_pressure_api5c3(7.0, 0.500, 80_000)["collapse_pressure_psi"]
        p_mid = collapse_pressure_api5c3(7.0, 0.362, 80_000)["collapse_pressure_psi"]
        p_thin = collapse_pressure_api5c3(7.0, 0.250, 80_000)["collapse_pressure_psi"]
        assert p_thick > p_mid > p_thin

    def test_rejects_wall_too_thick(self):
        with pytest.raises(ValueError):
            collapse_pressure_api5c3(7.0, 4.0, 80_000)

    def test_n80_boundaries_match_published_api_table(self):
        """Pin the API 5C3 regime boundaries for N-80.
        Published values (API Bull 5C3 1994, Table 2):
            dt_yp = 13.38, dt_pt = 22.47, dt_te = 31.02."""
        from petropt.drilling.tubulars import _api5c3_coefficients

        c = _api5c3_coefficients(80_000)
        assert c["dt_yp"] == pytest.approx(13.38, abs=0.02)
        assert c["dt_pt"] == pytest.approx(22.47, abs=0.02)
        assert c["dt_te"] == pytest.approx(31.02, abs=0.02)
        # F and G must also match published values
        assert c["F"] == pytest.approx(1.998, abs=0.005)
        assert c["G"] == pytest.approx(0.0434, abs=0.0005)
        # Boundaries must be strictly increasing — transition regime
        # is unreachable if dt_pt >= dt_te.
        assert c["dt_yp"] < c["dt_pt"] < c["dt_te"]

    def test_transition_regime_is_reachable(self):
        """Pipe with D/t between dt_pt (22.47) and dt_te (31.02) for N-80
        must be classified as 'transition'. 7" OD × 0.230" wall → D/t = 30.4."""
        res = collapse_pressure_api5c3(7.0, 0.230, 80_000)
        assert res["regime"] == "transition"
        assert res["collapse_pressure_psi"] > 0
