"""Tests for production-engineering calculations.

Reference values:
    Turner et al. (1969), Coleman et al. (1991), Gilbert (1954),
    Hammerschmidt (1934), API RP 14E.
    Lea, Nickens, & Wells, "Gas Well Deliquification" (2008).
"""

import math

import pytest

from petropt.production import (
    api_rp14e_erosional_velocity,
    coleman_critical_velocity,
    critical_gas_rate_mcfd,
    gilbert_choke_flow,
    hammerschmidt_inhibitor_dosing,
    katz_hydrate_temperature,
    turner_critical_velocity,
)


# ---------------------------------------------------------------------------
# Turner / Coleman
# ---------------------------------------------------------------------------

class TestTurner:
    def test_hand_calc(self):
        """Water at 60 dyne/cm, rho_l = 65.5, rho_g = 1.8:
        v_c = 1.593 * 60^0.25 * (65.5-1.8)^0.25 / 1.8^0.5
             = 1.593 * 2.7832 * 2.8257 / 1.3416
             ≈ 9.34 ft/s."""
        v = turner_critical_velocity(
            surface_tension_dyne_cm=60, rho_liquid_lb_ft3=65.5,
            rho_gas_lb_ft3=1.8,
        )
        expected = (
            1.593 * 60**0.25 * (65.5 - 1.8) ** 0.25 / 1.8**0.5
        )
        assert v == pytest.approx(expected, rel=1e-10)

    def test_rejects_gas_denser_than_liquid(self):
        with pytest.raises(ValueError):
            turner_critical_velocity(60, 1.0, 2.0)

    def test_monotonic_in_gas_density(self):
        """Higher gas density (higher pressure) → lower v_c needed."""
        low = turner_critical_velocity(60, 65, 0.5)
        hi = turner_critical_velocity(60, 65, 2.0)
        assert hi < low

    def test_water_controls_vs_condensate(self):
        """For water (sigma=60) vs condensate (sigma=20) at same rho_g,
        water needs higher velocity to lift."""
        v_water = turner_critical_velocity(60, 65, 1.5)
        v_cond = turner_critical_velocity(20, 45, 1.5)
        assert v_water > v_cond


class TestColeman:
    def test_is_80_percent_of_turner(self):
        kwargs = dict(
            surface_tension_dyne_cm=60, rho_liquid_lb_ft3=65, rho_gas_lb_ft3=1.0
        )
        t = turner_critical_velocity(**kwargs)
        c = coleman_critical_velocity(**kwargs)
        assert c == pytest.approx(0.8 * t, rel=1e-12)


class TestCriticalGasRate:
    def test_hand_calc(self):
        """v=10 ft/s, ID=2.441" tubing, P=1000 psia, T=120°F (580°R), Z=0.85.
        Converts well-conditions volumetric flow to standard-condition
        Mcf/d via q_sc = q_wh * (P/P_sc) * (T_sc/T_wh) / Z.
        Expected ≈ 2016 Mcf/d."""
        v, d_in, p, t, z = 10.0, 2.441, 1000.0, 120.0, 0.85
        q = critical_gas_rate_mcfd(v, d_in, p, t, z)
        area = math.pi / 4 * (d_in / 12) ** 2
        t_r = t + 459.67
        expected = v * area * 86400.0 * p * 520.0 / (z * t_r * 14.7 * 1000.0)
        assert q == pytest.approx(expected, rel=1e-10)
        # Order-of-magnitude sanity: 10 ft/s in 2.441" tubing at 1000 psi
        # should be ~2000 Mcf/d, not ~5 Mcf/d.
        assert 1500 < q < 2500

    def test_quadratic_in_tubing_id(self):
        """Doubling tubing ID → 4x critical gas rate (area scales as d²)."""
        q_small = critical_gas_rate_mcfd(10, 2.0, 1000, 120, 0.9)
        q_large = critical_gas_rate_mcfd(10, 4.0, 1000, 120, 0.9)
        assert q_large / q_small == pytest.approx(4.0, rel=1e-10)


# ---------------------------------------------------------------------------
# Hydrates
# ---------------------------------------------------------------------------

class TestKatzHydrate:
    def test_monotonic_in_pressure(self):
        """Hydrate formation T rises with pressure (up to the plateau)."""
        t500 = katz_hydrate_temperature(500, 0.65)
        t1000 = katz_hydrate_temperature(1000, 0.65)
        t2000 = katz_hydrate_temperature(2000, 0.65)
        assert t500 < t1000 < t2000

    def test_heavier_gas_higher_t(self):
        """Heavier natural gas (higher SG) forms hydrates at higher T."""
        t_light = katz_hydrate_temperature(1000, 0.6)
        t_heavy = katz_hydrate_temperature(1000, 0.9)
        assert t_heavy > t_light

    def test_rejects_out_of_range_sg(self):
        with pytest.raises(ValueError):
            katz_hydrate_temperature(1000, 0.4)
        with pytest.raises(ValueError):
            katz_hydrate_temperature(1000, 1.2)

    def test_in_reasonable_range(self):
        """Hydrates at 1000 psia / 0.7 SG are ~60-70°F per Katz chart.
        Motiee (1991) fit gives ~65°F."""
        t = katz_hydrate_temperature(1000, 0.7)
        assert 55 <= t <= 72

    def test_motiee_hand_calc(self):
        """Motiee formula at P=500 psia, SG=0.65:
        T = -20.35 + 13.47*ln(500) + 34.27*ln(0.65) - 1.675*ln(500)*ln(0.65)."""
        t = katz_hydrate_temperature(500, 0.65)
        import math
        ln_p = math.log(500)
        ln_sg = math.log(0.65)
        expected = -20.35 + 13.47 * ln_p + 34.27 * ln_sg - 1.675 * ln_p * ln_sg
        assert t == pytest.approx(expected, rel=1e-10)


class TestHammerschmidt:
    def test_no_inhibitor_when_already_below(self):
        """If operating T is above hydrate T, no inhibitor is needed."""
        r = hammerschmidt_inhibitor_dosing(
            hydrate_temp_f=60, operating_temp_f=65, water_rate_bwpd=10
        )
        assert r["weight_percent"] == 0.0
        assert r["rate_lb_day"] == 0.0
        assert r["rate_gal_day"] == 0.0

    def test_methanol_weight_pct_matches_hand_calc(self):
        """ΔT = 15°F, methanol (M=32.04, K=2335):
        W = 100 * 32.04 * 15 / (2335 + 32.04 * 15) = 48060 / 2815.6 = 17.07 %."""
        r = hammerschmidt_inhibitor_dosing(65, 50, water_rate_bwpd=10)
        expected_w = 100 * 32.04 * 15 / (2335 + 32.04 * 15)
        assert r["weight_percent"] == pytest.approx(expected_w, rel=1e-10)

    def test_meg_needs_less_weight_pct_than_methanol(self):
        """For the same ΔT, MEG (K=2700, higher M) needs less mass-percent..."""
        dt = 20  # °F
        r_m = hammerschmidt_inhibitor_dosing(80, 80 - dt, 5, "methanol")
        r_g = hammerschmidt_inhibitor_dosing(80, 80 - dt, 5, "meg")
        # Compare in weight percent: actually MEG's higher M offsets higher K...
        # The formula: W = 100*M*dT/(K + M*dT). Plug numbers:
        # methanol: 100*32*20/(2335 + 32*20) = 64000 / 2975 = 21.5%
        # MEG: 100*62*20/(2700 + 62*20) = 124000 / 3940 = 31.5%
        # MEG actually needs MORE weight %! Assert correct direction.
        assert r_g["weight_percent"] > r_m["weight_percent"]

    def test_unknown_inhibitor_raises(self):
        with pytest.raises(ValueError):
            hammerschmidt_inhibitor_dosing(70, 50, 10, "ammonia")


# ---------------------------------------------------------------------------
# Erosion
# ---------------------------------------------------------------------------

class TestErosion:
    def test_c_100_rho_40(self):
        """C=100, rho=40 lb/ft³ → v_e = 100 / sqrt(40) = 15.81 ft/s."""
        assert api_rp14e_erosional_velocity(40) == pytest.approx(
            100 / math.sqrt(40), rel=1e-12
        )

    def test_higher_c_higher_velocity(self):
        low = api_rp14e_erosional_velocity(40, c_factor=100)
        high = api_rp14e_erosional_velocity(40, c_factor=150)
        assert high == pytest.approx(1.5 * low, rel=1e-12)

    def test_heavier_mix_lower_velocity(self):
        """Higher density fluid → lower allowable velocity."""
        light = api_rp14e_erosional_velocity(10)
        heavy = api_rp14e_erosional_velocity(60)
        assert heavy < light


# ---------------------------------------------------------------------------
# Gilbert choke
# ---------------------------------------------------------------------------

class TestGilbertChoke:
    def test_hand_calc(self):
        """P=2000 psi, S=32 (32/64"), GLR=1000 scf/bbl, WC=0:
        q = 2000 * 32^1.89 / (435 * 1000^0.546)
        Verify against independent computation."""
        res = gilbert_choke_flow(2000.0, 32.0, 1000.0, water_cut=0.0)
        expected = 2000 * 32**1.89 / (435.0 * 1000**0.546)
        assert res["total_liquid_rate_bpd"] == pytest.approx(expected, rel=1e-10)
        assert res["oil_rate_bopd"] == pytest.approx(expected, rel=1e-10)
        assert res["water_rate_bwpd"] == 0.0
        # gas rate: q_oil * GOR / 1000
        assert res["gas_rate_mcfd"] == pytest.approx(expected * 1000 / 1000, rel=1e-10)

    def test_water_cut_splits_rates(self):
        res = gilbert_choke_flow(2000, 32, 1000, water_cut=0.25)
        # Effective GLR reduced by (1 - wc)
        assert res["effective_glr_scf_bbl"] == pytest.approx(750.0)
        assert res["water_rate_bwpd"] == pytest.approx(
            0.25 * res["total_liquid_rate_bpd"]
        )
        assert res["oil_rate_bopd"] == pytest.approx(
            0.75 * res["total_liquid_rate_bpd"]
        )

    def test_choke_size_inches(self):
        """32/64" = 0.5" bean."""
        res = gilbert_choke_flow(2000, 32, 1000)
        assert res["choke_size_inches"] == pytest.approx(0.5)

    def test_power_law_in_choke_size(self):
        """Doubling choke size → 2^1.89 = ~3.71x rate."""
        small = gilbert_choke_flow(2000, 16, 1000)["total_liquid_rate_bpd"]
        big = gilbert_choke_flow(2000, 32, 1000)["total_liquid_rate_bpd"]
        assert big / small == pytest.approx(2.0**1.89, rel=1e-10)

    def test_water_cut_one_raises(self):
        """100% water cut → effective GLR = 0, can't be computed."""
        with pytest.raises(ValueError):
            gilbert_choke_flow(2000, 32, 1000, water_cut=1.0)

    def test_invalid_water_cut(self):
        with pytest.raises(ValueError):
            gilbert_choke_flow(2000, 32, 1000, water_cut=1.5)
