"""Tests for petrophysics log interpretation correlations.

Reference values checked against:
    Archie (1942), Simandoux (1963), Poupon-Leveaux (1971).
    Timur (1968), Coates (1991).
    Schlumberger, "Log Interpretation Principles/Applications" (1989).
    Crain's Petrophysical Handbook (worked examples).
"""

import math

import pytest

from petropt.petrophysics import (
    archie_sw,
    density_porosity,
    effective_porosity,
    gamma_ray_index,
    hydrocarbon_pore_thickness,
    indonesian_sw,
    net_pay,
    neutron_density_porosity,
    coates_permeability,
    timur_permeability,
    simandoux_sw,
    sonic_porosity,
    vshale_from_gr,
)


# ---------------------------------------------------------------------------
# Vshale
# ---------------------------------------------------------------------------

class TestVshale:
    def test_linear_midpoint(self):
        """IGR = 0.5 with linear method → Vshale = 0.5."""
        assert vshale_from_gr(gr=70, gr_clean=20, gr_shale=120, method="linear") == pytest.approx(0.5)

    def test_larionov_tertiary_softer_than_linear(self):
        """Larionov Tertiary must under-read vs linear (IGR > 0)."""
        igr = 0.5
        v_linear = vshale_from_gr(70, 20, 120, "linear")
        v_lt = vshale_from_gr(70, 20, 120, "larionov_tertiary")
        assert v_lt < v_linear
        # Larionov Tertiary at IGR=0.5: 0.083*(2^1.85 - 1) ≈ 0.217
        assert v_lt == pytest.approx(0.083 * (2 ** (3.7 * igr) - 1), rel=1e-4)

    def test_larionov_older_between_tertiary_and_linear(self):
        """Older Larionov reads higher than Tertiary (older rocks less radioactive per Vsh)."""
        v_lt = vshale_from_gr(70, 20, 120, "larionov_tertiary")
        v_lo = vshale_from_gr(70, 20, 120, "larionov_older")
        v_lin = vshale_from_gr(70, 20, 120, "linear")
        assert v_lt < v_lo < v_lin

    def test_clavier_midrange(self):
        """Clavier at IGR=0.5: 1.7 - sqrt(3.38 - 1.44) ≈ 0.308."""
        v = vshale_from_gr(70, 20, 120, "clavier")
        expected = 1.7 - math.sqrt(3.38 - (0.5 + 0.7) ** 2)
        assert v == pytest.approx(expected, rel=1e-6)

    def test_clamps_below_clean(self):
        """GR below clean baseline → Vshale = 0."""
        assert vshale_from_gr(gr=10, gr_clean=20, gr_shale=120) == 0.0

    def test_clamps_above_shale(self):
        """GR above shale baseline → Vshale = 1."""
        assert vshale_from_gr(gr=200, gr_clean=20, gr_shale=120) == 1.0

    def test_equal_baselines_raises(self):
        with pytest.raises(ValueError):
            vshale_from_gr(gr=50, gr_clean=80, gr_shale=80)

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError):
            vshale_from_gr(70, 20, 120, method="bayes")

    def test_igr_helper(self):
        assert gamma_ray_index(70, 20, 120) == pytest.approx(0.5)
        assert gamma_ray_index(10, 20, 120) == 0.0
        assert gamma_ray_index(200, 20, 120) == 1.0


# ---------------------------------------------------------------------------
# Porosity
# ---------------------------------------------------------------------------

class TestPorosity:
    def test_density_porosity_sandstone(self):
        """rho_b = 2.35, sandstone matrix, fresh water:
        phi = (2.65 - 2.35)/(2.65 - 1.0) = 0.1818."""
        phi = density_porosity(rhob=2.35)
        assert phi == pytest.approx(0.3 / 1.65, rel=1e-6)

    def test_density_porosity_limestone(self):
        """rho_b = 2.50, limestone matrix 2.71:
        phi = (2.71 - 2.50)/(2.71 - 1.0) = 0.1228."""
        phi = density_porosity(rhob=2.50, rho_matrix=2.71)
        assert phi == pytest.approx(0.21 / 1.71, rel=1e-6)

    def test_density_porosity_zero_at_matrix(self):
        """rho_b equal to matrix → zero porosity."""
        assert density_porosity(rhob=2.65) == 0.0

    def test_density_porosity_clamps_high(self):
        """rho_b < fluid → physically impossible, clamps to 1."""
        assert density_porosity(rhob=0.5) == 1.0

    def test_sonic_wyllie(self):
        """dt=85, dt_ma=55.5, dt_f=189 (defaults):
        phi = (85-55.5)/(189-55.5) = 0.2210."""
        assert sonic_porosity(dt=85.0) == pytest.approx(29.5 / 133.5, rel=1e-6)

    def test_sonic_raymer(self):
        """dt=85, dt_ma=55.5: phi = 0.625 * 29.5/85 = 0.2169."""
        phi = sonic_porosity(dt=85.0, method="raymer")
        assert phi == pytest.approx(0.625 * 29.5 / 85.0, rel=1e-6)

    def test_sonic_unconsolidated_raymer_less_than_wyllie(self):
        """At high dt (unconsolidated sand) Raymer-Hunt-Gardner reads
        lower than Wyllie — the whole reason RHG exists (Wyllie
        over-predicts porosity in soft rock)."""
        w = sonic_porosity(110.0, method="wyllie")
        r = sonic_porosity(110.0, method="raymer")
        assert r < w

    def test_sonic_unknown_method(self):
        with pytest.raises(ValueError):
            sonic_porosity(85.0, method="gassmann")

    def test_neutron_density_rms(self):
        """NPHI=0.30, DPHI=0.10: phi_ND = sqrt((0.09 + 0.01)/2) = 0.2236 — gas effect averages crossplot."""
        phi = neutron_density_porosity(phi_neutron=0.30, phi_density=0.10)
        assert phi == pytest.approx(math.sqrt(0.05), rel=1e-6)

    def test_neutron_density_equal(self):
        """Equal neutron and density porosity → returns that value."""
        assert neutron_density_porosity(0.20, 0.20) == pytest.approx(0.20, rel=1e-6)

    def test_effective_porosity(self):
        """phi_t = 0.25, Vsh = 0.20 → phi_e = 0.20."""
        assert effective_porosity(phi_total=0.25, vshale=0.20) == pytest.approx(0.20)

    def test_effective_porosity_pure_shale(self):
        assert effective_porosity(0.20, 1.0) == 0.0


# ---------------------------------------------------------------------------
# Water saturation
# ---------------------------------------------------------------------------

class TestSaturation:
    def test_archie_hand_calculation(self):
        """Textbook example: Rw=0.05, phi=0.20, Rt=20, a=m=n=default.
        Sw = sqrt(1*0.05/(0.04*20)) = sqrt(0.0625) = 0.25."""
        assert archie_sw(rt=20.0, phi=0.20, rw=0.05) == pytest.approx(0.25, rel=1e-6)

    def test_archie_100_percent_water(self):
        """When Rt equals F*Rw (fully water-saturated), Sw = 1."""
        phi, rw, a, m = 0.20, 0.05, 1.0, 2.0
        f = a / phi**m  # formation factor
        rt = f * rw  # Ro — resistivity at 100% brine
        sw = archie_sw(rt=rt, phi=phi, rw=rw, a=a, m=m, n=2.0)
        assert sw == pytest.approx(1.0, rel=1e-6)

    def test_archie_high_resistivity_low_sw(self):
        """Very high Rt → low Sw (hydrocarbon-bearing)."""
        sw = archie_sw(rt=500.0, phi=0.20, rw=0.05)
        assert sw < 0.1

    def test_archie_invalid_phi(self):
        with pytest.raises(ValueError):
            archie_sw(rt=20.0, phi=1.5, rw=0.05)

    def test_simandoux_reduces_to_archie_when_no_shale(self):
        """Vsh=0 → Simandoux must return Archie Sw."""
        kwargs = dict(rt=20.0, phi=0.20, rw=0.05)
        sw_ar = archie_sw(**kwargs)
        sw_sm = simandoux_sw(vshale=0.0, rsh=2.0, **kwargs)
        assert sw_sm == pytest.approx(sw_ar, rel=1e-6)

    def test_simandoux_shaly_lower_sw_than_archie(self):
        """Simandoux attributes part of the formation conductance to the
        shale term, so less is credited to brine → the computed Sw for
        the same Rt is LOWER than pure Archie. This is the standard
        shaly-sand correction (Crain, Log Analyst)."""
        sw_ar = archie_sw(rt=20.0, phi=0.20, rw=0.05)
        sw_sm = simandoux_sw(rt=20.0, phi=0.20, rw=0.05, vshale=0.3, rsh=2.0)
        assert sw_sm < sw_ar

    def test_simandoux_arbitrary_n_matches_bisection(self):
        """n=2 closed-form must match n=2 bisection path within tolerance."""
        sw2 = simandoux_sw(rt=15.0, phi=0.15, rw=0.04, vshale=0.25, rsh=3.0, n=2.0)
        sw_bis = simandoux_sw(rt=15.0, phi=0.15, rw=0.04, vshale=0.25, rsh=3.0, n=2.0001)
        assert sw2 == pytest.approx(sw_bis, rel=1e-3)

    def test_indonesian_reduces_to_archie_when_no_shale(self):
        sw_ar = archie_sw(rt=20.0, phi=0.20, rw=0.05)
        sw_in = indonesian_sw(rt=20.0, phi=0.20, rw=0.05, vshale=0.0, rsh=2.0)
        assert sw_in == pytest.approx(sw_ar, rel=1e-6)

    def test_indonesian_both_shaly_corrections_below_archie(self):
        """Both shaly-sand equations should sit below pure Archie for
        the same Rt because they credit some conductance to shale."""
        kwargs = dict(rt=20.0, phi=0.18, rw=0.05, vshale=0.35, rsh=2.0)
        sw_ar = archie_sw(rt=kwargs["rt"], phi=kwargs["phi"], rw=kwargs["rw"])
        sw_sm = simandoux_sw(**kwargs)
        sw_in = indonesian_sw(**kwargs)
        assert sw_sm < sw_ar
        assert sw_in < sw_ar
        # Both should be in physical range
        assert 0.0 <= sw_sm <= 1.0
        assert 0.0 <= sw_in <= 1.0

    def test_simandoux_pure_shale_limit(self):
        """Vsh=1 (pure shale): Simandoux reduces to Sw = Rsh/Rt, not 1.0."""
        sw = simandoux_sw(rt=10.0, phi=0.20, rw=0.05, vshale=1.0, rsh=2.0)
        assert sw == pytest.approx(0.2, rel=1e-6)  # 2.0 / 10.0

    def test_simandoux_bisection_matches_closed_form(self):
        """Arbitrary-n bisection with bracketing should match n=2 closed
        form within iteration tolerance."""
        kwargs = dict(rt=15.0, phi=0.15, rw=0.04, vshale=0.25, rsh=3.0)
        sw_closed = simandoux_sw(n=2.0, **kwargs)
        sw_bis = simandoux_sw(n=2.000001, **kwargs)
        assert sw_bis == pytest.approx(sw_closed, abs=1e-6)

    def test_indonesian_bisection_matches_closed_form(self):
        kwargs = dict(rt=15.0, phi=0.15, rw=0.04, vshale=0.25, rsh=3.0)
        sw_closed = indonesian_sw(n=2.0, **kwargs)
        sw_bis = indonesian_sw(n=2.000001, **kwargs)
        assert sw_bis == pytest.approx(sw_closed, abs=1e-6)


# ---------------------------------------------------------------------------
# Permeability
# ---------------------------------------------------------------------------

class TestPermeability:
    def test_timur_hand_calc(self):
        """phi=20%, Swirr=20%: k = 0.136 * 20^4.4 / 20^2 ≈ 180.3 mD."""
        k = timur_permeability(phi=0.20, swirr=0.20)
        expected = 0.136 * 20.0**4.4 / 20.0**2
        assert k == pytest.approx(expected, rel=1e-6)

    def test_timur_monotonic_in_phi(self):
        """k increases with phi, holding Swirr fixed."""
        assert timur_permeability(0.10, 0.20) < timur_permeability(0.15, 0.20)
        assert timur_permeability(0.15, 0.20) < timur_permeability(0.25, 0.20)

    def test_timur_monotonic_in_swirr(self):
        """k decreases as Swirr rises (tighter rock holds more irreducible water)."""
        assert timur_permeability(0.20, 0.10) > timur_permeability(0.20, 0.30)

    def test_coates_hand_calc(self):
        """phi=20%, BVI=0.05, FFI=0.15, C=10:
        k = ((20/10)^2 * (0.15/0.05))^2 = (4*3)^2 = 144 mD."""
        k = coates_permeability(phi=0.20, bvi=0.05, ffi=0.15)
        assert k == pytest.approx(144.0, rel=1e-6)

    def test_coates_zero_ffi_zero_perm(self):
        assert coates_permeability(0.20, bvi=0.10, ffi=0.0) == 0.0

    def test_timur_invalid_phi(self):
        with pytest.raises(ValueError):
            timur_permeability(phi=1.5, swirr=0.2)

    def test_coates_invalid_bvi(self):
        with pytest.raises(ValueError):
            coates_permeability(phi=0.2, bvi=1.5, ffi=0.15)


# ---------------------------------------------------------------------------
# Pay & HPT
# ---------------------------------------------------------------------------

class TestNetPay:
    def test_simple_pay_flags(self):
        res = net_pay(
            depths=[9000, 9001, 9002, 9003, 9004, 9005],
            phi=[0.10, 0.12, 0.05, 0.15, 0.18, 0.02],
            sw=[0.30, 0.40, 0.60, 0.35, 0.25, 0.90],
            vshale=[0.20, 0.30, 0.60, 0.40, 0.20, 0.70],
        )
        assert res["pay_flags"] == [True, True, False, True, True, False]
        # 4 pay samples * 1 ft each = 4 ft net
        assert res["net_pay_ft"] == pytest.approx(4.0)
        # 5 steps of 1 ft + last sample = 6 ft gross
        assert res["gross_thickness_ft"] == pytest.approx(6.0)
        assert res["net_to_gross"] == pytest.approx(4.0 / 6.0)
        assert res["num_pay_samples"] == 4

    def test_averages_are_pay_weighted(self):
        res = net_pay(
            depths=[9000, 9001, 9002],
            phi=[0.20, 0.10, 0.04],
            sw=[0.3, 0.4, 0.9],
            vshale=[0.1, 0.2, 0.8],
        )
        # Samples 0 and 1 meet defaults (phi>=0.06, sw<=0.5, vsh<=0.5), sample 2 fails all
        assert res["avg_porosity_pay"] == pytest.approx((0.20 + 0.10) / 2)
        assert res["avg_sw_pay"] == pytest.approx((0.3 + 0.4) / 2)

    def test_no_pay_returns_none_averages(self):
        res = net_pay(
            depths=[9000, 9001, 9002],
            phi=[0.02, 0.03, 0.02],
            sw=[0.8, 0.9, 0.7],
            vshale=[0.7, 0.8, 0.9],
        )
        assert res["net_pay_ft"] == 0.0
        assert res["avg_porosity_pay"] is None

    def test_tight_cutoffs(self):
        """Tighter phi cutoff prunes marginal pay."""
        kwargs = dict(
            depths=[9000, 9001, 9002, 9003],
            phi=[0.08, 0.12, 0.07, 0.15],
            sw=[0.3, 0.3, 0.3, 0.3],
            vshale=[0.1, 0.1, 0.1, 0.1],
        )
        loose = net_pay(phi_cutoff=0.06, **kwargs)
        tight = net_pay(phi_cutoff=0.10, **kwargs)
        assert tight["num_pay_samples"] < loose["num_pay_samples"]

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            net_pay([9000, 9001], [0.1], [0.3, 0.3], [0.2, 0.2])

    def test_single_point_raises(self):
        with pytest.raises(ValueError):
            net_pay([9000], [0.1], [0.3], [0.2])

    def test_non_monotonic_depths_raises(self):
        with pytest.raises(ValueError, match="monotonic"):
            net_pay(
                depths=[9000, 9001, 9000.5, 9002],
                phi=[0.10, 0.12, 0.08, 0.15],
                sw=[0.3, 0.3, 0.3, 0.3],
                vshale=[0.2, 0.2, 0.2, 0.2],
            )

    def test_decreasing_depths_accepted(self):
        """Logs recorded bottom-up should be OK as long as strictly monotonic."""
        res = net_pay(
            depths=[9005, 9004, 9003, 9002, 9001, 9000],
            phi=[0.02, 0.18, 0.15, 0.05, 0.12, 0.10],
            sw=[0.9, 0.25, 0.35, 0.6, 0.4, 0.3],
            vshale=[0.7, 0.2, 0.4, 0.6, 0.3, 0.2],
        )
        assert res["num_pay_samples"] == 4

    def test_nan_input_raises(self):
        with pytest.raises(ValueError, match="finite"):
            net_pay(
                depths=[9000, 9001, 9002],
                phi=[0.10, float("nan"), 0.15],
                sw=[0.3, 0.3, 0.3],
                vshale=[0.2, 0.2, 0.2],
            )

    def test_out_of_range_phi_raises(self):
        with pytest.raises(ValueError, match="phi"):
            net_pay(
                depths=[9000, 9001, 9002],
                phi=[0.10, 1.2, 0.15],
                sw=[0.3, 0.3, 0.3],
                vshale=[0.2, 0.2, 0.2],
            )

    def test_invalid_cutoff_raises(self):
        with pytest.raises(ValueError):
            net_pay(
                depths=[9000, 9001, 9002],
                phi=[0.10, 0.12, 0.15],
                sw=[0.3, 0.3, 0.3],
                vshale=[0.2, 0.2, 0.2],
                phi_cutoff=1.5,
            )


class TestValidationExtras:
    def test_density_porosity_rejects_equal_matrix_fluid(self):
        with pytest.raises(ValueError, match="rho_fluid"):
            density_porosity(rhob=2.35, rho_matrix=2.0, rho_fluid=2.0)

    def test_density_porosity_rejects_inverted_matrix_fluid(self):
        with pytest.raises(ValueError, match="rho_matrix"):
            density_porosity(rhob=2.35, rho_matrix=1.0, rho_fluid=2.5)

    def test_sonic_wyllie_rejects_inverted_transit_times(self):
        with pytest.raises(ValueError, match="dt_fluid"):
            sonic_porosity(dt=85, dt_matrix=200, dt_fluid=50, method="wyllie")

    def test_vshale_rejects_nan(self):
        with pytest.raises(ValueError, match="finite"):
            vshale_from_gr(gr=float("nan"), gr_clean=20, gr_shale=120)

    def test_vshale_rejects_inf(self):
        with pytest.raises(ValueError, match="finite"):
            vshale_from_gr(gr=float("inf"), gr_clean=20, gr_shale=120)


class TestHPT:
    def test_basic(self):
        """50 ft * 0.2 phi * (1 - 0.3 Sw) * 0.8 NTG = 5.6 ft HPT."""
        assert hydrocarbon_pore_thickness(50, 0.2, 0.3, 0.8) == pytest.approx(5.6)

    def test_full_water(self):
        """Sw = 1 → HPT = 0."""
        assert hydrocarbon_pore_thickness(50, 0.2, 1.0) == 0.0

    def test_zero_ntg(self):
        assert hydrocarbon_pore_thickness(50, 0.2, 0.3, 0.0) == 0.0

    def test_invalid_thickness(self):
        with pytest.raises(ValueError):
            hydrocarbon_pore_thickness(-1, 0.2, 0.3)
