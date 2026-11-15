"""Randfälle, an denen Kernroutinen erfahrungsgemäß brechen (DEMO-PLAYBOOK §12): L = 1, β = 1 exakt, β = ½ ± 10⁻⁹, α am Rand nur als Band, L bis 40, β > 1 und β < 0,
Soll-Leitung mit nichtpositiver Lieferzeit, d = 0."""

import math

import numpy as np
import pytest

import lmr_dynamics as D
import lmr_evaluation as ev


def test_L1_has_an_empty_pipeline_and_no_beta_dependence():
    assert D.char_poly(1, 0.4, 0.9) == pytest.approx(D.char_poly(1, 0.4, -2.0))
    assert D.spectral_radius(1, 0.4, 0.9) == pytest.approx(0.6)
    i, u = D.impulse_response(1, 0.4, 0.9, 5)
    j, _w = D.impulse_response(1, 0.4, 0.0, 5)
    assert list(i) == pytest.approx(list(j)) and u[0] == 0.0
    # L = 1: die Leitungssumme ist leer, nur ΔW zählt mit dem Gewicht β in die Bestellung ein
    i2, _u2 = D.simulate(1, 0.4, 0.9, np.zeros(200), D.delta_w(10.0, 1, 10.0, 1))
    assert i2[-1] == pytest.approx(0.0, abs=1e-9)                                           # L̂ = L = 1: ΔW = 0, kein Dauerfehler


def test_beta_exactly_one_has_a_multiple_root_at_zero_and_the_matrix_agrees_within_a_band():
    for L in range(2, 13):
        for a in (0.37, 0.8, 1.3):
            assert D.spectral_radius(L, a, 1.0) == pytest.approx(abs(1.0 - a), abs=1e-12)
            assert D.spectral_radius_matrix(L, a, 1.0) == pytest.approx(abs(1.0 - a), abs=1e-6)     # mehrfache Wurzel 0: Eigenwerte nur auf 10⁻⁶


def test_beta_near_one_is_continuous_and_small_roots_follow_the_first_order_formula():
    """β = 1 + ε: die kleinen Wurzeln haben Betrag (α |ε| / |1 − α|)^(1/(L−1)) (Näherung erster Ordnung, gilt für α ungleich 1)."""
    a, eps, L = 0.5001, 1e-3, 6
    r = np.sort(np.abs(D.roots_of_P(L, a, 1.0 + eps)))
    approx = (a * eps / abs(1 - a)) ** (1.0 / (L - 1))
    assert r[0] == pytest.approx(approx, rel=0.3)
    assert D.spectral_radius(L, a, 1.0 + 1e-12) == pytest.approx(abs(1 - a), abs=1e-3)       # stetig in β: die kleinen Wurzeln wachsen wie ε^(1/(L−1)), die große bleibt


def test_alpha_two_and_alpha_krit_are_checked_only_as_bands():
    for L in (2, 3, 5, 8):
        assert D.spectral_radius(L, 2.0 * (1 - 1e-6), 0.5) < 1.0 < D.spectral_radius(L, 2.0 * (1 + 1e-6), 0.5)
        c = D.alpha_krit(L, 0.3)
        assert D.spectral_radius(L, c * (1 - 1e-6), 0.3) <= 1.0 + 1e-9 < D.spectral_radius(L, c * (1 + 1e-6), 0.3)
    assert ev.state_of(D.spectral_radius(4, 2.0, 0.5)) == "grenze"                           # α = 2 genau: ρ = 1 innerhalb von 10⁻⁹, nicht entscheidbar


def test_long_delay_up_to_40_with_np_roots_matches_the_matrix_and_the_closed_form_at_beta_zero():
    for L in (30, 40):
        c0 = D.alpha_krit(L, 0.0)
        assert c0 == pytest.approx(2.0 * math.sin(math.pi / (4 * L - 2)), abs=1e-9)
        assert D.spectral_radius(L, 0.8 * c0, 0.0) == pytest.approx(D.spectral_radius_matrix(L, 0.8 * c0, 0.0), abs=1e-8)


def test_beta_above_one_and_below_zero_are_handled():
    assert D.alpha_krit(4, 3.0) == pytest.approx(0.4, abs=1e-6)                              # 2/(2β − 1)
    assert D.alpha_krit(4, -2.0) < D.alpha_krit(4, -0.5) < D.alpha_krit(4, 0.0)
    assert D.spectral_radius(4, 0.2, 3.0) < 1.0 and D.spectral_radius(4, 0.2, -3.0) > 1.0


def test_soll_pipeline_with_nonpositive_estimated_delay_still_follows_the_formula():
    """L̂ ≤ 0: W* ist negativ, die Formel gilt (physikalisch sinnlos, die App weist darauf hin)."""
    dW = D.delta_w(10.0, 3, 10.0, 0)
    assert dW == 30.0
    i, _u = D.simulate(3, 0.4, 0.8, np.zeros(4000), dW)
    assert i[-1] == pytest.approx(-0.8 * 30.0, abs=1e-6)
    assert D.delta_w(10.0, 3, 10.0, -2) == pytest.approx(10 * 2 + 10 * 3)


def test_zero_mean_demand_gives_no_drift_and_no_noise():
    assert D.delta_w(0.0, 5, 0.0, 3) == 0.0 and D.drift_equilibrium(1.0, 0.0) == 0.0
    r = ev.analyse_drift(3, 0.4, 0.8, 0, 10.0, 0.0, 16)
    assert r.i_star == 0.0 and not np.any(r.i_noisy) and not np.any(r.i_clean) and r.t_eq is None


def test_settle_time_edge_cases():
    assert ev.settle_time(np.array([0.0, -5.0, -9.95, -10.0, -10.0]), -10.0) == 2
    assert ev.settle_time(np.array([-10.0, -10.0]), -10.0) == 0
    assert ev.settle_time(np.array([0.0, 0.0, 0.0]), -10.0) is None                           # kommt nie an
    assert ev.settle_time(np.zeros(4), 0.0) is None                                            # i* = 0: keine Zeit bis auf 1 % von 0


def test_analysis_at_the_slider_extremes_is_finite():
    for L in (1, 24):
        for a in (0.05, 4.0):
            for b in (-1.0, 2.0):
                an = ev.analyse(ev.Settings(L, a, b))
                assert np.all(np.isfinite(an.i)) and np.all(np.isfinite(an.u)) and len(an.t) == len(an.i)
                assert math.isfinite(an.rho) and (an.krit > 0)


def test_plot_horizon_bounds():
    assert ev.plot_horizon(0.5, 0.5, 10, 12, 3) == 30
    assert ev.plot_horizon(0.99, 0.99, 3000, 3000, 3) == 400
    assert 30 <= ev.plot_horizon(1.3, 1.5, None, None, 3) <= 150
    assert ev.plot_horizon(1.0, 1.0, None, None, 3) == 120                                      # nichts zum Abklingen: 40 L
