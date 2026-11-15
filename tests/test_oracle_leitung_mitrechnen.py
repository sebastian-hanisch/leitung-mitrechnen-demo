"""Orakel-Tests leitung-mitrechnen-demo: jede Kernroutine gegen einen anderen Rechenweg (ORACLE-PLAYBOOK §2, §5, §6; DEMO-PLAYBOOK §12).

Wege gegeneinander (Toleranzen an den gemessenen Abweichungen der Vormessung, mit Reserve):
 1. Koeffizienten von P gegen das charakteristische Polynom der Zustandsmatrix (np.poly), P(1) und P(−1) gegen die Formel.
 2. Spektralradius aus den Wurzeln von P gegen die Eigenwerte der Zustandsmatrix.
 3. Stoßantwort: direkte Rekursion (Kern) gegen Verzögerungsliste, Filter und Brüche.
 4. Grenze α_krit: Wurzeln auf dem Einheitskreis (Kern) gegen Gitter und Bisektion; L = 2 gegen die Jury-Bedingungen.
 5. Gleichgewicht bei falscher Soll-Leitung: Gauß mit Brüchen gegen Simulation, Filter und die Formel −β·ΔW.
 6. Streuung: Lyapunov-Gleichung gegen Simulation (Band) und gegen die Formel bei β = 1.
 7. Doppelwurzel ohne Leitung (Brüche); SplitMix64 gegen Referenzwerte.
 8. Gegenproben für Absolutaussagen (ORACLE-PLAYBOOK §6): ungerades L, β < 0, β = ½ mit zufälligen Werten ohne Gitter.
Zufall: fester Zufalls-Seed; plattformabhängige Zahlen (lange Läufe, Gleitkomma-Summen) nur als Band."""

import math
from fractions import Fraction as F

import numpy as np
import pytest

import lmr_dynamics as D
import lmr_exact as X
import lmr_rng as R


def _pair_cases(n_L=range(2, 9), n_beta=20, n_alpha=12):
    for L in n_L:
        for beta in np.linspace(-1.5, 2.5, n_beta):
            if abs(beta - 1.0) < 1e-9:
                continue
            for alpha in np.linspace(0.0537, 2.9731, n_alpha):
                yield L, float(alpha), float(beta)


# --- 1. und 2. Polynom und Spektralradius -----------------------------------------------------------------------------------------

def test_polynomial_coefficients_equal_the_characteristic_polynomial_of_the_matrix():
    worst, n = 0.0, 0
    for L in range(2, 13):
        for alpha in (0.21, 0.47, 0.93, 1.3, 1.97):
            for beta in (0.0, 0.3, 0.55, 0.8, 1.0):
                worst = max(worst, float(np.max(np.abs(np.poly(D.state_matrix(L, alpha, beta)) - D.char_poly(L, alpha, beta)))))
                n += 1
    assert n == 275 and worst < 1e-10


def test_P_at_plus_and_minus_one_against_the_formulas_2000_random_cases():
    rng = np.random.default_rng(11)
    worst1 = worstm = 0.0
    for _ in range(2000):
        L = int(rng.integers(1, 41))
        a, b = float(rng.uniform(0.01, 4.0)), float(rng.uniform(-2.0, 3.0))
        p = D.char_poly(L, a, b)
        worst1 = max(worst1, abs(np.polyval(p, 1.0) - a))
        worstm = max(worstm, abs(np.polyval(p, -1.0) - D.p_at_minus_one(L, a, b)))
    assert worst1 < 1e-12 and worstm < 1e-12


def test_spectral_radius_from_polynomial_roots_equals_matrix_eigenvalues():
    worst, n = 0.0, 0
    for L, a, b in _pair_cases():
        worst = max(worst, abs(D.spectral_radius(L, a, b) - D.spectral_radius_matrix(L, a, b)))
        n += 1
    assert n == 1680 and worst < 1e-9


# --- 3. Stoßantwort ---------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("L,alpha,beta", [(1, 0.7, 0.3), (2, 1.2, 0.55), (3, 0.6, 0.3), (5, 0.9, 1.0), (7, 0.3, 0.0), (4, 2.2, 0.8)])
def test_impulse_response_four_ways_agree(L, alpha, beta):
    T = 40
    i_sim, u_sim = D.impulse_response(L, alpha, beta, T)
    i_fil = D.impulse_stock_fast(L, alpha, beta, T)
    ii, uu = X.impulse_exact(L, alpha, beta, T)
    assert list(i_fil) == pytest.approx(list(i_sim), rel=1e-9, abs=1e-9)
    assert list(i_sim[1:]) == pytest.approx([float(v) for v in ii], rel=1e-9, abs=1e-9)
    assert list(u_sim) == pytest.approx([float(v) for v in uu], rel=1e-9, abs=1e-9)
    # die Bilanz u_s = i_{s+L} − i_{s+L−1} (+ e_0 bei s + L − 1 = 0) verbindet Bestand und Bestellung (Zähleinheit: Perioden)
    for s in range(1, T - L):
        assert u_sim[s] == pytest.approx(i_sim[s + L] - i_sim[s + L - 1], abs=1e-9)


def test_decay_time_by_recursion_equals_decay_time_by_filter():
    for L, a, b in ((2, 1.2, 0.55), (3, 0.6, 0.3), (6, 0.4, 0.8), (12, 0.12, 0.0), (5, 0.85, 1.0), (4, 0.4, 1.0), (3, 0.55, 0.9)):
        rho = D.spectral_radius(L, a, b)
        T = D.decay_horizon(rho, L)
        i, _u = D.impulse_response(L, a, b, T)
        assert D.decay_time_of(i) == D.decay_time(L, a, b), (L, a, b)


# --- 4. Grenze ---------------------------------------------------------------------------------------------------------------------

def test_alpha_krit_by_circle_crossings_equals_grid_and_bisection():
    rng = np.random.default_rng(3)
    n_checked = 0
    for _ in range(40):
        L = int(rng.integers(2, 14))
        b = float(rng.uniform(-1.0, 2.0))
        scan = D.alpha_krit_scan(L, b, hi=8.0, n=4000)
        if math.isinf(scan):
            continue
        assert D.alpha_krit(L, b) == pytest.approx(scan, abs=1e-6), (L, b)
        n_checked += 1
    assert n_checked >= 35


def test_alpha_krit_L2_equals_the_jury_limit_for_14_betas():
    betas = (-2.0, -1.0, -0.5, 0.0, 0.25, 0.51, 0.6, 0.75, 0.9, 1.0, 1.25, 1.5, 2.0, 3.0)
    assert len(betas) == 14
    for b in betas:
        assert D.alpha_krit(2, b) == pytest.approx(D.jury_alpha_krit_L2(b), abs=1e-8), b


def test_alpha_krit_zero_beta_equals_the_closed_form_2_sin():
    for L in (1, 2, 3, 4, 6, 8, 12):
        assert D.alpha_krit(L, 0.0) == pytest.approx(2.0 * math.sin(math.pi / (4 * L - 2)), abs=1e-9)


def test_alpha_krit_beta_zero_by_the_matrix_eigenvalues_instead_of_the_polynomial():
    """Dritter Weg: Eigenwerte der Zustandsmatrix knapp unter und knapp über der Grenze."""
    for L, b in ((3, 0.3), (6, -0.5), (5, 1.25), (12, 0.8)):
        c = D.alpha_krit(L, b)
        assert D.spectral_radius_matrix(L, c * (1 - 1e-6), b) <= 1.0 + 1e-9
        assert D.spectral_radius_matrix(L, c * (1 + 1e-6), b) > 1.0


# --- 5. Gleichgewicht bei falscher Soll-Leitung ------------------------------------------------------------------------------------

def test_equilibrium_in_fractions_equals_minus_beta_dW_in_720_cases():
    n = 0
    for L in range(1, 7):
        for alpha in ("3/10", "3/5", "9/10", "7/5"):
            for beta in ("-1/3", "0", "1/3", "3/5", "1", "3/2"):
                for Wstar in (10 * (L - 2), 10 * L, 10 * (L + 1), 7 * (L - 1), 0):
                    dW = F(10 * (L - 1) - Wstar)
                    i_star, stored = X.steady_state_exact(L, alpha, beta, dW)
                    assert i_star == -F(beta) * dW and all(v == 0 for v in stored), (L, alpha, beta, Wstar)
                    n += 1
    assert n == 720


DRIFT_CASES = [(4, 0.4, 0.55, 3), (4, 0.4, 1.0, 3), (4, 0.4, 1.0, 5), (4, 0.4, 0.0, 3), (6, 0.3, 1.0, 5),
               (3, 0.7, 0.6, 2), (2, 0.9, 1.5, 1), (4, 0.55, 1.3, 3), (5, 0.35, 0.8, 2)]


def test_simulation_reaches_the_exact_equilibrium_in_9_of_9_stable_cases():
    for L, a, b, Lh in DRIFT_CASES:
        assert D.spectral_radius(L, a, b) < 1.0
        dW = D.delta_w(10.0, L, 10.0, Lh)
        i, _u = D.simulate(L, a, b, np.zeros(3000), dW)
        assert i[-1] == pytest.approx(-b * dW, abs=1e-9), (L, a, b, Lh)
        step = D.drift_step_response(L, a, b, dW, 3000)
        assert list(step) == pytest.approx(list(i), abs=1e-8)                      # Filter und Rekursion stimmen auch im Übergang


def test_noisy_simulation_mean_stays_within_three_standard_errors_of_the_exact_value():
    """4 Fälle, 400 000 Perioden, Bedarfsrauschen s = 0,2 d, Zufalls-Seed 16; Schwingung aus dem Filter, Dauerfehler aus dem Schrittfilter (Superposition)."""
    T, burn = 400_000, 1000
    e = D.noise_series(T, 0.2, 10.0, 16)
    for L, a, b, Lh in DRIFT_CASES[:4]:
        dW = D.delta_w(10.0, L, 10.0, Lh)
        i = D.filter_response(L, a, b, e) + D.drift_step_response(L, a, b, dW, T)
        mean, se = D.batch_mean_and_se(i[burn:])
        assert abs(mean + b * dW) <= 3.0 * se, (L, a, b, Lh, mean, se)


def test_noisy_filter_superposition_equals_the_direct_simulation():
    e = D.noise_series(2000, 0.2, 10.0, 16)
    for L, a, b, Lh in DRIFT_CASES[:3]:
        dW = D.delta_w(10.0, L, 10.0, Lh)
        i_direct, _u = D.simulate(L, a, b, e, dW)
        i_super = D.filter_response(L, a, b, e) + D.drift_step_response(L, a, b, dW, 2000)
        assert list(i_super) == pytest.approx(list(i_direct), abs=1e-8)


# --- 6. Streuung -------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("beta", [0.0, 0.55, 1.0])
def test_lyapunov_variance_equals_simulation_within_4_percent(beta):
    L, a = 3, 0.45
    vu, vi = D.noise_variances(L, a, beta)
    su, si = D.simulated_variances(L, a, beta, 400_000, 5)
    assert su == pytest.approx(vu, rel=0.04) and si == pytest.approx(vi, rel=0.04)


def test_variance_at_beta_one_equals_the_closed_form_for_L_1_3_8():
    worst = 0.0
    for L in (1, 3, 8):
        for alpha in np.linspace(0.113, 1.887, 19):
            vu, vi = D.noise_variances(L, float(alpha), 1.0)
            cu, ci = D.noise_variances_beta1(L, float(alpha))
            worst = max(worst, abs(vu - cu), abs(vi - ci) / max(1.0, ci))
    assert worst < 1e-8


# --- 7. Doppelwurzel, SplitMix64 -----------------------------------------------------------------------------------------------------

def test_double_root_without_pipeline_exact_for_L_2_to_30():
    for L in range(2, 31):
        assert X.doubled_root_check(L) == (True, True), L
    for L in (3, 6, 12):
        assert D.spectral_radius(L, D.best_alpha_without_pipeline(L), 0.0) == pytest.approx(1 - 1 / L, abs=1e-6)


def test_smallest_power_of_the_matrix_is_2L_at_beta_half_alpha_2_for_L_2_to_9():
    for L in range(2, 10):
        A, _c = X.matrix_exact(L, "2", "1/2")
        assert X.smallest_power_identity(A) == 2 * L


def test_splitmix64_reference_values_and_array_equals_class():
    g = R.SplitMix64(0)
    assert [g.next_u64() for _ in range(3)] == [0xE220A8397B1DCDAF, 0x6E789E6AA1B965F4, 0x06C45D188009454F]
    seed = 123456789
    cls = R.SplitMix64(seed)
    assert [int(v) for v in R.splitmix_array(seed, 50)] == [cls.next_u64() for _ in range(50)]
    cls = R.SplitMix64(seed)
    assert list(R.normal_array(seed, 30)) == pytest.approx([cls.normal() for _ in range(30)], abs=1e-12)


# --- 8. Gegenproben für Absolutaussagen ----------------------------------------------------------------------------------------------

def test_odd_L_has_the_limit_2_for_beta_in_half_to_one_2000_random_pairs_without_grid():
    rng = np.random.default_rng(21)
    for _ in range(2000):
        L = int(rng.integers(1, 20)) * 2 + 1                       # ungerades L von 3 bis 39
        b = float(rng.uniform(0.5, 1.0))
        a_below = rng.uniform(0.001, 2.0 * (1 - 1e-6), size=12)    # keine Gitterpunkte
        assert D.rho_batch(L, a_below, b).max() <= 1.0 + 1e-9, (L, b)
        assert D.rho_batch(L, [2.0 * (1 + 1e-6)], b)[0] > 1.0, (L, b)


def test_beta_negative_has_no_stable_piece_above_the_limit_5000_random_alphas():
    rng = np.random.default_rng(22)
    for _ in range(500):
        L = int(rng.integers(2, 14))
        b = float(rng.uniform(-2.0, -0.01))
        c = D.alpha_krit(L, b)
        a_above = rng.uniform(c * (1 + 1e-6), 8.0, size=10)
        assert D.rho_batch(L, a_above, b).min() > 1.0, (L, b, c)


def test_beta_half_is_stable_for_every_alpha_below_2_3000_random_pairs_up_to_L_60():
    rng = np.random.default_rng(23)
    for _ in range(3000):
        L = int(rng.integers(1, 61))
        a = float(rng.uniform(0.001, 2.0 * (1 - 1e-6)))
        assert D.spectral_radius(L, a, 0.5) < 1.0, (L, a)


def test_beta_half_at_alpha_2_all_roots_lie_on_the_unit_circle():
    for L in (2, 3, 4, 5, 7, 8, 12, 25):
        r = np.abs(D.roots_of_P(L, 2.0, 0.5))
        assert r.min() == pytest.approx(1.0, abs=1e-9) and r.max() == pytest.approx(1.0, abs=1e-9)
