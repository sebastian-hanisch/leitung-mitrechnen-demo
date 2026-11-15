"""Kern gegen Handrechnung (Mini-Instanzen, Brüche) und Einheiten: Polynom, Zustandsmatrix, Spektralradius, Simulation, Filter, Abklingzeit, Grenze, Drift.

Die sieben Handbeispiele des Plans stehen hier als Brüche; die Parameter der Handbeispiele, die Zweierpotenzen sind (β = ½, α = 1, α = 2), sind Sätze
in exakter Rechnung (Brüche) oder Gleitkomma nur mit einem Rundungsrauschen und einem Band."""

import math
from fractions import Fraction as F

import numpy as np
import pytest

import lmr_dynamics as D
import lmr_exact as X

# --------------------------------------------------------------------------------------------------------------------------------
# Handbeispiel 1 und 2: L = 2, Matrix, Polynom, Jury
# --------------------------------------------------------------------------------------------------------------------------------


def test_L2_matrix_polynomial_and_jury_limits_by_hand():
    A, c = X.matrix_exact(2, "2", "1/2")
    assert A == [[1, 1], [-2, -1]] and c == [-2, -1]
    A2 = X.matmul(A, A)
    assert A2 == [[-1, 0], [0, -1]]                       # A^2 = −I
    assert X.matmul(A2, A2) == X.identity(2)              # A^4 = I: P = z^2 + 1, Wurzeln ±i, Periode 4
    # Jury: α_krit = 1 (β = 0), 2 (β = ½), 4 (β = ¾, Maximum), 2 (β = 1)
    assert [D.jury_alpha_krit_L2(b) for b in (0.0, 0.5, 0.75, 1.0)] == [1.0, 2.0, 4.0, 2.0]


def test_L2_trace_and_determinant_are_the_polynomial_coefficients():
    for a, b in ((1.2, 0.55), (0.7, 0.2), (3.0, 0.8)):
        p = D.char_poly(2, a, b)
        assert p == pytest.approx([1.0, -(1.0 - a * b), a * (1.0 - b)], abs=1e-15)
        A = D.state_matrix(2, a, b)
        assert np.trace(A) == pytest.approx(1.0 - a * b) and np.linalg.det(A) == pytest.approx(a * (1.0 - b))


def test_hand_example_L2_alpha_6_5_beta_11_20(hand):
    ii, uu = X.impulse_exact(hand["L"], hand["alpha"], hand["beta"], 4)
    assert ii == hand["i"] and uu == hand["u"]
    Am, _c = X.matrix_exact(2, hand["alpha"], hand["beta"])
    assert Am[0][0] + Am[1][1] == F(17, 50)                                     # Spur 1 − αβ
    assert Am[0][0] * Am[1][1] - Am[0][1] * Am[1][0] == F(27, 50)               # Determinante α (1 − β)
    # Gleitkomma-Kern gegen die Brüche
    i, u = D.impulse_response(2, 1.2, 0.55, 3)
    assert i[1:4] == pytest.approx([float(v) for v in hand["i"][:3]], abs=1e-12)
    assert u == pytest.approx([float(v) for v in hand["u"][:3]], abs=1e-12)
    assert D.spectral_radius(2, 1.2, 0.55) == pytest.approx(math.sqrt(0.54))     # komplexes Paar: ρ = √det
    assert D.spectral_radius(2, 1.2, 0.0) == pytest.approx(math.sqrt(1.2))       # ohne Leitung wächst die Schwingung um 9,54 % je Periode


def test_hand_example_beta_half_alpha_6_5_stoss_response_in_fractions():
    ii, uu = X.impulse_exact(2, "6/5", "1/2", 8)
    assert ii == [F(-1), F(-1), F(1, 5), F(17, 25), F(19, 125), F(-217, 625), F(-719, 3125), F(1817, 15625)]
    assert uu[1] == F(6, 5) and uu[2] == F(12, 25)
    assert D.spectral_radius(2, 1.2, 0.5) == pytest.approx(math.sqrt(3 / 5))
    assert D.decay_time(2, 1.2, 0.5) == 19                                       # Abklingzeit 19 Perioden (Schätzung ln 0,01 / ln ρ = 18,0)
    assert D.decay_estimate(math.sqrt(3 / 5)) == pytest.approx(18.0, abs=0.05)


# --------------------------------------------------------------------------------------------------------------------------------
# Handbeispiel 3: L = 3, β = ½, α = 2 (Grenze); Handbeispiel 4: Parität
# --------------------------------------------------------------------------------------------------------------------------------


def test_L3_beta_half_alpha_2_matrix_has_order_6_and_P_is_z3_plus_1():
    A, _ = X.matrix_exact(3, "2", "1/2")
    assert A == [[1, 0, 1], [-2, -1, -1], [0, 1, 0]]
    assert X.smallest_power_identity(A) == 6
    assert D.char_poly(3, 2.0, 0.5) == pytest.approx([1.0, 0.0, 0.0, 1.0])
    # P(z) = (z + 1)(z² − z + 1): Wurzeln −1 und e^(±iπ/3), alle auf dem Einheitskreis
    r = np.sort_complex(D.roots_of_P(3, 2.0, 0.5))
    assert np.abs(r) == pytest.approx([1.0, 1.0, 1.0], abs=1e-12)
    assert D.spectral_radius(3, 1.0 + 1e-12, 0.5) == pytest.approx(0.872175, abs=1e-6)


def test_parity_same_setting_L2_stable_L3_unstable():
    """α = 3, β = 4/5: L = 2 hat die Grenze 10/3 (stabil, det = 3/5), L = 3 hat P(−1) = α − 2 = 1 und die Grenze 2 (instabil)."""
    assert D.jury_alpha_krit_L2(0.8) == pytest.approx(10 / 3)
    assert D.spectral_radius(2, 3.0, 0.8) == pytest.approx(math.sqrt(3 / 5))
    assert D.p_at_minus_one(3, 3.0, 0.8) == pytest.approx(1.0)
    assert D.p_at_minus_one(2, 3.0, 0.8) == pytest.approx(2.0 + 3.0 * (1.0 - 1.6))
    r3 = D.spectral_radius(3, 3.0, 0.8)
    assert r3 > 1.0 and r3 == pytest.approx(1.6267, abs=1e-4)
    assert D.alpha_krit(3, 0.8) == pytest.approx(2.0, abs=1e-9)


# --------------------------------------------------------------------------------------------------------------------------------
# Handbeispiel 5: β = 1
# --------------------------------------------------------------------------------------------------------------------------------


def test_beta_one_polynomial_and_deadbeat_in_fractions():
    for L in (2, 3, 4):
        p = D.char_poly(L, 0.6, 1.0)
        assert p == pytest.approx([1.0, -0.4] + [0.0] * (L - 1))
        assert D.spectral_radius(L, 0.6, 1.0) == pytest.approx(0.4, abs=1e-12)
    for L in range(1, 8):
        ii, _ = X.impulse_exact(L, "1", "1", L + 4)
        assert ii[:L] == [F(-1)] * L and all(v == 0 for v in ii[L:])      # L Perioden auf −1, ab t = L + 1 exakt 0
    # dasselbe mit Rundungsrauschen im Gleitkomma (kein exaktes α = 1): die Abklingzeit bleibt L
    for L in (2, 3, 6, 12):
        assert D.decay_time(L, 1.0 + 1e-9, 1.0) == L


# --------------------------------------------------------------------------------------------------------------------------------
# Handbeispiel 6: Drift
# --------------------------------------------------------------------------------------------------------------------------------


def _hand_drift_sequence(alpha, beta, dW, L, T):
    """Von Hand in Brüchen: i_t und u_t aus i_0 = 0, ohne Bedarfsabweichung (Verzögerungsliste, ohne Matrix)."""
    alpha, beta, dW = F(alpha), F(beta), F(dW)
    u, i = [], [F(0)]
    for t in range(T):
        S = sum((u[t - j] for j in range(1, L) if t - j >= 0), F(0))
        u.append(-alpha * (i[t] + beta * (S + dW)))
        i.append(i[t] + (u[t + 1 - L] if t + 1 - L >= 0 else F(0)))
    return i, u


def test_drift_by_hand_L2_Lhat1_beta1_alpha_3_5():
    d, L, Lh = F(10), 2, 1
    dW = d * (L - 1) - d * (Lh - 1)
    assert dW == 10 and -F(1) * dW == -10                                         # i* = −β ΔW = −10
    i, u = _hand_drift_sequence("3/5", "1", dW, L, 8)
    assert u[0] == -6 and u[1] == F(-12, 5) and u[2] == F(-24, 25)
    assert i[1] == 0 and i[2] == -6 and i[3] == F(-42, 5)
    # der Abstand zum Grenzwert fällt wie (2/5)^t; i_7 = −9,959
    assert float(i[7]) == pytest.approx(-9.959, abs=5e-4)
    # Gleitkomma-Kern und Formel
    dWf = D.delta_w(10.0, 2, 10.0, 1)
    assert dWf == 10.0 and D.drift_equilibrium(1.0, dWf) == -10.0
    ifl, ufl = D.simulate(2, 0.6, 1.0, np.zeros(8), dWf)
    assert list(ifl[:4]) == pytest.approx([0.0, 0.0, -6.0, -8.4])
    assert list(ufl[:3]) == pytest.approx([-6.0, -2.4, -0.96])


def test_drift_equilibria_by_hand_for_other_beta_and_Lhat():
    d = 10.0
    assert D.drift_equilibrium(0.5, D.delta_w(d, 2, d, 1)) == pytest.approx(-5.0)       # β = ½: −5
    assert D.drift_equilibrium(1.0, D.delta_w(d, 2, d, 3)) == pytest.approx(10.0)       # L̂ = 3 (zu lang geschätzt): +10
    assert D.drift_equilibrium(0.5, D.delta_w(d, 2, d, 3)) == pytest.approx(5.0)
    assert D.drift_equilibrium(0.0, D.delta_w(d, 2, d, 1)) == 0.0                       # β = 0: nichts
    assert D.delta_w(10.0, 4, 7.0, 4) == pytest.approx(3 * 3)                           # falscher Mittelbedarf: (d − d̂)(L − 1)


# --------------------------------------------------------------------------------------------------------------------------------
# Einheiten
# --------------------------------------------------------------------------------------------------------------------------------


def test_char_poly_L1_and_leading_coefficients():
    assert D.char_poly(1, 0.4, 0.3) == pytest.approx([1.0, -0.6])                       # z − (1 − α): β fällt heraus
    p = D.char_poly(4, 0.7, 0.2)
    assert p[0] == 1.0 and p[1] == pytest.approx(-(1 - 0.14)) and p[-1] == pytest.approx(0.7 * 0.8) and p[2] == 0.0 == p[3]


def test_state_matrix_float_equals_fraction_matrix():
    for L in (1, 2, 3, 5):
        A, c = X.matrix_exact(L, F(7, 10), F(3, 10))
        assert np.array(D.state_matrix(L, 0.7, 0.3)) == pytest.approx(np.array([[float(v) for v in row] for row in A]), abs=1e-15)
        assert D.readout(L, 0.7, 0.3) == pytest.approx([float(v) for v in c])


def test_spectral_radius_ways_agree_on_the_hand_cases():
    for L, a, b in ((2, 1.2, 0.55), (3, 0.7, 0.3), (5, 2.2, 0.8), (12, 0.12, 0.0)):
        assert D.spectral_radius(L, a, b) == pytest.approx(D.spectral_radius_matrix(L, a, b), abs=1e-10)
        assert D.rho_batch(L, [a], b)[0] == pytest.approx(D.spectral_radius(L, a, b), abs=1e-10)


def test_simulate_against_naive_delay_list_and_filter():
    """Verzögerungsliste mit expliziter Summe, ohne fortgeschriebene Leitungssumme, gegen simulate() und das Filter."""
    rng = np.random.default_rng(5)
    for L, a, b, dW in ((1, 0.5001, 0.3, 4.0), (2, 1.2, 0.55, 0.0), (5, 0.6, 0.9, -3.0), (12, 0.8, 1.1, 2.5)):
        e = rng.normal(size=60)
        i, u = D.simulate(L, a, b, e, dW)
        il, ul = [0.0], []
        for t in range(60):
            ul.append(-a * (il[t] + b * (sum(ul[t - j] for j in range(1, L) if t - j >= 0) + dW)))
            il.append(il[t] + (ul[t + 1 - L] if t + 1 - L >= 0 else 0.0) - e[t])
        assert list(i) == pytest.approx(il, abs=1e-9) and list(u) == pytest.approx(ul, abs=1e-9)
        if dW == 0.0:
            assert D.filter_response(L, a, b, e) == pytest.approx(i, abs=1e-9)


def test_decay_time_of_toy_sequences():
    assert D.decay_time_of(np.array([0.0, -1.0, 0.5, 0.011, 0.009, 0.0])) == 3          # letzte Periode über 1 % der Spitze (|0,011| > 0,01)
    assert D.decay_time_of(np.zeros(5)) == 0
    assert D.decay_time_of(np.array([0.0, 1.0, 0.0, 0.0, 0.5, 0.0])) == 4               # späte Wiederkehr zählt mit
    assert D.decay_time_of(np.array([0.0, -1.0, -0.5, 0.0]), frac=0.6) == 1


def test_decay_time_is_none_without_decay():
    assert D.decay_time(3, 3.0, 0.8) is None                                           # instabil
    assert D.decay_time(3, 2.0, 0.5) is None                                           # ρ = 1 (auf dem Kreis)
    assert D.decay_time(1, 1.0 + 1e-9, 0.4) == 1                                       # L = 1, ρ ≈ 0: ein Schritt


def test_decay_estimate_edge_cases():
    assert D.decay_estimate(0.0) == 0.0 and D.decay_estimate(1.0) == math.inf and D.decay_estimate(1.3) == math.inf
    assert D.decay_estimate(0.1) == pytest.approx(2.0)                                  # ln 0,01 / ln 0,1


def test_best_alpha_without_pipeline_values_and_L1():
    assert D.best_alpha_without_pipeline(1) == 1.0
    assert D.best_alpha_without_pipeline(2) == pytest.approx(0.25) and D.best_alpha_without_pipeline(3) == pytest.approx(4 / 27)
    assert D.best_alpha_without_pipeline(12) == pytest.approx(0.032, abs=1e-4)


def test_impulse_response_shape_and_first_values():
    i, u = D.impulse_response(3, 0.6, 0.3, 10)
    assert i.shape == (11,) and u.shape == (10,)
    assert i[0] == 0.0 and i[1] == -1.0 and u[0] == 0.0
    assert u[1] == pytest.approx(0.6)                                                  # u_1 = −α i_1
    i0, u0 = D.impulse_response(3, 0.6, 0.3, 0)
    assert i0.shape == (1,) and u0.shape == (0,)


def test_noise_series_is_reproducible_and_scaled():
    a = D.noise_series(50000, 0.2, 10.0, 16)
    b = D.noise_series(50000, 0.2, 10.0, 16)
    assert np.array_equal(a, b) and not np.array_equal(a, D.noise_series(50000, 0.2, 10.0, 17))
    assert abs(float(a.mean())) < 0.05 and float(a.std()) == pytest.approx(2.0, rel=0.02)
    assert not np.any(D.noise_series(100, 0.0, 10.0, 16))


def test_batch_mean_and_standard_error_on_a_constant_and_a_ramp():
    m, se = D.batch_mean_and_se(np.full(400, 3.0))
    assert m == 3.0 and se == 0.0
    m2, _se2 = D.batch_mean_and_se(np.arange(400.0))
    assert m2 == pytest.approx(199.5)


def test_noise_variances_unstable_is_infinite_and_beta1_closed_form():
    assert D.noise_variances(3, 0.8, 0.0) == (math.inf, math.inf)
    vu, vi = D.noise_variances(3, 0.3, 1.0)
    assert vu == pytest.approx(0.3 / 1.7, abs=1e-12) and vi == pytest.approx(2 + 1 / (0.3 * 1.7), abs=1e-9)
    assert D.noise_variances_beta1(3, 0.3) == pytest.approx((vu, vi), abs=1e-9)


def test_circle_crossings_are_sorted_and_contain_the_known_limit():
    for L, b, expect in ((3, 0.0, 0.6180339887), (2, 0.75, 4.0), (13, 0.8, 2.0)):
        c = D.circle_crossings(L, b)
        assert c == sorted(c) and any(abs(x - expect) < 1e-6 for x in c)
    assert D.alpha_krit(1, 0.9) == pytest.approx(2.0, abs=1e-9) and D.alpha_krit(1, -1.5) == pytest.approx(2.0, abs=1e-9)
