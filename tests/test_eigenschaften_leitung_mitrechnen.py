"""Eigenschaftstests (metamorph, ORACLE-PLAYBOOK §5.2): Aussagen, die für das lineare Lager gelten müssen, ohne dass man das Ergebnis kennt.

Je Eigenschaft 20 bis 50 Zufallsinstanzen mit festem Zufalls-Seed; jede Aussage gilt in diesem Modell (Meldeverzug 0, ohne Klemmung)."""

import numpy as np
import pytest

import lmr_dynamics as D
import lmr_evaluation as ev


def _rng(seed):
    return np.random.default_rng(seed)


def test_linearity_a_scaled_shock_scales_the_path_and_two_shocks_add_up():
    rng = _rng(1)
    for _ in range(30):
        L = int(rng.integers(1, 10))
        a, b = float(rng.uniform(0.1, 1.5)), float(rng.uniform(-0.5, 1.5))
        e1, e2 = rng.normal(size=40), rng.normal(size=40)
        c = float(rng.uniform(0.2, 5.0))
        i1, u1 = D.simulate(L, a, b, e1)
        i2, u2 = D.simulate(L, a, b, e2)
        ic, uc = D.simulate(L, a, b, c * e1)
        isum, usum = D.simulate(L, a, b, e1 + e2)
        assert ic == pytest.approx(c * i1, rel=1e-9, abs=1e-9) and uc == pytest.approx(c * u1, rel=1e-9, abs=1e-9)
        assert isum == pytest.approx(i1 + i2, rel=1e-9, abs=1e-9) and usum == pytest.approx(u1 + u2, rel=1e-9, abs=1e-9)


def test_time_invariance_a_shifted_shock_shifts_the_response():
    rng = _rng(2)
    for _ in range(25):
        L = int(rng.integers(1, 9))
        a, b = float(rng.uniform(0.1, 1.2)), float(rng.uniform(0.0, 1.3))
        k = int(rng.integers(1, 12))
        e = np.zeros(60)
        e[0] = 1.0
        es = np.zeros(60)
        es[k] = 1.0
        i0, _ = D.simulate(L, a, b, e)
        ik, _ = D.simulate(L, a, b, es)
        assert ik[k:] == pytest.approx(i0[: len(ik) - k], abs=1e-9) and not np.any(ik[:k])


def test_beta_one_gives_rho_abs_one_minus_alpha_for_L_1_to_40():
    rng = _rng(3)
    for L in range(1, 41):
        a = float(rng.uniform(0.05, 3.0))
        assert D.spectral_radius(L, a, 1.0) == pytest.approx(abs(1.0 - a), abs=1e-12), L


def test_L1_ignores_beta():
    rng = _rng(4)
    for _ in range(30):
        a = float(rng.uniform(0.05, 4.0))
        r = [D.spectral_radius(1, a, float(b)) for b in rng.uniform(-3, 3, size=5)]
        assert r == pytest.approx([abs(1 - a)] * 5, abs=1e-12)


def test_odd_L_limit_never_exceeds_2_for_any_beta_on_the_grid():
    rng = _rng(5)
    for _ in range(40):
        L = int(rng.integers(1, 20)) * 2 + 1
        b = float(rng.uniform(-2.0, 3.0))
        assert D.alpha_krit(L, b) <= 2.0 + 1e-7, (L, b)                                      # P(−1) = α − 2: α = 2 ist immer eine Grenze oder darüber stabil


def test_even_L_beta_above_half_limit_is_at_most_2_over_2beta_minus_1():
    rng = _rng(6)
    for _ in range(40):
        L = int(rng.integers(1, 12)) * 2
        b = float(rng.uniform(0.51, 3.0))
        assert D.alpha_krit(L, b) <= 2.0 / (2.0 * b - 1.0) + 1e-7, (L, b)


def test_drift_is_linear_in_beta_and_in_L_minus_Lhat_and_independent_of_alpha():
    rng = _rng(7)
    for _ in range(25):
        L = int(rng.integers(1, 8))
        a = float(rng.uniform(0.1, 1.0))
        b = float(rng.uniform(0.1, 1.0))
        Lh = int(rng.integers(max(1, L - 3), L + 4))
        if D.spectral_radius(L, a, b) >= 0.995 or D.spectral_radius(L, a, 2 * b) >= 0.995:
            continue
        dW = D.delta_w(10.0, L, 10.0, Lh)
        assert dW == pytest.approx(10.0 * (L - Lh))
        i1, _ = D.simulate(L, a, b, np.zeros(6000), dW)
        i2, _ = D.simulate(L, a, 2 * b, np.zeros(6000), dW)
        i3, _ = D.simulate(L, min(1.1, 1.3 * a), b, np.zeros(6000), dW)
        if D.spectral_radius(L, min(1.1, 1.3 * a), b) >= 0.995:
            continue
        assert i1[-1] == pytest.approx(-b * 10.0 * (L - Lh), abs=1e-6)
        assert i2[-1] == pytest.approx(2 * i1[-1], abs=1e-6)                                    # linear in β
        assert i3[-1] == pytest.approx(i1[-1], abs=1e-6)                                        # unabhängig von α


def test_no_drift_when_the_estimate_is_right_or_beta_is_zero():
    rng = _rng(8)
    for _ in range(25):
        L = int(rng.integers(1, 9))
        a, b = float(rng.uniform(0.1, 0.7)), float(rng.uniform(0.0, 1.0))
        assert D.drift_equilibrium(b, D.delta_w(10.0, L, 10.0, L)) == 0.0
        assert D.drift_equilibrium(0.0, D.delta_w(10.0, L, 10.0, L + 2)) == 0.0
        assert D.drift_step_response(L, a, 0.0, 17.0, 300) == pytest.approx(np.zeros(301), abs=0.0)


def test_orders_in_equilibrium_equal_the_mean_demand_and_scale_with_d():
    for L, a, b, Lh in ((4, 0.4, 0.55, 3), (3, 0.7, 0.6, 2), (5, 0.35, 0.8, 2)):
        for c in (0.5, 3.0, 7.7):
            dW = D.delta_w(10.0 * c, L, 10.0 * c, Lh)
            i, u = D.simulate(L, a, b, np.zeros(4000), dW)
            assert u[-1] == pytest.approx(0.0, abs=1e-8)                                       # u = 0: Bestellung gleich Mittelbedarf
            assert i[-1] == pytest.approx(c * D.drift_equilibrium(b, D.delta_w(10.0, L, 10.0, Lh)), abs=1e-6)   # d → c d skaliert den Dauerfehler


def test_P_at_one_equals_alpha_and_the_pipeline_sum_order_does_not_matter():
    rng = _rng(9)
    for _ in range(40):
        L = int(rng.integers(1, 30))
        a, b = float(rng.uniform(0.05, 3.0)), float(rng.uniform(-1.0, 2.0))
        assert float(np.polyval(D.char_poly(L, a, b), 1.0)) == pytest.approx(a, abs=1e-12)
    # vertauschte Stoßreihenfolge: zwei Stöße in beiden Reihenfolgen geben die Summe der Einzelantworten
    e = np.zeros(50)
    e[3], e[11] = 1.0, 2.0
    f = np.zeros(50)
    f[3], f[11] = 2.0, 1.0
    ie, _ = D.simulate(4, 0.4, 0.8, e)
    i3, _ = D.simulate(4, 0.4, 0.8, np.eye(50)[3])
    i11, _ = D.simulate(4, 0.4, 0.8, np.eye(50)[11])
    assert ie == pytest.approx(1.0 * i3 + 2.0 * i11, abs=1e-9)
    assert D.simulate(4, 0.4, 0.8, f)[0] == pytest.approx(2.0 * i3 + 1.0 * i11, abs=1e-9)


def test_same_seed_same_noise_and_same_drift_result():
    a = ev.analyse_drift(4, 0.4, 0.8, -1, 10.0, 0.1, 16)
    b = ev.analyse_drift(4, 0.4, 0.8, -1, 10.0, 0.1, 16)
    c = ev.analyse_drift(4, 0.4, 0.8, -1, 10.0, 0.1, 17)
    assert np.array_equal(a.i_noisy, b.i_noisy) and not np.array_equal(a.i_noisy, c.i_noisy)
    assert np.array_equal(a.i_clean, c.i_clean)                                                 # ohne Rauschen hängt die Bahn nicht am Zufalls-Seed


def test_stock_variance_at_beta_one_is_at_least_L_with_the_minimum_at_alpha_one():
    """var(i)/var(e) = (L − 1) + 1/(α (2 − α)) ≥ L, Gleichheit bei α = 1 (nur mit Rundungsrauschen geprüft)."""
    rng = _rng(10)
    for _ in range(40):
        L = int(rng.integers(1, 12))
        a = float(rng.uniform(0.05, 1.95))
        assert D.noise_variances(L, a, 1.0)[1] >= L - 1e-9
        assert D.noise_variances(L, 1.0 + 1e-6, 1.0)[1] == pytest.approx(L, abs=1e-9)


def test_noise_variance_ratios_do_not_depend_on_the_noise_level():
    """Bedarfsrauschen mit doppelter Standardabweichung: Bestand und Bestellung mit doppelter Standardabweichung, die Verhältnisse bleiben (Linearität)."""
    e = D.noise_series(20_000, 1.0, 1.0, 3)
    i1 = D.filter_response(5, 0.3, 1.0, e)
    i2 = D.filter_response(5, 0.3, 1.0, 2.0 * e)
    assert float(i2.var()) / float((2.0 * e).var()) == pytest.approx(float(i1.var()) / float(e.var()), rel=1e-9)
