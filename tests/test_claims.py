"""Claims: jede Zahl der README-Tabelle „Befunde“ wird hier nachgerechnet (DEMO-PLAYBOOK §4, §11a). Zählungen und Bänder, nie ein einzelner Seed als Befund.

Plattformabhängige Zahlen (lange Läufe, Gleitkomma-Summen) stehen als Band; die Zahlen unten sind die des Entwicklungsrechners. Die Parameter der Messungen
sind nirgends Zweierpotenzen außer den benannten Sätzen (β = ½ als Satz, α = 1 nur in Brüchen oder mit einem Rundungsrauschen von 10⁻⁹)."""

import math

import numpy as np
import pytest

import lmr_constants as C
import lmr_dynamics as D
import lmr_evaluation as ev
import lmr_exact as X

# --------------------------------------------------------------------------------------------------------------------------------
# Satz β = ½ und die Grenze darunter
# --------------------------------------------------------------------------------------------------------------------------------


def test_beta_half_limit_is_2_for_10_of_10_L():
    Ls = (1, 2, 3, 4, 5, 6, 7, 8, 12, 13)
    assert len(Ls) == 10 and all(abs(D.alpha_krit(L, 0.5) - 2.0) < 1e-6 for L in Ls)
    # Gegenprobe gegen das Gleitkomma-Artefakt: β = ½ ± 10⁻⁹ ändert die Grenze für L ≤ 8 um weniger als 10⁻³
    for L in range(2, 9):
        assert abs(D.alpha_krit(L, 0.5 + 1e-9) - 2.0) < 1e-3 and abs(D.alpha_krit(L, 0.5 - 1e-9) - 2.0) < 1e-3, L


def test_beta_half_is_stable_on_the_whole_grid_30_by_199():
    grid = np.linspace(0.0137, 1.9871, 199)
    bad = [(L, float(a)) for L in range(1, 31) for a in grid if D.spectral_radius(L, float(a), 0.5) >= 1.0]
    assert len(grid) * 30 == 5970 and bad == []


def test_beta_half_alpha_2_matrix_order_is_2L_in_8_of_8_and_roots_are_on_the_circle_in_8_of_8():
    assert [X.smallest_power_identity(X.matrix_exact(L, "2", "1/2")[0]) for L in range(2, 10)] == [2 * L for L in range(2, 10)]
    for L in (2, 3, 4, 5, 7, 8, 12, 25):
        r = np.abs(D.roots_of_P(L, 2.0, 0.5))
        assert r.min() == pytest.approx(1.0, abs=1e-9) and r.max() == pytest.approx(1.0, abs=1e-9)


def test_below_beta_half_the_limit_rises_and_is_steeper_for_large_L():
    expect = {(0.49, 3): 1.889, (0.49, 6): 1.612, (0.49, 12): 1.149, (0.49, 24): 0.697, (0.495, 24): 0.920}
    for (b, L), v in expect.items():
        assert D.alpha_krit(L, b) == pytest.approx(v, abs=6e-4), (b, L)
    assert D.alpha_krit(24, 0.5) == pytest.approx(2.0, abs=1e-6)
    # steiler: der Abstand zu 2 bei β = 0,49 wächst mit L
    gaps = [2.0 - D.alpha_krit(L, 0.49) for L in (3, 6, 12, 24)]
    assert gaps == sorted(gaps)


# --------------------------------------------------------------------------------------------------------------------------------
# Parität und die Grenze außerhalb von 0 bis 1
# --------------------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("beta", [0.6, 0.8])
def test_parity_odd_L_limit_is_2_in_19_of_19_even_L_above_2_in_20_of_20(beta):
    odd = [D.alpha_krit(L, beta) for L in range(3, 40, 2)]
    even = [D.alpha_krit(L, beta) for L in range(2, 41, 2)]
    assert len(odd) == 19 and sum(abs(x - 2.0) < 1e-6 for x in odd) == 19
    assert len(even) == 20 and sum(x > 2.0 + 1e-6 for x in even) == 20


def test_parity_values_at_beta_08():
    expect = {2: 3.333, 4: 2.395, 6: 2.168, 12: 2.044, 40: 2.004}
    for L, v in expect.items():
        assert D.alpha_krit(L, 0.8) == pytest.approx(v, abs=6e-4), L
    assert max(D.alpha_krit(L, 0.8) for L in range(2, 41, 2)) == D.alpha_krit(2, 0.8)       # der größte Wert gehört zu L = 2
    assert D.alpha_krit(2, 0.75) == pytest.approx(4.0, abs=1e-9)                           # Maximum der L = 2 Grenze bei β = ¾


def test_L2_limit_equals_the_jury_formula_in_14_of_14():
    betas = (-2.0, -1.0, -0.5, 0.0, 0.25, 0.51, 0.6, 0.75, 0.9, 1.0, 1.25, 1.5, 2.0, 3.0)
    assert len(betas) == 14 and max(abs(D.alpha_krit(2, b) - D.jury_alpha_krit_L2(b)) for b in betas) < 1e-8


def test_beta_below_zero_the_limit_falls_with_abs_beta_and_no_stable_piece_above_in_140_of_140():
    assert [D.alpha_krit(3, b) for b in (0.0, -0.25, -1.0, -2.0)] == pytest.approx([0.618, 0.469, 0.274, 0.177], abs=6e-4)
    assert round(D.alpha_krit(12, 0.0), 3) == 0.136 and D.alpha_krit(12, -1.0) == pytest.approx(0.054, abs=6e-4)
    betas = (-2.0, -1.8, -1.5, -1.2, -1.0, -0.9, -0.75, -0.6, -0.5, -0.4, -0.25, -0.1, -0.05, -0.01)
    cells = 0
    for b in betas:
        for L in (1, 2, 3, 4, 5, 6, 7, 8, 12, 13):
            c = D.alpha_krit(L, b)
            al = np.linspace(c * (1 + 1e-6), 8.0, 4000)
            assert D.rho_batch(L, al, b).min() > 1.0, (L, b)
            cells += 1
    assert cells == 140


def test_beta_above_one_even_L_limit_is_2_over_2beta_minus_1_in_25_of_25_odd_L_above_it_in_12_of_12():
    n = 0
    for L in (2, 4, 6, 8, 12):
        for b in (1.1, 1.25, 1.5, 2.0, 3.0):
            assert D.alpha_krit(L, b) == pytest.approx(2.0 / (2.0 * b - 1.0), abs=1e-6), (L, b)
            n += 1
    assert n == 25
    m = 0
    for L in (3, 5, 7, 13):
        for b in (1.1, 1.5, 3.0):
            assert D.alpha_krit(L, b) > 2.0 / (2.0 * b - 1.0) + 1e-6, (L, b)
            m += 1
    assert m == 12
    assert [D.alpha_krit(3, b) for b in (1.1, 1.5, 3.0)] == pytest.approx([2.0, 1.281, 0.432], abs=6e-4)
    assert [D.alpha_krit(13, b) for b in (1.1, 1.5, 3.0)] == pytest.approx([1.747, 1.013, 0.401], abs=6e-4)
    assert D.alpha_krit(7, 1.1) == pytest.approx(1.914, abs=6e-4) and D.alpha_krit(5, 1.1) == pytest.approx(2.0, abs=1e-6)


def test_L1_has_no_beta_dependence():
    worst = max(abs(D.spectral_radius(1, float(a), float(b)) - abs(1.0 - a)) for a in np.linspace(0.1, 3.0, 30) for b in np.linspace(-3.0, 3.0, 25))
    assert worst < 1e-12


# --------------------------------------------------------------------------------------------------------------------------------
# Abklingzeiten
# --------------------------------------------------------------------------------------------------------------------------------


def test_best_setting_without_pipeline_decay_11_18_40_84_172_and_with_beta_1_exactly_L():
    expect = {2: 11, 3: 18, 6: 40, 12: 84, 24: 172}
    for L, n in expect.items():
        a = D.best_alpha_without_pipeline(L)
        assert D.decay_time(L, a, 0.0) == n and D.decay_time(L, a * (1 + 1e-6), 0.0) == n and D.decay_time(L, a * (1 - 1e-6), 0.0) == n, L
    for L in (2, 3, 6, 12):                                                                # der kleinste Spektralradius ohne Leitung ist 1 − 1/L (Gitter von 2000 Werten)
        grid = np.linspace(1e-3, 0.999 * D.alpha_krit(L, 0.0), 2000)
        rmin = float(D.rho_batch(L, grid, 0.0).min())
        assert 1 - 1 / L - 1e-7 <= rmin <= 1 - 1 / L + 1e-3, L
    ratios = [expect[L] / L for L in expect]
    assert min(ratios) == 5.5 and max(ratios) == pytest.approx(172 / 24)                  # das 5,5- bis 7,2-Fache von L
    Ls = (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 16, 20, 24)
    assert len(Ls) == 14 and all(D.decay_time(L, 1.0 + 1e-9, 1.0) == L for L in Ls)


def test_hand_example_decay_16_with_beta_055_19_with_beta_half_and_growth_without_pipeline():
    assert D.spectral_radius(2, 1.2, 0.55) == pytest.approx(0.735, abs=5e-4) and D.decay_time(2, 1.2, 0.55) == 16
    assert D.spectral_radius(2, 1.2, 0.5) == pytest.approx(0.775, abs=5e-4) and D.decay_time(2, 1.2, 0.5) == 19
    assert D.spectral_radius(2, 1.2, 0.0) == pytest.approx(1.0954, abs=5e-5)              # +9,54 % je Periode


def test_peak_at_beta_one_gets_sharper_with_L_alpha_08():
    betas = (1.0, 0.99, 0.9, 1.1)
    expect = {3: [5, 5, 7, 8], 6: [8, 12, 18, 14], 12: [14, 24, 36, 26]}
    for L, row in expect.items():
        got = [D.decay_time(L, 0.8, b) for b in betas]
        assert got == row, L
        assert [D.decay_time(L, 0.8 * (1 + 1e-6), b) for b in betas] == row              # nicht an der Rundung der Schwelle
    assert D.spectral_radius(12, 0.8, 1.0) == pytest.approx(0.200, abs=5e-4)
    assert D.spectral_radius(12, 0.8, 0.99) == pytest.approx(0.688, abs=5e-4) and D.spectral_radius(12, 0.8, 0.9) == pytest.approx(0.837, abs=5e-4)
    # Anteil der Spitze: das Verhältnis der Abklingzeit bei β = 0,99 zu der bei β = 1 steigt mit L
    ratios = [expect[L][1] / expect[L][0] for L in (3, 6, 12)]
    assert ratios == sorted(ratios) and ratios[0] == 1.0


def test_beta_one_is_not_always_the_fastest_L3_alpha_045_beta_09_beats_beta_1():
    assert D.decay_time(3, 0.45, 0.9) == 8 and D.decay_time(3, 0.45, 1.0) == 10
    assert D.decay_time(3, 0.45 * (1 + 1e-6), 0.9) == 8 and D.decay_time(3, 0.45 * (1 + 1e-6), 1.0) == 10


def test_beta_half_near_alpha_2_decay_is_very_long_and_the_estimate_is_below_in_12_of_12():
    big = {(3, 1.99): 5842, (6, 1.99): 43208, (12, 1.99): 338186, (2, 1.99): 1858}
    for (L, a), n in big.items():
        assert D.decay_time(L, a, 0.5) == n, (L, a)
    assert D.spectral_radius(3, 1.99, 0.5) == pytest.approx(0.999164, abs=5e-7)
    gaps = []
    for L in (2, 3, 6, 12):
        for a in (1.01, 1.51, 1.91, 1.99):
            t = D.decay_time(L, a, 0.5)
            if t > 100:
                gaps.append(100.0 * (1.0 - D.decay_estimate(D.spectral_radius(L, a, 0.5)) / t))
    assert len(gaps) == 12 and 0.5 < min(gaps) and max(gaps) < 7.0                         # gemessen 0,97 bis 5,87 %
    assert D.decay_time(2, 1.01, 0.5) == 15                                                # kurze Abklingzeit: die Schätzung liegt 10,1 % darunter (nicht gezählt)
    assert 100.0 * (1.0 - D.decay_estimate(D.spectral_radius(2, 1.01, 0.5)) / 15) == pytest.approx(10.1, abs=0.1)


def test_preset_numbers_in_the_readme():
    assert D.spectral_radius(12, 0.12, 0.0) == pytest.approx(0.992, abs=5e-4) and D.decay_time(12, 0.12, 0.0) == 585
    assert D.spectral_radius(12, 0.85, 1.0) == pytest.approx(0.15, abs=1e-9) and D.decay_time(12, 0.85, 1.0) == 14
    assert D.spectral_radius(4, 2.2, 0.8) == pytest.approx(0.954, abs=5e-4) and D.decay_time(4, 2.2, 0.8) == 96
    assert D.spectral_radius(4, 2.2, 0.0) == pytest.approx(1.402, abs=5e-4)
    assert D.spectral_radius(5, 2.2, 0.8) == pytest.approx(1.08, abs=5e-3)                # ungerades L mit derselben Einstellung: über der Grenze 2, instabil
    assert D.spectral_radius(6, 0.4, 0.3) == pytest.approx(0.972, abs=5e-4) and D.decay_time(6, 0.4, 0.3) == 163
    assert D.spectral_radius(6, 0.4, 0.0) == pytest.approx(1.046, abs=5e-4)


# --------------------------------------------------------------------------------------------------------------------------------
# Drift
# --------------------------------------------------------------------------------------------------------------------------------


def test_drift_examples_table_L4_beta_in_periods_of_demand():
    d = 10.0
    for beta, expect in ((1.0, [-2, -1, 0, 1, 2]), (0.6, [-1.2, -0.6, 0, 0.6, 1.2]), (0.0, [0, 0, 0, 0, 0])):
        got = [D.drift_equilibrium(beta, D.delta_w(d, 4, d, Lh)) / d for Lh in (2, 3, 4, 5, 6)]
        assert got == pytest.approx(expect, abs=1e-12), beta


def test_drift_simulation_is_exact_after_20000_periods_in_9_of_9():
    cases = [(4, 0.4, 0.55, 3), (4, 0.4, 1.0, 3), (4, 0.4, 1.0, 5), (4, 0.4, 0.0, 3), (6, 0.3, 1.0, 5),
             (3, 0.7, 0.6, 2), (2, 0.9, 1.5, 1), (4, 0.55, 1.3, 3), (5, 0.35, 0.8, 2)]
    worst = 0.0
    for L, a, b, Lh in cases:
        dW = D.delta_w(10.0, L, 10.0, Lh)
        i, _u = D.simulate(L, a, b, np.zeros(20000), dW)
        worst = max(worst, abs(float(i[-1]) + b * dW))
    assert len(cases) == 9 and worst < 1e-9                                               # gemessen 8,9·10⁻¹⁵


def test_drift_noise_means_within_three_standard_errors_in_12_of_12_four_settings_three_seeds():
    cases = [(4, 0.4, 0.55, 3), (4, 0.4, 1.0, 3), (4, 0.4, 1.0, 5), (4, 0.4, 0.0, 3)]
    n, worst = 0, 0.0
    for seed in (16, 17, 18):
        e = D.noise_series(400_000, 0.2, 10.0, seed)
        for L, a, b, Lh in cases:
            dW = D.delta_w(10.0, L, 10.0, Lh)
            i = D.filter_response(L, a, b, e) + D.drift_step_response(L, a, b, dW, 400_000)
            mean, se = D.batch_mean_and_se(i[1000:])
            worst = max(worst, abs(mean + b * dW) / se)
            n += 1
    assert n == 12 and worst <= 3.0


def test_drift_settling_time_19_and_13_periods_L4_alpha_04():
    for b, expect in ((0.55, 19), (1.0, 13)):
        dW = D.delta_w(10.0, 4, 10.0, 3)
        assert ev.settle_time(D.drift_step_response(4, 0.4, b, dW, 3000), -b * dW) == expect, b


def test_drift_does_not_depend_on_alpha():
    dW = D.delta_w(10.0, 3, 10.0, 2)
    for a in (0.3, 0.6, 0.9, 1.4):
        assert D.spectral_radius(3, a, 0.8) < 1.0
        i, _u = D.simulate(3, a, 0.8, np.zeros(20000), dW)
        assert float(i[-1]) == pytest.approx(-0.8 * dW, abs=1e-6)


# --------------------------------------------------------------------------------------------------------------------------------
# Preis der Ruhe
# --------------------------------------------------------------------------------------------------------------------------------


def test_trade_off_table_L3_values():
    expect = {
        0.3: [(0.350, 3.889), (0.222, 3.763), (0.176, 3.961)],
        0.45: [(1.042, 5.147), (0.400, 3.407), (0.290, 3.434)],
        0.6: [(13.286, 36.905), (0.636, 3.373), (0.429, 3.190)],
    }
    for a, rows in expect.items():
        for b, (vu, vi) in zip((0.0, 0.55, 1.0), rows):
            u, i = D.noise_variances(3, a, b)
            assert u == pytest.approx(vu, abs=1.5e-3) and i == pytest.approx(vi, abs=1.5e-3), (a, b)
    assert D.noise_variances(3, 0.8, 0.0)[0] == math.inf                                   # ohne Leitung bei α = 0,8 instabil (Grenze 0,618)
    # Bestandsstreuung hat bei α = 0,45 ihr Minimum bei β = 0,55, nicht bei β = 1
    vi = [D.noise_variances(3, 0.45, b)[1] for b in (0.0, 0.55, 1.0)]
    assert vi[1] < vi[2] < vi[0]


def test_trade_off_verdict_rows_from_the_app_functions():
    rows = ev.tradeoff_rows(3, 0.45)
    kind, text = ev.verdict_tradeoff(3, 0.45, rows)
    assert kind == "info" and "Kleinste Bestellstreuung bei β = 1,00" in text and "kleinste Bestandsstreuung bei β = 0,55" in text


# --------------------------------------------------------------------------------------------------------------------------------
# Presets: erwartete Verdikt-Zeile je Preset (kein Preset ohne Wirkung meldet eine Wirkung)
# --------------------------------------------------------------------------------------------------------------------------------


def _main(preset):
    p = C.PRESETS[preset]
    a = ev.analyse(ev.Settings(p["L"], p["alpha"], p["beta"]))
    return ev.verdict_main(p["L"], p["alpha"], p["beta"], a.rho, a.rho0, a.decay, a.decay0)


def test_each_preset_shows_the_expected_main_verdict_line():
    kind, text = _main("Von Hand (L = 2, β = 0,55)")
    assert kind == "success" and "ohne Leitung (β = 0) ist ρ = 1,095" in text and "mit β = 0,55 ist ρ = 0,735, der Stoß klingt in 16 Perioden" in text
    kind, text = _main("Ohne Leitung (β = 0)")
    assert kind == "success" and "ohne Leitung (β = 0) ist ρ = 0,992, der Stoß klingt in 585 Perioden" in text and "mit β" not in text
    kind, text = _main("β = 1, fast Deadbeat")
    assert kind == "success" and "mit β = 1,00 ist ρ = 0,150, der Stoß klingt in 14 Perioden" in text
    kind, text = _main("Gerade und ungerade L (β = 0,8)")
    assert kind == "success" and "ohne Leitung (β = 0) ist ρ = 1,402" in text and "mit β = 0,80 ist ρ = 0,954, der Stoß klingt in 96 Perioden" in text
    kind, text = _main("Leitung zu wenig (β = 0,3)")
    assert kind == "success" and "ohne Leitung (β = 0) ist ρ = 1,046" in text and "mit β = 0,30 ist ρ = 0,972, der Stoß klingt in 163 Perioden" in text
    kind, text = _main("Drift (L̂ = L − 1)")
    assert kind == "success" and "mit β = 1,00 ist ρ = 0,600, der Stoß klingt in 13 Perioden" in text


def test_drift_preset_verdict_and_every_other_preset_has_no_dauerfehler_at_correct_estimate():
    p = C.PRESETS["Drift (L̂ = L − 1)"]
    dr = ev.analyse_drift(p["L"], p["alpha"], p["beta"], p["lhat_delta"], 10.0, 0.0, C.DEFAULT_SEED)
    rho = D.spectral_radius(p["L"], p["alpha"], p["beta"])
    kind, text = ev.verdict_drift(p["L"], p["beta"], p["L"] + p["lhat_delta"], 10.0, 10.0, dr.i_star, dr.t_eq, rho)
    assert kind == "warning" and "1,00 Perioden Bedarf unter dem Soll (−10,0 Stück bei d = 10)" in text and "in 13 Perioden auf 1 %" in text
    for name, q in C.PRESETS.items():
        if q["lhat_delta"] == 0:
            r = ev.analyse_drift(q["L"], q["alpha"], q["beta"], 0, 10.0, 0.0, C.DEFAULT_SEED)
            assert r.i_star == 0.0, name                                                  # richtig geschätzt: kein Dauerfehler


def test_beta_one_alpha_09_decay_is_at_the_rounding_edge_and_the_preset_avoids_it():
    """Bei α = 0,9, β = 1 ist ρ² = 0,01 genau die Schwelle: ρ = 0,1 gerundet entscheidet über 13 oder 14 Perioden (Befund gegenüber dem Plan, README)."""
    assert D.decay_time(12, 0.9, 1.0) in (13, 14)
    for a in (0.85, 0.8):                                                                      # ρ² = 0,0225 bzw. 0,04 über 0,01, ρ³ = 0,0034 bzw. 0,008 darunter: L + 2 = 14
        assert D.decay_time(12, a, 1.0) == D.decay_time(12, a * (1 + 1e-6), 1.0) == 14
    assert C.PRESETS["β = 1, fast Deadbeat"]["alpha"] == 0.85
