"""Verdikte an von Hand gesetzten Werten (DEMO-PLAYBOOK §12): jede Zahl im Text und jede Schwelle der Art steht hier.

Ein Verdikt ist eine reine Funktion (Art, Text) und bekommt Zahlen, keine Einstellungen; der Test prüft die Grenzen (ρ unter 1, auf 1, über 1) und jede genannte Zahl direkt."""

import math

import lmr_evaluation as ev

# --------------------------------------------------------------------------------------------------------------------------------
# Hilfsfunktionen
# --------------------------------------------------------------------------------------------------------------------------------


def test_german_number_format():
    assert ev.de(0.7348, 3) == "0,735" and ev.de(-10.0, 1) == "−10,0" and ev.de(2.0, 0) == "2"


def test_state_thresholds_are_the_tolerance_around_one():
    assert ev.state_of(0.5) == "stabil"
    assert ev.state_of(1.0 - 2e-9) == "stabil" and ev.state_of(1.0 - 5e-10) == "grenze"
    assert ev.state_of(1.0) == "grenze" and ev.state_of(1.0 + 5e-10) == "grenze"
    assert ev.state_of(1.0 + 2e-9) == "instabil"


def test_periods_names_none_as_no_decay():
    assert ev.periods(16) == "16 Perioden" and ev.periods(None) == "kein Abklingen"


def test_describe_each_branch():
    assert ev.describe(0.735, 16) == "ist ρ = 0,735, der Stoß klingt in 16 Perioden auf 1 % ab"
    assert ev.describe(1.0954, None) == "ist ρ = 1,095: die Schwingung wächst auf lange Sicht um 9,5 % je Periode"
    assert ev.describe(1.0, None) == "liegt ρ bei 1: die Schwingung klingt weder ab noch wächst"
    assert "praktisch nicht ab" in ev.describe(0.9999999, None) and "4 Millionen Perioden" in ev.describe(0.9999999, None)


# --------------------------------------------------------------------------------------------------------------------------------
# Kernfrage 1
# --------------------------------------------------------------------------------------------------------------------------------


def test_main_verdict_stable_with_pipeline_unstable_without():
    kind, text = ev.verdict_main(2, 1.2, 0.55, 0.735, 1.0954, 16, None)
    assert kind == "success"
    assert text.startswith("L = 2, α = 1,20: ohne Leitung (β = 0) ist ρ = 1,095: die Schwingung wächst auf lange Sicht um 9,5 % je Periode; ")
    assert "mit β = 0,55 ist ρ = 0,735, der Stoß klingt in 16 Perioden auf 1 % ab" in text and "ohne Meldeverzug und ohne Klemmung" in text


def test_main_verdict_unstable_with_pipeline_is_a_warning():
    kind, text = ev.verdict_main(5, 2.2, 0.8, 1.15, 1.4, None, None)
    assert kind == "warning" and "mit β = 0,80 ist ρ = 1,150: die Schwingung wächst auf lange Sicht um 15,0 % je Periode" in text


def test_main_verdict_on_the_unit_circle_is_info():
    kind, text = ev.verdict_main(3, 2.0, 0.5, 1.0 + 3e-10, 1.5, None, None)
    assert kind == "info" and "mit β = 0,50 liegt ρ bei 1: die Schwingung klingt weder ab noch wächst" in text


def test_main_verdict_without_pipeline_says_so_and_has_no_comparison():
    kind, text = ev.verdict_main(12, 0.12, 0.0, 0.992, 0.992, 585, 585)
    assert kind == "success" and "ohne Leitung (β = 0) ist ρ = 0,992, der Stoß klingt in 585 Perioden auf 1 % ab" in text
    assert "mit β" not in text and "Setzen Sie β auf einen Wert über 0" in text
    assert ev.verdict_main(12, 0.2, 0.0, 1.05, 1.05, None, None)[0] == "warning"


# --------------------------------------------------------------------------------------------------------------------------------
# Grenzkarte
# --------------------------------------------------------------------------------------------------------------------------------


def test_grenzkarte_verdict_counts_beta_half_and_names_parity():
    rows = [{"L": 3, "a05": 2.0, "a08": 2.0}, {"L": 4, "a05": 2.0 + 1e-9, "a08": 2.3946}, {"L": 6, "a05": 1.9, "a08": 2.168}]
    kind, text = ev.verdict_grenzkarte(rows)
    assert kind == "info" and "α_krit = 2 bei 2 von 3 gewählten L" in text
    assert "ungerades L (L = 3: 2,000)" in text and "gerades L (L = 4: 2,395, L = 6: 2,168)" in text and "nicht erklärt" in text


def test_grenzkarte_verdict_with_only_odd_or_only_even_L():
    _k, text = ev.verdict_grenzkarte([{"L": 5, "a05": 2.0, "a08": 2.0}])
    assert "gerades L (keines gewählt)" in text and "bei 1 von 1 gewählten L" in text
    _k, text = ev.verdict_grenzkarte([{"L": 2, "a05": 2.0, "a08": 3.3333}])
    assert "ungerades L (keines gewählt)" in text and "L = 2: 3,333" in text


# --------------------------------------------------------------------------------------------------------------------------------
# β = 1, Empfindlichkeit
# --------------------------------------------------------------------------------------------------------------------------------


def test_beta1_verdict_numbers():
    kind, text = ev.verdict_beta1(12, 0.032, 84, 0.85, 0.15, 14)
    assert kind == "success"
    assert "α* = 0,032 (84 Perioden bis 1 %)" in text and "Mit β = 1 und α = 0,85 ist ρ = 0,150, der Stoß klingt in 14 Perioden auf 1 % ab" in text
    assert "ρ = |1 − α| = 0,150 für jedes L" in text
    assert ev.verdict_beta1(12, 0.032, None, 1.95, 0.95, None)[0] == "success"            # 0,95 < 1: stabil
    assert ev.verdict_beta1(12, 0.032, 84, 2.1, 1.1, None)[0] == "warning"


def test_sensitivity_verdict_beta_one_fastest_and_slower_neighbour():
    kind, text = ev.verdict_sensitivity(12, 0.8, {1.0: 14, 0.99: 24, 0.9: 36, 1.1: 26})
    assert kind == "info" and "β = 1: 14 Perioden, β = 0,99: 24 Perioden, β = 0,9: 36 Perioden, β = 1,1: 26 Perioden" in text
    assert "β = 1 ist unter diesen vier die schnellste Wahl" in text and "(β = 0,99) verlängert das auf 24 Perioden" in text


def test_sensitivity_verdict_tie_with_the_neighbour_does_not_claim_a_loss():
    _k, text = ev.verdict_sensitivity(3, 0.8, {1.0: 5, 0.99: 5, 0.9: 7, 1.1: 8})
    assert "schnellste Wahl" in text and "ändert die Abklingzeit hier nicht" in text and "verlängert" not in text


def test_sensitivity_verdict_other_beta_is_faster():
    _k, text = ev.verdict_sensitivity(3, 0.45, {1.0: 10, 0.99: 10, 0.9: 8, 1.1: 12})
    assert "nicht die schnellste" in text and "schneller: β = 0,90 mit 8 Perioden" in text


def test_sensitivity_verdict_without_any_decay_is_a_warning():
    kind, text = ev.verdict_sensitivity(4, 2.2, {1.0: None, 0.99: None, 0.9: None, 1.1: None})
    assert kind == "warning" and "kein Abklingen" in text and "bei keinem der vier β" in text
    assert ev.verdict_sensitivity(4, 2.2, {1.0: None, 0.99: None, 0.9: 36, 1.1: None})[1].count("kein Abklingen") == 3


# --------------------------------------------------------------------------------------------------------------------------------
# Drift
# --------------------------------------------------------------------------------------------------------------------------------


def test_drift_verdict_below_the_target():
    kind, text = ev.verdict_drift(4, 1.0, 3, 10.0, 10.0, -10.0, 13, 0.6)
    assert kind == "warning"
    assert "Geschätzte Lieferzeit 3 statt 4, β = 1,00" in text and "1,00 Perioden Bedarf unter dem Soll (−10,0 Stück bei d = 10); er stellt sich in 13 Perioden auf 1 % ein" in text
    assert "geschätzter Mittelbedarf" not in text and "In diesem Modell hängt der Fehler nicht an α (bei ρ unter 1)" in text


def test_drift_verdict_above_the_target_and_wrong_mean_demand():
    _k, text = ev.verdict_drift(4, 0.6, 5, 10.0, 7.0, 6.0, None, 0.5)
    assert "0,60 Perioden Bedarf über dem Soll (6,0 Stück bei d = 10)" in text and "geschätzter Mittelbedarf 7,0 statt 10,0" in text
    assert "stellt sich" not in text                                                       # keine Zeit ohne t_eq


def test_drift_verdict_zero_error_is_a_success():
    kind, text = ev.verdict_drift(4, 0.0, 3, 10.0, 10.0, 0.0, None, 0.9)
    assert kind == "success" and "kein Dauerfehler" in text
    assert ev.verdict_drift(4, 1.0, 4, 10.0, 10.0, -0.0, None, 0.6)[0] == "success"


def test_drift_verdict_unstable_has_no_equilibrium():
    kind, text = ev.verdict_drift(5, 0.8, 4, 10.0, 10.0, -8.0, None, 1.2)
    assert kind == "warning" and "kein Gleichgewicht" in text and "−0,80 Perioden Bedarf" in text


# --------------------------------------------------------------------------------------------------------------------------------
# Preis der Ruhe
# --------------------------------------------------------------------------------------------------------------------------------


def test_tradeoff_verdict_names_the_smallest_per_quantity():
    rows = {0.0: (1.042, 5.147), 0.55: (0.4, 3.407), 1.0: (0.2903, 3.434)}
    kind, text = ev.verdict_tradeoff(3, 0.45, rows)
    assert kind == "info"
    assert "β = 0,00: 1,042 und 5,15; β = 0,55: 0,400 und 3,41; β = 1,00: 0,290 und 3,43" in text
    assert text.endswith("Kleinste Bestellstreuung bei β = 1,00, kleinste Bestandsstreuung bei β = 0,55.")


def test_tradeoff_verdict_with_unstable_entries_and_with_none_stable():
    inf = math.inf
    _k, text = ev.verdict_tradeoff(12, 0.45, {0.0: (inf, inf), 0.55: (0.699, 19.54), 1.0: (0.29, 12.43)})
    assert "β = 0,00: instabil und instabil" in text and "kleinste Bestandsstreuung bei β = 1,00" in text
    kind, text = ev.verdict_tradeoff(12, 0.9, {0.0: (inf, inf), 1.0: (inf, inf)})
    assert kind == "warning" and "bei keinem der β der Tabelle" in text
