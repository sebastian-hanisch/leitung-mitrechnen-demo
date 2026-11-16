# Die Leitung mitrechnen – was die Bestellregel von der Ware unterwegs weiß – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-leitung-mitrechnen-demo.streamlit.app/)**

Stück 2 der Linie **Rückkopplung und Verzögerung** in der „Konzepte“-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“. Vorgänger ist das Lager ganz ohne Leitung, [`lager-lieferzeit-demo`](https://github.com/sebastian-hanisch/lager-lieferzeit-demo): Dort braucht die beste Einstellung schon bei L = 3 nur α = 0,148 und bei L = 12 nur 0,032,
und selbst sie klingt erst nach dem 5,5- bis 7,2fachen der Lieferzeit ab (Schwelle 1 %, L = 2 bis 24; Zahlen unten). Dieses Stück **rechnet die Ware unterwegs mit** (Kante `fix`) und fragt, was das bringt, was es kostet und wie genau die Rechnung stimmen muss.
Weiter geht es mit [`meldeverzug-demo`](https://github.com/sebastian-hanisch/meldeverzug-demo) (Meldeverzug), [`streuende-lieferzeit-demo`](https://github.com/sebastian-hanisch/streuende-lieferzeit-demo) (streuende Lieferzeit) und [`frequenzgang-lager-demo`](https://github.com/sebastian-hanisch/frequenzgang-lager-demo) (Rauschen und Frequenz);
das Lager mit Klemmung und Chaos steht in [`lager-chaos-demo`](https://github.com/sebastian-hanisch/lager-chaos-demo), die stetige Verzögerung in [`verzoegerung-hopf-demo`](https://github.com/sebastian-hanisch/verzoegerung-hopf-demo).
Alle Formeln und Befunde sind **eigene Herleitung und Messung in diesem Modell**; die Fachliteratur zu Regeln mit Leitungsrückführung ist nicht gelesen, eine Neuheit wird nicht behauptet.

## Warum dieses Problem

Ein Lager bestellt jede Periode nach einer Regel nach, und die Bestellung kommt erst nach L Perioden an. Ohne Leitung (β = 0) sieht die Regel nur den Bestand und bestellt weiter nach, obwohl die Ware noch unterwegs ist: das Lager schwingt auf, sobald α die Grenze 2·sin(π/(4L − 2)) überschreitet (bei L = 12: 0,136).
Zählt die Regel die Ware unterwegs mit dem Gewicht β mit, ändert sich die Stabilität von Grund auf. Die Demo fragt: **Wie viel bringt das, wo ist der Haken, und wie genau muss β stimmen?** Der Haken hat drei Teile, die gemessen sind: Oberhalb von β = ½ hängt die Grenze an der Parität von L;
die Spitze bei β = 1 ist scharf (L = 12, α = 0,8: 14 Perioden bei β = 1, 24 bei β = 0,99); und eine falsch geschätzte Lieferzeit lässt einen Dauerfehler von −β·d·(L − L̂) im Bestand zurück.

## Modell

Alle Größen sind Abweichungen vom Gleichgewicht; Zähleinheit ist die Periode. i Bestand minus Soll, u Bestellung minus mittlerer Bedarf d, e Bedarfsabweichung, L Lieferzeit, α Anpassung, β Leitungsgewicht, ΔW = d(L − 1) − W* der Fehler der Soll-Leitung W* = d̂(L̂ − 1) (0, wenn die Soll-Leitung stimmt).

- **i_{t+1} = i_t + u_{t+1−L} − e_t**
- **u_t = −α · (i_t + β · (u_{t−1} + … + u_{t−L+1} + ΔW))**
- Meldeverzug 0, keine Klemmung (Bestellungen dürfen unter den Mittelbedarf fallen), konstanter Mittelbedarf, feste Lieferzeit.
- **Charakteristisches Polynom:** P(z) = z^L − (1 − αβ)·z^(L−1) + α(1 − β); stabil heißt: alle Wurzeln liegen im Einheitskreis, der **Spektralradius ρ** ist kleiner als 1. P(1) = α (der Bestand verliert die Stabilität nie über z = 1) und P(−1) = (−1)^L·(2 − αβ) + α(1 − β), bei ungeradem L also α − 2.
- **Abklingzeit:** Bedarfsstoß e₀ = 1, danach nichts mehr; die letzte Periode t, in der |i_t| noch über 1 % von max|i| liegt. Die Schätzung ln 0,01 / ln ρ steht nur als Vergleichswert daneben.
- **ρ gleich 1** (|ρ − 1| ≤ 10⁻⁹) heißt in der App: die Schwingung klingt weder ab noch wächst; die Rundung entscheidet dort über Bruchteile von 10⁻⁹.
- Die Formeln stehen im Expander 📐 der App; die App zeigt L, α, β, den Schätzfehler L̂ − L, den geschätzten Mittelbedarf, das Bedarfsrauschen und den Zufalls-Seed als Regler. **Mittelbedarf d = 10** ist fest (nur eine Skala).

## Methodik

Jede Größe hat mindestens zwei Rechenwege; die Tests vergleichen sie.

| Größe | Weg im Kern | Zweiter Weg (Test) |
|---|---|---|
| Spektralradius ρ | Wurzeln von P (`numpy.roots`; die (L−1)-fache Wurzel 0 bei β = 1 ist exakt) | Eigenwerte der Zustandsmatrix; Begleitmatrix von P gestapelt (`rho_batch`) |
| Grenze α_krit | Wurzeln auf dem Einheitskreis: α(θ) = −z^(L−1)(z − 1) / (β z^(L−1) + 1 − β) für z = e^(iθ), Nullstellen von Im α, dann Probe knapp dahinter; **kein Gitter in α, keine Obergrenze** | Gitter in α mit Bisektion auf dem Spektralradius; für L = 2 die Jury-Bedingungen geschlossen |
| Stoßantwort | direkte Rekursion mit fortgeschriebener Leitungssumme | Verzögerungsliste mit expliziter Summe; Filter I(z) = −E(z)·N(z)/P(z) (`scipy.signal.lfilter`); exakte Brüche (`fractions`) |
| Abklingzeit | Filter über einen Horizont bis ρ^t unter 10⁻⁶ (höchstens 4 Millionen Perioden) | direkte Rekursion; Brüche für das Handbeispiel |
| Dauerfehler | Formel i* = −β·ΔW | Gauß mit Brüchen auf (I − A)x = b; Simulation; Schrittfilter; Simulation mit Bedarfsrauschen |
| Streuung bei weißem Bedarf | Lyapunov-Gleichung (`scipy.linalg.solve_discrete_lyapunov`) | Simulation über das Filter (Band); Formel bei β = 1 |

Zufall nur über den eigenen **SplitMix64** (`lmr_rng.py`, vektorisiert gegen die Klasse geprüft, Normalwerte nach Box-Muller). **Nichts ist vorgerechnet:** jede Eingabe der App rechnet live. Gemessen auf dem Entwicklungsrechner (nicht festgeschrieben): die Grenzkarte mit 6 Lieferzeiten 0,6 Sekunden, jeder andere Abschnitt unter 0,15 Sekunden; ein App-Lauf nach einem Preset-Klick dauert im AppTest etwa 0,1 Sekunden, der erste Lauf etwa 2 Sekunden.

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen gelten in diesem Modell (Meldeverzug 0, ohne Klemmung) bei den genannten Parametern; Zählungen sind „k von n“ über die genannten Zellen. Die Tests prüfen Zählungen und Bänder, kein Zahlenwert hängt an der letzten Stelle.
Presets und Voreinstellungen sind nie Zweierpotenzen, außer β = ½ als Satz, β = 1 als Satz und gemessene Spitze, und α = 1 nur in Brüchen oder mit α = 1 + 10⁻⁹; der Regler lässt sie zu, und die Messtabellen enthalten β = 2, −1, −2 und −0,25.

| Frage | Befund | Test |
|---|---|---|
| Stimmen die Rechenwege? | Koeffizienten von P gegen das charakteristische Polynom der Zustandsmatrix: größte Abweichung 3,0·10⁻¹⁴ in 275 Fällen (L = 2 bis 12). P(1) = α und die Formel für P(−1): 1,8·10⁻¹⁵ und 8,9·10⁻¹⁶ in je 2000 Zufallsfällen. Spektralradius aus den Wurzeln gegen Eigenwerte: größte Abweichung 1,9·10⁻¹⁴ in 1680 Fällen (L = 2 bis 8, β von −1,5 bis 2,5, α von 0,054 bis 2,97) | `test_polynomial_coefficients_equal_the_characteristic_polynomial_of_the_matrix`, `test_P_at_plus_and_minus_one_against_the_formulas_2000_random_cases`, `test_spectral_radius_from_polynomial_roots_equals_matrix_eigenvalues` (Schwellen 10⁻¹⁰ bis 10⁻⁹) |
| β = ½: wo liegt die Grenze? | **α_krit = 2 für 10 von 10 Lieferzeiten** (L = 1 bis 8, 12, 13). ρ < 1 auf allen **5970** Gitterpunkten (L = 1 bis 30, 199 Werte von α zwischen 0,0137 und 1,9871); dazu 3000 Zufallspaare bis L = 60 ohne Gitter. Bei α = 2 ist P = z^L + 1: A^(2L) = I exakt in Brüchen für 8 von 8 (L = 2 bis 9, kleinste Potenz 2L), alle Wurzeln auf dem Einheitskreis für 8 von 8 (L = 2, 3, 4, 5, 7, 8, 12, 25). β = ½ ± 10⁻⁹ ändert die Grenze für L = 2 bis 8 um weniger als 10⁻³ | `test_beta_half_limit_is_2_for_10_of_10_L`, `test_beta_half_is_stable_on_the_whole_grid_30_by_199`, `test_beta_half_alpha_2_matrix_order_is_2L_in_8_of_8_and_roots_are_on_the_circle_in_8_of_8`, `test_beta_half_is_stable_for_every_alpha_below_2_3000_random_pairs_up_to_L_60` |
| Unterhalb von ½? | Die Grenze steigt mit β, und je größer L, desto steiler: β = 0,49: **1,889 (L = 3), 1,612 (L = 6), 1,149 (L = 12), 0,697 (L = 24)**; β = 0,495 bei L = 24: 0,920; β = 0,5: 2,000 | `test_below_beta_half_the_limit_rises_and_is_steeper_for_large_L` |
| Parität oberhalb von ½ | β = 0,6 und 0,8: **ungerades L: α_krit = 2,000 in 38 von 38 Zellen** (L = 3, 5, …, 39), **gerades L: über 2 in 40 von 40 Zellen** (L = 2, 4, …, 40), mit L fallend: β = 0,8: 3,333 (L = 2), 2,395 (4), 2,168 (6), 2,044 (12), 2,004 (40). Für L = 2 ist die Grenze exakt min(1/(1 − β), 2/(2β − 1)) (Jury): 14 von 14 Werten von β zwischen −2 und 3 auf 10⁻⁸, Maximum 4 bei β = ¾. Gegenprobe: ungerades L bis 39, 2000 zufällige Paare (L, β in [½, 1]) mit je 12 zufälligen α ohne Gitter: stabil unter 2, instabil knapp über 2. **Warum sich gerades und ungerades L trennen, ist nicht erklärt** (gemessen: bei ungeradem L ist P(−1) = α − 2) | `test_parity_odd_L_limit_is_2_in_19_of_19_even_L_above_2_in_20_of_20`, `test_parity_values_at_beta_08`, `test_L2_limit_equals_the_jury_formula_in_14_of_14`, `test_odd_L_has_the_limit_2_for_beta_in_half_to_one_2000_random_pairs_without_grid` |
| β unter 0 | α_krit fällt unter den Wert bei β = 0 und mit \|β\|: L = 3: 0,618 (β = 0), 0,469 (−0,25), 0,274 (−1), 0,177 (−2); L = 12: 0,136 (β = 0), 0,054 (−1). **Kein stabiles Stück oberhalb der Grenze in 140 von 140 Zellen** (14 Werte von β, 10 Werte von L; je 4000 Punkte bis α = 8); Gegenprobe 5000 zufällige α (500 Paare) | `test_beta_below_zero_the_limit_falls_with_abs_beta_and_no_stable_piece_above_in_140_of_140`, `test_beta_negative_has_no_stable_piece_above_the_limit_5000_random_alphas` |
| β über 1 | **Gerades L: α_krit = 2/(2β − 1) in 25 von 25 Zellen** (L = 2, 4, 6, 8, 12; β = 1,1, 1,25, 1,5, 2, 3), unter 2. **Ungerades L liegt über dieser Kurve 2/(2β − 1) in 12 von 12 Zellen** (L = 3, 5, 7, 13; β = 1,1, 1,5, 3): L = 3: 2,000 (β = 1,1), 1,281 (1,5), 0,432 (3); L = 13: 1,747, 1,013, 0,401; L = 7, β = 1,1: 1,914. Eine geschlossene Formel für den ungeraden Fall ist nicht hergeleitet. Für L = 1 ist ρ = \|1 − α\| für jedes β (größte Abweichung unter 10⁻¹² auf 750 Paaren) | `test_beta_above_one_even_L_limit_is_2_over_2beta_minus_1_in_25_of_25_odd_L_above_it_in_12_of_12`, `test_L1_has_no_beta_dependence` |
| Wie schnell klingt es ohne Leitung ab, wie mit? | Beste Einstellung ohne Leitung (Doppelwurzel bei z* = (L−1)/L, α* = ((L−1)/L)^(L−1)/L, exakt in Brüchen für 29 von 29, L = 2 bis 30; der kleinste Spektralradius auf einem Gitter von 2000 Werten von α ist dort 1 − 1/L, geprüft für L = 2, 3, 6, 12): Abklingzeit **11 (L = 2), 18 (3), 40 (6), 84 (12), 172 (24)** Perioden, das 5,5- bis 7,2fache von L. Mit β = 1 und α = 1 (gerechnet mit α = 1 + 10⁻⁹) genau **L Perioden in 14 von 14 Lieferzeiten** (L = 2 bis 12, 16, 20, 24); in Brüchen ist der Bestand bei α = 1 für L Perioden −1 und ab t = L + 1 exakt 0 (L = 1 bis 7) | `test_double_root_without_pipeline_exact_for_L_2_to_30`, `test_best_setting_without_pipeline_decay_11_18_40_84_172_and_with_beta_1_exactly_L`, `test_beta_one_polynomial_and_deadbeat_in_fractions` |
| Handbeispiel (L = 2, α = 1,2) | Ohne Leitung ρ = 1,095 (+9,5 % je Periode); mit β = 0,55 **ρ = 0,735, Abklingzeit 16 Perioden**; mit β = ½ ρ = 0,775 und 19 Perioden (Handwert in Brüchen: i₁ bis i₈ = −1, −1, 1/5, 17/25, 19/125, −217/625, −719/3125, 1817/15625) | `test_hand_example_L2_alpha_6_5_beta_11_20`, `test_hand_example_beta_half_alpha_6_5_stoss_response_in_fractions`, `test_hand_example_decay_16_with_beta_055_19_with_beta_half_and_growth_without_pipeline` |
| Spitze bei β = 1 | Abklingzeit bei α = 0,8 für β = 1 / 0,99 / 0,9 / 1,1: **L = 3: 5 / 5 / 7 / 8; L = 6: 8 / 12 / 18 / 14; L = 12: 14 / 24 / 36 / 26** Perioden (gleiche Werte bei α·(1 ± 10⁻⁶)). ρ bei L = 12: 0,200 (β = 1), 0,688 (0,99), 0,837 (0,9). **Nicht überall ist β = 1 am schnellsten:** L = 3, α = 0,45: 8 Perioden bei β = 0,9 gegen 10 bei β = 1 | `test_peak_at_beta_one_gets_sharper_with_L_alpha_08`, `test_beta_one_is_not_always_the_fastest_L3_alpha_045_beta_09_beats_beta_1` |
| β = ½ nahe α = 2 | „Stabil“ heißt hier „klingt kaum ab“: α = 1,99, β = ½: **5842 Perioden (L = 3, ρ = 0,999164), 43 208 (L = 6), 338 186 (L = 12)**, 1858 (L = 2). Die Schätzung ln 0,01 / ln ρ liegt in **12 von 12 Zellen mit Abklingzeit über 100** (L = 2, 3, 6, 12; α = 1,01, 1,51, 1,91, 1,99) um 1,0 bis 5,9 % darunter (die Zellen mit kürzerer Abklingzeit sind nicht gezählt; α = 1,01, L = 2: 10,1 % darunter bei 15 Perioden) | `test_beta_half_near_alpha_2_decay_is_very_long_and_the_estimate_is_below_in_12_of_12` |
| Presets der App | Ohne Leitung (L = 12, α = 0,12): ρ = 0,992, **585 Perioden**. β = 1, α = 0,85, L = 12: ρ = 0,150, **14 Perioden**. L = 4, α = 2,2, β = 0,8: ρ = 0,954, 96 Perioden (ohne Leitung ρ = 1,402; dieselbe Einstellung mit L = 5: ρ = 1,08, instabil). L = 6, α = 0,4, β = 0,3: ρ = 0,972, 163 Perioden (ohne Leitung ρ = 1,046) | `test_preset_numbers_in_the_readme`, `test_each_preset_shows_the_expected_main_verdict_line` |
| Dauerfehler bei falscher Lieferzeit | **i\* = −β·ΔW exakt in 720 von 720 Fällen** (Gauß mit Brüchen; L = 1 bis 6, α ∈ {3/10, 3/5, 9/10, 7/5}, β ∈ {−1/3, 0, 1/3, 3/5, 1, 3/2}, fünf Soll-Leitungen darunter „Mittelbedarf 7 statt 10“; alle gespeicherten Bestellungen 0). Beispiel L = 4, d = 10, in Perioden Bedarf für L̂ = 2 bis 6: β = 1: −2, −1, 0, 1, 2; β = 3/5: −1,2, −0,6, 0, 0,6, 1,2; β = 0: überall 0. Simulation gegen den exakten Wert: **9 von 9 stabilen Fällen**, größte Abweichung 8,9·10⁻¹⁵ nach 20 000 Perioden. Mit Bedarfsrauschen (Standardabweichung 0,2 d, 400 000 Perioden, Batch-Mittel 40 × 10 000): drei Zufalls-Seeds (16, 17, 18) mit je vier Einstellungen, Abstand des Mittels zum exakten Wert **höchstens 0,82 Standardfehler** (die vier Einstellungen eines Seeds teilen sich die Rauschfolge: es sind drei unabhängige Läufe). Zeit bis auf 1 % von \|i\*\| (L = 4, α = 0,4, L̂ = 3): 19 Perioden bei β = 0,55, 13 bei β = 1 | `test_equilibrium_in_fractions_equals_minus_beta_dW_in_720_cases`, `test_drift_examples_table_L4_beta_in_periods_of_demand`, `test_drift_simulation_is_exact_after_20000_periods_in_9_of_9`, `test_drift_noise_means_within_three_standard_errors_in_12_of_12_four_settings_three_seeds`, `test_drift_settling_time_19_and_13_periods_L4_alpha_04` |
| Preis der Ruhe (weißer Bedarf, L = 3) | var(u)/var(e) und var(i)/var(e) für β = 0 / 0,55 / 1: **α = 0,45: 1,042 und 5,147 / 0,400 und 3,407 / 0,290 und 3,434**; α = 0,3: 0,350 und 3,889 / 0,222 und 3,763 / 0,176 und 3,961; α = 0,6: 13,286 und 36,905 / 0,636 und 3,373 / 0,429 und 3,190; α = 0,8 ohne Leitung instabil. Bei α = 0,45 ist die Bestandsstreuung unter den drei β der Tabelle bei β = 0,55 am kleinsten (3,407 gegen 3,434 bei β = 1); das Minimum über alle β liegt bei etwa β = 0,72. Bei β = 1 gilt var(u)/var(e) = α/(2 − α) und var(i)/var(e) = (L − 1) + 1/(α(2 − α)): größte Abweichung 8,2·10⁻¹⁴ (L = 1, 3, 8; 19 Werte von α zwischen 0,113 und 1,887); Minimum der Bestandsstreuung L bei α = 1. Simulation (400 000 Perioden, Zufalls-Seed 5, α = 0,45) liegt höchstens 1,4 % neben der Lyapunov-Lösung (Messung dieser README, der Test prüft das Band 4 %) | `test_trade_off_table_L3_values`, `test_variance_at_beta_one_equals_the_closed_form_for_L_1_3_8`, `test_lyapunov_variance_equals_simulation_within_4_percent`, `test_stock_variance_at_beta_one_is_at_least_L_with_the_minimum_at_alpha_one` |

Ergebnis je Preset (Verdikt der App, Meldeverzug 0): Von Hand: stabil, 16 Perioden; Ohne Leitung: stabil, 585 Perioden; β = 1: stabil, 14 Perioden; Gerade und ungerade L: stabil, 96 Perioden; Leitung zu wenig: stabil, 163 Perioden; Drift: der Bestand bleibt 1,00 Perioden Bedarf unter dem Soll, in 13 Perioden auf 1 % (`test_each_preset_shows_the_expected_main_verdict_line`, `test_drift_preset_verdict_and_every_other_preset_has_no_dauerfehler_at_correct_estimate`).

### Befunde und Korrekturen gegenüber dem Plan

- **β = 1, α = 0,9 ist eine Rundungskante.** Der Plan nannte „α = 0,9: ρ = 0,100, 14 Perioden“. Dort ist ρ² = 0,01, genau die Schwelle von 1 %; die Gleitkommadarstellung von 1 − 0,9 entscheidet, ob i_{L+2} über oder unter der Schwelle liegt (hier 13 Perioden, in der Vormessung 14). Das Preset „β = 1, fast Deadbeat“ und der Regler im Abschnitt β = 1 starten deshalb mit **α = 0,85** (ρ² = 0,0225, ρ³ = 0,0034: robust 14 Perioden; auch α = 0,8: 14).
- **Zweierpotenzen in Presets.** β = ½ (Plan: Preset „Von Hand“) steht als **β = 0,55**, β = ¼ (Plan: Preset „Leitung zu wenig“) als **β = 0,3**; der Zielkonflikt wird bei α = 0,45 statt α = 0,5 gemessen. Die Zahlen des Plans bei β = ½ (ρ = 0,775, 19 Perioden; Handwert in Brüchen) und α = 0,5 (V1b: 1,667 / 0,500 / 0,333) sind nicht Teil der Messung dieser Demo; die neuen Zahlen stehen oben. β = 1 bleibt Preset: dort liegt die gemessene Spitze, und α ist keine Zweierpotenz.
- **Zeit bis zum Gleichgewicht.** Die Vormessung nannte 12 Perioden (L = 4, α = 0,4, β = 1, L̂ = 3); hier sind es 13 (erste Periode, ab der |i − i\*| dauerhaft höchstens 1 % von |i\*| ist; die Zählweise der Vormessung ist nicht identisch). Die neue Messung gilt.
- **Dauerfehler: 720 statt 2880.** Die Vormessung zählte 2880 Fälle mit Meldeverzug s = 0 bis 3; diese Demo hat nur s = 0. Die Aussage „unabhängig von s“ ist hier nicht gemessen (Plan Stück 3).
- **Rauschen im Dauerfehler.** Die Vormessung nutzte den numpy-Generator (ein Seed, höchstens 1,56 Standardfehler); hier SplitMix64 mit drei Zufalls-Seeds (höchstens 0,82 Standardfehler).
- **Abklingzeit nahe α = 2:** Der Plan nannte „1,1 bis 6,2 % darunter“ bei α = 1, 1,5, 1,9, 1,99; da α = 1 eine Zweierpotenz und α = 1,5 ein Dualbruch (3/2) ist, wird bei α = 1,01, 1,51, 1,91, 1,99 gemessen: 1,0 bis 5,9 % darunter, 12 von 12 Zellen mit Abklingzeit über 100.
- **Werte unterhalb von ½:** Der Plan (aus der Vormessung mit Gitter in α) nannte 1,890, 1,613, 1,150 bei β = 0,49; das Kurvenverfahren ohne Gitter gibt 1,889, 1,612, 1,149 (Unterschied ein Tausendstel, durch das α-Gitter der Vormessung erklärt; die Gittermethode stimmt mit dem Kurvenverfahren überein: größte Abweichung 2,0·10⁻¹¹ in 40 von 40 Zufallsfällen mit Grenze unter 8; der Test verlangt 10⁻⁶ in mindestens 35 Fällen).
- **Reglerbereich α:** bis 4 statt bis 2,5 des Plans, damit L = 2 (Grenze bis 4 bei β = ¾) und β > 1 erreichbar sind.
- **Parität:** Die Aussage der Gliederung „gerades L, β > ½: Grenze über 2“ gilt nur für ½ < β < 1; für β > 1 liegt die Grenze bei geradem L unter 2 (25 von 25 Zellen, oben).
- **Vorlage der Abklingzeit im Handbeispiel:** die Abklingzeit ist die letzte Periode über der Schwelle (19 bei β = ½), nicht die erste darunter; beide Lesarten unterscheiden sich um eine Periode.

## Quellen

| Quelle | Stufe | Was hier verwendet wird |
|---|---|---|
| Hamiche u. a., arXiv 1903.04256 (CoDIT 2019) | im Wortlaut gelesen (2026-11-15) | nur als Umgebung: ein Smith-Prädiktor für ein Lager ohne Stabilitätsbeweis, ohne falsch geschätzte Lieferzeit; kein Wert dieser Demo stammt daraus |
| Yü, Cai, Gao, arXiv 2208.04459 | nur der Anfang gelesen | nur genannt: die Formel dort gehört zu einem anderen Modell und wird nicht nachgebaut |
| Sterman 1989, Management Science 35(3) | nur das Abstract gelesen | ein Gegenüber ohne Parameter: Anker-und-Anpassung beschreibt Entscheidungen im Beer Game ökonometrisch gut; Fehlwahrnehmung der Rückkopplung, vor allem Unempfindlichkeit gegen die Rückwirkung eigener Entscheidungen. **Keine Gleichsetzung mit β** |
| Disney/Towill u. a., Dejonckheere u. a. 2003, Croson u. a. 2014 | **nicht gelesen** (nur Suchtreffer, teils 403) | nichts; ob β = ½, die Parität oder der Dauerfehler dort stehen, ist offen |

## Ehrliche Grenzen

- **Ein Modell:** Meldeverzug 0, feste Lieferzeit, ohne Klemmung, konstanter Mittelbedarf. Schon eine Periode Meldeverzug hebt die Grenze 2 bei β = 1 nicht mehr (Vormessung für L = 3 und 6: rund 0,7); das ist in dieser Demo **nicht gerechnet** und gehört zu [`meldeverzug-demo`](https://github.com/sebastian-hanisch/meldeverzug-demo).
- **Erstes Überschreiten, nicht „danach immer instabil“:** α_krit ist das kleinste α mit einer Wurzel außerhalb des Einheitskreises. Dass oberhalb nichts wieder stabil wird, ist nur für β unter 0 gemessen (140 von 140 Zellen, 5000 Zufalls-α); für die übrigen β nicht behauptet.
- **Parität nur gemessen:** die Trennung von geradem und ungeradem L ist nicht erklärt; für gerades L zwischen ½ und 1 und für ungerades L über 1 gibt es keine geschlossene Formel, nur Zahlen (und für ungerades L zwischen ½ und 1 die Grenze 2 aus P(−1) = α − 2).
- **ρ = 1:** bei α = 2, β = ½ klingt die Schwingung nicht ab und wächst nicht; nahe daran heißt „stabil“ nur „klingt kaum ab“ (338 186 Perioden bei L = 12, α = 1,99). Die Schätzung ln 0,01 / ln ρ ist ein Vergleichswert, kein Gesetz; für die Doppelwurzel ohne Leitung klingt der Stoß langsamer ab als die Schätzung.
- **Abklingzeit ist eine Wahl:** 1 % der Spitze, ein Bedarfsstoß; bei anderen Brüchen oder Stößen anderer Form nicht gemessen. An der Schwelle (ρ² = 0,01 bei β = 1, α = 0,9) entscheidet die Rundung.
- **Dauerfehler:** nur für konstanten Mittelbedarf, einmal geschätzte Soll-Leitung und ohne Klemmung; nachgeführte Schätzer, sich ändernder Mittelbedarf und Klemmung sind nicht gemessen. Der Fehler hängt in diesem Modell nicht an α, solange das Gleichgewicht existiert (ρ unter 1).
- **Rauschen:** weißes Bedarfsrauschen; Saison und Resonanz sind nicht Teil dieser Demo ([`frequenzgang-lager-demo`](https://github.com/sebastian-hanisch/frequenzgang-lager-demo)). Der Zielkonflikt ist nur für L = 3 in der Tabelle der Vormessung geprüft; in der App ist L frei, die Tests prüfen die Zahlen oben und die Formel bei β = 1.
- **Streuende Lieferzeit:** nicht gemessen, ob die Leitungsrechnung dort noch wirkt ([`streuende-lieferzeit-demo`](https://github.com/sebastian-hanisch/streuende-lieferzeit-demo)).
- **Eine Regel, kein Verhalten von Menschen.** Das Preset „Leitung zu wenig“ zeigt nur, was das Modell bei β = 0,3 tut.
- **Plattform:** Zahlen mit sehr langen Läufen (Abklingzeiten über 10 000 Perioden, Simulationen mit 400 000 Perioden) und Gleitkomma-Summen stehen in den Tests als Band oder mit Toleranz; die Zahlen oben sind die des Entwicklungsrechners.
- **Nicht mit echtem Mausklick getestet:** die App ist mit AppTest und im Browser über die Presets geprüft; Plotly-Interaktionen sind gesperrt (keine Auswahl).

## Verifikation

Die Suite hat **163 Tests** (`pytest --collect-only -q tests`); sie läuft in etwa einer halben Minute.

- **Kern (`tests/test_dynamics.py`):** die sieben Handbeispiele in Brüchen (L = 2 mit Jury-Grenzen 1, 2, 4, 2; A⁴ = I; α = 6/5 mit β = 11/20 und β = ½ als Stoßantwort; L = 3 mit A⁶ = I; Parität; β = 1 mit Deadbeat; Drift mit i\* = −10 und der Folge u₀ bis u₃), Einheiten (Polynom, Zustandsmatrix, Rekursion gegen Verzögerungsliste und Filter, Abklingzeit an Spielfolgen, Doppelwurzel, Rauschen).
- **Orakel (`tests/test_oracle_leitung_mitrechnen.py`):** Koeffizienten gegen `numpy.poly` der Zustandsmatrix, P(1) und P(−1), Wurzeln gegen Eigenwerte (1680 Fälle), Stoßantwort auf vier Wegen (Rekursion, Filter, Brüche, Bilanz u_s = i_{s+L} − i_{s+L−1}), Abklingzeit Rekursion gegen Filter, α_krit gegen Gitter und Bisektion (40 Zufallsfälle) und gegen die Jury-Bedingungen (14 Werte), α_krit bei β = 0 gegen 2·sin(π/(4L − 2)), Gleichgewicht in Brüchen (720 Fälle) gegen Simulation und Schrittfilter,
  Streuung gegen Lyapunov und Simulation, Doppelwurzel in Brüchen (L = 2 bis 30), SplitMix64 gegen Referenzwerte, Gegenproben für Absolutaussagen (ungerades L, β unter 0, β = ½) mit zufälligen Werten ohne Gitter.
- **Eigenschaften (`tests/test_eigenschaften_leitung_mitrechnen.py`):** Linearität, Zeitinvarianz, β = 1 gibt ρ = |1 − α| für L = 1 bis 40, L = 1 ohne β, ungerades L: α_krit höchstens 2 (40 Zufallspaare, β von −2 bis 3), gerades L und β über ½: höchstens 2/(2β − 1) (40 Zufallspaare), Dauerfehler linear in β und in L − L̂ und unabhängig von α, ohne Dauerfehler bei richtiger Schätzung oder β = 0, Skalierung d → c·d, P(1) = α, vertauschte Stoßreihenfolge, gleicher Zufalls-Seed gleiche Bahn, Bestandsstreuung bei β = 1 mindestens L.
- **Randfälle (`tests/test_edge_cases.py`):** L = 1, β = 1 exakt (mehrfache Wurzel 0, Eigenwerte nur auf 10⁻⁶), β nahe 1, α = 2 und α_krit nur als Band, L bis 40, β über 1 und unter 0, L̂ ≤ 0, d = 0, Regler am Rand.
- **Claims (`tests/test_claims.py`):** jede Zahl der Tabelle oben als Zählung oder Band (Spalte „Test“), je Preset die erwartete Verdikt-Zeile. **Verdikte (`tests/test_verdicts.py`):** jede Schwelle und jede Zahl im Text an handgesetzten Werten.
- **Presets und Oberfläche:** Presets vollständig, nie α oder β ½ oder ¼ (β = 1 und β = 0 benannt), AppTest-Rauchtest (jeder Button, jeder Regler am Minimum und Maximum, alle zusammen, jedes Preset, leere und volle Auswahl der Grenzkarte, instabiler Fall im Drift-Abschnitt, Permalink), Layout (Schnellstart in der Seitenleiste, aktives Preset, jeder Regler in `SETTING_SPECS`, der Drift-Regler im Hauptbereich), Footer, Laufzeit-Importe.
- **Nicht im Test:** die Rechenzeiten (Messung auf dem Entwicklungsrechner), die Darstellung der Diagramme im Browser, die Zahlen der Vormessung, die diese Demo nicht neu misst (V1b bei α = 0,5, Meldeverzug).

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App: Stoßantwort und Wurzeln, Kernfragen, fünf Experimente (Grenzkarte über β, β = 1, Wie genau muss β stimmen, Drift, Preis der Ruhe), Grenzen, Formeln |
| `lmr_dynamics.py` | Kern: Polynom, Zustandsmatrix, Spektralradius, Grenze (zwei Wege), Stoßantwort, Abklingzeit, Simulation, Dauerfehler, Streuung |
| `lmr_exact.py` | exakte Rechnung in Brüchen (Handbeispiele, Gleichgewicht, kleinste Potenz A^k = I) |
| `lmr_rng.py` | SplitMix64 (Kopie aus `lager-chaos-demo`, dort aus `restaurant-tischmix-demo`) |
| `lmr_evaluation.py` | Auswertungen der Abschnitte und ihre Verdikte (reine Funktionen) |
| `lmr_constants.py` | Regler-Grenzen, feste Annahmen, Farben, Presets |
| `lmr_presets.py` | Permalink (Begrenzen), Presets (Schnellstart in der Seitenleiste, aktives Preset hervorgehoben), Zufalls-Seed-Button |
| `lmr_layout.py` | Fester Kopf (Stylesheet) |
| `lmr_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `requirements.txt` | Laufzeit-Bibliotheken (ungepinnt): streamlit, numpy, scipy, plotly |
| `requirements-dev.txt` | Laufzeit plus pytest |
| `tests/` | Kern, Orakel, Eigenschaften, Randfälle, Claims, Verdikte, Presets, Layout, AppTest, Footer, Laufzeit-Importe |

## Bewusst nicht umgesetzt

- **Meldeverzug, streuende Lieferzeit, Frequenzgang:** gehören zu den Nachbar-Stücken der Linie ([`meldeverzug-demo`](https://github.com/sebastian-hanisch/meldeverzug-demo), [`streuende-lieferzeit-demo`](https://github.com/sebastian-hanisch/streuende-lieferzeit-demo), [`frequenzgang-lager-demo`](https://github.com/sebastian-hanisch/frequenzgang-lager-demo)); hier Meldeverzug 0, feste Lieferzeit, weißer Bedarf.
- **Klemmung, Grenzzyklus, Chaos:** [`lager-chaos-demo`](https://github.com/sebastian-hanisch/lager-chaos-demo) (dort nur β = 0).
- **Doppelwurzel und beste Einstellung ohne Leitung als eigenes Thema:** gehören zu [`lager-lieferzeit-demo`](https://github.com/sebastian-hanisch/lager-lieferzeit-demo); hier nur die Zahlen als Vergleichsbasis.
- **Erklärung der Parität, geschlossene Formel für gerades L zwischen ½ und 1 und ungerades L über 1:** nur gemessen, nicht bewiesen.
- **Nachgeführte Schätzer für Lieferzeit und Mittelbedarf, Reglerentwurf, Parameterschätzung aus Daten:** nicht Teil der Linie.
- **Der Name „Smith-Prädiktor“ und ein Vergleich mit der Fachliteratur:** nicht gelesen; belegt ist nur: bei β = 1 steht die Lieferzeit nicht mehr im Spektralradius (eigene Herleitung).
- **Anwendungen und Optimum** (Kette mit Prognose, Kosten, Bestandspolitiken): [`bullwhip-demo`](https://github.com/sebastian-hanisch/bullwhip-demo), [`lager-und-haendler-demo`](https://github.com/sebastian-hanisch/lager-und-haendler-demo); hier eine Einzelschleife ohne Prognose und Kosten.
- **PDF-Export und Erklärseite „verständlich erklärt“:** nicht eingebaut.

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly, NumPy und SciPy.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zur Reihe: [Rückkopplung und Verzögerung: das Lager](https://sebastianhanisch.net/konzepte-rueckkopplung.html).
