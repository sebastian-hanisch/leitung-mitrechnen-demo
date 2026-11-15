"""Konstanten: Regler-Grenzen, feste Annahmen, Farben und Presets.

Presets nie mit α oder β als Zweierpotenz (½, ¼, 2, ...): dort rechnet der Rechner in Dualbrüchen exakt. Im linearen Modell entsteht daraus
kein Artefakt, aber die Regel gilt portfolioweit (Dynamische Systeme, Chaos im Lager); die Sätze „β = ½“ und „β = 1“ stehen als Sätze in den Texten und werden
in den Tests mit Rundungsrauschen geprüft. β = 1 ist als Preset geblieben: die gemessene Spitze der Abklingzeit liegt genau dort (siehe README)."""

L_MIN, L_MAX, DEFAULT_L = 1, 24, 2
ALPHA_MIN, ALPHA_MAX, DEFAULT_ALPHA = 0.05, 4.0, 1.2
BETA_MIN, BETA_MAX, DEFAULT_BETA = -1.0, 2.0, 0.55
LHAT_DELTA_MIN, LHAT_DELTA_MAX = -3, 3          # geschätzte Lieferzeit minus tatsächliche
D_MEAN = 10.0                                   # mittlerer Bedarf je Periode (nur Skala; Bestand ist in Perioden Bedarf lesbar)
DHAT_MIN, DHAT_MAX = 5.0, 15.0                  # geschätzter Mittelbedarf
NOISE_MIN, NOISE_MAX = 0.0, 0.3                 # Bedarfsrauschen: Standardabweichung in Vielfachen von d
SEED_MAX, DEFAULT_SEED = 999999, 16
DRIFT_T = 120                                   # Perioden, die der Drift-Reiter zeigt
IMPULSE_T_MAX = 400                             # höchstens so viele Perioden im Stoßantwort-Diagramm
GRID_BETAS = tuple(round(-1.0 + 0.02 * k, 2) for k in range(151))   # β von −1 bis 2 für die Grenzkarte (enthält 0,5 als Gitterpunkt)
SENS_BETAS = tuple(round(0.5 + 0.01 * k, 2) for k in range(101))    # β von 0,5 bis 1,5 für die Empfindlichkeit
SENS_L = (3, 6, 12)
BETA1_L = (1, 3, 6, 12)
GRENZKARTE_L_DEFAULT = (3, 4, 12, 13)
TRADEOFF_ALPHAS = (0.3, 0.45, 0.6, 0.8)
TRADEOFF_BETAS = (0.0, 0.55, 1.0)

COLOR_WITH = "#2F6B65"
COLOR_WITHOUT = "#C4553B"
COLOR_NEUTRAL = "#4c78a8"
COLOR_GRID = "rgba(128,128,128,0.35)"
L_COLORS = {1: "#7f7f7f", 2: "#9467bd", 3: "#4c78a8", 4: "#e45756", 5: "#72b7b2", 6: "#54a24b", 7: "#b279a2", 8: "#f58518", 12: "#2F6B65", 13: "#9d755d"}

PRESETS = {
    "Von Hand (L = 2, β = 0,55)": {"L": 2, "alpha": 1.2, "beta": 0.55, "lhat_delta": 0},
    "Ohne Leitung (β = 0)": {"L": 12, "alpha": 0.12, "beta": 0.0, "lhat_delta": 0},
    "β = 1, fast Deadbeat": {"L": 12, "alpha": 0.85, "beta": 1.0, "lhat_delta": 0},
    "Gerade und ungerade L (β = 0,8)": {"L": 4, "alpha": 2.2, "beta": 0.8, "lhat_delta": 0},
    "Leitung zu wenig (β = 0,3)": {"L": 6, "alpha": 0.4, "beta": 0.3, "lhat_delta": 0},
    "Drift (L̂ = L − 1)": {"L": 4, "alpha": 0.4, "beta": 1.0, "lhat_delta": -1},
}
PRESET_HELP = {
    "Von Hand (L = 2, β = 0,55)": "Lieferzeit 2, Anpassung 1,2: ohne Leitung wächst die Schwingung um rund 9,5 % je Periode, mit β = 0,55 klingt sie ab. Das Handbeispiel der Demo, mit β = 0,55 statt ½ (Zweierpotenz-Regel).",
    "Ohne Leitung (β = 0)": "Lieferzeit 12, Anpassung 0,12, Leitung nicht mitgerechnet: knapp unter der Grenze, der Stoß klingt über hundert Perioden ab. Der Ausgangspunkt des Vergleichs.",
    "β = 1, fast Deadbeat": "Lieferzeit 12, Anpassung 0,85, Leitung voll mitgerechnet: die Lieferzeit verschwindet aus der Charakteristik, der Spektralradius ist |1 − α| = 0,15.",
    "Gerade und ungerade L (β = 0,8)": "Lieferzeit 4 (gerade), Anpassung 2,2, β = 0,8: stabil. Mit Lieferzeit 5 (ungerade) und derselben Einstellung wäre die Grenze genau 2: instabil.",
    "Leitung zu wenig (β = 0,3)": "Lieferzeit 6, Anpassung 0,4, nur 30 % der Leitung gezählt: ohne Leitung wäre das instabil, mit β = 0,3 klingt der Stoß langsam ab. Zeigt nur, was das Modell bei wenig Leitung tut.",
    "Drift (L̂ = L − 1)": "Lieferzeit 4, β = 1, die Regel nimmt die Lieferzeit 3 an: der Bestand bleibt dauerhaft eine Periode Bedarf unter dem Soll (im Abschnitt „Drift“).",
}
