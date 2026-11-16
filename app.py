"""Die Leitung mitrechnen – was die Bestellregel von der Ware unterwegs weiß - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Stück 2 der Linie „Rückkopplung und Verzögerung“: das lineare Lager (Meldeverzug 0, ohne Klemmung) mit Leitungsgewicht β.
Aufbau und Regeln stehen im DEMO-PLAYBOOK.md (§2). Lauffähig mit: streamlit run app.py
"""

import streamlit as st

import lmr_constants as C
import lmr_dynamics as D
import lmr_evaluation as ev
from lmr_layout import dunkles_design, kopf_css
from lmr_presets import (
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sidebar_presets,
    sync_query_params,
)
from lmr_visualization import (
    build_beta1_rho,
    build_drift,
    build_grenzkarte,
    build_impulse,
    build_roots,
    build_sensitivity,
    build_stock_compare,
)

st.set_page_config(page_title="Die Leitung mitrechnen – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _analysis(settings):
    return ev.analyse(settings)


@st.cache_data(show_spinner=False)
def _grenzkarte(Ls):
    return ev.grenzkarte_curves(Ls), ev.grenzkarte_rows(Ls)


@st.cache_data(show_spinner=False)
def _beta1_curves(alphas):
    return ev.beta1_rho_curves(alphas)


@st.cache_data(show_spinner=False)
def _sensitivity(L, alpha):
    return ev.sensitivity_curve(L, alpha), ev.sensitivity_cells(L, alpha)


@st.cache_data(show_spinner=False)
def _sens_table(alpha):
    return {L: ev.sensitivity_cells(L, alpha) for L in C.SENS_L}


@st.cache_data(show_spinner=False)
def _drift(L, alpha, beta, lhat_delta, d_hat, s_rel, seed):
    return ev.analyse_drift(L, alpha, beta, lhat_delta, d_hat, s_rel, seed)


@st.cache_data(show_spinner=False)
def _tradeoff(L, alpha):
    return ev.tradeoff_rows(L, alpha), ev.tradeoff_table(L)


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in lmr_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("📦 Die Leitung mitrechnen – was die Bestellregel von der Ware unterwegs weiß")
st.markdown(
    """
Eine Bestellregel für ein Lager mit **Lieferzeit L** kann die **Ware unterwegs** (schon bestellt, noch nicht angekommen) mitzählen, und zwar mit dem **Leitungsgewicht β**: β = 0 ignoriert sie, β = 1 zählt sie voll.
Das Stück rechnet in diesem Modell nach, was das an der Stabilität ändert (**eigene Herleitung und Messung, keine Aussage über Quellen**), was es an Abklingzeit bringt und was es kostet: eine genaue Einstellung von β und ein **Dauerfehler**, wenn die Lieferzeit falsch geschätzt ist.
Vorgänger ist das Lager ganz ohne Leitung ([`lager-lieferzeit-demo`](https://github.com/sebastian-hanisch/lager-lieferzeit-demo)); das Lager mit Klemmung und Chaos steht in [`lager-chaos-demo`](https://github.com/sebastian-hanisch/lager-chaos-demo).

Weiter unten: **❓ Kernfragen**, die Experimente **🔬 Grenzkarte über β**, **🔬 β = 1**, **🔬 Wie genau muss β stimmen?**, **🔬 Drift** und **🔬 Preis der Ruhe**, **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""
)

# erklaerseite:einfach:start (erzeugt von tools/set_demo_explainer.py (Website-Repo) aus tools/content/explainers/ware-unterwegs-erklaert.json, nicht von Hand ändern)
with st.expander("💬 In einfachen Worten"):
    st.markdown(
        """
Stellen Sie sich ein Lager vor, das nachbestellt, obwohl die Ware erst nach einer festen Lieferzeit ankommt. Wer nur auf das Regal schaut, bestellt ein zweites Mal, was schon unterwegs ist, und das Lager kann sich aufschaukeln. Die Demo rechnet in einem einfachen Modell nach, was sich ändert, wenn die Regel die **Ware unterwegs** mit einem einstellbaren **Leitungsgewicht** mitzählt: Bis zu welcher Reaktionsstärke bleibt das Lager ruhig, wie schnell klingt ein Bedarfsstoß ab, und was passiert, wenn die Lieferzeit falsch geschätzt ist? Es ist eine Rechenregel mit fester Lieferzeit, ohne Meldeverzug und ohne Bestellgrenze nach unten, kein Abbild von Menschen. Die Ergebnisse gelten nur in diesem Modell.

[Ausführlich und verständlich erklärt, mit Fachteil](https://sebastianhanisch.net/ware-unterwegs-erklaert.html)
        """
    )
# erklaerseite:einfach:end

with st.expander("So funktioniert die Leitung mitrechnen", expanded=True):
    st.markdown(
        """
1. **Die Regel:** Jede Periode bestellt das Lager so viel mehr oder weniger als den mittleren Bedarf, wie es die Abweichung vom Soll verlangt: `u = −α · (Bestand minus Soll + β · Ware unterwegs minus Soll-Leitung)`. Bestellungen kommen nach L Perioden an.
2. **Handbeispiel (Preset „Von Hand“, L = 2, α = 6/5, β = 11/20 ≈ 0,55):** Ein Bedarfsstoß von 1 lässt den Bestand auf i₁ = −1 fallen, die Bestellung u₀ ist 0. Dann ist u₁ = −α · i₁ = 6/5, der Bestand bleibt bei i₂ = i₁ + u₀ = −1, und mit der Leitung u₂ = −α · (i₂ + β · u₁) = 51/125; i₃ = i₂ + u₁ = 1/5. Die Matrix hat Spur 1 − αβ = 17/50 und Determinante α(1 − β) = 27/50: Spektralradius ρ = √0,54 ≈ 0,735. Ohne Leitung (β = 0) ist die Determinante α = 6/5 und ρ = √1,2 ≈ 1,095: die Schwingung wächst.
3. **Was die Demo zeigt:** ob die Einstellung stabil ist (ρ unter 1: alle Wurzeln des Polynoms P liegen im Einheitskreis), wie schnell ein Stoß abklingt, und wie das von β und von der Parität der Lieferzeit L abhängt. Jede Zahl kommt aus mindestens zwei Rechenwegen (Wurzeln von P, Eigenwerte der Zustandsmatrix, Brüche; siehe README).
        """
    )

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Einstellungen")
    st.markdown("**Lager und Regel**")
    L = st.slider("Lieferzeit L (Perioden)", *bounds("L_slider"), key="L_slider",
                  help="Wie viele Perioden eine Bestellung bis zur Ankunft braucht. Die Parität (gerade oder ungerade) zählt bei β über ½.")
    alpha = st.slider("Anpassung α", *bounds("alpha_slider"), key="alpha_slider", step=0.01,
                      help="Anteil der gemeldeten Abweichung, den die Bestellung je Periode ausgleicht.")
    beta = st.slider("Leitungsgewicht β", *bounds("beta_slider"), key="beta_slider", step=0.01,
                     help="Anteil der Ware unterwegs, den die Regel mitzählt: 0 gar nicht, 1 voll. Werte unter 0 und über 1 sind erlaubt (Experimente).")
    st.caption("Meldeverzug 0 und keine Klemmung in diesem Stück. Das Handbeispiel hat β = 11/20, nicht ½ (Zweierpotenz-Regel der Linie).")

settings = ev.Settings(int(L), float(alpha), float(beta), int(st.session_state["lhat_delta_slider"]))
sync_query_params({"L_slider": int(L), "alpha_slider": float(alpha), "beta_slider": float(beta), "lhat_delta_slider": settings.lhat_delta})

a = _analysis(ev.Settings(settings.L, settings.alpha, settings.beta))

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ergebnis: die Stoßantwort")
st.caption("Ein einmaliger Bedarfsstoß von einer Einheit über dem Mittelbedarf (e₀ = 1), danach nichts mehr. „Abklingzeit“: die letzte Periode, in der der Bestand noch über 1 % seiner Spitze liegt.")
m1, m2, m3, m4 = st.columns(4)
# Das Vorzeichen des Deltas muss ein ASCII-Minus sein: Streamlit erkennt „−“ (U+2212) nicht als negativ und zeigt den Pfeil falsch herum.
m1.metric("Spektralradius ρ mit β", ev.de(a.rho, 4), delta=f"{a.rho - a.rho0:+.4f} gegen β = 0".replace(".", ","), delta_color="inverse")
m2.metric("Abklingzeit mit β", ev.periods(a.decay) if a.decay is not None else ("kein Abklingen" if a.rho >= 1 - D.TOL_ONE else "über 4 Mio. Perioden"))
m3.metric("Grenze α_krit mit β", "keine unter 60" if a.krit == float("inf") else ev.de(a.krit, 3))
m4.metric("Grenze α_krit ohne Leitung", ev.de(a.krit0, 3))
c1, c2 = st.columns([3, 2])
c1.plotly_chart(build_impulse(a), width="stretch", key="impulse_chart")
c2.plotly_chart(build_roots(a), width="stretch", key="roots_chart")
st.caption("Rechts: die Wurzeln des Polynoms P(z) in der komplexen Ebene; alle innerhalb des Einheitskreises heißt stabil. Gestrichelt links und gekreuzt rechts: dieselbe Einstellung ohne Leitung.")

# --- Kernfragen ----------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfragen")
st.markdown("**1. Wie viel bringt die mitgerechnete Leitung?** (mit der Einstellung in der Seitenleiste)")
kind, text = ev.verdict_main(settings.L, settings.alpha, settings.beta, a.rho, a.rho0, a.decay, a.decay0)
getattr(st, kind)(text)

curve_k2, cells_k2 = _sensitivity(settings.L, settings.alpha)
st.markdown("**2. Wie genau muss β stimmen?** (bei der Lieferzeit und der Anpassung der Seitenleiste; ausgeführt im Abschnitt unten)")
kind, text = ev.verdict_sensitivity(settings.L, settings.alpha, cells_k2)
getattr(st, kind)(text)

st.markdown("**3. Was kostet eine falsch geschätzte Lieferzeit?** (die Regel nimmt L − 1 statt L an; mittlerer Bedarf d = 10; ausgeführt im Abschnitt „Drift“)")
dr0 = _drift(settings.L, settings.alpha, settings.beta, -1, C.D_MEAN, 0.0, C.DEFAULT_SEED)
kind, text = ev.verdict_drift(settings.L, settings.beta, settings.L - 1, C.D_MEAN, C.D_MEAN, dr0.i_star, dr0.t_eq, a.rho)
getattr(st, kind)(text)

st.markdown("---")

# --- Experiment 2: Grenzkarte über β -------------------------------------------------------------------------------------------------------

st.subheader("🔬 Grenzkarte über β: wo kippt das Lager?")
st.caption("Die Grenze α_krit (darüber hat P eine Wurzel außerhalb des Einheitskreises) gegen β, Meldeverzug 0. Beide Rechenwege der Grenze (Wurzeln auf dem Einheitskreis und Gitter mit Bisektion) sind in den Tests gegeneinander geprüft.")
gk_L = st.multiselect("Lieferzeiten L (höchstens 6)", list(range(C.L_MIN, C.L_MAX + 1)), default=list(C.GRENZKARTE_L_DEFAULT), max_selections=6, key="gk_L")
if gk_L:
    with st.spinner("Rechne die Grenzen …"):
        curves, rows = _grenzkarte(tuple(sorted(gk_L)))
    st.plotly_chart(build_grenzkarte(C.GRID_BETAS, curves), width="stretch", key="grenzkarte_chart")
    kind, text = ev.verdict_grenzkarte(rows)
    getattr(st, kind)(text)
    st.caption("Gestrichelt senkrecht β = ½, waagerecht α = 2, gepunktet 2/(2β − 1): für gerades L und β > 1 liegt die Grenze darauf (Messung in den Tests), für ungerades L über dieser Kurve.")
else:
    st.info("Wählen Sie mindestens eine Lieferzeit.")

st.markdown("---")

# --- Experiment 3: β = 1 -------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 β = 1: die Lieferzeit verschwindet aus der Charakteristik")
st.caption("Bei β = 1 ist P(z) = z^(L−1) · (z − (1 − α)): der Spektralradius ist |1 − α| für jedes L. Verglichen mit der besten Einstellung ohne Leitung (β = 0, Doppelwurzel bei α* = ((L−1)/L)^(L−1)/L); die Lieferzeit L steht in der Seitenleiste.")
alpha1 = st.slider("Anpassung α bei β = 1", 0.1, 1.9, 0.85, 0.01, key="beta1_alpha", help="Nahe 1 klingt der Stoß fast in L Perioden ab (α = 1 genau steht als Handbeispiel in Brüchen im README).")
alphas_grid = tuple(round(0.02 * k, 2) for k in range(1, 100))
curves1 = _beta1_curves(alphas_grid)
d1, d2 = st.columns(2)
d1.plotly_chart(build_beta1_rho(alphas_grid, curves1), width="stretch", key="beta1_rho_chart")
a_star = D.best_alpha_without_pipeline(settings.L)
T1 = 90
i_b1, _u = D.impulse_response(settings.L, alpha1, 1.0, T1)
i_b0, _u0 = D.impulse_response(settings.L, a_star, 0.0, T1)
d2.plotly_chart(build_stock_compare(list(range(T1 + 1)), [
    (f"β = 1, α = {ev.de(alpha1)}", i_b1, C.COLOR_WITH, "solid"),
    (f"β = 0, α* = {ev.de(a_star, 3)}", i_b0, C.COLOR_WITHOUT, "dash"),
]), width="stretch", key="beta1_stock_chart")
kind, text = ev.verdict_beta1(settings.L, a_star, D.decay_time(settings.L, a_star, 0.0), alpha1, D.spectral_radius(settings.L, alpha1, 1.0), D.decay_time(settings.L, alpha1, 1.0))
getattr(st, kind)(text)
st.caption("Links: alle Kurven liegen auf |1 − α|. Rechts: die ersten 90 Perioden; ist die Abklingzeit länger, läuft die Kurve über den Ausschnitt hinaus.")

st.markdown("---")

# --- Experiment 4: Wie genau muss β stimmen? -----------------------------------------------------------------------------------------------

st.subheader("🔬 Wie genau muss β stimmen?")
st.caption("Abklingzeit gegen β in der Umgebung von β = 1, für die Lieferzeit und die Anpassung der Seitenleiste; darunter dieselbe Tabelle für L = 3, 6 und 12.")
st.plotly_chart(build_sensitivity(curve_k2, settings.L), width="stretch", key="sensitivity_chart")
kind, text = ev.verdict_sensitivity(settings.L, settings.alpha, cells_k2)
getattr(st, kind)(text)
table4 = _sens_table(settings.alpha)
lab = {1.0: "β = 1", 0.99: "β = 0,99", 0.9: "β = 0,9", 1.1: "β = 1,1"}
rows4 = [{"L": L, **{lab[b]: ("kein Abklingen" if v is None else v) for b, v in cells.items()}} for L, cells in table4.items()]
st.dataframe(rows4, hide_index=True, width="content")
st.caption(f"Abklingzeit in Perioden bei α = {ev.de(settings.alpha)}. Die Spitze bei β = 1 wird mit L schärfer; nicht überall ist β = 1 am schnellsten (die Zählung steht im README).")

st.markdown("---")

# --- Experiment 5: Drift -------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Drift: die Lieferzeit falsch geschätzt")
st.caption("Die Soll-Leitung ist W* = d̂ · (L̂ − 1) mit geschätzter Lieferzeit L̂ und geschätztem Mittelbedarf d̂; die wahre Leitung ist d · (L − 1), der Mittelbedarf d = 10. Im Gleichgewicht ist die Bestellung gleich d, und der Bestand bleibt um −β · ΔW neben dem Soll.")
g1, g2, g3 = st.columns(3)
dl = g1.slider("Schätzfehler der Lieferzeit L̂ − L", *bounds("lhat_delta_slider"), key="lhat_delta_slider", help="Die Regel nimmt L̂ = L + dieser Wert an; 0 heißt richtig geschätzt.")
dhat = g2.slider("Geschätzter Mittelbedarf d̂", C.DHAT_MIN, C.DHAT_MAX, 10.0, 0.5, key="dhat_slider", help="Wahrer Mittelbedarf d = 10; ein anderer Wert verschiebt die Soll-Leitung ebenfalls.")
noise = g3.slider("Bedarfsrauschen s (Standardabweichung in Vielfachen von d)", C.NOISE_MIN, C.NOISE_MAX, 0.1, 0.01, key="noise_slider")
sc1, sc2 = st.columns([1, 3])
seed = sc1.number_input("Zufalls-Seed", 0, C.SEED_MAX, key="seed_input", step=1)
sc2.button("🎲 Neuer Zufalls-Seed", on_click=randomize_seed)
if settings.L + int(dl) < 1:
    st.caption("Die geschätzte Lieferzeit liegt unter 1: die Formel gilt, ist aber physikalisch sinnlos.")
dr = _drift(settings.L, settings.alpha, settings.beta, int(dl), float(dhat), float(noise), int(seed))
if ev.state_of(a.rho) != "stabil":
    st.warning(f"Bei ρ = {ev.de(a.rho, 3)} (nicht unter 1) erreicht der Bestand kein Gleichgewicht, die Reihe wird deshalb nicht gezeichnet.")
else:
    st.plotly_chart(build_drift(dr, C.D_MEAN, settings.beta), width="stretch", key="drift_chart")
kind, text = ev.verdict_drift(settings.L, settings.beta, settings.L + int(dl), C.D_MEAN, float(dhat), dr.i_star, dr.t_eq, a.rho)
getattr(st, kind)(text)
st.caption(f"Mittlere Bestellung in der zweiten Hälfte der {C.DRIFT_T} Perioden: {ev.de(dr.mean_order, 2)} (Mittelbedarf {ev.de(C.D_MEAN, 0)}). Zufalls-Seed {int(seed)}; die Rauschfolge ist aus dem eigenen SplitMix64 gezogen.")

st.markdown("---")

# --- Experiment 6: Preis der Ruhe ----------------------------------------------------------------------------------------------------------

st.subheader("🔬 Preis der Ruhe: Streuung bei weißem Bedarfsrauschen")
st.caption("Bedarf Periode für Periode unabhängig mit Varianz var(e); gerechnet wird das stationäre Verhältnis aus der Lyapunov-Gleichung. Kurztabelle; die Frequenzabhängigkeit steht in [`frequenzgang-lager-demo`](https://github.com/sebastian-hanisch/frequenzgang-lager-demo).")
rows6, table6 = _tradeoff(settings.L, 0.45)
kind, text = ev.verdict_tradeoff(settings.L, 0.45, rows6)
getattr(st, kind)(text)
tbl = [{"α": ev.de(r["alpha"]), "β": ev.de(r["beta"]), "ρ": ev.de(r["rho"], 4), "var(u)/var(e)": ("instabil" if r["var_u"] == float("inf") else ev.de(r["var_u"], 4)),
        "var(i)/var(e)": ("instabil" if r["var_i"] == float("inf") else ev.de(r["var_i"], 4))} for r in table6]
st.dataframe(tbl, hide_index=True, width="content")
st.caption(f"Lieferzeit L = {settings.L} aus der Seitenleiste. Bei β = 1 gilt var(u)/var(e) = α/(2 − α) und var(i)/var(e) = (L − 1) + 1/(α(2 − α)) (in den Tests gegen die Lyapunov-Gleichung geprüft).")

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Meldeverzug 0** (die Regel sieht den Bestand sofort) | Schon eine Periode Meldeverzug hebt die Grenze 2 bei β = 1 nicht mehr: sie liegt dann bei rund 0,7 (Vormessung für L = 3 und 6, in dieser Demo nicht gerechnet); „stabil“ heißt dort oft „klingt kaum ab“. | [`meldeverzug-demo`](https://github.com/sebastian-hanisch/meldeverzug-demo) |
| **Feste Lieferzeit** | Streut die Lieferzeit, ändert sich die Grenze; ob die Leitungsrechnung dort noch wirkt, ist nicht gemessen. | [`streuende-lieferzeit-demo`](https://github.com/sebastian-hanisch/streuende-lieferzeit-demo) |
| **Linear, ohne Klemmung** | Mit der Klemmung (Bestellung nicht unter 0) kann das Lager hinter der Grenze chaotisch werden; dort ist β = 0 gemessen, mit β ungleich 0 nicht. | [`lager-chaos-demo`](https://github.com/sebastian-hanisch/lager-chaos-demo) |
| **Konstanter Mittelbedarf, einmal geschätzte Soll-Leitung** | Der Dauerfehler −β·ΔW bleibt; nachgeführte Schätzer und sich ändernder Mittelbedarf sind nicht gemessen. | – |
| **Weißes Bedarfsrauschen** (Experiment Preis der Ruhe) | Bei Saison und Resonanz zählt der Frequenzgang, nicht das Mittel über alle Frequenzen. | [`frequenzgang-lager-demo`](https://github.com/sebastian-hanisch/frequenzgang-lager-demo) |
| **Eine Regel, kein Verhalten von Menschen** | Das Preset „Leitung zu wenig“ zeigt nur, was das Modell bei β = 0,3 tut. Sterman (1989; nur das Abstract gelesen) beschreibt Fehlwahrnehmungen der Rückkopplung, vor allem eine Unempfindlichkeit gegen die Rückwirkung der eigenen Entscheidungen; ob das einem β unter 1 entspricht, ist nicht untersucht. | – |
| **Grenze α = 2 bei β = ½ heißt ρ = 1** | In diesem Fall klingt die Schwingung nicht ab und wächst nicht; die Rundung entscheidet dann über Bruchteile von 10⁻⁹. | – |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    # erklaerseite:lesehilfe:start (erzeugt von tools/set_demo_explainer.py (Website-Repo) aus tools/content/explainers/ware-unterwegs-erklaert.json, nicht von Hand ändern)
    st.markdown(
        r"""
**Lesehilfe.** Gegeben sind die Lieferzeit $L$, die Anpassung $\alpha$, das Leitungsgewicht $\beta$ und die Bedarfsabweichung $e_t$ (bei Stoß: $e_0 = 1$, sonst 0). Gesucht sind die Stabilitätsgrenze, die Abklingzeit eines Stoßes und der Dauerfehler bei falscher Soll-Leitung. Alle Größen sind Abweichungen vom Gleichgewicht, Zähleinheit ist die Periode.

| Symbol | Bedeutung |
|---|---|
| $i_t$ | Bestand minus Soll am Ende der Periode $t$ |
| $u_t$ | Bestellung in Periode $t$ minus mittlerer Bedarf $d$ |
| $e_t$ | Bedarfsabweichung in Periode $t$ |
| $L$, $\alpha$, $\beta$ | Lieferzeit in Perioden; Anpassung (Anteil der Abweichung, den die Bestellung ausgleicht); Leitungsgewicht (Anteil der Ware unterwegs, den die Regel mitzählt) |
| $\Delta W$, $W^{*}$ | Fehler der Soll-Leitung $W^{*} = \hat d\,(\hat L - 1)$; $\Delta W = d\,(L-1) - W^{*}$ |
| $P$, $\rho$ | charakteristisches Polynom; Spektralradius, der größte Betrag einer Wurzel von $P$ |

[Schritt für Schritt erklärt, mit Herleitung und Literatur](https://sebastianhanisch.net/ware-unterwegs-erklaert.html#fachleute)
        """
    )
    # erklaerseite:lesehilfe:end
    st.markdown(
        r"""
**Gegeben und gesucht.** Gegeben sind die Lieferzeit $L$, die Anpassung $\alpha$, das Leitungsgewicht $\beta$ und die Bedarfsabweichung $e_t$ (bei Stoß: $e_0 = 1$, sonst 0).
Gesucht sind die Stabilitätsgrenze, die Abklingzeit eines Stoßes und der Dauerfehler bei falscher Soll-Leitung. Alle Aussagen gelten in diesem Modell (Meldeverzug 0, ohne Klemmung) und sind eigene Herleitung, gegen mehrere Rechenwege geprüft.

| Symbol | Bedeutung |
|---|---|
| $i_t$ | Bestand minus Soll am Ende der Periode $t$ |
| $u_t$ | Bestellung in Periode $t$ minus mittlerer Bedarf $d$ |
| $e_t$ | Bedarfsabweichung in Periode $t$ |
| $L$ | Lieferzeit in Perioden |
| $\alpha$ | Anpassung: Anteil der Abweichung, den die Bestellung ausgleicht |
| $\beta$ | Leitungsgewicht: Anteil der Ware unterwegs, den die Regel mitzählt |
| $\Delta W$ | Fehler der Soll-Leitung $W^{*}$ |
| $\rho$ | Spektralradius: größter Betrag einer Wurzel von $P$ |

**Modell.**

$$i_{t+1} = i_t + u_{t+1-L} - e_t$$

$$u_t = -\alpha\Bigl(i_t + \beta\bigl(\textstyle\sum_{j=1}^{L-1} u_{t-j} + \Delta W\bigr)\Bigr)$$

Der Bestand $i_t$ steigt um die ankommende Bestellung $u_{t+1-L}$ und fällt um den Bedarf $e_t$. Die Bestellung gleicht den Bestand zu einem Anteil $\alpha$ aus und rechnet die Ware unterwegs (die Summe der letzten $L-1$ Bestellungen) mit dem Gewicht $\beta$ mit.
Für $L = 1$ ist die Summe leer. Die Soll-Leitung ist $W^{*} = \hat d\,(\hat L - 1)$ mit geschätzter Lieferzeit $\hat L$ und geschätztem Mittelbedarf $\hat d$, also $\Delta W = d\,(L-1) - W^{*}$.

**Charakteristisches Polynom.** Mit dem Zustand $(i_t, u_{t-1}, \dots, u_{t-L+1})$ hat die Zustandsmatrix für $\Delta W = 0$ das charakteristische Polynom

$$P(z) = z^{L} - (1-\alpha\beta)\,z^{L-1} + \alpha\,(1-\beta)$$

Stabil heißt: alle Wurzeln liegen im Einheitskreis, $\rho < 1$. Die Übertragungsfunktion vom Bedarf zum Bestand ist

$$I(z) = -\frac{N(z)}{P(z)}\,E(z),\qquad N(z) = z^{L-1} + \alpha\beta\,\bigl(z^{L-2} + \dots + 1\bigr)$$

Es gilt $P(1) = \alpha$ (der Bestand verliert die Stabilität nie über $z = 1$) und

$$P(-1) = (-1)^{L}\,(2-\alpha\beta) + \alpha\,(1-\beta)$$

Bei ungeradem $L$ ist das $\alpha - 2$, also hat $P$ bei $\alpha = 2$ für jedes $\beta$ eine Wurzel bei $-1$.

**Sonderfälle.** Bei $\beta = 1$ ist $P(z) = z^{L-1}\bigl(z-(1-\alpha)\bigr)$, die Lieferzeit steht nicht mehr im Spektralradius $\rho = \lvert 1-\alpha\rvert$. Bei $\beta = \tfrac12$ und $\alpha = 2$ ist $P(z) = z^{L} + 1$:
alle Wurzeln liegen auf dem Einheitskreis. Ohne Leitung ($\beta = 0$) hat $P$ bei $\alpha^{*} = \tfrac1L\bigl(\tfrac{L-1}{L}\bigr)^{L-1}$ eine Doppelwurzel bei $z^{*} = \tfrac{L-1}{L}$ (exakt in Brüchen geprüft); gemessen ist dort auch der kleinste Spektralradius über $\alpha$, $1 - \tfrac1L$.
Für $L = 2$ ist die Grenze geschlossen (Jury-Bedingungen): $\alpha_{\mathrm{krit}} = \min\bigl(\tfrac{1}{1-\beta},\ \tfrac{2}{2\beta-1}\bigr)$ für $\beta > \tfrac12$ ($\tfrac{2}{2\beta-1}$ entfällt bei $\beta \le \tfrac12$, $\tfrac{1}{1-\beta}$ bei $\beta \ge 1$).

**Dauerfehler.** Bei konstantem Fehler $\Delta W$ ist im Gleichgewicht $u = 0$ und

$$i^{*} = -\beta\,\Delta W$$

unabhängig von $\alpha$, solange das Gleichgewicht existiert (ρ unter 1).

**Streuung bei weißem Bedarf.** Mit der Zustandsmatrix $A$ und $b = (-1, 0, \dots, 0)^{\top}$ löst die stationäre Kovarianz $\Sigma = A\,\Sigma\,A^{\top} + b\,b^{\top}$; $\operatorname{var}(i)/\operatorname{var}(e) = \Sigma_{11}$ und $\operatorname{var}(u)/\operatorname{var}(e) = c\,\Sigma\,c^{\top}$ mit $c = (-\alpha, -\alpha\beta, \dots, -\alpha\beta)$. Bei $\beta = 1$:

$$\frac{\operatorname{var}(u)}{\operatorname{var}(e)} = \frac{\alpha}{2-\alpha},\qquad \frac{\operatorname{var}(i)}{\operatorname{var}(e)} = (L-1) + \frac{1}{\alpha\,(2-\alpha)}$$

**Literatur.** Keine Quelle ist für diese Herleitungen gelesen; die Fachliteratur zu Bestandsregeln mit Leitungsrückführung (Disney/Towill und andere) ist nicht gelesen, eine Neuheit wird nicht behauptet.
Gelesen sind nur Hamiche u. a. (arXiv 1903.04256, ein Smith-Prädiktor für ein Lager, ohne Stabilitätsbeweis) und der Anfang von arXiv 2208.04459 (nicht nachgebaut).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zur Reihe: [Rückkopplung und Verzögerung: das Lager](https://sebastianhanisch.net/konzepte-rueckkopplung.html)."
)
