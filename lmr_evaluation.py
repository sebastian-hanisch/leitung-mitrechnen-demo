"""Auswertungen der Abschnitte der App und ihre Verdikte (reine Funktionen: (Art, Text); die App ruft st.<Art>(Text)).

Alle Zahlen kommen aus `lmr_dynamics`; die Texte nennen die Voraussetzung (dieses Modell, Meldeverzug 0, ohne Klemmung, Stoß auf 1 % der Spitze)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

import lmr_constants as C
import lmr_dynamics as D


def de(x: float, nd: int = 2) -> str:
    """Zahl mit deutschem Dezimalkomma (und Minuszeichen), für alle sichtbaren Texte."""
    return f"{x:.{nd}f}".replace(".", ",").replace("-", "−")


def periods(n: int | None) -> str:
    return "kein Abklingen" if n is None else f"{n} Perioden"


@dataclass(frozen=True)
class Settings:
    L: int = C.DEFAULT_L
    alpha: float = C.DEFAULT_ALPHA
    beta: float = C.DEFAULT_BETA
    lhat_delta: int = 0


@dataclass(frozen=True)
class Analysis:
    settings: Settings
    rho: float
    rho0: float            # gleiche Einstellung ohne Leitung (β = 0)
    krit: float            # α_krit mit β
    krit0: float           # α_krit ohne Leitung
    decay: int | None
    decay0: int | None
    estimate: float        # ln 0,01 / ln ρ (nur Vergleichswert)
    t: np.ndarray          # Perioden 0..T
    i: np.ndarray          # Bestand mit β
    i0: np.ndarray         # Bestand ohne Leitung
    u: np.ndarray          # Bestellung mit β (u_0..u_{T-1})
    u0: np.ndarray
    roots: np.ndarray
    roots0: np.ndarray


def plot_horizon(rho: float, rho0: float, decay: int | None, decay0: int | None, L: int) -> int:
    """Länge des Stoßantwort-Diagramms: bis zum Abklingen (mit Reserve), höchstens IMPULSE_T_MAX; ist nichts stabil, bis das Wachstum 10^3 erreicht (höchstens 150)."""
    ds = [x for x in (decay, decay0) if x is not None]
    if ds:
        return int(min(C.IMPULSE_T_MAX, max(30, 1.15 * max(ds) + 5)))
    worst = max(rho, rho0)
    if worst > 1.0 + D.TOL_ONE:
        return int(min(150, max(30, 7.0 / math.log(worst) + L + 10)))
    return int(min(C.IMPULSE_T_MAX, max(30, 40 * L)))


def analyse(s: Settings) -> Analysis:
    rho = D.spectral_radius(s.L, s.alpha, s.beta)
    rho0 = D.spectral_radius(s.L, s.alpha, 0.0)
    decay = D.decay_time(s.L, s.alpha, s.beta)
    decay0 = D.decay_time(s.L, s.alpha, 0.0)
    T = plot_horizon(rho, rho0, decay, decay0, s.L)
    i, u = D.impulse_response(s.L, s.alpha, s.beta, T)
    i0, u0 = D.impulse_response(s.L, s.alpha, 0.0, T)
    return Analysis(
        s, rho, rho0, D.alpha_krit(s.L, s.beta), D.alpha_krit(s.L, 0.0), decay, decay0,
        D.decay_estimate(rho), np.arange(T + 1), i, i0, u, u0,
        D.roots_of_P(s.L, s.alpha, s.beta), D.roots_of_P(s.L, s.alpha, 0.0),
    )


# --------------------------------------------------------------------------------------------------------------------------------
# Verdikte (reine Funktionen)
# --------------------------------------------------------------------------------------------------------------------------------

def state_of(rho: float) -> str:
    """'stabil' (ρ < 1 − 10^-9), 'grenze' (|ρ − 1| ≤ 10^-9: klingt weder ab noch wächst) oder 'instabil'."""
    if rho < 1.0 - D.TOL_ONE:
        return "stabil"
    if rho <= 1.0 + D.TOL_ONE:
        return "grenze"
    return "instabil"


def describe(rho: float, decay: int | None) -> str:
    """Ein Halbsatz zu einem Spektralradius: stabil mit Abklingzeit, Grenze oder wachsend."""
    st = state_of(rho)
    if st == "stabil":
        if decay is None:
            return f"ist ρ = {de(rho, 6)}: der Stoß klingt praktisch nicht ab (länger als {D.DECAY_T_CAP // 1_000_000} Millionen Perioden bis 1 %)"
        return f"ist ρ = {de(rho, 3)}, der Stoß klingt in {decay} Perioden auf 1 % ab"
    if st == "grenze":
        return "liegt ρ bei 1: die Schwingung klingt weder ab noch wächst"
    return f"ist ρ = {de(rho, 3)}: die Schwingung wächst auf lange Sicht um {de(100.0 * (rho - 1.0), 1)} % je Periode"


def verdict_main(L: int, alpha: float, beta: float, rho: float, rho0: float, decay: int | None, decay0: int | None) -> tuple[str, str]:
    """Kernfrage 1: Wie viel bringt die mitgerechnete Leitung? Vergleich der Einstellung (L, α, β) mit derselben ohne Leitung (β = 0)."""
    head = f"L = {L}, α = {de(alpha)}"
    st = state_of(rho)
    kind = {"stabil": "success", "grenze": "info", "instabil": "warning"}[st]
    if abs(beta) < 1e-12:
        return kind, f"{head}, ohne Leitung (β = 0) {describe(rho, decay)}. Setzen Sie β auf einen Wert über 0, um die Leitung mitzurechnen."
    return kind, f"{head}: ohne Leitung (β = 0) {describe(rho0, decay0)}; mit β = {de(beta)} {describe(rho, decay)}. Gilt in diesem Modell, ohne Meldeverzug und ohne Klemmung."


def verdict_grenzkarte(rows: list[dict]) -> tuple[str, str]:
    """rows: je gewähltem L {'L', 'a05', 'a08'} mit α_krit bei β = 0,5 und β = 0,8. Zählt, bei wie vielen L die Grenze bei β = ½ gleich 2 ist (Toleranz 10^-6), und nennt die Parität bei β = 0,8."""
    n = len(rows)
    k = sum(1 for r in rows if abs(r["a05"] - 2.0) < 1e-6)
    odd = [r for r in rows if r["L"] % 2 == 1]
    even = [r for r in rows if r["L"] % 2 == 0]
    odd_txt = ", ".join(f"L = {r['L']}: {de(r['a08'], 3)}" for r in odd) or "keines gewählt"
    even_txt = ", ".join(f"L = {r['L']}: {de(r['a08'], 3)}" for r in even) or "keines gewählt"
    text = (f"β = ½: α_krit = 2 bei {k} von {n} gewählten L. β = 0,8: ungerades L ({odd_txt}), gerades L ({even_txt}). "
            "Eigene Messung in diesem Modell (Meldeverzug 0); warum sich gerades und ungerades L trennen, ist hier nicht erklärt.")
    return "info", text


def verdict_beta1(L: int, alpha_star: float, decay0: int | None, alpha: float, rho1: float, decay1: int | None) -> tuple[str, str]:
    """Experiment β = 1: beste Einstellung ohne Leitung gegen β = 1 mit dem gewählten α."""
    tail = describe(rho1, decay1)
    kind = "success" if state_of(rho1) == "stabil" else "warning"
    return kind, (f"L = {L}: ohne Leitung ist die beste Einstellung α* = {de(alpha_star, 3)} ({periods(decay0)} bis 1 %). "
                  f"Mit β = 1 und α = {de(alpha)} {tail}. Bei β = 1 steht die Lieferzeit nicht mehr im Spektralradius: ρ = |1 − α| = {de(abs(1.0 - alpha), 3)} für jedes L.")


def verdict_sensitivity(L: int, alpha: float, cells: dict[float, int | None]) -> tuple[str, str]:
    """cells: Abklingzeit je β (Schlüssel 1,0, 0,99, 0,9, 1,1; None = kein Abklingen). Meldet, ob β = 1 hier die schnellste dieser vier Wahlen ist."""
    labels = {1.0: "1", 0.99: "0,99", 0.9: "0,9", 1.1: "1,1"}
    parts = ", ".join(f"β = {labels[b]}: {periods(cells[b])}" for b in labels)
    finite = {b: v for b, v in cells.items() if v is not None}
    best = min(finite, key=lambda b: (finite[b], abs(b - 1.0))) if finite else None
    if best is None:
        return "warning", f"L = {L}, α = {de(alpha)}: bei keinem der vier β ({parts}) klingt der Stoß ab."
    if finite.get(1.0) is not None and finite[1.0] <= min(finite.values()):
        low = ""
        if finite.get(0.99) is not None:
            low = (f" Eine Abweichung von 0,01 nach unten (β = 0,99) verlängert das auf {finite[0.99]} Perioden." if finite[0.99] > finite[1.0]
                   else " Eine Abweichung von 0,01 nach unten (β = 0,99) ändert die Abklingzeit hier nicht.")
        return "info", f"L = {L}, α = {de(alpha)}: {parts}. β = 1 ist unter diesen vier die schnellste Wahl.{low}"
    return "info", f"L = {L}, α = {de(alpha)}: {parts}. Hier ist β = 1 nicht die schnellste dieser vier Wahlen (schneller: β = {de(best, 2)} mit {finite[best]} Perioden)."


def verdict_drift(L: int, beta: float, L_hat: int, d: float, d_hat: float, i_star: float, t_eq: int | None, rho: float) -> tuple[str, str]:
    """Kosten der falsch geschätzten Soll-Leitung: Dauerfehler i* = −β ΔW (in Perioden Bedarf), Zeit bis auf 1 % von |i*|."""
    if state_of(rho) != "stabil":
        return "warning", f"Bei ρ = {de(rho, 3)} (nicht unter 1) gibt es kein Gleichgewicht, das der Bestand erreicht; der Dauerfehler −β·ΔW = {de(i_star / d, 2)} Perioden Bedarf gilt nur für stabile Einstellungen."
    periods_bedarf = i_star / d
    if abs(i_star) < 1e-9:
        return "success", (f"Geschätzte Lieferzeit {L_hat} (tatsächlich {L}), β = {de(beta)}: kein Dauerfehler, der Bestand liegt im Gleichgewicht auf dem Soll "
                           "(ΔW = 0 oder β = 0). Dauerfehler und Zeit gelten für diese Einstellung und konstanten Mittelbedarf.")
    sign = "unter" if periods_bedarf < 0 else "über"
    wait = f"; er stellt sich in {t_eq} Perioden auf 1 % ein" if t_eq is not None else ""
    hat = f", geschätzter Mittelbedarf {de(d_hat, 1)} statt {de(d, 1)}" if abs(d_hat - d) > 1e-9 else ""
    return "warning", (f"Geschätzte Lieferzeit {L_hat} statt {L}{hat}, β = {de(beta)}: im Gleichgewicht liegt der Bestand dauerhaft "
                       f"{de(abs(periods_bedarf), 2)} Perioden Bedarf {sign} dem Soll ({de(i_star, 1)} Stück bei d = {de(d, 0)}){wait}. "
                       "In diesem Modell hängt der Fehler nicht an α (bei ρ unter 1) und ändert die Stabilität nicht.")


def verdict_tradeoff(L: int, alpha: float, rows: dict[float, tuple[float, float]]) -> tuple[str, str]:
    """rows: β -> (var(u)/var(e), var(i)/var(e)) bei weißem Bedarfsrauschen; inf, wenn nicht stabil. Nennt die kleinste Streuung je Größe unter den β der Tabelle."""
    parts = []
    for b, (vu, vi) in rows.items():
        parts.append(f"β = {de(b)}: {de(vu, 3) if math.isfinite(vu) else 'instabil'} und {de(vi, 2) if math.isfinite(vi) else 'instabil'}")
    fin = {b: v for b, v in rows.items() if math.isfinite(v[0])}
    if not fin:
        return "warning", f"L = {L}, α = {de(alpha)}: bei keinem der β der Tabelle ist die Rückkopplung stabil."
    bu = min(fin, key=lambda b: fin[b][0])
    bi = min(fin, key=lambda b: fin[b][1])
    return "info", (f"L = {L}, α = {de(alpha)} (var(u)/var(e) und var(i)/var(e), weißer Bedarf): {'; '.join(parts)}. "
                    f"Kleinste Bestellstreuung bei β = {de(bu)}, kleinste Bestandsstreuung bei β = {de(bi)}.")


# --------------------------------------------------------------------------------------------------------------------------------
# Auswertungen der Experimente
# --------------------------------------------------------------------------------------------------------------------------------

def sensitivity_curve(L: int, alpha: float, betas=C.SENS_BETAS) -> list[dict]:
    """Spektralradius und Abklingzeit je β für feste (L, α)."""
    return [{"beta": b, "rho": D.spectral_radius(L, alpha, b), "decay": D.decay_time(L, alpha, b)} for b in betas]


def sensitivity_cells(L: int, alpha: float) -> dict[float, int | None]:
    return {b: D.decay_time(L, alpha, b) for b in (1.0, 0.99, 0.9, 1.1)}


def beta1_rho_curves(alphas: np.ndarray, Ls=C.BETA1_L) -> dict[int, np.ndarray]:
    return {L: np.array([D.spectral_radius(L, float(a), 1.0) for a in alphas]) for L in Ls}


def grenzkarte_curves(Ls: tuple[int, ...], betas=C.GRID_BETAS) -> dict[int, np.ndarray]:
    return {L: D.alpha_krit_curve(L, betas) for L in Ls}


def grenzkarte_rows(Ls: tuple[int, ...]) -> list[dict]:
    return [{"L": L, "a05": D.alpha_krit(L, 0.5), "a08": D.alpha_krit(L, 0.8)} for L in Ls]


@dataclass(frozen=True)
class DriftResult:
    dW: float
    i_star: float
    t: np.ndarray
    i_noisy: np.ndarray
    i_clean: np.ndarray
    t_eq: int | None
    mean_order: float      # mittlere Bestellung d + u über die zweite Hälfte der Periode


def settle_time(i: np.ndarray, i_star: float, tol: float = 0.01) -> int | None:
    """Erste Periode, ab der |i_t − i*| dauerhaft höchstens tol·|i*| beträgt (bis zum Ende von i); None, wenn i* = 0 oder die Reihe nie dort ankommt."""
    if abs(i_star) < 1e-12:
        return None
    bad = np.nonzero(np.abs(i - i_star) > tol * abs(i_star))[0]
    if bad.size == 0:
        return 0
    last = int(bad.max())
    return None if last >= i.size - 1 else last + 1


def analyse_drift(L: int, alpha: float, beta: float, lhat_delta: int, d_hat: float, s_rel: float, seed: int, T: int = C.DRIFT_T) -> DriftResult:
    d = C.D_MEAN
    dW = D.delta_w(d, L, d_hat, L + lhat_delta)
    i_star = D.drift_equilibrium(beta, dW)
    rho = D.spectral_radius(L, alpha, beta)
    if state_of(rho) != "stabil":
        z = np.zeros(T + 1)
        return DriftResult(dW, i_star, np.arange(T + 1), z, z, None, d)
    e = D.noise_series(T, s_rel, d, seed)
    i_noisy, u = D.simulate(L, alpha, beta, e, dW)
    Th = D.decay_horizon(rho, L)
    long_clean = D.drift_step_response(L, alpha, beta, dW, min(Th, D.DECAY_T_CAP))
    t_eq = settle_time(long_clean, i_star)
    i_clean = long_clean[: T + 1] if long_clean.size >= T + 1 else np.concatenate([long_clean, np.full(T + 1 - long_clean.size, long_clean[-1])])
    return DriftResult(dW, i_star, np.arange(T + 1), i_noisy, i_clean, t_eq, d + float(u[T // 2:].mean()))


def tradeoff_rows(L: int, alpha: float, betas=C.TRADEOFF_BETAS) -> dict[float, tuple[float, float]]:
    return {b: D.noise_variances(L, alpha, b) for b in betas}


def tradeoff_table(L: int, alphas=C.TRADEOFF_ALPHAS, betas=C.TRADEOFF_BETAS) -> list[dict]:
    rows = []
    for a in alphas:
        for b in betas:
            vu, vi = D.noise_variances(L, a, b)
            rows.append({"alpha": a, "beta": b, "rho": D.spectral_radius(L, a, b), "var_u": vu, "var_i": vi})
    return rows
