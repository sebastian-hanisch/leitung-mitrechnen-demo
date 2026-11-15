"""Kern: das lineare Lager mit Leitungsgewicht (nur numpy und scipy.signal; Meldeverzug s = 0, ohne Klemmung).

Modell (alle Größen Abweichungen vom Gleichgewicht, Zähleinheit Periode):
    i_{t+1} = i_t + u_{t+1-L} - e_t
    u_t     = -alpha * (i_t + beta * (sum_{j=1..L-1} u_{t-j} + dW))
i Bestand minus Soll, u Bestellung minus mittlerer Bedarf d, e Bedarfsabweichung, L Lieferzeit, alpha Anpassung, beta Leitungsgewicht,
dW = d (L - 1) - W* der Fehler der Soll-Leitung W* (0, wenn sie stimmt).

Eigene Herleitung in diesem Modell (nicht aus einer Quelle; die Fachliteratur zu Regeln mit Leitungsrückführung ist nicht gelesen):
    P(z) = z^L - (1 - alpha beta) z^(L-1) + alpha (1 - beta)        charakteristisches Polynom (Stabilität: alle Wurzeln im Einheitskreis)
    I(z) = -E(z) N(z) / P(z),  N(z) = z^(L-1) + alpha beta (z^(L-2) + ... + 1)      Übertragungsfunktion Bedarf -> Bestand
    P(1) = alpha,  P(-1) = (-1)^L (2 - alpha beta) + alpha (1 - beta)
Jede Größe hat mindestens zwei Rechenwege (Wurzeln von P, Eigenwerte der Zustandsmatrix, Verzögerungsliste, Filter, Brüche); die Tests
vergleichen sie (tests/test_oracle_leitung_mitrechnen.py).

Aufbau: Polynom und Zustandsmatrix, Spektralradius, Grenze alpha_krit (zwei Wege), Stoßantwort und Abklingzeit, Simulation mit
Bedarfsrauschen und Fehler der Soll-Leitung, Streuung (Lyapunov-Gleichung)."""

from __future__ import annotations

import math

import numpy as np
from scipy.linalg import solve_discrete_lyapunov
from scipy.signal import lfilter

import lmr_rng

TOL_ONE = 1e-9          # |rho - 1| darunter heißt "auf dem Einheitskreis" (die Schwingung klingt weder ab noch wächst)
DECAY_FRACTION = 0.01   # Abklingzeit: der Bestand bleibt ab der Periode unter 1 % seiner Spitze
DECAY_T_CAP = 4_000_000  # Obergrenze des Horizonts der Abklingzeit (Perioden)


# --------------------------------------------------------------------------------------------------------------------------------
# Polynom, Zustandsmatrix, Spektralradius
# --------------------------------------------------------------------------------------------------------------------------------

def char_poly(L: int, alpha: float, beta: float) -> np.ndarray:
    """Koeffizienten von P(z), höchste Potenz zuerst (Länge L + 1)."""
    p = np.zeros(L + 1)
    p[0] = 1.0
    p[1] = -(1.0 - alpha * beta)
    p[L] += alpha * (1.0 - beta)
    return p


def state_matrix(L: int, alpha: float, beta: float) -> np.ndarray:
    """Zustand x_t = (i_t, u_{t-1}, ..., u_{t-L+1}); x_{t+1} = A x_t (ohne Bedarfsstoß). Die Bestellung u_t = c x_t steht in `readout`."""
    c = readout(L, alpha, beta)
    A = np.zeros((L, L))
    if L == 1:
        A[0, 0] = 1.0 + c[0]
        return A
    A[0, 0] = 1.0
    A[0, L - 1] += 1.0          # u_{t+1-L} = u_{t-(L-1)}: die älteste gespeicherte Bestellung kommt an
    A[1, :] = c
    for j in range(2, L):
        A[j, j - 1] = 1.0
    return A


def readout(L: int, alpha: float, beta: float) -> np.ndarray:
    """Die Bestellung u_t als Zeile c mit u_t = c . x_t (x_t wie in `state_matrix`)."""
    c = np.full(L, -alpha * beta)
    c[0] = -alpha
    return c


def roots_of_P(L: int, alpha: float, beta: float) -> np.ndarray:
    """Alle L Wurzeln von P (np.roots; exakte Nullen in den letzten Koeffizienten werden als Wurzeln 0 geführt)."""
    return np.roots(char_poly(L, alpha, beta))


def spectral_radius(L: int, alpha: float, beta: float) -> float:
    """Spektralradius aus den Wurzeln von P. Bei beta = 1 ist der Koeffizient alpha (1 - beta) exakt 0: die (L-1)-fache Wurzel 0 ist exakt."""
    r = roots_of_P(L, alpha, beta)
    return float(np.max(np.abs(r))) if r.size else 0.0


def spectral_radius_matrix(L: int, alpha: float, beta: float) -> float:
    """Zweiter Weg: Spektralradius der Zustandsmatrix (Eigenwerte)."""
    return float(np.max(np.abs(np.linalg.eigvals(state_matrix(L, alpha, beta)))))


def _companion_batch(L: int, alphas: np.ndarray, beta: float) -> np.ndarray:
    a = np.asarray(alphas, dtype=float)
    c = np.zeros((a.size, L))
    c[:, 0] = -(1.0 - a * beta)
    c[:, L - 1] += a * (1.0 - beta)
    M = np.zeros((a.size, L, L))
    M[:, 0, :] = -c
    for k in range(1, L):
        M[:, k, k - 1] = 1.0
    return M


def rho_batch(L: int, alphas, beta: float) -> np.ndarray:
    """Spektralradius für viele alpha auf einmal (Eigenwerte der Begleitmatrizen von P)."""
    return np.abs(np.linalg.eigvals(_companion_batch(L, np.asarray(alphas, float), beta))).max(axis=1)


def p_at_minus_one(L: int, alpha: float, beta: float) -> float:
    """P(-1) = (-1)^L (2 - alpha beta) + alpha (1 - beta): bei ungeradem L gleich alpha - 2."""
    return (-1.0) ** L * (2.0 - alpha * beta) + alpha * (1.0 - beta)


# --------------------------------------------------------------------------------------------------------------------------------
# Grenze alpha_krit: zwei unabhängige Wege
# --------------------------------------------------------------------------------------------------------------------------------

def _alpha_on_circle(L: int, beta: float, theta: np.ndarray) -> np.ndarray:
    """Die Wurzel z = e^{i theta} auf dem Einheitskreis löst P(z) = 0 für alpha = -z^(L-1) (z - 1) / (beta z^(L-1) + 1 - beta) (komplex)."""
    z = np.exp(1j * theta)
    zl = z ** (L - 1)
    return -zl * (z - 1.0) / (beta * zl + 1.0 - beta)


def circle_crossings(L: int, beta: float, n_theta: int = 4000) -> list[float]:
    """Alle positiven reellen alpha, für die P eine Wurzel auf dem Einheitskreis hat, aufsteigend (Weg 1: Kurve alpha(theta), Nullstellen von Im alpha).

    Vorzeichenwechsel von Im alpha auf einem Gitter in theta, dann vektorisierte Bisektion; dazu der Punkt theta = pi (z = -1).
    Polstellen (beta > 1: Nenner 0) fallen durch die Grenze |alpha| < 60 heraus."""
    th = np.linspace(1e-6, math.pi, n_theta)
    a = _alpha_on_circle(L, beta, th)
    s = np.sign(a.imag)
    idx = np.nonzero(s[:-1] * s[1:] < 0)[0]
    out: list[float] = []
    if idx.size:
        lo, hi = th[idx], th[idx + 1]
        slo = s[idx]
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            smid = np.sign(_alpha_on_circle(L, beta, mid).imag)
            same = smid == slo
            lo = np.where(same, mid, lo)
            hi = np.where(same, hi, mid)
        vals = _alpha_on_circle(L, beta, 0.5 * (lo + hi))
        out += [float(v.real) for v in vals if v.real > 0 and abs(v) < 60 and abs(v.imag) < 1e-6 * max(1.0, abs(v))]
    v_pi = _alpha_on_circle(L, beta, np.array([math.pi]))[0]
    if abs(v_pi) < 60 and v_pi.real > 0 and abs(v_pi.imag) < 1e-9:
        out.append(float(v_pi.real))
    return sorted(out)


def alpha_krit(L: int, beta: float, n_theta: int = 4000) -> float:
    """Kleinstes alpha > 0, ab dem P eine Wurzel außerhalb des Einheitskreises hat (Weg 1: Wurzeln auf dem Kreis, dann Probe knapp dahinter).

    Kein Gitter in alpha und keine Obergrenze der Suche; gibt es keine Grenze unter 60, kommt inf zurück."""
    for c in circle_crossings(L, beta, n_theta):
        if c > 1e-12 and rho_batch(L, [c * (1.0 + 1e-7)], beta)[0] > 1.0:
            return c
    return math.inf


def alpha_krit_scan(L: int, beta: float, hi: float = 4.0, n: int = 4000) -> float:
    """Weg 2 (Gitter in alpha, dann Bisektion auf dem Spektralradius der Begleitmatrix von P): kleinstes alpha mit rho > 1 + 1e-9. inf, wenn es bis `hi` keines gibt."""
    al = np.linspace(1e-4, hi, n)
    bad = np.nonzero(rho_batch(L, al, beta) > 1.0 + 1e-9)[0]
    if bad.size == 0:
        return math.inf
    j = int(bad[0])
    lo, up = (float(al[j - 1]) if j > 0 else 1e-4), float(al[j])
    for _ in range(50):
        mid = 0.5 * (lo + up)
        if rho_batch(L, [mid], beta)[0] > 1.0 + 1e-12:
            up = mid
        else:
            lo = mid
    return up


def jury_alpha_krit_L2(beta: float) -> float:
    """L = 2 geschlossen (Jury-Bedingungen: |alpha (1 - beta)| < 1 und P(-1) = 2 + alpha (1 - 2 beta) > 0; P(1) = alpha > 0)."""
    cands = []
    if beta != 1.0:
        cands.append(1.0 / abs(1.0 - beta))
    if beta > 0.5:
        cands.append(2.0 / (2.0 * beta - 1.0))
    return min(cands) if cands else 2.0


def alpha_krit_curve(L: int, betas) -> np.ndarray:
    """alpha_krit(beta) für viele beta (Weg 1)."""
    return np.array([alpha_krit(L, float(b), 1500) for b in betas])


def best_alpha_without_pipeline(L: int) -> float:
    """beta = 0: das alpha mit dem kleinsten Spektralradius: Doppelwurzel z* = (L-1)/L bei alpha* = z*^(L-1) / L (L >= 2; eigene Herleitung, gegen Brüche geprüft)."""
    if L < 2:
        return 1.0
    zs = (L - 1.0) / L
    return zs ** (L - 1) / L


# --------------------------------------------------------------------------------------------------------------------------------
# Stoßantwort, Abklingzeit, Simulation
# --------------------------------------------------------------------------------------------------------------------------------

def simulate(L: int, alpha: float, beta: float, e, dW: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Direkte Rekursion (Weg 1): Bestand i_0..i_T (Länge T + 1) und Bestellung u_0..u_{T-1} bei Bedarfsabweichung e_0..e_{T-1}; Start i_0 = 0, alle u vor 0 gleich 0.

    Die Leitungssumme S_t = u_{t-1} + ... + u_{t-L+1} wird fortgeschrieben (S_{t+1} = S_t + u_t - u_{t-L+1})."""
    e = np.asarray(e, dtype=float)
    T = e.size
    i = np.zeros(T + 1)
    u = np.zeros(T)
    S = 0.0
    el = e.tolist()
    ul: list[float] = []
    il = [0.0]
    cur = 0.0
    for t in range(T):
        ut = -alpha * (cur + beta * (S + dW))
        ul.append(ut)
        old = ul[t - L + 1] if t - L + 1 >= 0 else 0.0
        S += ut - old
        arrive = ul[t + 1 - L] if t + 1 - L >= 0 else 0.0
        cur = cur + arrive - el[t]
        il.append(cur)
    i[:] = il
    u[:] = ul
    return i, u


def transfer_coeffs(L: int, alpha: float, beta: float) -> tuple[np.ndarray, np.ndarray]:
    """Filter (b, a) in z^-1 mit i = lfilter(b, a, e): I(z) = -E(z) N(z)/P(z), durch z^L gekürzt (Weg 2)."""
    b = np.zeros(L + 1)
    b[1] = -1.0
    b[2:] = -alpha * beta
    return b, char_poly(L, alpha, beta)


def filter_response(L: int, alpha: float, beta: float, e) -> np.ndarray:
    """Bestand i_0..i_T bei Bedarfsabweichung e_0..e_{T-1} über die Übertragungsfunktion (Länge T + 1: i_0 = 0)."""
    e = np.asarray(e, dtype=float)
    b, a = transfer_coeffs(L, alpha, beta)
    x = np.concatenate([e, [0.0]])
    return lfilter(b, a, x)


def impulse_response(L: int, alpha: float, beta: float, T: int) -> tuple[np.ndarray, np.ndarray]:
    """Bedarfsstoß e_0 = 1 (sonst 0): Bestand i_0..i_T und Bestellung u_0..u_{T-1} (direkte Rekursion)."""
    e = np.zeros(T)
    if T:
        e[0] = 1.0
    return simulate(L, alpha, beta, e)


def impulse_stock_fast(L: int, alpha: float, beta: float, T: int) -> np.ndarray:
    """Bestand i_0..i_T nach dem Stoß über das Filter (schnell, für lange Horizonte)."""
    e = np.zeros(T)
    if T:
        e[0] = 1.0
    return filter_response(L, alpha, beta, e)


def decay_time_of(i: np.ndarray, frac: float = DECAY_FRACTION) -> int:
    """Letzte Periode t mit |i_t| > frac * max|i| (0, wenn der Bestand nie die Schwelle übersteigt). i beginnt mit i_0."""
    a = np.abs(i)
    peak = a.max() if a.size else 0.0
    if peak <= 0:
        return 0
    above = np.nonzero(a > frac * peak)[0]
    return int(above.max()) if above.size else 0


def decay_horizon(rho: float, L: int) -> int:
    """Horizont, bis zu dem der Stoß sicher unter 10^-6 seiner Anfangshöhe abgeklungen ist (ln 10^6 / -ln rho, mindestens 40 L + 200)."""
    if rho <= 0.0:
        return 40 * L + 200
    return int(min(DECAY_T_CAP, max(40 * L + 200, 14.0 / -math.log(rho) + 20 * L)))


def decay_time(L: int, alpha: float, beta: float) -> int | None:
    """Abklingzeit des Bestands nach einem Bedarfsstoß in Perioden (letzte Periode über 1 % der Spitze); None, wenn rho >= 1 (kein Abklingen).

    Bedingung: rho < 1 - TOL_ONE und der Horizont (siehe decay_horizon) liegt unter DECAY_T_CAP; sonst None."""
    rho = spectral_radius(L, alpha, beta)
    if rho >= 1.0 - TOL_ONE:
        return None
    T = decay_horizon(rho, L)
    if T >= DECAY_T_CAP:
        return None
    return decay_time_of(impulse_stock_fast(L, alpha, beta, T))


def decay_estimate(rho: float) -> float:
    """Schätzung ln 0,01 / ln rho (nur ein Vergleichswert; kein Gesetz: bei Doppelwurzel und bei dichten Wurzeln weicht die gemessene Zeit ab)."""
    if rho <= 0.0:
        return 0.0
    if rho >= 1.0:
        return math.inf
    return math.log(DECAY_FRACTION) / math.log(rho)


# --------------------------------------------------------------------------------------------------------------------------------
# Falsch geschätzte Soll-Leitung
# --------------------------------------------------------------------------------------------------------------------------------

def delta_w(d: float, L: int, d_hat: float, L_hat: float) -> float:
    """dW = d (L - 1) - W*, W* = d_hat (L_hat - 1): die Soll-Leitung mit geschätzter Lieferzeit und geschätztem Mittelbedarf."""
    return d * (L - 1) - d_hat * (L_hat - 1)


def drift_equilibrium(beta: float, dW: float) -> float:
    """Bestand minus Soll im Gleichgewicht (u = 0): i* = -beta dW (eigene Herleitung; gegen Brüche geprüft)."""
    return -beta * dW


def drift_step_response(L: int, alpha: float, beta: float, dW: float, T: int) -> np.ndarray:
    """Bestand i_0..i_T bei konstantem Fehler dW der Soll-Leitung ohne Bedarfsrauschen (Filter, I(z) = -alpha beta dW step(z) / P(z); Weg 2)."""
    b = np.zeros(L + 1)
    b[L] = -alpha * beta * dW
    return lfilter(b, char_poly(L, alpha, beta), np.ones(T + 1))


def noise_series(T: int, s_rel: float, d: float, seed: int) -> np.ndarray:
    """Bedarfsabweichung e_0..e_{T-1} ~ Normal(0, (s_rel d)^2) aus dem eigenen SplitMix64."""
    return s_rel * d * lmr_rng.normal_array(seed, T)


def batch_mean_and_se(x: np.ndarray, n_batches: int = 40) -> tuple[float, float]:
    """Mittel und Standardfehler über n_batches gleich lange Blöcke (Batch-Mittel; trägt die Autokorrelation des Bestands)."""
    m = x.size // n_batches
    blocks = x[: m * n_batches].reshape(n_batches, m).mean(axis=1)
    return float(blocks.mean()), float(blocks.std(ddof=1) / math.sqrt(n_batches))


# --------------------------------------------------------------------------------------------------------------------------------
# Streuung bei weißem Bedarfsrauschen
# --------------------------------------------------------------------------------------------------------------------------------

def noise_variances(L: int, alpha: float, beta: float) -> tuple[float, float]:
    """(var(u)/var(e), var(i)/var(e)) im stationären Zustand bei weißem Bedarfsrauschen aus der Lyapunov-Gleichung S = A S A^T + B B^T (Weg 1).

    Nur für ρ < 1; sonst (inf, inf). Zustand x_t = (i_t, u_{t-1}, ..., u_{t-L+1}); e_t geht mit -1 in i_{t+1}."""
    if spectral_radius(L, alpha, beta) >= 1.0 - TOL_ONE:
        return math.inf, math.inf
    A = state_matrix(L, alpha, beta)
    B = np.zeros((L, 1))
    B[0, 0] = -1.0
    S = solve_discrete_lyapunov(A, B @ B.T)
    c = readout(L, alpha, beta)
    return float(c @ S @ c), float(S[0, 0])


def noise_variances_beta1(L: int, alpha: float) -> tuple[float, float]:
    """Geschlossen bei beta = 1 (eigene Herleitung, gegen die Lyapunov-Gleichung für L = 1, 3, 8 geprüft): var(u)/var(e) = alpha/(2-alpha), var(i)/var(e) = (L-1) + 1/(alpha (2-alpha))."""
    return alpha / (2.0 - alpha), (L - 1) + 1.0 / (alpha * (2.0 - alpha))


def simulated_variances(L: int, alpha: float, beta: float, T: int, seed: int, burn: int = 2000) -> tuple[float, float]:
    """Dritter Weg: Simulation über das Filter mit Normalrauschen (SplitMix64); (var(u)/var(e), var(i)/var(e)) nach `burn` Einschwingperioden.

    u folgt aus der Bilanz u_s = i_{s+L} - i_{s+L-1} + e_{s+L-1}."""
    e = lmr_rng.normal_array(seed, T)
    i = filter_response(L, alpha, beta, e)
    u = i[L:] - i[L - 1:-1] + e[L - 1:]      # u_s für s = 0..T-L  (i hat die Länge T+1)
    uu = u[burn:]
    ii = i[burn:]
    return float(uu.var()) / float(e.var()), float(ii.var()) / float(e.var())
