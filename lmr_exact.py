"""Exakte Rechnung in Brüchen (nur Standardbibliothek): dasselbe Lager wie in `lmr_dynamics`, ohne Rundung.

Zweck: Handbeispiele und Gegenproben. Die Zustandsmatrix ist hier unabhängig von `lmr_dynamics` aufgebaut (Fractions, eigene Schleifen);
das Gleichgewicht löst ein Gauß-Verfahren mit Brüchen. Alle Funktionen nehmen int, Fraction oder str (z. B. "6/5") als Parameter."""

from __future__ import annotations

from fractions import Fraction as F


def fr(x) -> F:
    """Parameter als Bruch: Strings wie "6/5" und ganze Zahlen exakt, Gleitkommazahlen über ihre Dezimaldarstellung (0.55 -> 11/20)."""
    if isinstance(x, F):
        return x
    if isinstance(x, float):
        return F(repr(x))
    return F(x)


def matrix_exact(L: int, alpha, beta) -> tuple[list[list[F]], list[F]]:
    """Zustandsmatrix A (x_t = (i_t, u_{t-1}, ..., u_{t-L+1})) und Zeile c der Bestellung u_t = c . x_t, in Brüchen."""
    alpha, beta = fr(alpha), fr(beta)
    c = [-alpha * beta] * L
    c[0] = -alpha
    if L == 1:
        return [[F(1) + c[0]]], c
    A = [[F(0)] * L for _ in range(L)]
    A[0][0] = F(1)
    A[0][L - 1] += 1
    A[1] = c[:]
    for j in range(2, L):
        A[j][j - 1] = F(1)
    return A, c


def matmul(X: list[list[F]], Y: list[list[F]]) -> list[list[F]]:
    n = len(X)
    return [[sum((X[r][m] * Y[m][q] for m in range(n)), F(0)) for q in range(n)] for r in range(n)]


def identity(n: int) -> list[list[F]]:
    return [[F(int(r == q)) for q in range(n)] for r in range(n)]


def smallest_power_identity(A: list[list[F]], limit: int = 200) -> int | None:
    """Kleinstes k >= 1 mit A^k = I (exakt), sonst None bis `limit`."""
    n = len(A)
    M = [row[:] for row in A]
    eye = identity(n)
    for k in range(1, limit + 1):
        if M == eye:
            return k
        M = matmul(M, A)
    return None


def impulse_exact(L: int, alpha, beta, T: int) -> tuple[list[F], list[F]]:
    """Bedarfsstoß e_0 = 1: Bestand i_1..i_T und Bestellung u_0..u_{T-1} in Brüchen (u_0 = 0: der Stoß steht erst in i_1)."""
    A, c = matrix_exact(L, alpha, beta)
    x = [F(0)] * L
    x[0] = F(-1)
    ii, uu = [x[0]], [F(0)]
    for _ in range(1, T):
        uu.append(sum((c[k] * x[k] for k in range(L)), F(0)))
        x = [sum((A[r][q] * x[q] for q in range(L)), F(0)) for r in range(L)]
        ii.append(x[0])
    return ii, uu


def steady_state_exact(L: int, alpha, beta, dW) -> tuple[F, list[F]]:
    """Gleichgewicht bei konstantem Fehler dW der Soll-Leitung: löst (I - A) x = b mit b = (0, -alpha beta dW, 0, ...) und dem Beitrag -alpha beta dW in u.

    Der Zustand ist (i, u_{t-1}, ..., u_{t-L+1}); die neue Bestellung ist u = c . x - alpha beta dW. Rückgabe (i*, [alle gespeicherten Bestellungen])."""
    alpha, beta, dW = fr(alpha), fr(beta), fr(dW)
    A, c = matrix_exact(L, alpha, beta)
    off = -alpha * beta * dW                       # Anteil von dW an jeder neuen Bestellung
    n = L
    # x' = A x + g mit g = (0, off, 0, ...) (L >= 2) bzw. x' = A x + off (L = 1, Bestellung geht sofort in i ein)
    g = [F(0)] * n
    if L == 1:
        g[0] = off
    else:
        g[1] = off
    # (I - A) x = g, Gauß mit Brüchen
    M = [[(F(int(r == q)) - A[r][q]) for q in range(n)] + [g[r]] for r in range(n)]
    for col in range(n):
        piv = next(r for r in range(col, n) if M[r][col] != 0)
        M[col], M[piv] = M[piv], M[col]
        pv = M[col][col]
        M[col] = [v / pv for v in M[col]]
        for r in range(n):
            if r != col and M[r][col] != 0:
                f = M[r][col]
                M[r] = [a - f * b for a, b in zip(M[r], M[col])]
    x = [M[r][n] for r in range(n)]
    return x[0], x[1:]


def doubled_root_check(L: int) -> tuple[bool, bool]:
    """beta = 0, alpha* = (1/L) ((L-1)/L)^(L-1): P(z*) = 0 und P'(z*) = 0 bei z* = (L-1)/L, exakt (nur L >= 2)."""
    zs = F(L - 1, L)
    a = zs ** (L - 1) / L
    P = zs ** L - zs ** (L - 1) + a
    dP = L * zs ** (L - 1) - (L - 1) * zs ** (L - 2)
    return P == 0, dP == 0
