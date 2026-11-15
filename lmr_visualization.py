"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben."""

from __future__ import annotations

import math

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import lmr_constants as C


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=360):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.25), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def build_impulse(a, title_with: str | None = None):
    """Stoßantwort nach einem Bedarfsstoß e_0 = 1: Bestand (oben) und Bestellung (unten), mit Leitung (durchgezogen) und ohne (gestrichelt)."""
    s = a.settings
    with_name = title_with or f"mit Leitung (β = {s.beta:.2f})".replace(".", ",")
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, subplot_titles=("Bestand minus Soll", "Bestellung minus Mittelbedarf"))
    show_ref = abs(s.beta) > 1e-12
    fig.add_trace(go.Scatter(x=a.t, y=a.i, mode="lines", line=dict(color=C.COLOR_WITH, width=3), name=with_name, legendgroup="mit",
                             hovertemplate="Periode %{x}: %{y:.4f}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=a.t[:-1], y=a.u, mode="lines", line=dict(color=C.COLOR_WITH, width=3), name=with_name, legendgroup="mit", showlegend=False,
                             hovertemplate="Periode %{x}: %{y:.4f}<extra></extra>"), row=2, col=1)
    if show_ref:
        fig.add_trace(go.Scatter(x=a.t, y=a.i0, mode="lines", line=dict(color=C.COLOR_WITHOUT, width=2, dash="dash"), name="ohne Leitung (β = 0)", legendgroup="ohne",
                                 hovertemplate="Periode %{x}: %{y:.4f}<extra></extra>"), row=1, col=1)
        fig.add_trace(go.Scatter(x=a.t[:-1], y=a.u0, mode="lines", line=dict(color=C.COLOR_WITHOUT, width=2, dash="dash"), name="ohne Leitung (β = 0)", legendgroup="ohne",
                                 showlegend=False, hovertemplate="Periode %{x}: %{y:.4f}<extra></extra>"), row=2, col=1)
    fig.update_xaxes(title_text="Periode", row=2, col=1)
    fig.add_hline(y=0, line_width=1, line_color=C.COLOR_GRID)
    return _base(fig, 460)


def _circle_xy(n=200):
    th = np.linspace(0, 2 * math.pi, n)
    return np.cos(th), np.sin(th)


def build_roots(a):
    """Wurzeln von P in der komplexen Ebene mit dem Einheitskreis: außerhalb heißt instabil. Gleiche Maßstäbe über scaleanchor (autorange) plus zwei unsichtbare Eckpunkte."""
    cx, cy = _circle_xy()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cx, y=cy, mode="lines", line=dict(color=C.COLOR_GRID, width=2), name="Einheitskreis", hoverinfo="skip"))
    if abs(a.settings.beta) > 1e-12:
        fig.add_trace(go.Scatter(x=a.roots0.real, y=a.roots0.imag, mode="markers", marker=dict(color=C.COLOR_WITHOUT, size=9, symbol="x"), name="ohne Leitung (β = 0)",
                                 hovertemplate="%{x:.3f} + %{y:.3f} i<extra></extra>"))
    fig.add_trace(go.Scatter(x=a.roots.real, y=a.roots.imag, mode="markers", marker=dict(color=C.COLOR_WITH, size=10), name="mit Leitung",
                             hovertemplate="%{x:.3f} + %{y:.3f} i<extra></extra>"))
    ext = float(max(1.25, 1.1 * max(np.abs(a.roots).max(), np.abs(a.roots0).max())))
    fig.add_trace(go.Scatter(x=[-ext, ext], y=[-ext, ext], mode="markers", marker=dict(opacity=0), showlegend=False, hoverinfo="skip"))
    fig.update_xaxes(title_text="Realteil", scaleanchor="y", scaleratio=1)
    fig.update_yaxes(title_text="Imaginärteil")
    return _base(fig, 420)


def build_grenzkarte(betas, curves: dict[int, np.ndarray], y_max: float = 4.5):
    """α_krit gegen β für die gewählten L; gestrichelt: die Gerade α = 2 und die Kurve 2/(2β−1) (gerades L, β > 1); senkrecht: β = ½."""
    fig = go.Figure()
    b = np.asarray(betas, dtype=float)
    for L, y in curves.items():
        yy = np.where(np.isfinite(y) & (y <= y_max), y, np.nan)
        fig.add_trace(go.Scatter(x=b, y=yy, mode="lines", line=dict(color=C.L_COLORS.get(L, C.COLOR_NEUTRAL), width=2.5), name=f"L = {L}",
                                 hovertemplate="β = %{x:.2f}: α_krit = %{y:.3f}<extra>L = " + str(L) + "</extra>"))
    bb = b[b > 0.5 + 1e-9]
    fig.add_trace(go.Scatter(x=bb, y=np.minimum(2.0 / (2.0 * bb - 1.0), y_max), mode="lines", line=dict(color="rgba(80,80,80,0.8)", width=1.5, dash="dot"),
                             name="2/(2β − 1)", hoverinfo="skip"))
    fig.add_hline(y=2.0, line_width=1, line_dash="dash", line_color=C.COLOR_GRID)
    fig.add_vline(x=0.5, line_width=1, line_dash="dash", line_color=C.COLOR_GRID)
    fig.update_xaxes(title_text="Leitungsgewicht β")
    fig.update_yaxes(title_text="Grenze α_krit (darüber instabil)", range=[0, y_max])
    return _base(fig, 400)


def build_beta1_rho(alphas, curves: dict[int, np.ndarray]):
    """β = 1: Spektralradius gegen α für mehrere L: alle Kurven liegen auf |1 − α| (gestrichelt)."""
    fig = go.Figure()
    for k, (L, y) in enumerate(curves.items()):
        fig.add_trace(go.Scatter(x=alphas, y=y, mode="lines", line=dict(color=C.L_COLORS.get(L, C.COLOR_NEUTRAL), width=4 - k * 0.7), name=f"L = {L}",
                                 hovertemplate="α = %{x:.2f}: ρ = %{y:.3f}<extra>L = " + str(L) + "</extra>"))
    fig.add_trace(go.Scatter(x=alphas, y=np.abs(1.0 - np.asarray(alphas)), mode="lines", line=dict(color="rgba(30,30,30,0.9)", width=1.5, dash="dash"), name="|1 − α|", hoverinfo="skip"))
    fig.add_hline(y=1.0, line_width=1, line_dash="dash", line_color=C.COLOR_GRID)
    fig.update_xaxes(title_text="Anpassung α")
    fig.update_yaxes(title_text="Spektralradius ρ")
    return _base(fig, 360)


def build_stock_compare(t, series: list[tuple[str, np.ndarray, str, str]]):
    """Bestand nach einem Bedarfsstoß für mehrere Einstellungen: (Name, Werte, Farbe, Strichart)."""
    fig = go.Figure()
    for name, y, color, dash in series:
        fig.add_trace(go.Scatter(x=t[: len(y)], y=y, mode="lines", line=dict(color=color, width=2.5, dash=dash), name=name, hovertemplate="Periode %{x}: %{y:.4f}<extra></extra>"))
    fig.add_hline(y=0, line_width=1, line_color=C.COLOR_GRID)
    fig.update_xaxes(title_text="Periode")
    fig.update_yaxes(title_text="Bestand minus Soll")
    return _base(fig, 340)


def build_sensitivity(rows: list[dict], L: int):
    """Abklingzeit (Perioden bis 1 %) gegen β bei festem α und L; Lücken, wo der Stoß nicht abklingt."""
    b = [r["beta"] for r in rows]
    y = [np.nan if r["decay"] is None else r["decay"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=b, y=y, mode="lines+markers", marker=dict(size=4), line=dict(color=C.COLOR_WITH, width=2.5), name=f"L = {L}",
                             hovertemplate="β = %{x:.2f}: %{y} Perioden<extra></extra>"))
    fig.add_vline(x=1.0, line_width=1, line_dash="dash", line_color=C.COLOR_GRID)
    fig.update_xaxes(title_text="Leitungsgewicht β")
    fig.update_yaxes(title_text="Abklingzeit (Perioden bis 1 %)", type="log")
    return _base(fig, 340)


def build_drift(res, d: float, beta: float):
    """Bestand minus Soll in Perioden Bedarf über der Zeit: mit Rauschen, ohne Rauschen und der exakte Dauerfehler −β·ΔW."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=res.t, y=res.i_noisy / d, mode="lines", line=dict(color="rgba(76,120,168,0.55)", width=1.5), name="mit Bedarfsrauschen"))
    fig.add_trace(go.Scatter(x=res.t, y=res.i_clean / d, mode="lines", line=dict(color=C.COLOR_WITH, width=3), name="ohne Rauschen"))
    fig.add_trace(go.Scatter(x=[res.t[0], res.t[-1]], y=[res.i_star / d] * 2, mode="lines", line=dict(color=C.COLOR_WITHOUT, width=2, dash="dash"), name="−β·ΔW (exakt)"))
    fig.add_hline(y=0, line_width=1, line_color=C.COLOR_GRID)
    fig.update_xaxes(title_text="Periode")
    fig.update_yaxes(title_text="Bestand minus Soll (Perioden Bedarf)")
    return _base(fig, 360)
