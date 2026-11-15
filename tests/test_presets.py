"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen, keine Zweierpotenz als α oder β außer dem benannten β = 1."""

import math

import lmr_constants as C
import lmr_evaluation as ev
import lmr_presets as P


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_slider_bounds():
    for name, p in C.PRESETS.items():
        assert C.L_MIN <= p["L"] <= C.L_MAX and C.ALPHA_MIN <= p["alpha"] <= C.ALPHA_MAX, name
        assert C.BETA_MIN <= p["beta"] <= C.BETA_MAX and C.LHAT_DELTA_MIN <= p["lhat_delta"] <= C.LHAT_DELTA_MAX, name


def test_default_preset_equals_the_default_settings():
    first = C.PRESETS[next(iter(C.PRESETS))]
    assert ev.Settings(first["L"], first["alpha"], first["beta"]) == ev.Settings()
    assert first["lhat_delta"] == 0 and first["L"] == C.DEFAULT_L


def test_setting_specs_cover_every_preset_key():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)


def _is_power_of_two(x: float) -> bool:
    if x <= 0:
        return False
    m, _e = math.frexp(x)
    return m == 0.5


def test_no_alpha_or_beta_preset_is_a_power_of_two_except_the_named_beta_1_and_beta_0():
    """α nie; β = ½, ¼ nie (β = 1 steht als Satz und gemessene Spitze, β = 0 heißt „ohne Leitung“ und ist keine Zweierpotenz im Sinn der Regel)."""
    for name, p in C.PRESETS.items():
        assert not _is_power_of_two(p["alpha"]), name
        if p["beta"] not in (0.0, 1.0):
            assert not _is_power_of_two(abs(p["beta"])), name


def test_beta_half_is_never_a_preset_but_a_grid_point():
    assert all(abs(p["beta"] - 0.5) > 1e-9 for p in C.PRESETS.values())
    assert min(abs(b - 0.5) for b in C.GRID_BETAS) < 1e-9                                  # die Grenzkarte enthält β = ½ als Gitterpunkt


def test_grids_and_defaults_are_consistent():
    assert len(C.GRID_BETAS) == 151 and C.GRID_BETAS[0] == -1.0 and C.GRID_BETAS[-1] == 2.0
    assert len(C.SENS_BETAS) == 101 and C.SENS_BETAS[0] == 0.5 and C.SENS_BETAS[-1] == 1.5
    assert set(C.GRENZKARTE_L_DEFAULT) <= set(range(C.L_MIN, C.L_MAX + 1))
    assert C.PRESETS["Drift (L̂ = L − 1)"]["lhat_delta"] == -1
