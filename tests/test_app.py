"""End-to-end Smoke-Test via Streamlits offiziellem AppTest-Framework: laedt app.py mit den
Standardeinstellungen, klickt jeden Preset-Button und faehrt jeden Slider an seine Grenzen und prueft, dass
kein Python-Fehler auftritt - insbesondere `streamlit.errors.StreamlitDuplicateElementId` (mehrere
st.plotly_chart-Aufrufe ohne eindeutiges key= koennen zufaellig identischen Inhalt rendern und kollidieren)
und `StreamlitAPIException` bei Slidern mit berechneten Grenzen (min == max)."""

import os

from streamlit.testing.v1 import AppTest

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = int(os.environ.get("APPTEST_TIMEOUT", "300"))  # unter Last (parallele Läufe) reichen 180 s nicht immer; Umgebungsvariable zum Anpassen
AUTOPLAY_MARKER = "\u25b6"  # Auto-Play-Buttons (Animation mit sleep) im Smoke-Test auslassen


def _fresh_app() -> AppTest:
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return at


def test_app_loads_without_exception():
    _fresh_app()


def test_every_button_click_does_not_raise():
    labels = [b.label for b in _fresh_app().button]
    assert labels, "keine Buttons gefunden"
    for index, label in enumerate(labels):
        if AUTOPLAY_MARKER in label:
            continue
        at = _fresh_app()
        at.button[index].click().run(timeout=TIMEOUT)
        assert not at.exception, (label, [str(e) for e in at.exception])


def test_every_slider_at_its_min_and_max_does_not_raise():
    labels = [s.label for s in _fresh_app().slider]
    assert labels, "keine Slider gefunden"
    for index, label in enumerate(labels):
        for edge in ("min", "max"):
            at = _fresh_app()
            slider = at.slider[index]
            value = slider.min if edge == "min" else slider.max
            if isinstance(slider.value, int):
                value = int(value)
            slider.set_value(value).run(timeout=TIMEOUT)
            assert not at.exception, (label, edge, [str(e) for e in at.exception])


# --- Ergänzungen zu dieser Demo (der Anfang der Datei bleibt die Vorlage) ------------------------------------------------------------

import pytest  # noqa: E402

import lmr_constants as C  # noqa: E402


def _texts(at):
    return " ".join(x.value for kind in (at.success, at.info, at.warning, at.error) for x in kind)


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_each_preset_runs_and_shows_the_main_verdict(name):
    at = _fresh_app()
    at.sidebar.button(key=f"preset_{name}").click().run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    text = _texts(at)
    p = C.PRESETS[name]
    assert f"L = {p['L']}, α = " in text and "Geschätzte Lieferzeit" in text and "Abklingzeit" in " ".join(m.label for m in at.metric)
    assert at.slider(key="lhat_delta_slider").value == p["lhat_delta"]


def test_all_sliders_at_the_minimum_and_at_the_maximum_together():
    for edge in ("min", "max"):
        at = _fresh_app()
        for s in at.slider:
            v = s.min if edge == "min" else s.max
            s.set_value(int(v) if isinstance(s.value, int) else v)
        at.run(timeout=TIMEOUT)
        assert not at.exception, (edge, [str(e) for e in at.exception])


def test_the_limit_map_handles_empty_and_full_selection():
    at = _fresh_app()
    at.multiselect(key="gk_L").set_value([]).run(timeout=TIMEOUT)
    assert not at.exception and any("mindestens eine Lieferzeit" in i.value for i in at.info)
    at.multiselect(key="gk_L").set_value([1, 2, 3, 4, 5, 24]).run(timeout=TIMEOUT)
    assert not at.exception and any("bei 6 von 6 gewählten L" in i.value for i in at.info)


def test_drift_section_in_the_unstable_case_shows_a_warning_and_no_chart_error():
    at = _fresh_app()
    at.slider(key="L_slider").set_value(5)
    at.slider(key="alpha_slider").set_value(1.5)
    at.slider(key="beta_slider").set_value(0.0)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    assert any("erreicht der Bestand kein Gleichgewicht" in w.value for w in at.warning)


def test_nonpositive_estimated_delay_shows_the_note():
    at = _fresh_app()
    at.slider(key="L_slider").set_value(2)
    at.slider(key="lhat_delta_slider").set_value(-3)
    at.run(timeout=TIMEOUT)
    assert not at.exception and any("physikalisch sinnlos" in c.value for c in at.caption)


def test_footer_and_permalink_do_not_break_the_app():
    at = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)
    at.query_params["L"] = "99"        # wird auf die Reglergrenze begrenzt
    at.query_params["beta"] = "0.8"
    at.run(timeout=TIMEOUT)
    assert not at.exception and at.slider(key="L_slider").value == C.L_MAX and at.slider(key="beta_slider").value == pytest.approx(0.8)
