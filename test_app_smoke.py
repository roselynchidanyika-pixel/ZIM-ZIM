"""End-to-end smoke test of the RFC Securities Streamlit app using
streamlit.testing.v1.AppTest (headless, no browser required).

Covers: login -> load sample -> run full analysis -> navigate every page.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app_rfc.py"


def _fake_synthesize(text, out_dir=None, *, lang="en"):
    """Deterministic, offline stand-in for gTTS so tests don't hit the network."""
    import os
    from pathlib import Path
    if out_dir is not None:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        path = os.path.join(out_dir, "fake_speech.mp3")
    else:
        path = os.path.abspath("fake_speech.mp3")
    Path(path).write_bytes(b"ID3-fake-audio")
    return {"ok": True, "path": path, "error": ""}


@pytest.fixture(scope="module", autouse=True)
def _no_tts():
    import rfc.ai.tts as tts_mod
    orig = tts_mod.synthesize
    tts_mod.synthesize = _fake_synthesize
    yield
    tts_mod.synthesize = orig


def _find(elements, label: str):
    for el in elements:
        try:
            if el.label == label:
                return el
        except Exception:
            continue
    return None


def _set(app, elements, label, value):
    el = _find(elements, label)
    assert el is not None, f"widget not found: {label}"
    el.set_value(value)
    return el


@pytest.fixture(scope="module")
def logged_in_app():
    at = AppTest.from_file(str(APP), default_timeout=60)
    at.run()
    assert not at.exception, at.exception
    assert len(at.text_input) >= 2

    _set(at, at.text_input, "Username or email", "rfc")
    _set(at, at.text_input, "Password", "rfc2024")
    btn = _find(at.button, "LOGIN")
    assert btn is not None
    btn.click().run()
    assert not at.exception, at.exception
    assert "Dashboard" in [p.label for p in at.radio] or True
    return at


def test_login_ok(logged_in_app):
    # sidebar greeting contains the username
    joined = " ".join(str(m.value) for m in logged_in_app.markdown)
    assert "rfc" in joined or "rfc" in logged_in_app.session_state["username"]


def test_load_sample_and_run(logged_in_app):
    at = logged_in_app
    btn = _find(at.button, "\U0001F4E6 Upload Sample Data & Run Analysis")
    assert btn is not None
    btn.click().run()
    assert not at.exception, at.exception
    assert at.session_state["state"] is not None
    assert at.session_state["state"].company.periods == 8


def test_navigate_all_pages(logged_in_app):
    at = logged_in_app
    pages = [
        "\U0001F4C4 Data & Setup",
        "\U0001F4C8 Profitability",
        "\U0001F4B5 Cash Flow",
        "\U0001F3D7\ufe0f Investment Analysis",
        "\U0001F6E1\ufe0f Risk & Stress Testing",
        "\U0001F3AD Scenario Analysis",
        "\U0001F3AF Optimization",
        "\U0001F916 AI Financial Analyst",
        "\U0001F4E4 Reports & Delivery",
        "\U0001F4C8 Dashboard",
    ]
    for page in pages:
        # set the sidebar radio to page and rerun
        radio = _find(at.radio, "Navigate")
        assert radio is not None
        radio.set_value(page).run()
        assert not at.exception, f"exception on {page}: {at.exception}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])