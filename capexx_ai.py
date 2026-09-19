"""
CAPEXX AI AGENT - Interpretation, Narration & Voice Layer
=========================================================
The robot is the COMMUNICATION layer.  It never calculates anything:
all numbers come from capexx_engine.run_analysis().  This module turns the
engine results into plain-language explanations, a spoken narration script,
and downloadable audio.  If text-to-speech is unavailable the application
degrades gracefully and never crashes.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from capexx_engine import AnalysisBundle, fmt_ccy, fmt_pct, DEMO_DISCLAIMER

AUDIO_DIR = os.path.join("outputs", "audio")


def _fmt_ccy(v, ccy) -> str:
    return fmt_ccy(v, ccy)


def _fmt_pct(v) -> str:
    return fmt_pct(v if v is not None else 0.0)


# --------------------------------------------------------------------------
# AI INTERPRETATION (WHAT / WHY / SO WHAT / WHAT IF / MONITOR)
# --------------------------------------------------------------------------


def generate_ai_interpretation(bundle: AnalysisBundle) -> str:
    from capexx_engine import _ai_interpretation_text
    return _ai_interpretation_text(bundle)


def why_did_capexx_text(bundle: AnalysisBundle) -> str:
    p = bundle.project_input
    m = bundle.metrics
    f = lambda v: _fmt_ccy(v, p.reporting_currency)
    lines = [
        "CAPEXX reached this result by taking the project proposal and running "
        "it through ONE central cash-flow engine.",
        "",
        f"1. INPUT -> The model used {f(p.total_capex())} of bloc capital "
        f"expenditure spread over {p.construction_period} construction year(s), "
        f"base revenue of {f(p.annual_revenue)} ({p.revenue_currency}), recurring "
        f"operating costs of {f(p.annual_opex)} ({p.opex_currency or p.reporting_currency}) "
        f"and a {p.discount_rate:.1f}% discount rate (WACC).",
        "",
        "2. CALCULATION -> These inputs became yearly free cash flows "
        "(CAPEX, working capital, tax, depreciation, salvage, inflation and "
        "expected currency movements are all captured). NPV, IRR, MIRR, "
        "payback and profitability index are computed directly on those flows.",
        "",
        f"3. EVIDENCE -> Base-case NPV = {f(m.npv)}; IRR = {_fmt_pct(m.irr)}; "
        f"risk score = {bundle.risk.score:.0f}/100 ({bundle.risk.level}); "
        f"foreign-exchange risk = {bundle.fx['fx_risk_level']} with "
        f"{bundle.fx['mismatch_count']} mismatched currency stream(s).",
        "",
        f"4. INTERPRETATION -> The decision engine applied transparent rules "
        f"(financial, risk and scenario pillars). The outcome is "
        f"{bundle.decision.status} with a decision score of "
        f"{bundle.decision.score:.0f}.",
        "",
        "Every conclusion in this explanation can be traced back to one of "
        "those four steps. Nothing is guessed.",
    ]
    return "\n".join(lines)


def monitoring_points(bundle: AnalysisBundle) -> List[str]:
    from capexx_engine import monitoring_points as _mp
    return _mp(bundle)


# --------------------------------------------------------------------------
# NARRATION SCRIPT (dynamic - uses the real engine results)
# --------------------------------------------------------------------------


def _readable_ccy(v, ccy) -> str:
    """Human-friendly speech: 'twenty six point five million US dollars'."""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return str(v)
    sign = "negative " if v < 0 else ""
    av = abs(v)
    if av >= 1e9:
        return f"{sign}{av/1e9:.1f} billion {ccy}"
    if av >= 1e6:
        return f"{sign}{av/1e6:.1f} million {ccy}"
    if av >= 1e3:
        return f"{sign}{av/1e3:.1f} thousand {ccy}"
    return f"{sign}{av:.0f} {ccy}"


def _section(h: str, body: str) -> str:
    return f"{h}.\n{body}\n"


def build_narration_text(bundle: AnalysisBundle,
                         include_disclaimer: bool = True) -> str:
    p = bundle.project_input
    m = bundle.metrics
    ccy = p.reporting_currency
    npv = m.npv
    irr = m.irr
    mirr = m.mirr
    payback = m.payback

    out: List[str] = []

    out.append(_section(
        "Introduction",
        "CAPEXX AI has completed the analysis of the selected capital project."))

    out.append(_section(
        "Project overview",
        f"The project analysed is {p.project_name}, a {p.project_type.lower()} "
        f"project located in {p.country}, {p.location}. The reporting currency "
        f"is {ccy}. The total capital expenditure is "
        f"{_readable_ccy(p.total_capex(), ccy)} committed over "
        f"{p.construction_period} construction years, with an expected delay of "
        f"{p.expected_delay_years or 0} years."))

    if "GZU" in p.organisation.upper() or "GZU" in p.project_name.upper() or \
            "Great Zimbabwe" in p.organisation.title():
        out.append(_section(
            "Demonstration notice",
            "Please note that this is a hypothetical demonstration analysis "
            "using invented assumptions. It must not be interpreted as actual "
            "Great Zimbabwe University financial data."))

    out.append(_section(
        "Financial performance",
        f"The calculated net present value is {_readable_ccy(npv, ccy)}. "
        f"This means the project {'creates' if npv >= 0 else 'does not create'} "
        f"that amount of value above the required return of {m.wacc:.1f} "
        f"percent. The internal rate of return is {_fmt_pct(irr)}, and the "
        f"modified internal rate of return is {_fmt_pct(mirr)}, based on the "
        f"modelled financing and reinvestment assumptions. The estimated "
        f"payback period is {payback:.1f} years." if payback else
        f"The calculated net present value is {_readable_ccy(npv, ccy)}. "
        f"The internal rate of return is {_fmt_pct(irr)}, the modified "
        f"internal rate of return is {_fmt_pct(mirr)}, and the project does "
        f"not recover its investment within the modelled horizon."))

    out.append(_section(
        "Risk findings",
        f"CAPEXX identified an overall risk level of {bundle.risk.level}, "
        f"with a risk score of {bundle.risk.score:.0f} out of 100. "
        f"The primary risks are {', '.join(r['driver'] for r in sorted(bundle.sensitivity, key=lambda r: r['delta'], reverse=True)[:3])}. "
        f"Under a 20 percent increase in capital costs, the net present value "
        f"moves to {_readable_ccy(next((s['npv'] for s in bundle.stress if 'CAPEX +20%' in s['name']), npv), ccy)}."))

    exp = bundle.fx["exposures"]
    mism = [e for e in exp if e["mismatch"]]
    if mism:
        names = ", ".join(f"{e['component']} in {e['currency']}" for e in mism[:3])
        out.append(_section(
            "Currency findings",
            f"The project has foreign exchange exposure. {names}, among "
            f"others, are denominated away from the reporting currency {ccy}. "
            f"The engine has incorporated the expected currency movements into "
            f"the cash flows. The currency risk level is "
            f"{bundle.fx['fx_risk_level']}."))
    else:
        out.append(_section(
            "Currency findings",
            "The engine detected no material currency mismatch. Financial flows "
            "are effectively matched with the reporting currency."))

    sc = bundle.scenarios
    out.append(_section(
        "Scenario analysis",
        f"Under the base scenario the net present value is "
        f"{_readable_ccy(sc['BASE CASE']['npv'], ccy)}. "
        f"Under the optimistic scenario, with improved revenues and controlled "
        f"costs, the value becomes {_readable_ccy(sc['OPTIMISTIC']['npv'], ccy)}. "
        f"Under the pessimistic scenario it falls to "
        f"{_readable_ccy(sc['PESSIMISTIC']['npv'], ccy)}. Under the extreme "
        f"stress scenario, with combined adverse shocks, the value becomes "
        f"{_readable_ccy(sc['EXTREME STRESS']['npv'], ccy)}."))

    def _what_block():
        t = []
        t.append(("What happened", f"The model outcome is {bundle.decision.status}. "
                  f"Base-case net present value is {_readable_ccy(npv, ccy)}."))
        if npv >= 0 and (irr or 0) >= m.wacc:
            why = (f"This happened because expected cash flows exceed the "
                   f"required return of {m.wacc:.1f} percent, producing "
                   f"positive value after the initial investment.")
        elif npv >= 0:
            why = (f"The project is only marginally value-accretive: it is "
                   f"sensitive to assumptions and risks.")
        else:
            why = (f"This happened because expected cash flows are not enough "
                   f"to cover the required return of {m.wacc:.1f} percent and "
                   f"the initial investment.")
        t.append(("Why", why))
        so_what = ("Management can proceed with confidence under the stated "
                   "assumptions, subject to the monitoring points in this "
                   "briefing." if bundle.decision.status == "ACCEPT" else
                   "Management should treat this as a conditional outcome and "
                   "resolve the identified gaps before committing capital." if
                   bundle.decision.status == "REVIEW" else
                   "Under the stated assumptions the project is not "
                   "recommended; material improvements would be required.")
        t.append(("So what", so_what))
        t.append(("What if", f"If the sensitive drivers change by 10 percent, "
                  f"the net present value can move by up to "
                  f"{_readable_ccy(max((r['delta'] for r in bundle.sensitivity), default=0.0), ccy)}."))
        t.append(("What should management monitor",
                  "; ".join(monitoring_points(bundle)[:4])))
        return "\n".join(f"{h}.\n{b}" for h, b in t)

    out.append(_section("AI interpretation", _what_block()))

    out.append(_section(
        "Final model outcome",
        f"CAPEXX AI has completed the investment assessment. The model outcome "
        f"is {bundle.decision.status}. This outcome is based on the project's "
        f"calculated financial returns, risk exposure, scenario performance "
        f"and stress test results."))

    if include_disclaimer:
        out.append(_section(
            "Disclaimer",
            "CAPEXX AI is a decision support system. Its outputs depend on the "
            "quality and assumptions of the supplied data. The model does not "
            "replace professional financial, engineering, legal, tax or "
            "investment advice."))

    return "\n\n".join(out)


def section_map(bundle: AnalysisBundle) -> List[Dict[str, str]]:
    """Split the narration into the documented sections for the robot panel."""
    full = build_narration_text(bundle)
    chunks = full.split("\n\n")
    result: List[Dict[str, str]] = []
    for c in chunks:
        c = c.strip()
        if not c:
            continue
        title = c.split(".\n")[0].replace(".", "").strip()[:36] or "Section"
        result.append({"title": title, "text": c})
    return result


# --------------------------------------------------------------------------
# TEXT-TO-SPEECH (graceful degradation)
# --------------------------------------------------------------------------


def tts_available() -> tuple[bool, str]:
    try:
        from gtts import gTTS  # noqa: F401
        try:
            gTTS(text="test", lang="en")
            return True, "Google Text-to-Speech (gTTS) available."
        except Exception:
            return False, "TTS engine failed to initialise (network/test)."
    except Exception:
        return False, "gTTS package not installed."


def synthesize(text: str, out_path: Optional[str] = None,
               lang: str = "en") -> Dict[str, Any]:
    """Generate an MP3 of the narration. Returns {ok, path, error}."""
    try:
        from gtts import gTTS
    except Exception:
        return {"ok": False, "path": None,
                "error": "gTTS is not installed. Voice service unavailable."}
    if not text or not text.strip():
        return {"ok": False, "path": None, "error": "Empty narration text."}
    try:
        if not out_path:
            os.makedirs(AUDIO_DIR, exist_ok=True)
            out_path = os.path.join(AUDIO_DIR, "capexx_project_analysis.mp3")
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        import tempfile
        tts = gTTS(text=text, lang=lang)
        fd, tmp = tempfile.mkstemp(suffix=".mp3")
        os.close(fd)
        try:
            tts.save(tmp)
            with open(tmp, "rb") as src, open(out_path, "wb") as dst:
                dst.write(src.read())
        finally:
            try:
                os.remove(tmp)
            except OSError:
                pass
        return {"ok": True, "path": out_path, "error": None}
    except Exception as ex:
        return {"ok": False, "path": None,
                "error": f"Voice generation failed: {ex}"}


def read_aloud(text: str, out_path: Optional[str] = None) -> Dict[str, Any]:
    """Alias used by the robot panel and ASK AI module."""
    return synthesize(text, out_path)


# --------------------------------------------------------------------------
# ROBOT SCRIPTED LINES (used when no analysis exists yet)
# --------------------------------------------------------------------------

ROBOT_WELCOME = ("Welcome to CAPEXX AI Agent. I am your capital project "
                 "decision assistant. I analyse project cash flows, investment "
                 "returns, risks, currencies, scenarios and financial "
                 "performance to support capital investment decisions.")

ROBOT_GUIDANCE_UPLOAD = ("Please upload your project data or enter the project "
                         "assumptions manually. I will validate the information "
                         "before performing the analysis.")

ROBOT_GUIDANCE_VALIDATED = ("Your project data has been successfully "
                            "validated. The model is ready to analyse the "
                            "capital project.")

ROBOT_GUIDANCE_READY = ("The project is ready. Select Run Analysis to "
                        "evaluate financial performance, risk, currency "
                        "exposure and scenario outcomes.")

ROBOT_MARKET = ("Markets move. Costs change. Currencies fluctuate. Capital "
                "projects are therefore exposed to uncertainty. CAPEXX "
                "evaluates how these changes could affect project value and "
                "investment decisions.")

ROBOT_DEMO_NOTICE = ("This is a demonstration analysis using hypothetical "
                     "project assumptions. It should not be interpreted as "
                     "actual Great Zimbabwe University financial data.")


# --------------------------------------------------------------------------
# DATA TRANSPARENCY HELPERS
# --------------------------------------------------------------------------


def data_status_tag(date_str: str) -> str:
    """Classify a data point as CURRENT / RECENT / HISTORICAL from its date."""
    try:
        d = datetime.fromisoformat(str(date_str).replace("Z", "+00:00"))
        days = (datetime.utcnow().replace(tzinfo=None) - d.replace(tzinfo=None)).days
    except Exception:
        return "UNKNOWN"
    if days < 7:
        return "CURRENT"
    if days < 180:
        return "RECENT"
    return "HISTORICAL"