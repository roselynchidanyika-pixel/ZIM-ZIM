"""
RFC SECURITIES — AI Financial Modelling Platform
================================================
TURN UNCERTAINTY INTO PROFITABILITY.  MODEL. PREDICT. OPTIMIZE.

One unified AI financial modelling agent connecting five engines:

    PROFITABILITY  ->  CASH FLOW  ->  INVESTMENT  ->  RISK  ->  OPTIMIZATION

with a Robotic AI Assistant (text-to-speech) across the whole system.
"""
from __future__ import annotations

import copy
import hashlib
import io
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from rfc.config import (
    BRAND, TAGLINE, SUBTAGLINE, MODEL_CHAIN, DEEP_GREEN, GREEN, GOLD,
    GOLD_LIGHT, WHITE, OFF_WHITE, GREEN_MINT, RED, AMBER,
    SUPPORTED_CURRENCIES, fmt_currency, fmt_pct, fmt_num,
)
from rfc.engine import CompanyInput, run_full_analysis
from rfc.models.investment import ProjectSpec, evaluate_project
from rfc.models.optimization import optimise_price, allocate_capital, select_projects
from rfc.data import auth, loader, samples
from rfc.ai import analyst as rfc_analyst
from rfc.ai import tts as rfc_tts
from rfc.reports import build_html_report, build_json_export, csv_export_for

logging.basicConfig(level=logging.WARNING)

ROOT = Path(__file__).resolve().parent
OUTPUT_AU = ROOT / "outputs" / "rfc_audio"
OUTPUT_AU.mkdir(parents=True, exist_ok=True)

st.set_page_config(
    page_title="RFC Securities — AI Financial Modelling Platform",
    layout="wide",
    page_icon="\U0001F4C8",
    initial_sidebar_state="expanded",
)

MODEL_STEPS = ["PROFITABILITY", "CASH FLOW", "INVESTMENT", "RISK", "OPTIMIZATION"]


# ══════════════════════════════════════════════════════════════════════════
# THEME / CSS — white background, deep green + gold accents
# ══════════════════════════════════════════════════════════════════════════
def inject_css():
    st.markdown(f"""
<style>
:root{{--green:{DEEP_GREEN};--green2:{GREEN};--gold:{GOLD};}}
header[data-testid="stHeader"]{{background:{DEEP_GREEN}!important;}}
header [data-testid="stHeader"] *{{color:#fff!important;}}
section[data-testid="stSidebar"]{{background:linear-gradient(180deg,{DEEP_GREEN},#10523C)!important;}}
section[data-testid="stSidebar"] *{{color:#fff!important;}}
section[data-testid="stSidebar"] label{{color:#C9DED3!important;}}
div[data-testid="stToolbar"]{{background:{DEEP_GREEN}!important;}}
.stApp{{background:#fff;}}
h1,h2,h3,h4{{color:{DEEP_GREEN};}}
hr{{border-color:#E3EAE5;}}

.rfc-hero{{background:linear-gradient(135deg,{DEEP_GREEN},{GREEN});color:#fff;
  border-radius:14px;padding:1.6rem 2rem;margin:.4rem 0 1rem;}}
.rfc-hero h1{{margin:0;font-size:2.1rem;color:{GOLD_LIGHT};letter-spacing:.05em;}}
.rfc-hero .tag{{color:{GOLD_LIGHT};letter-spacing:.28em;font-size:.8rem;font-weight:600;}}
.rfc-hero .sub{{color:#C9DED3;font-size:.95rem;margin-top:.35rem;letter-spacing:.14em;}}
.rfc-chain{{display:flex;flex-wrap:wrap;gap:.3rem;align-items:center;margin:.4rem 0 1rem;
  font-size:.78rem;font-weight:700;}}
.rfc-chain .c{{background:{GREEN_MINT};color:{DEEP_GREEN};border:1px solid {GREEN};
  border-radius:8px;padding:.28rem .6rem;}}
.rfc-chain .a{{color:#9C7C1E;}}

.rfc-card{{background:#FFFFFF;border:1px solid #E3EAE5;border-left:5px solid {GOLD};
  border-radius:10px;padding:1rem 1.2rem;margin:.5rem 0;}}
.rfc-card h4{{margin:0 0 .25rem;color:{DEEP_GREEN};}}
.rfc-card p{{margin:.2rem 0;color:#334155;font-size:.9rem;}}
.rfc-kpi{{background:{OFF_WHITE};border-top:4px solid {GOLD};border-radius:10px;
  padding:1rem 1.1rem;}}
.rfc-kpi .lbl{{font-size:.72rem;color:#5B7266;text-transform:uppercase;letter-spacing:.05em;}}
.rfc-kpi .val{{font-size:1.5rem;font-weight:800;color:{DEEP_GREEN};margin:.1rem 0;}}
.rfc-kpi .sub{{font-size:.74rem;color:#5B7266;}}
.rfc-good{{color:{GREEN}!important;}} .rfc-bad{{color:{RED}!important;}}
.rfc-banner{{border-radius:10px;padding:1rem 1.3rem;font-size:1.05rem;font-weight:700;
  text-align:center;margin:.5rem 0;}}
.rfc-banner-ok{{background:{GREEN_MINT};color:{DEEP_GREEN};border:2px solid {GREEN};}}
.rfc-banner-warn{{background:#FDF3E0;color:#7A5B12;border:2px solid {AMBER};}}
.rfc-note{{background:{OFF_WHITE};border-left:4px solid {GOLD};border-radius:8px;
  padding:.75rem 1rem;line-height:1.5;font-size:.9rem;margin:.5rem 0;}}
.rfc-robot-card{{background:linear-gradient(135deg,{DEEP_GREEN},{GREEN});color:#fff;
  border:1px solid {GOLD};border-radius:14px;padding:1.1rem 1.3rem;margin:.6rem 0}}
.rfc-robot-card h4{{color:{GOLD_LIGHT};margin:0 0 .3rem;}}
.rfc-robot-card p{{color:#fff;font-size:.9rem;line-height:1.5;margin:.2rem 0;}}
@media (max-width:800px){{.rfc-hero h1{{font-size:1.5rem;}}}}
</style>
""", unsafe_allow_html=True)


def kpi(label, value, sub="", good=None):
    cls = " rfc-good" if good is True else (" rfc-bad" if good is False else "")
    st.markdown(
        f'<div class="rfc-kpi"><div class="lbl">{label}</div>'
        f'<div class="val{cls}">{value}</div>'
        f'<div class="sub">{sub}</div></div>', unsafe_allow_html=True)


def card(title, body):
    st.markdown(f'<div class="rfc-card"><h4>{title}</h4><p>{body}</p></div>',
                unsafe_allow_html=True)


def banner(text, ok=True):
    cls = "rfc-banner-ok" if ok else "rfc-banner-warn"
    st.markdown(f'<div class="rfc-banner {cls}">{text}</div>',
                unsafe_allow_html=True)


def note(text):
    st.markdown(f'<div class="rfc-note">{text}</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════════
def _init_state():
    defaults = {
        "logged_in": False, "username": "", "role": "Guest",
        "ci": None, "projects": None,
        "state": None,
        "custom_scenario": None,
        "risk_cfg": {"mc_iterations": 1000, "seed": 2026},
        "opt_cfg": {},
        "_welcome_done": False, "_welcome_path": None, "_welcome_played": False,
        "_speak_cache": {},
        "asked": None, "asked_answer": None,
        "toast_shown": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()


def load_sample():
    st.session_state["ci"] = samples.sample_company()
    st.session_state["projects"] = list(samples.sample_projects())
    st.session_state["state"] = None
    st.session_state["custom_scenario"] = None


def recompute(custom_scenario=None, risk_cfg=None, opt_cfg=None,
              new_ci=None, show_progress=True):
    ci = new_ci if new_ci is not None else st.session_state["ci"]
    if ci is None:
        st.warning("Load data first (samples or upload).")
        return None
    steps = ["Profitability model", "Cash-flow model", "Investment model",
             "Risk & stress model", "Optimization model", "AI narrative"]
    pbar = None
    if show_progress:
        pbar = st.progress(0, text="Running the interconnected models\u2026")
    st.session_state["custom_scenario"] = custom_scenario or st.session_state.get(
        "custom_scenario")
    st.session_state["risk_cfg"] = dict(risk_cfg or st.session_state["risk_cfg"])
    st.session_state["opt_cfg"] = dict(opt_cfg or st.session_state["opt_cfg"])
    n = len(steps)
    for i, s in enumerate(steps):
        if pbar:
            pbar.progress((i + 1) / n, text=s + " \u2026")
        time.sleep(0.02)
    state = run_full_analysis(
        ci,
        projects=st.session_state["projects"] or [],
        scenario_custom=st.session_state["custom_scenario"],
        risk_config=st.session_state["risk_cfg"],
        opt_config=st.session_state["opt_cfg"],
        computed_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    )
    st.session_state["state"] = state
    if pbar:
        pbar.empty()
    return state


def get_state():
    return st.session_state.get("state")


def has_state():
    return st.session_state.get("state") is not None


def require_state() -> bool:
    if not has_state():
        st.info("\U0001F441\ufe0f Load sample data or upload your financials, then "
                "press **Run Full Analysis**. The five models are interconnected, "
                "so one click runs everything.")
        return False
    return True


# ══════════════════════════════════════════════════════════════════════════
# ROBOT / TTS
# ══════════════════════════════════════════════════════════════════════════
def speak_path(text: str) -> str | None:
    """Generate (once) and return the cached MP3 path for a text."""
    key = hashlib.sha1((text or "").encode("utf-8")).hexdigest()
    cache = st.session_state.setdefault("_speak_cache", {})
    if key in cache and Path(cache[key]).exists():
        return cache[key]
    path = OUTPUT_AU / f"{key}.mp3"
    if path.exists():
        cache[key] = str(path)
        return str(path)
    res = rfc_tts.synthesize(text, out_dir=str(OUTPUT_AU))
    if res["ok"]:
        cache[key] = res["path"]
        return res["path"]
    return None


def robot_panel(text: str, *, headline="RFC Securities Robotic AI Assistant",
                autoplay=False, show_text=True):
    """Visible robotic assistant panel reading ``text`` aloud."""
    st.markdown(
        f'<div class="rfc-robot-card"><h4>\U0001F916 {headline}</h4>'
        f'<p>{rfc_analyst.ROBOT_WELCOME}</p></div>',
        unsafe_allow_html=True)
    st.markdown(
        "<small>Controls: <b>Play</b> (start) · <b>Pause</b>/<b>Resume</b> · "
        "<b>Stop</b> · <b>Mute</b> · <b>Restart</b> · speed selector, and "
        "download the MP3.</small>", unsafe_allow_html=True)
    path = speak_path(text)
    if not path:
        st.info("\U0001F507 Voice service is unavailable right now (no internet "
                "or gTTS unavailable). The robot text is shown below.")
        if show_text:
            st.code(text, language=None)
        return
    html = rfc_tts.audio_player_html(path, autoplay=autoplay,
                                     headline=headline,
                                     visible_text=text if show_text else "")
    st.components.v1.html(html, height=320)


def robot_welcome_block():
    if not st.session_state["_welcome_done"]:
        st.session_state["_welcome_path"] = speak_path(rfc_analyst.ROBOT_WELCOME)
        st.session_state["_welcome_done"] = True
    autoplay = not st.session_state.get("_welcome_played", False)
    if autoplay:
        st.session_state["_welcome_played"] = True
    st.markdown(
        f'<div class="rfc-robot-card"><h4>\U0001F916 {BRAND} Robotic AI Assistant</h4>'
        f'<p style="font-size:1.05rem;font-weight:700;color:{GOLD_LIGHT};">'
        f'"{rfc_analyst.ROBOT_WELCOME}"</p></div>',
        unsafe_allow_html=True)
    if st.session_state["_welcome_path"] and Path(st.session_state["_welcome_path"]).exists():
        wpath = st.session_state["_welcome_path"]
        html = rfc_tts.audio_player_html(
            wpath, autoplay=autoplay, headline="RFC Securities Robotic AI Assistant",
            visible_text=rfc_analyst.ROBOT_WELCOME)
        st.components.v1.html(html, height=300)
    else:
        st.info("\U0001F507 Voice service unavailable — the robot welcomes you "
                "in text: **Welcome to RFC Securities. How may I help you?**")


# ══════════════════════════════════════════════════════════════════════════
# PLOT HELPERS
# ══════════════════════════════════════════════════════════════════════════
def money_df(df: pd.DataFrame, exclude=()) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if c in exclude or not pd.api.types.is_numeric_dtype(out[c]):
            continue
        out[c] = out[c].map(lambda v: f"{v:,.0f}")
    return out


def styled_plot(fig: go.Figure, title: str = "", height: int = 380):
    fig.update_layout(
        title=title, template="plotly_white", height=height,
        font=dict(family="Segoe UI, Arial", color="#0F172A"),
        paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
        margin=dict(l=40, r=20, t=52, b=40),
        title_font=dict(color=DEEP_GREEN, size=15),
    )
    fig.update_xaxes(gridcolor="#E9EFEA")
    fig.update_yaxes(gridcolor="#E9EFEA")
    st.plotly_chart(fig, width="stretch")


# ══════════════════════════════════════════════════════════════════════════
# LOGIN PAGE
# ══════════════════════════════════════════════════════════════════════════
def login_page():
    inject_css()
    st.markdown(f"""
<div class="rfc-hero" style="text-align:center;padding:2.4rem 2rem;">
  <div class="tag">R F C &nbsp;S E C U R I T I E S</div>
  <h1 style="margin:.4rem 0;font-size:2.5rem;">TURN UNCERTAINTY INTO PROFITABILITY</h1>
  <div class="sub" style="font-size:1rem;letter-spacing:.3em;">MODEL. PREDICT. OPTIMIZE.</div>
</div>
""", unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1, 1.4, 1])
    with c2:
        st.markdown("#### \U0001F510 Sign in to your workspace")
        tab_login, tab_register = st.tabs(["LOGIN", "REGISTER"])
        with tab_login:
            user = st.text_input("Username or email", key="lg_user")
            pw = st.text_input("Password", type="password", key="lg_pw")
            if st.button("LOGIN", type="primary", width="stretch"):
                res = auth.verify(user, pw)
                if res["ok"]:
                    st.session_state["logged_in"] = True
                    st.session_state["username"] = res["username"]
                    st.session_state["role"] = res["role"]
                    st.session_state["_welcome_done"] = False
                    st.rerun()
                else:
                    st.error(res["message"])
        with tab_register:
            ru = st.text_input("Choose username", key="rg_user")
            rp = st.text_input("Choose password", type="password", key="rg_pw")
            if st.button("REGISTER", width="stretch"):
                res = auth.register(ru, rp)
                if res["ok"]:
                    st.success(res["message"])
                else:
                    st.warning(res["message"])
        st.markdown(
            '<div class="rfc-note"><b>Demo access</b><br>'
            "Admin: <code>rfc</code> / <code>rfc2024</code><br>"
            "Analyst: <code>demo</code> / <code>rfc2024</code></div>",
            unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════
PAGES = [
    "\U0001F4C8 Dashboard",
    "\U0001F4C4 Data & Setup",
    "\U0001F4C8 Profitability",
    "\U0001F4B5 Cash Flow",
    "\U0001F3D7\ufe0f Investment Analysis",
    "\U0001F6E1\ufe0f Risk & Stress Testing",
    "\U0001F3AD Scenario Analysis",
    "\U0001F3AF Optimization",
    "\U0001F916 AI Financial Analyst",
    "\U0001F4E4 Reports & Delivery",
]


def sidebar():
    with st.sidebar:
        st.markdown(
            f"<div style='background:#FFFFFF14;border-radius:10px;padding:.7rem .9rem;'>"
            f"<div style='font-weight:800;color:{GOLD_LIGHT};letter-spacing:.12em;'>"
            f"RFC SECURITIES</div>"
            f"<div style='font-size:.68rem;color:#C9DED3;letter-spacing:.18em;'>"
            f"{TAGLINE}</div>"
            f"<div style='font-size:.62rem;color:#C9DED3;margin-top:.3rem;'>"
            f"{st.session_state['username']} · {st.session_state['role']}</div>"
            f"</div>", unsafe_allow_html=True)
        page = st.radio("Navigate", PAGES, label_visibility="collapsed")
        st.divider()
        if st.session_state.get("state") is not None:
            st.success("Analysis ready")
        else:
            st.warning("No analysis yet")
        st.caption(f"{BRAND} © 2026 · {SUBTAGLINE}")
        st.caption("Decision-support only · Not financial advice")
        return page


# ══════════════════════════════════════════════════════════════════════════
# DATA & SETUP
# ══════════════════════════════════════════════════════════════════════════
def data_page():
    st.subheader("\U0001F4C4 Data & Model Setup")
    st.markdown(f'<div class="rfc-chain">' + "".join(
        [f'<span class="c">{s}</span>' for s in MODEL_STEPS]) +
        '<span class="a"> \u27A1 all interconnected</span></div>',
        unsafe_allow_html=True)
    note("The five models share **one** central financial modelling engine. "
         "Load or upload financial data once, then **Run Full Analysis** — "
         "profitability feeds cash flow, cash flow feeds the investment "
         "metrics, risk re-runs the whole chain under shocks, and "
         "optimization maximises the result under your constraints.")

    t_sample, t_upload, t_assume = st.tabs(
        ["\U0001F4E6 Sample Data", "\u2B06\ufe0f Upload Data",
         "\u2699\ufe0f Assumptions & Inputs"])

    with t_sample:
        c1, c2 = st.columns([1.3, 1])
        with c1:
            st.markdown("### Sample datasets")
            st.markdown("Two downloadable sample datasets let you test the "
                        "platform immediately:")
            st.markdown("1. **Financials** — an 8-year manufacturing company "
                        "model (revenue, costs, capex, per-unit data).")
            st.markdown("2. **Projects** — a 6-project capital-budgeting "
                        "catalogue for investment & optimization.")
            st.download_button(
                "\U0001F4E5 Download Sample Financials (CSV)",
                data=samples.financials_csv_bytes(),
                file_name="rfc_sample_financials.csv", mime="text/csv")
            st.download_button(
                "\U0001F4E5 Download Sample Financials (Excel)",
                data=samples.financials_xlsx_bytes(),
                file_name="rfc_sample_financials.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            st.download_button(
                "\U0001F4E5 Download Sample Projects (CSV)",
                data=samples.projects_csv_bytes(),
                file_name="rfc_sample_projects.csv", mime="text/csv")
        with c2:
            st.markdown("### Load sample instantly")
            if st.button("\U0001F4E6 Upload Sample Data", type="primary",
                         width="stretch"):
                load_sample()
                st.success("Sample data loaded — press **Run Full Analysis** "
                           "below.")
            if st.session_state.get("projects"):
                st.info(f"\u2713 {len(st.session_state['projects'])} projects "
                        "in the catalogue.")
        if st.session_state.get("ci") is not None:
            st.dataframe(money_df(samples.financials_dataframe(
                st.session_state["ci"])),
                width="stretch", hide_index=True)

    with t_upload:
        st.markdown("### Upload your financial data")
        st.markdown("CSV or Excel. Auto-detected as either a **financials** "
                    "file (Period, Revenue, COGS, Opex, Depreciation, Capex + "
                    "optional Units/Price/Unit_Variable_Cost) or a **projects** "
                    "file (Project, Initial_Investment, Year1..YearN).")
        uploaded = st.file_uploader(
            "Choose a CSV or Excel file", type=["csv", "xlsx", "xls"],
            key="up_file")
        if uploaded:
            res = loader.parse_upload(uploaded.getvalue(), uploaded.name)
            for e in res["errors"]:
                st.error(e)
            for w in res["warnings"]:
                st.warning(w)
            if res["ok"]:
                if res["kind"] == "financials":
                    st.session_state["ci"] = res["ci"]
                    st.success(f"Financials loaded: **{st.session_state['ci'].company_name}** "
                               f"({st.session_state['ci'].periods} periods).")
                elif res["kind"] == "projects":
                    st.session_state["projects"] = res["projects"]
                    st.success(f"Loaded **{len(res['projects'])}** projects from "
                               f"{uploaded.name}.")
                st.dataframe(money_df(res["table"]), width="stretch",
                             hide_index=True)

    with t_assume:
        if st.session_state.get("ci") is None:
            st.info("Load a sample or upload financials first to edit "
                    "assumptions.")
        else:
            ci = st.session_state["ci"]
            st.caption("These global assumptions are used by every model.")
            c1, c2, c3 = st.columns(3)
            with c1:
                ci.company_name = st.text_input(
                    "Company name", value=ci.company_name)
                ci.currency = st.selectbox(
                    "Reporting currency", SUPPORTED_CURRENCIES,
                    index=SUPPORTED_CURRENCIES.index(ci.currency)
                    if ci.currency in SUPPORTED_CURRENCIES else 0)
                ci.initial_cash = st.number_input(
                    "Opening cash balance", value=float(ci.initial_cash),
                    step=100_000.0, format="%.0f")
            with c2:
                ci.tax_rate = st.number_input("Tax rate %", 0.0, 100.0,
                                              value=ci.tax_rate * 100,
                                              step=1.0) / 100
                ci.wacc = st.number_input("Discount rate (WACC) %", 1.0, 100.0,
                                          value=ci.wacc * 100, step=0.5) / 100
                ci.inflation = st.number_input("Inflation %", 0.0, 30.0,
                                               value=ci.inflation * 100,
                                               step=0.5) / 100
            with c3:
                ci.receivable_days = st.number_input(
                    "Receivables days", 0.0, 365.0, value=ci.receivable_days,
                    step=1.0)
                ci.payable_days = st.number_input(
                    "Payables days", 0.0, 365.0, value=ci.payable_days,
                    step=1.0)
                ci.inventory_days = st.number_input(
                    "Inventory days", 0.0, 365.0, value=ci.inventory_days,
                    step=1.0)
            c1, c2, c3 = st.columns(3)
            with c1:
                ci.debt_ratio = st.number_input(
                    "Debt ratio %", 0.0, 100.0, value=ci.debt_ratio * 100,
                    step=1.0) / 100
            with c2:
                ci.debt_interest_rate = st.number_input(
                    "Debt interest rate %", 0.0, 30.0,
                    value=ci.debt_interest_rate * 100, step=0.5) / 100
            with c3:
                ci.terminal_growth = st.number_input(
                    "Terminal growth %", 0.0, 10.0,
                    value=ci.terminal_growth * 100, step=0.5) / 100

    st.divider()
    errs = st.session_state["ci"].validate() if st.session_state.get("ci") else []
    if errs and st.session_state.get("ci"):
        for e in errs:
            st.error(e)
    elif st.session_state.get("ci"):
        if st.button("\U0001F680 RUN FULL ANALYSIS", type="primary",
                     width="stretch", key="run_full"):
            with st.spinner("Running the interconnected models\u2026"):
                st.session_state["ci"] = st.session_state["ci"]
                recompute(show_progress=True)
            st.success("Full analysis complete — everything is now computed "
                       "and interconnected.")


# ══════════════════════════════════════════════════════════════════════════
# DASHBOARD
# ══════════════════════════════════════════════════════════════════════════
def dashboard():
    st.markdown(f"""
<div class="rfc-hero">
  <div class="tag">R F C &nbsp;S E C U R I T I E S</div>
  <h1>{TAGLINE}</h1>
  <div class="sub">{SUBTAGLINE} · {MODEL_CHAIN}</div>
</div>
""", unsafe_allow_html=True)
    st.markdown('<div class="rfc-chain">' + "".join(
        [f'<span class="c">{s}</span><span style="color:#B8C5BD;"> → </span>'
         for s in MODEL_STEPS[:-1]]) +
        f'<span class="c">{MODEL_STEPS[-1]}</span></div>',
        unsafe_allow_html=True)

    c1, c2 = st.columns([1.4, 1])
    with c1:
        st.markdown("### Your financial modelling cockpit")
        st.markdown("One unified AI agent across five interconnected models: "
                    "profitability, cash flow, investment, risk and "
                    "optimization — with a robotic AI analyst and "
                    "downloadable reports.")
        with st.expander("What can I do here?", expanded=True):
            st.markdown("\u2022 **Upload Sample Data** — load the built-in "
                        "sample instantly (button below).\n"
                        "\u2022 Publish **CSV/Excel** financials and a project "
                        "catalogue.\n"
                        "\u2022 Press **Run Full Analysis** once — the models "
                        "feed each other automatically.\n"
                        "\u2022 Stress-test under **custom scenarios**, "
                        "**optimize** selection/pricing/capital, and ask the "
                        "**robot** anything.")
        c1a, c1b = st.columns(2)
        with c1a:
            if st.button("\U0001F4E6 Upload Sample Data", type="primary",
                         width="stretch", key="dash_sample"):
                load_sample()
                st.success("Sample data loaded.")
        with c1b:
            if st.button("\U0001F4C4 Go to Data & Setup", width="stretch"):
                st.session_state["_page"] = "\U0001F4C4 Data & Setup"
                st.rerun()
    with c2:
        if st.session_state.get("ci") is None:
            card("Get started", "Load the sample dataset or upload your own "
                 "financials to unlock the full platform.")
        else:
            ci = st.session_state["ci"]
            card("Active model input",
                 f"<b>{ci.company_name}</b> · {ci.currency}<br>"
                 f"{ci.periods} modelled periods · " +
                 (f"{len(st.session_state['projects'])} projects"
                  if st.session_state.get("projects") else "no project catalogue"))

    if has_state():
        st.divider()
        st.markdown("### Key results at a glance")
        state = get_state()
        bc = state.business_case
        p = state.profitability
        cf = state.cashflow
        risk = state.risk

        def _kpi_items():
            yield ("Business NPV", fmt_currency(bc.npv_, state.company.currency),
                   f"WACC {bc.wacc * 100:.1f}%", bc.npv_ >= 0)
            yield ("IRR", fmt_pct(bc.irr_ * 100) if bc.irr_ is not None else "n/a",
                   "vs cost of capital", (bc.irr_ or 0) >= bc.wacc)
            yield ("Net margin (Y1)", fmt_pct(p.net_margin),
                   fmt_pct(p.operating_margin) + " operating", True)
            yield ("Total FCF", fmt_currency(cf.total_fcf, state.company.currency),
                   f"{state.company.periods}-yr plan", cf.total_fcf >= 0)
            if risk.mc_p5 is not None:
                yield ("P5 NPV (MC)", fmt_currency(risk.mc_p5, state.company.currency),
                       f"P(neg) {risk.mc_prob_negative:.0f}%",
                       risk.mc_p5 >= 0)
            else:
                yield ("P5 NPV (MC)", "n/a", "run risk model", None)
            if state.optimization and state.optimization.selection:
                yield ("Optimized NPV",
                       fmt_currency(state.optimization.selection.total_npv,
                                    state.company.currency),
                       "selected under budget",
                       state.optimization.selection.total_npv >= 0)
            else:
                yield ("Optimized NPV", "n/a", "run Optimization", None)

        cols = st.columns(6)
        for col, (lbl, val, sub, good) in zip(cols, _kpi_items()):
            with col:
                kpi(lbl, val, sub, good=good)
        st.divider()
        robot_welcome_block()
    else:
        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            if st.button("\U0001F4E6 Upload Sample Data & Run Analysis",
                         type="primary", width="stretch"):
                load_sample()
                with st.spinner("Running the interconnected models\u2026"):
                    recompute(show_progress=True)
                st.rerun()
        with c2:
            st.info("No analysis yet. Load sample data or go to **Data & Setup** "
                    "to upload your own files.")
        st.divider()
        robot_welcome_block()


# ══════════════════════════════════════════════════════════════════════════
# PROFITABILITY PAGE
# ══════════════════════════════════════════════════════════════════════════
def profitability_page():
    st.subheader("\U0001F4C8 Profitability Model")
    if not require_state():
        return
    state = get_state()
    p = state.profitability
    ccy = state.company.currency

    c1, c2, c3, c4, c5 = st.columns(5)
    kpi("Gross margin", fmt_pct(p.gross_margin), f"{fmt_pct(p.ebitda_margin)} EBITDA", True)
    kpi("Operating margin", fmt_pct(p.operating_margin), "EBIT / revenue")
    kpi("Net margin", fmt_pct(p.net_margin), f"Year {p.years[-1]} est. {p.net_income[-1] / p.revenue[-1] * 100:.1f}%")
    kpi("Break-even revenue", (fmt_currency(p.breakeven_revenue, ccy)
                               if p.breakeven_revenue is not None else "n/a"),
        f"Fixed cost {fmt_currency(p.fixed_cost, ccy)}")
    kpi("Margin of safety", (f"{p.margin_of_safety_pct:.0f}%" if p.margin_of_safety_pct is not None else "n/a"),
        f"Operating leverage {p.operating_leverage:.1f}x" if p.operating_leverage else "")

    fig = go.Figure()
    fig.add_trace(go.Bar(x=p.years, y=p.revenue, name="Revenue",
                         marker_color=GREEN))
    fig.add_trace(go.Bar(x=p.years, y=p.cogs, name="COGS (variable)",
                         marker_color="#A8C9B8"))
    fig.add_trace(go.Bar(x=p.years, y=p.opex, name="Fixed opex",
                         marker_color="#E0D6A8"))
    fig.add_trace(go.Scatter(x=p.years, y=p.net_income, name="Net income",
                             mode="lines+markers", line=dict(color=GOLD, width=3)))
    fig.update_layout(barmode="group")
    styled_plot(fig, "Revenue build-up vs net income")

    c1, c2 = st.columns(2)
    with c1:
        mdf = p.margin_df()
        fig2 = go.Figure()
        for col, color in [("Gross margin %", GREEN), ("EBITDA margin %", "#4E9F7A"),
                           ("Operating margin %", GOLD), ("Net margin %", DEEP_GREEN)]:
            fig2.add_trace(go.Scatter(x=p.years, y=mdf[col], mode="lines+markers",
                                      name=col, line=dict(color=color, width=2.5)))
        styled_plot(fig2, "Margins over time", height=330)
    with c2:
        st.markdown("#### Break-even & operating leverage")
        if p.breakeven_revenue is not None:
            cur = p.revenue[0]
            fig3 = go.Figure()
            fig3.add_trace(go.Bar(x=["Break-even revenue", f"Actual revenue (Y{p.years[0]})"],
                                  y=[p.breakeven_revenue, cur],
                                  marker_color=[AMBER if p.breakeven_revenue > cur else GREEN,
                                                GREEN], textfont_size=12))
            styled_plot(fig3, "Revenue break-even comparison", height=330)
        else:
            note("Break-even needs per-unit (price/volume/unit variable cost) "
                 "data. A contribution-margin proxy was used.")
        with st.expander("Clear model assumptions (profitability)"):
            st.json(p.assumptions)

    st.markdown("#### Profitability drivers")
    rows = []
    for d in p.drivers:
        rows.append({"Driver": d["driver"],
                     "Base": (fmt_num(d["base"]) if d["base"] is not None else "n/a"),
                     "Final": (fmt_num(d["final"]) if d["final"] is not None else "n/a"),
                     "What drives it": d["effect"]})
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    with st.expander("AI explanation — Profitability", expanded=True):
        st.markdown(rfc_analyst.build_narrative(state)["profitability"])
    st.divider()
    robot_panel(rfc_analyst.build_narrative(state)["profitability"],
                headline="Profitability read-aloud",
                autoplay=False)


# ══════════════════════════════════════════════════════════════════════════
# CASH FLOW PAGE
# ══════════════════════════════════════════════════════════════════════════
def cashflow_page():
    st.subheader("\U0001F4B5 Cash-Flow Forecasting Model")
    if not require_state():
        return
    state = get_state()
    cf = state.cashflow
    p = state.profitability
    ccy = state.company.currency

    c1, c2, c3, c4, c5 = st.columns(5)
    kpi("Total free cash flow", fmt_currency(cf.total_fcf, ccy),
        f"{state.company.periods}-year total")
    kpi("Closing cash balance", fmt_currency(cf.cash_balance[-1], ccy),
        f"Year {p.years[-1]}")
    kpi("Operating CF (Y1)", fmt_currency(cf.operating_cash_flow[0], ccy),
        f"Y{p.years[-1]}: {fmt_currency(cf.operating_cash_flow[-1], ccy)}")
    kpi("Current ratio (Y1)",
        f"{cf.current_ratio[0]:.2f}" if cf.current_ratio[0] == cf.current_ratio[0] else "n/a",
        f"Quick {cf.quick_ratio[0]:.2f}")
    kpi("Capex (total)", fmt_currency(float(cf.capex.sum()), ccy), "investing")

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=p.years, y=cf.operating_cash_flow, name="Operating CF",
                             marker_color=GREEN))
        fig.add_trace(go.Bar(x=p.years, y=cf.investing_cash_flow, name="Investing CF",
                             marker_color="#D8B23A"))
        fig.add_trace(go.Bar(x=p.years, y=cf.financing_cash_flow, name="Financing CF",
                             marker_color="#A8C9B8"))
        fig.add_trace(go.Scatter(x=p.years, y=cf.fcf, name="Free cash flow",
                                 mode="lines+markers", line=dict(color=DEEP_GREEN, width=3)))
        styled_plot(fig, "Cash-flow statement (OCF / ICF / FCF)", height=360)
    with c2:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=p.years, y=cf.cash_balance, name="Cash balance",
                                  fill="tozeroy", line=dict(color=GREEN, width=3)))
        fig2.add_trace(go.Scatter(x=p.years, y=cf.current_ratio, name="Current ratio",
                                  yaxis="y2", line=dict(color=GOLD, width=2.5, dash="dash")))
        fig2.update_layout(yaxis2=dict(overlaying="y", side="right"))
        styled_plot(fig2, "Projected cash balance & liquidity", height=360)

    st.markdown("#### Working capital footprint")
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=p.years, y=cf.receivables, name="Receivables",
                              fill="tozeroy", line=dict(color=GREEN)))
    fig3.add_trace(go.Scatter(x=p.years, y=cf.inventory, name="Inventory",
                              fill="tonexty", line=dict(color=GOLD)))
    fig3.add_trace(go.Scatter(x=p.years, y=-cf.payables, name="Payables",
                              fill="tozeroy", line=dict(color="#9CC0AC")))
    styled_plot(fig3, "Working capital components", height=330)

    with st.expander("Cash-flow statement (detailed)", expanded=True):
        st.dataframe(money_df(cf.statement_df()), width="stretch",
                     hide_index=True)
    with st.expander("Liquidity ratios by period"):
        st.dataframe(money_df(cf.liquidity_df()), width="stretch",
                     hide_index=True)
    with st.expander("Clear model assumptions (cash flow)"):
        st.json(cf.assumptions)

    with st.expander("AI explanation — Cash Flow", expanded=True):
        st.markdown(rfc_analyst.build_narrative(state)["cashflow"])
    st.divider()
    robot_panel(rfc_analyst.build_narrative(state)["cashflow"],
                headline="Cash-flow read-aloud", autoplay=False)


# ══════════════════════════════════════════════════════════════════════════
# INVESTMENT PAGE
# ══════════════════════════════════════════════════════════════════════════
def investment_page():
    st.subheader("\U0001F3D7\ufe0f Investment & Capital Budgeting Model")
    if not require_state():
        return
    state = get_state()
    bc = state.business_case
    ccy = state.company.currency
    wacc = state.company.wacc

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    kpi("NPV", fmt_currency(bc.npv_, ccy), f"WACC {wacc * 100:.1f}%",
        good=bc.npv_ >= 0)
    kpi("IRR", fmt_pct(bc.irr_ * 100) if bc.irr_ is not None else "n/a",
        f"MIRR {fmt_pct(bc.mirr_ * 100) if bc.mirr_ else 'n/a'}")
    kpi("Profitability index", f"{bc.pi_:.2f}" if bc.pi_ is not None else "n/a",
        "PV inflows / outlay")
    kpi("Payback", f"{bc.payback_:.1f} y" if bc.payback_ else "n/a",
        f"Discounted {bc.discounted_payback_:.1f} y")
    kpi("EAA", fmt_currency(bc.eaa_, ccy) if bc.eaa_ is not None else "n/a",
        "annualised NPV")
    kpi("ARR", fmt_pct(bc.arr_ * 100) if bc.arr_ is not None else "n/a",
        "avg return / avg investment")

    banner(("The business plan creates value above the required return."
            if bc.npv_ >= 0 else
            "The business plan does not cover the required return today."),
           ok=bc.npv_ >= 0)

    with st.expander("Metric → formula → interpretation"):
        rows = [
            ["NPV", "Σ CFₜ/(1+r)ᵗ − C₀", fmt_currency(bc.npv_, ccy),
             "Yes — value created" if bc.npv_ >= 0 else "No — required return not earned"],
            ["IRR", "Rate making NPV = 0", fmt_pct(bc.irr_ * 100) if bc.irr_ else "n/a",
             f"vs WACC {wacc * 100:.1f}%"],
            ["MIRR", "Financed + reinvested return", fmt_pct(bc.mirr_ * 100) if bc.mirr_ else "n/a",
             "More conservative single return"],
            ["PI", "PV inflows / |C₀|", f"{bc.pi_:.2f}" if bc.pi_ else "n/a",
             "Value per unit invested"],
            ["ARR", "Avg return / avg investment", fmt_pct(bc.arr_ * 100) if bc.arr_ else "n/a",
             "Accounting-based check"],
            ["Payback", "Years to recover C₀", f"{bc.payback_:.1f} y" if bc.payback_ else "n/a",
             f"Undiscounted; {bc.discounted_payback_:.1f} y discounted"],
            ["EAA", "NPV ÷ annuity factor", fmt_currency(bc.eaa_, ccy) if bc.eaa_ else "n/a",
             "Annualised value over the plan"],
            ["DCF", "PV of return stream", fmt_currency(bc.dcf_value_, ccy), "Gross value of returns"],
            ["Break-even performance", "m: NPV=0", f"{bc.break_even_performance_:.2f}x" if bc.break_even_performance_ else "n/a",
             "Performance must stay above this multiple of plan"],
        ]
        st.dataframe(pd.DataFrame(rows, columns=["Metric", "Formula", "Result",
                                                 "Interpretation"]),
                     width="stretch", hide_index=True)

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        fig.add_trace(go.Waterfall(
            x=["Investment", *[f"Y{y}" for y in state.profitability.years]],
            y=[-bc.initial_investment, *bc.return_stream.tolist()],
            measure=["absolute"] + ["relative"] * (state.company.periods),
            connector=dict(line=dict(color="#C9A227")),
            increasing=dict(marker_color=GREEN), decreasing=dict(marker_color="#D8B23A"),
            totals=dict(marker_color=GOLD)))
        styled_plot(fig, "Business-case investment waterfall (returns)", height=380)
    with c2:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=list(range(len(bc.full_timeline))),
                                  y=np.cumsum(bc.full_timeline),
                                  mode="lines+markers", name="Cumulative cash flow",
                                  line=dict(color=DEEP_GREEN, width=3), fill="tozeroy"))
        fig2.add_hline(y=0, line_dash="dash", line_color=GOLD)
        fig2.update_layout(xaxis_title="Year", yaxis_title=f"Amount ({ccy})")
        styled_plot(fig2, "Cumulative cash flow & payback", height=380)
        st.caption(f"Payback ≈ {bc.payback_:.1f} years where the blue line "
                   "crosses zero.")

    if state.project_metrics:
        st.markdown("#### Project catalogue — full metrics")
        rows = []
        for m in state.project_metrics:
            rows.append({
                "Project": m.name,
                "Cost": m.full_timeline[0],
                "NPV": m.npv_,
                "IRR %": m.irr_ * 100 if m.irr_ is not None else None,
                "MIRR %": m.mirr_ * 100 if m.mirr_ is not None else None,
                "PI": m.pi_,
                "ARR %": m.arr_ * 100 if m.arr_ is not None else None,
                "Payback y": m.payback_,
                "Disc. payback y": m.discounted_payback_,
                "EAA": m.eaa_,
                "Break-even x": m.break_even_performance_,
            })
        disp = pd.DataFrame(rows)
        st.dataframe(money_df(disp, exclude=("Project",)), width="stretch",
                     hide_index=True)

        fig3 = go.Figure(go.Bar(
            x=[m.name for m in state.project_metrics],
            y=[m.npv_ for m in state.project_metrics],
            marker_color=[GREEN if m.npv_ >= 0 else "#C0392B"
                          for m in state.project_metrics]))
        styled_plot(fig3, "Project NPV comparison", height=360)

        choice = st.selectbox("Inspect a project's cash flows",
                              [m.name for m in state.project_metrics])
        pm = next(m for m in state.project_metrics if m.name == choice)
        fig4 = go.Figure()
        fig4.add_trace(go.Bar(x=["C₀", *[f"Y{i+1}" for i in range(pm.life)]],
                              y=pm.full_timeline, marker_color=[
                                  "#D8B23A" if v < 0 else GREEN
                                  for v in pm.full_timeline]))
        fig4.add_trace(go.Scatter(x=["C₀", *[f"Y{i+1}" for i in range(pm.life)]],
                                  y=np.cumsum(pm.full_timeline), name="Cumulative",
                                  mode="lines+markers", line=dict(color=DEEP_GREEN, width=3)))
        styled_plot(fig4, f"Cash-flow schedule — {choice}", height=360)
    else:
        note("No project catalogue loaded — upload a projects file (Project, "
             "initial investment, Year1..YearN) to score alternatives.")

    with st.expander("Clear model assumptions (investment)"):
        st.json({"discount_rate_pct": wacc * 100,
                 "initial_investment": bc.initial_investment,
                 "terminal_growth_pct": state.company.terminal_growth * 100,
                 "reinvestment_rate": "= WACC",
                 "method": "All capex treated as upfront at t=0; returns are "
                           "operating cash flows (NI + D&A − ΔNWC)."})

    with st.expander("AI explanation — Investment", expanded=True):
        st.markdown(rfc_analyst.build_narrative(state)["investment"])
    st.divider()
    robot_panel(rfc_analyst.build_narrative(state)["investment"],
                headline="Investment read-aloud", autoplay=False)


# ══════════════════════════════════════════════════════════════════════════
# RISK & STRESS PAGE
# ══════════════════════════════════════════════════════════════════════════
def risk_page():
    st.subheader("\U0001F6E1\ufe0f Risk & Stress-Testing Model")
    if not require_state():
        return
    state = get_state()
    risk = state.risk
    ccy = state.company.currency

    c1, c2, c3, c4, c5 = st.columns(5)
    kpi("Base NPV", fmt_currency(risk.base_npv, ccy), "unshocked")
    kpi("Optimistic NPV", fmt_currency([s["npv"] for s in risk.scenarios
                                        if s["name"] == "Optimistic"][0], ccy),
        "+15% revenue / −8% costs")
    kpi("Pessimistic NPV", fmt_currency([s["npv"] for s in risk.scenarios
                                         if s["name"] == "Pessimistic"][0], ccy),
        "−15% revenue / +12% costs", good=None)
    kpi("P5 NPV (Monte Carlo)",
        fmt_currency(risk.mc_p5, ccy) if risk.mc_p5 is not None else "n/a",
        f"P(negative) {risk.mc_prob_negative:.0f}%" if risk.mc_prob_negative is not None else "")
    kpi("MC median", fmt_currency(risk.mc_p50, ccy) if risk.mc_p50 is not None else "n/a",
        "across paths")

    mc_iter = st.number_input("Monte Carlo iterations", 100, 10000,
                              value=int(state.risk_config.get("mc_iterations", 1000)),
                              step=100)
    rerun_cols = st.columns(3)
    with rerun_cols[0]:
        if st.button("Re-run risk engine", type="primary"):
            recompute(risk_cfg={"mc_iterations": int(mc_iter),
                                "seed": 2026})
            st.rerun()

    fig = go.Figure(go.Bar(
        x=[s["name"] for s in risk.scenarios],
        y=[s["npv"] for s in risk.scenarios],
        marker_color=[GREEN if s["npv"] >= 0 else "#C0392B"
                      for s in risk.scenarios]))
    styled_plot(fig, "Scenario NPVs (full-model recompute)")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Stress tests — one shock at a time")
        st.dataframe(pd.DataFrame([{
            "Stress": s["name"], "NPV": fmt_currency(s["npv"], ccy),
            "Impact %": f"{s['impact_pct']:+.1f}%"} for s in risk.stress]),
            width="stretch", hide_index=True)
        fig2 = go.Figure(go.Bar(
            x=[s["name"].split(" ")[0].replace("\u2212", "-") for s in risk.stress],
            y=[s["impact"] for s in risk.stress],
            marker_color=[GREEN if s["impact"] >= 0 else "#C0392B"
                          for s in risk.stress]))
        styled_plot(fig2, "Stress-test NPV impact vs base", height=340)
    with c2:
        st.markdown("#### Sensitivity (tornado)")
        t = risk.tornado[:8]
        fig3 = go.Figure(go.Bar(
            x=[abs(r["delta"]) for r in t],
            y=[r["driver"] for r in t], orientation="h",
            marker_color=[GOLD if r["delta"] >= 0 else AMBER for r in t]))
        styled_plot(fig3, "NPV sensitivity by driver", height=340)
        note("The longer the bar, the more the plan depends on that driver. "
             "WACC and revenue usually dominate.")

    if len(risk.mc_samples):
        hist = go.Figure()
        hist.add_trace(go.Histogram(x=risk.mc_samples, nbinsx=50,
                                    marker_color=GREEN, opacity=0.85))
        for pct, c in [("P5", RED), ("P50", GOLD), ("P95", AMBER)]:
            val = {"P5": risk.mc_p5, "P50": risk.mc_p50, "P95": risk.mc_p95}[pct]
            hist.add_vline(x=val, line_dash="dash", line_color=c,
                           annotation_text=f"{pct}: {fmt_currency(val, ccy)}",
                           annotation_position="top")
        hist.add_vline(x=0, line_color=DEEP_GREEN, line_width=2)
        styled_plot(hist, f"Monte Carlo NPV distribution ({risk.mc_iterations} paths)")

    with st.expander("Clear model assumptions (risk & stress)"):
        st.json(risk.assumptions)

    with st.expander("AI explanation — Risk & Stress", expanded=True):
        st.markdown(rfc_analyst.build_narrative(state)["risk"])
    st.divider()
    robot_panel(rfc_analyst.build_narrative(state)["risk"],
                headline="Risk read-aloud", autoplay=False)


# ══════════════════════════════════════════════════════════════════════════
# SCENARIO PAGE
# ══════════════════════════════════════════════════════════════════════════
def scenario_page():
    st.subheader("\U0001F3AD Scenario Analysis — custom scenario builder")
    if not require_state():
        return
    state = get_state()
    ccy = state.company.currency

    st.markdown("Build a **custom scenario** on top of the base model. Every "
                "change re-runs the whole interconnected chain.")
    c1, c2, c3 = st.columns(3)
    with c1:
        rev_mult = st.slider("Revenue multiplier", 0.6, 1.4, 1.0, 0.01,
                             format="%.2f")
        var_mult = st.slider("Variable cost multiplier", 0.6, 1.4, 1.0, 0.01,
                             format="%.2f")
    with c2:
        fixed_mult = st.slider("Fixed opex multiplier", 0.6, 1.4, 1.0, 0.01,
                               format="%.2f")
        wacc_pp = st.slider("WACC change (pp)", -5.0, 8.0, 0.0, 0.5)
    with c3:
        delay = st.slider("Delay (years)", 0.0, 3.0, 0.0, 0.5)
        sc_name = st.text_input("Scenario name", value="Custom")
    if st.button("Run custom scenario", type="primary"):
        custom = {"name": sc_name.strip() or "Custom",
                  "rev_mult": float(rev_mult),
                  "var_mult": float(var_mult),
                  "fixed_mult": float(fixed_mult),
                  "wacc_pp": float(wacc_pp),
                  "delay": float(delay),
                  "desc": f"{sc_name.strip() or 'Custom'} — revenue ×{rev_mult:.2f}, "
                          f"variable ×{var_mult:.2f}, fixed ×{fixed_mult:.2f}, "
                          f"WACC {wacc_pp:+.1f}pp, delay {delay:.1f}y."}
        with st.spinner("Re-running the full model under your scenario\u2026"):
            recompute(custom_scenario=custom)
        st.success("Scenario applied and stored.")

    state = get_state()
    risk = state.risk
    st.markdown("#### Scenario board")
    rows = [{"Scenario": s["name"], "Description": s["description"],
             "NPV": s["npv"]} for s in risk.scenarios]
    st.dataframe(pd.DataFrame([{**r, "NPV": fmt_currency(r.pop("NPV"), ccy)}
                               for r in rows]), width="stretch", hide_index=True)
    fig = go.Figure(go.Bar(
        x=[s["name"] for s in risk.scenarios],
        y=[s["npv"] for s in risk.scenarios],
        marker_color=[GREEN if s["npv"] >= 0 else "#C0392B"
                      for s in risk.scenarios]))
    fig.add_hline(y=0, line_dash="dash", line_color=GOLD)
    styled_plot(fig, "Full scenario comparison (base + optimist + pessimist + custom)")
    st.caption("Custom scenario persists in the analysis until you change or "
               "reload it.")

    with st.expander("AI explanation — Scenarios", expanded=True):
        st.markdown(rfc_analyst.build_narrative(state)["risk"])
    st.divider()
    robot_panel(rfc_analyst.build_narrative(state)["risk"],
                headline="Scenario read-aloud", autoplay=False)


# ══════════════════════════════════════════════════════════════════════════
# OPTIMIZATION PAGE
# ══════════════════════════════════════════════════════════════════════════
def optimization_page():
    st.subheader("\U0001F3AF Optimization Model")
    if not require_state():
        return
    state = get_state()
    ccy = state.company.currency

    tab_sel, tab_px, tab_alloc = st.tabs(
        ["\U0001F4B0 Project selection", "\U0001F4C9 Pricing",
         "\U0001F4C8 Capital allocation"])

    with tab_sel:
        st.markdown("Maximise **total NPV** across the project catalogue, "
                    "subject to a capital budget (binary MILP).")
        if not state.project_metrics:
            note("No project catalogue loaded. Upload a projects file to use "
                 "selection.")
        else:
            total_cost = sum(abs(p.full_timeline[0]) for p in state.project_metrics)
            budget = st.number_input("Capital budget", 0.0, total_cost * 2,
                                     value=total_cost * 0.60,
                                     step=100_000.0, format="%.0f")
            if st.button("Run project-selection optimiser", type="primary"):
                sel = select_projects(state.project_metrics, budget)
                st.session_state["opt_cfg"] = {**st.session_state.get("opt_cfg", {}),
                                               "selection_budget": budget}
                recompute(opt_cfg=st.session_state["opt_cfg"])
                st.session_state["state"] = st.session_state["state"]
                state = get_state()
                st.success("Optimisation complete — results below.")
            s = state.optimization.selection if state.optimization else None
            if s:
                c1, c2, c3, c4 = st.columns(4)
                kpi("Total budget", fmt_currency(s.budget, ccy), "")
                kpi("Budget used", fmt_currency(s.used_budget, ccy),
                    f"idle {fmt_currency(s.idling, ccy)}")
                kpi("Selected NPV", fmt_currency(s.total_npv, ccy),
                    f"{len(s.selected)} of {len(state.project_metrics)} projects")
                kpi("NPV uplift vs all", fmt_currency(s.uplift_npv, ccy), "selected − all")
                sel_cols = st.columns(2)
                with sel_cols[0]:
                    st.markdown("**Selected:** " +
                                ", ".join(s.selected) or "none")
                with sel_cols[1]:
                    fig = go.Figure(go.Bar(
                        x=[m.name for m in state.project_metrics],
                        y=[m.npv_ for m in state.project_metrics],
                        marker_color=[GREEN if m.name in s.selected
                                      else "#C9D4CC" for m in state.project_metrics]))
                    fig.add_hline(y=0, line_dash="dash", line_color=GOLD)
                    styled_plot(fig, "Optimised portfolio (selected in green)",
                                height=320)

    with tab_px:
        st.markdown("Find the **profit-maximising selling price** using an "
                    "isoelastic demand curve (elasticity < −1).")
        ci = state.company
        p0 = float(ci.price[0]) if ci.price is not None else 45.0
        vc0 = float(ci.unit_var_cost[0]) if ci.unit_var_cost is not None else 24.0
        q0 = float(ci.units[0]) if ci.units is not None else 1.0
        c1, c2, c3, c4 = st.columns(4)
        price = c1.number_input("Current price", 0.01, 1000.0, value=p0,
                                step=0.5, format="%.2f")
        vc = c2.number_input("Variable cost / unit", 0.01, 1000.0, value=vc0,
                             step=0.5, format="%.2f")
        qty = c3.number_input("Current volume (units)", 1.0, 1e9, value=q0,
                              step=1000.0, format="%.0f")
        elas = c4.number_input("Demand elasticity (e.g. −2.5)", -8.0, -1.1,
                               value=-2.5, step=0.1)
        if st.button("Run pricing optimiser", type="primary"):
            pr = optimise_price(price, vc, qty, elas)
            st.session_state["opt_cfg"] = {**st.session_state.get("opt_cfg", {}),
                                           "price_params": {
                                               "price": price, "vc": vc,
                                               "volume": qty, "elasticity": elas}}
            recompute(opt_cfg=st.session_state["opt_cfg"])
            state = get_state()
            st.success("Pricing optimisation complete.")
        pr = state.optimization.pricing if state.optimization else None
        if pr:
            c1, c2, c3, c4 = st.columns(4)
            kpi("Recommended price", fmt_currency(pr.optimal_price, ccy, 2),
                f"{pr.price_change_pct:+.1f}% vs current")
            kpi("Current contribution", fmt_currency(pr.current_contribution, ccy),
                f"qty {pr.current_volume:,.0f}")
            kpi("Optimised contribution", fmt_currency(pr.optimal_contribution, ccy),
                f"qty {pr.projected_volume:,.0f}")
            kpi("Uplift", fmt_currency(pr.uplift_contribution, ccy),
                f"{pr.uplift_pct:+.1f}%")
            px = np.linspace(max(pr.optimal_price * 0.5, vc), pr.optimal_price * 1.8, 200)
            q = qty * (px / price) ** elas
            cont = (px - vc) * q
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=px, y=cont, name="Contribution",
                                     line=dict(color=GREEN, width=3)))
            fig.add_vline(x=pr.optimal_price, line_dash="dash",
                          line_color=GOLD,
                          annotation_text="Optimal price",
                          annotation_position="top")
            styled_plot(fig, "Contribution curve vs price (isoelastic demand)",
                        height=340)
            note("Pricing uses the company's current price, variable cost and "
                 "volume from your dataset. The contribution uplift is on "
                 "variable margin, before fixed-cost changes.")

    with tab_alloc:
        st.markdown("Allocate capital across business units to maximise returns "
                    "(linear programming with unit caps).")
        c1, c2 = st.columns(2)
        budget = c1.number_input("Allocation budget", 0.0, 1e9,
                                 value=5_000_000.0, step=250_000.0,
                                 format="%.0f")
        units_txt = c2.text_area("Business units (one per line)",
                                 "Division A\nDivision B\nDivision C")
        units = [u.strip() for u in units_txt.splitlines() if u.strip()]
        rets_txt = c1.text_area("Expected return % (same order)",
                                "18\n12\n9")
        caps_txt = c2.text_area("Caps (same order)", "2,500,000\n2,000,000\n2,000,000")
        try:
            rets = [float(x) for x in rets_txt.replace(",", " ").split()]
            caps = [float(x) for x in caps_txt.replace(" ", ",").split(",") if x.strip()]
        except Exception:
            rets, caps = [], []
        if units and len(rets) == len(units) and st.button("Run allocation optimiser",
                                                           type="primary"):
            caps = [float(c) for c in caps] if len(caps) == len(units) else \
                [budget] * len(units)
            alloc = allocate_capital(units, rets, budget, caps)
            st.session_state["opt_cfg"] = {**st.session_state.get("opt_cfg", {}),
                                           "allocation_params": {
                                               "units": units,
                                               "returns_pct": rets,
                                               "budget": budget,
                                               "caps": caps}}
            recompute(opt_cfg=st.session_state["opt_cfg"])
            state = get_state()
            st.success("Allocation optimised.")
        a = state.optimization.allocation if state.optimization else None
        if a and a.units:
            fig = go.Figure(go.Bar(x=a.units, y=a.allocation,
                                   marker_color=[GREEN, GOLD, "#4E9F7A"]))
            styled_plot(fig, f"Optimal capital allocation (total return "
                             f"{fmt_currency(a.total_return, ccy)}", height=320)
            disp = pd.DataFrame({"Business unit": a.units,
                                 "Allocation": a.allocation,
                                 "Return %": a.returns_pct,
                                 "Cap": a.caps})
            st.dataframe(money_df(disp, exclude=("Business unit", "Return %")),
                         width="stretch", hide_index=True)

    with st.expander("AI explanation — Optimization", expanded=True):
        st.markdown(rfc_analyst.build_narrative(state)["optimization"] or
                    "Run one of the optimisers above to see the AI narrative.")
    st.divider()
    robot_panel(rfc_analyst.build_narrative(state)["optimization"] or
                "Run project selection, pricing or capital allocation to see "
                "the optimisation narrative.",
                headline="Optimization read-aloud", autoplay=False)


# ══════════════════════════════════════════════════════════════════════════
# AI ANALYST PAGE
# ══════════════════════════════════════════════════════════════════════════
def analyst_page():
    st.subheader("\U0001F916 AI Financial Analyst")
    if not require_state():
        return
    state = get_state()

    c1, c2 = st.columns([1.2, 1])
    with c1:
        st.markdown("Ask the analyst anything about your model. Answers are "
                    "computed directly from your results — never generic.")
        q = st.text_input("Your question", key="ask_q",
                          placeholder="e.g. Is the NPV positive?")
        if st.button("Ask the analyst", type="primary"):
            ans, _ = rfc_analyst.answer_or_fallback(q, state)
            st.session_state["asked_answer"] = ans
        if st.session_state.get("asked_answer"):
            st.markdown(f'<div class="rfc-note">{st.session_state["asked_answer"]}</div>',
                        unsafe_allow_html=True)
            st.divider()
            robot_panel(st.session_state["asked_answer"],
                        headline="Analyst answer — read aloud", autoplay=False)
    with c2:
        st.markdown("#### Suggested questions")
        for ex in rfc_analyst.EXAMPLE_QUESTIONS:
            if st.button(f"\U0001F4AC {ex}", key=f"ex_{ex}"):
                ans, _ = rfc_analyst.answer_or_fallback(ex, state)
                st.session_state["asked_answer"] = ans
                st.rerun()

    st.divider()
    st.markdown("#### Full narrative by section (read aloud)")
    narr = rfc_analyst.build_narrative(state)
    for sec, txt in narr.items():
        if not txt:
            continue
        label = f"🔊 {sec.title()}"
        with st.expander(f"{label}", expanded=False):
            st.markdown(f'<div class="rfc-note">{txt}</div>', unsafe_allow_html=True)

    st.divider()
    st.markdown("#### Full read-aloud report")
    full = rfc_analyst.full_narrative(state)
    if st.button("Generate full audio report"):
        with st.spinner("Synthesising the complete narrative\u2026"):
            path = speak_path(full)
        if path:
            st.success("Audio ready — use the player below.")
    robot_panel(full, headline="Complete model read-aloud", autoplay=False)


# ══════════════════════════════════════════════════════════════════════════
# REPORTS PAGE
# ══════════════════════════════════════════════════════════════════════════
def reports_page():
    st.subheader("\U0001F4E4 Reports & Delivery")
    if not require_state():
        return
    state = get_state()

    st.markdown("Export every result as a professional HTML report, JSON data, "
                "CSV tables, or spoken audio.")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("#### \U0001F4C4 Reports")
        html_rpt = build_html_report(state, username=st.session_state["username"])
        st.download_button("Download HTML report", data=html_rpt,
                           file_name=f"rfc_report_{state.company.company_name.replace(' ', '_')}.html",
                           mime="text/html", width="stretch")
        st.download_button("Download JSON export",
                           data=build_json_export(state),
                           file_name=f"rfc_export_{state.company.company_name.replace(' ', '_')}.json",
                           mime="application/json", width="stretch")
    with c2:
        st.markdown("#### \U0001F4C9 CSV tables")
        st.download_button("Income statement (CSV)",
                           data=csv_export_for(state, "income"),
                           file_name="rfc_income_statement.csv",
                           mime="text/csv", width="stretch")
        st.download_button("Cash-flow statement (CSV)",
                           data=csv_export_for(state, "cashflow"),
                           file_name="rfc_cashflow.csv",
                           mime="text/csv", width="stretch")
        has_projects = bool(state.project_metrics)
        if has_projects:
            st.download_button("Project metrics (CSV)",
                               data=csv_export_for(state, "projects"),
                               file_name="rfc_project_metrics.csv",
                               mime="text/csv", width="stretch")
    with c3:
        st.markdown("#### \U0001F916 Spoken audio")
        st.markdown("Generate a downloadable MP3 of the robot reading the "
                    "full analysis, then download it from the player.")
        full = rfc_analyst.full_narrative(state)
        robot_panel(full, headline="Read the full report aloud", autoplay=False)

    st.divider()
    st.markdown("#### Report preview")
    with st.expander("Preview the HTML report"):
        st.components.v1.html(build_html_report(
            state, username=st.session_state["username"]), height=900,
            scrolling=True)

    st.divider()
    note("RFC Securities generates decision-support analysis. It is not "
         "financial advice; validate assumptions before committing capital. "
         "All sample data is fictitious.")


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════
def main():
    if not st.session_state["logged_in"]:
        login_page()
        return

    inject_css()
    page = sidebar()

    if not st.session_state.get("toast_shown") and has_state() and \
            st.session_state["state"].company.company_name:
        st.toast(f"Analysis ready for {st.session_state['state'].company.company_name}")
        st.session_state["toast_shown"] = True

    if page == "\U0001F4C8 Dashboard":
        dashboard()
    elif page == "\U0001F4C4 Data & Setup":
        data_page()
    elif page == "\U0001F4C8 Profitability":
        profitability_page()
    elif page == "\U0001F4B5 Cash Flow":
        cashflow_page()
    elif page == "\U0001F3D7\ufe0f Investment Analysis":
        investment_page()
    elif page == "\U0001F6E1\ufe0f Risk & Stress Testing":
        risk_page()
    elif page == "\U0001F3AD Scenario Analysis":
        scenario_page()
    elif page == "\U0001F3AF Optimization":
        optimization_page()
    elif page == "\U0001F916 AI Financial Analyst":
        analyst_page()
    elif page == "\U0001F4E4 Reports & Delivery":
        reports_page()

    with st.sidebar:
        st.divider()
        if st.button("Log out", width="stretch"):
            st.session_state["logged_in"] = False
            st.rerun()


if __name__ == "__main__":
    main()