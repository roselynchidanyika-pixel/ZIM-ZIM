"""
CAPEXX AI AGENT — Streamlit Application
========================================
Royal-blue and white login · Cinematic intro · Market simulation ticker
· Full capital-project analysis dashboard · Portfolio comparison
· Exchange-rate board · ASK CAPEXX AI research · Report / Email / Audio
· 10-second decision status light · Robotic AI voice panel
"""
from __future__ import annotations

import copy, io, math, os, time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── local modules ────────────────────────────────────────────────────────────
from capexx_engine import (
    ProjectInput, AnalysisBundle, fmt_ccy, fmt_pct,
    CURRENCY_DETAILS, SUPPORTED_CURRENCIES, COUNTRY_LIST, PROJECT_TYPES,
    EXPECTED_COLUMNS, NUMERIC_COLUMNS, parse_upload,
    demo_project, demo_csv_bytes, try_fetch_live_rates,
    build_exchange_rate_board, market_snapshot, model_to_text,
    MARKET_DISCLAIMER, DEMO_DISCLAIMER, run_analysis,
    funding_priority, knapsack_select, sample_portfolio_projects,
    vision_pillar,
)
from capexx_ai import (
    build_narration_text, section_map, synthesize, tts_available, read_aloud,
    generate_ai_interpretation, why_did_capexx_text,
    ROBOT_WELCOME, ROBOT_DEMO_NOTICE, ROBOT_MARKET, data_status_tag,
)
from capexx_ask import research, EXAMPLE_QUESTIONS, SEARCH_MODES, live_fx_board
from capexx_auth import verify, register, SEED_ADMINS
from capexx_ml import (
    train_overrun_models, predict_overrun, demo_history_dataframe,
    validate_history_frame, demo_history_csv_bytes, history_template_csv_bytes,
    REQUIRED_COLUMNS, NUMERIC_FEATURES, TARGET, OVERRUN_THRESHOLD,
    RISK_LABELS, DEMO_DISCLAIMER as ML_DEMO_DISCLAIMER,
    explain_model_performance, explain_roc, explain_confusion,
    explain_feature_importance, explain_residuals, explain_calibration,
    explain_shapiro, explain_breusch_pagan, explain_durbin_watson,
    explain_vif, explain_trends,
)

# ── paths ────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parent
ASSETS     = ROOT / "assets"
OUTPUT_RPT = ROOT / "outputs" / "reports"
OUTPUT_AU  = ROOT / "outputs" / "audio"
OUTPUT_RPT.mkdir(parents=True, exist_ok=True)
OUTPUT_AU.mkdir(parents=True, exist_ok=True)

# ── page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CAPEXX AI AGENT",
    layout="wide",
    page_icon=str(ASSETS / "robot.png") if (ASSETS / "robot.png").exists() else "🏛️",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════════════
# CSS — Royal Blue + White + Electric accents + green/amber/red cards
# ═══════════════════════════════════════════════════════════════════════════════
def inject_css():
    st.markdown("""
<style>
/* ── global palette ─────────────────────────────────────── */
:root{--rb:#1547A0;--rb-light:#1E57B5;--rb-dark:#0D3080;
       --elec:#3B82F6;--white:#FFFFFF;--off:#F0F4FF;--ink:#0F172A;
       --green:#15803d;--amber:#d97706;--red:#dc2626;
       --green-bg:#DCFCE7;--amber-bg:#FEF3C7;--red-bg:#FEE2E2;}
header[data-testid="stHeader"]{background:var(--rb)!important;}
header [data-testid="stHeader"] *{color:#fff!important;}
section[data-testid="stSidebar"]{background:var(--rb)!important;}
section[data-testid="stSidebar"] *{color:#fff!important;}
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] .stMarkdown p{color:#fff!important;}
div[data-testid="stToolbar"]{background:var(--rb)!important;}

/* ── metric cards ──────────────────────────────────────── */
.metric-card{background:var(--off);border-left:5px solid var(--elec);
  border-radius:8px;padding:1rem 1.2rem;margin:.4rem 0;}
.metric-card h3{margin:0 0 4px 0;font-size:.85rem;color:var(--rb);}
.metric-card p{margin:0;font-size:1.55rem;font-weight:700;color:var(--ink);}
.metric-card small{color:#64748b;}

/* ── decision banners ──────────────────────────────────── */
.decision-banner{border-radius:10px;padding:1rem 1.4rem;font-size:1.15rem;
  font-weight:700;margin:.6rem 0;text-align:center;}
.dec-accept{background:var(--green-bg);color:var(--green);border:2px solid var(--green);}
.dec-review{background:var(--amber-bg);color:var(--amber);border:2px solid var(--amber);}
.dec-reject{background:var(--red-bg);color:var(--red);border:2px solid var(--red);}

/* ── status-light overlay (10-second travelling sweep) ─── */
@keyframes sweep{0%{clip-path:inset(0 100% 0 0)}100%{clip-path:inset(0 0 0 0)}}
@keyframes fadeOut{0%{opacity:1}85%{opacity:1}100%{opacity:0}}
#status-overlay{
  position:fixed;top:0;left:0;width:100%;height:100%;z-index:9999;
  display:flex;align-items:center;justify-content:center;
  pointer-events:none;
  animation:fadeOut 11s forwards;
}
#status-overlay .box{
  width:min(520px,88vw);border-radius:14px;padding:1.6rem 2rem;
  font-size:1.8rem;font-weight:800;text-align:center;
  color:#fff;
  box-shadow:0 0 40px rgba(0,0,0,.45);
  animation:sweep 2s ease-out forwards;
  clip-path:inset(0 100% 0 0);
}
.overlay-accept{background:linear-gradient(135deg,#15803d,#22c55e)!important;}
.overlay-review{background:linear-gradient(135deg,#d97706,#f59e0b)!important;}
.overlay-reject{background:linear-gradient(135deg,#dc2626,#ef4444)!important;}

/* ── robot panel ───────────────────────────────────────── */
.robot-box{background:linear-gradient(135deg,#0D3080,#1E57B5);
  color:#fff;border-radius:12px;padding:1.2rem;margin:.6rem 0;}
.robot-box h4{margin:0 0 6px;color:var(--elec);}
.robot-box p{margin:0;font-size:.92rem;line-height:1.5;}

/* ── pill badges ───────────────────────────────────────── */
.pill{display:inline-block;border-radius:14px;padding:2px 10px;font-size:.78rem;font-weight:600;margin:0 4px;}
.pill-live{background:#DCFCE7;color:#15803d;}
.pill-demo{background:#FEF3C7;color:#92400e;}
.pill-info{background:#E0E7FF;color:#1547A0;}
</style>
""", unsafe_allow_html=True)


def metric_card(label: str, value: str, sub: str = ""):
    sub_html = f"<small>{sub}</small>" if sub else ""
    st.markdown(
        f'<div class="metric-card"><h3>{label}</h3><p>{value}</p>{sub_html}</div>',
        unsafe_allow_html=True)


def decision_banner(status: str, score: float, grade: str, reasons: list[str]):
    cls = {"ACCEPT": "dec-accept", "REVIEW": "dec-review", "REJECT": "dec-reject"}[status]
    icon = {"ACCEPT": "🟢", "REVIEW": "🟠", "REJECT": "🔴"}[status]
    html = f'<div class="decision-banner {cls}">{icon} {grade} — Score {score:.0f}/100</div>'
    for r in reasons[:6]:
        html += f'<div style="padding:2px 12px;font-size:.88rem;color:#334155;">• {r}</div>'
    st.markdown(html, unsafe_allow_html=True)


def status_light_overlay(status: str):
    cls = {"ACCEPT": "overlay-accept", "REVIEW": "overlay-review", "REJECT": "overlay-reject"}[status]
    icon = {"ACCEPT": "🟢", "REVIEW": "🟠", "REJECT": "🔴"}[status]
    st.markdown(
        f"""<div id="status-overlay">
<div class="box {cls}">{icon} DECISION: {status}</div></div>""",
        unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SESSION STATE INIT
# ═══════════════════════════════════════════════════════════════════════════════
def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _init():
    for k, v in {
        "logged_in": False, "username": "", "role": "GUEST",
        "bundle": None, "portfolio": [],
        "compare_analyses": {},   # name -> AnalysisBundle (Compare & Select)
        "ask_history": [],
        "ml_history": None, "ml_art": None, "ml_source": "",
        "_welcomed": False, "_welcome_audio": None, "_welcome_error": "",
        "fx_user_rates": dict(ProjectInput().fx_rates),
        "fx_live_info": None,
    }.items():
        if k not in st.session_state:
            st.session_state[k] = v
_init()

# ═══════════════════════════════════════════════════════════════════════════════
# LOGIN PAGE  (Royal-blue + white)
# ═══════════════════════════════════════════════════════════════════════════════
def login_page():
    inject_css()
    logo = ASSETS / "capexx_logo.png"
    if logo.exists():
        st.image(str(logo), width=200)
    st.markdown("""
<div style="background:#1547A0;color:#fff;border-radius:12px;padding:1.4rem 2rem;margin-bottom:1rem;">
<h2 style="margin:0;color:#fff;">🏛️ CAPEXX AI AGENT</h2>
<p style="margin:4px 0 0;color:#93C5FD;font-size:.92rem;">
AI-Powered Capital Project Decision Intelligence</p>
</div>
""", unsafe_allow_html=True)

    st.markdown("""
<div style="background:#EEF2FF;border-left:4px solid #1547A0;border-radius:6px;padding:.7rem 1rem;margin-bottom:.8rem;">
<strong style="color:#1547A0;">DEMO CREDENTIALS — NOT FOR PRODUCTION</strong><br>
<span style="color:#334155;font-size:.88rem;">
Admin: <code>FraiserXX</code> / <code>M251232@1</code> &nbsp;|&nbsp;
Admin: <code>admin</code> / <code>admin123</code> &nbsp;|&nbsp;
Anyone may register as Guest.
</span>
</div>
""", unsafe_allow_html=True)

    tab_login, tab_register = st.tabs(["LOGIN", "REGISTER GUEST"])
    with tab_login:
        user = st.text_input("Username", key="login_user")
        pw   = st.text_input("Password", type="password", key="login_pw")
        if st.button("SIGN IN", type="primary", width="stretch"):
            res = verify(user, pw)
            if res["ok"]:
                st.session_state.logged_in = True
                st.session_state.username  = user
                st.session_state.role      = res.get("role", "GUEST")
                st.session_state["_welcomed"] = False   # robot re-welcomes
                st.session_state["_welcome_audio"] = None
                st.session_state["_welcome_error"] = ""
                st.rerun()
            else:
                st.error(res["message"])
    with tab_register:
        ru = st.text_input("Choose username", key="reg_user")
        rp = st.text_input("Choose password", type="password", key="reg_pw")
        if st.button("REGISTER", width="stretch"):
            res = register(ru, rp)
            if res["ok"]:
                st.success(res["message"])
            else:
                st.warning(res["message"])


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
PAGES = ["🏠 Home", "📊 Project Analysis", "📁 Portfolio", "⚖️ Compare & Select",
         "🧠 Overrun AI & Explainability", "💱 Exchange Rates",
         "🤖 ASK CAPEXX AI", "📤 Reports & Delivery", "ℹ️ About"]

def sidebar():
    with st.sidebar:
        st.markdown(
            f"### 🏛️ CAPEXX AI\n"
            f"**{st.session_state.username}** · {st.session_state.role}",
            unsafe_allow_html=False)
        page = st.radio("Navigate", PAGES, label_visibility="collapsed")
        st.divider()
        st.caption("CAPEXX AI AGENT © 2026")
        st.caption("Decision-support only · Not financial advice")
        return page


# ═══════════════════════════════════════════════════════════════════════════════
# HOME — Cinematic + Market simulation ticker + Robot
# ═══════════════════════════════════════════════════════════════════════════════
def cinematic_strip():
    imgs = [
        (ASSETS / "rbz.jpg",        "Reserve Bank of Zimbabwe, Harare — "
         "Markets move. Rates change. Capital projects carry real uncertainty."),
        (ASSETS / "gzu_innovation_hub.jpg",
         "GZU Innovation Hub · Great Zimbabwe University — "
         "HYPOTHETICAL DEMONSTRATION, not actual GZU financial data."),
        (ASSETS / "global_finance.jpg",
         "Global Finance — CAPEXX evaluates projects for any institution, "
         "anywhere in the world."),
        (ASSETS / "engineers.jpg",
         "Engineering & Delivery — Construction cost, delay, currency and "
         "risk, all in one engine."),
    ]
    for img, caption in imgs:
        if img.exists():
            st.image(str(img), width="stretch", caption=caption)
        else:
            st.caption(f"Image not found: {img.name}")


def market_ticker():
    st.markdown("---")
    st.markdown(
        '<div style="background:#1547A0;color:#fff;border-radius:8px;'
        'padding:.5rem 1rem;font-size:.78rem;margin-bottom:.4rem;">'
        '📈 MARKET SIMULATION — DEMONSTRATION ONLY · '
        'Prices shown are randomly generated within the app · '
        'NOT live market data · No live source connected</div>',
        unsafe_allow_html=True)
    df = market_snapshot()
    cols = st.columns(len(df))
    for col, (_, row) in zip(cols, df.iterrows()):
        chg = row["Change %"]
        colour = "#15803d" if chg >= 0 else "#dc2626"
        arrow  = "▲" if chg >= 0 else "▼"
        col.markdown(
            f'<div style="text-align:center;background:#F0F4FF;border-radius:8px;'
            f'padding:.5rem .3rem;">'
            f'<div style="font-size:.72rem;color:#64748b;">{row["Ticker"]}</div>'
            f'<div style="font-size:1.05rem;font-weight:700;">{row["Last"]:.4f}</div>'
            f'<div style="color:{colour};font-weight:600;font-size:.82rem;">'
            f'{arrow} {chg:+.2f}%</div></div>',
            unsafe_allow_html=True)


def robot_panel(bundle: AnalysisBundle | None = None,
                auto_welcome: bool = False):
    st.markdown("---")
    st.markdown('<div class="robot-box"><h4>🤖 CAPEXX AI Agent — Voice Panel</h4>',
                unsafe_allow_html=True)
    robot_img = ASSETS / "robot.png"
    if robot_img.exists():
        st.image(str(robot_img), width=70)
    if bundle is None:
        st.markdown(f"<p>{ROBOT_WELCOME}</p>", unsafe_allow_html=True)
        st.markdown(f"<p><em>{ROBOT_MARKET}</em></p>", unsafe_allow_html=True)
        if auto_welcome and not st.session_state["_welcomed"]:
            res = synthesize(ROBOT_WELCOME + "\n" + ROBOT_MARKET,
                             out_path=str(OUTPUT_AU / "capexx_welcome.mp3"))
            if res["ok"]:
                st.session_state["_welcome_audio"] = res["path"]
            else:
                st.session_state["_welcome_error"] = res["error"]
            st.session_state["_welcomed"] = True   # speak once, after login
        au = st.session_state.get("_welcome_audio")
        if au:
            st.markdown("👋 **Welcome announcement** "
                        f"<small>(for {st.session_state.username})"
                        "</small>",
                        unsafe_allow_html=True)
            st.audio(au, autoplay=auto_welcome)
            if st.button("🔊 REPLAY WELCOME", key="robot_welcome"):
                res = read_aloud(ROBOT_WELCOME + "\n" + ROBOT_MARKET)
                if res["ok"]:
                    st.audio(res["path"])
        else:
            if st.button("🔊 READ WELCOME ALOUD", key="robot_welcome"):
                res = read_aloud(ROBOT_WELCOME + "\n" + ROBOT_MARKET)
                if res["ok"]:
                    st.audio(res["path"])
                else:
                    st.info(f"⚠️ VOICE SERVICE UNAVAILABLE — {res['error']}")
            elif st.session_state.get("_welcome_error"):
                st.info("🔇 Voice service unavailable right now — tap the "
                        "button above to retry the welcome audio.")
    else:
        narr = build_narration_text(bundle)
        secs = section_map(bundle)
        st.markdown(f"<p><strong>Project:</strong> "
                    f"{bundle.project_input.project_name}</p>",
                    unsafe_allow_html=True)
        st.markdown(f"<p><strong>Decision:</strong> {bundle.decision.grade}</p>",
                    unsafe_allow_html=True)
        with st.expander("View narration script"):
            st.code(narr)
        if st.button("🔊 READ FINDINGS ALOUD", key="robot_read"):
            au_path = str(OUTPUT_AU / "capexx_project_analysis.mp3")
            res = synthesize(narr, out_path=au_path)
            if res["ok"]:
                st.audio(res["path"])
            else:
                st.info(f"⚠️ VOICE SERVICE UNAVAILABLE — {res['error']}\n\n"
                        "The narration script is shown above.")
        avail, msg = tts_available()
        st.markdown(f'<small style="color:#93C5FD;">TTS status: {msg}</small>',
                    unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


def home_page():
    cinematic_strip()
    market_ticker()
    robot_panel(st.session_state.bundle, auto_welcome=True)


# ═══════════════════════════════════════════════════════════════════════════════
# FORM BUILDERS
# ═══════════════════════════════════════════════════════════════════════════════
def _ccy_opts():
    return SUPPORTED_CURRENCIES

def _risk_opts():
    return [0, 1, 2]

def manual_input_form() -> ProjectInput | None:
    prj = ProjectInput()
    st.subheader("Project Details")
    c1, c2, c3 = st.columns(3)
    with c1:
        prj.project_name = st.text_input("Project name *", value="")
        prj.organisation = st.text_input("Organisation *", value="")
        prj.country      = st.selectbox("Country *", [""] + COUNTRY_LIST)
        prj.location     = st.text_input("Location", value="")
    with c2:
        prj.project_type = st.selectbox("Project type", PROJECT_TYPES)
        prj.description  = st.text_area("Description", value="")
        prj.reporting_currency = st.selectbox("Reporting currency", _ccy_opts(),
                                               index=_ccy_opts().index("USD"))
    with c3:
        prj.project_life          = st.number_input("Project life (years)", 1, 50, 10)
        prj.construction_period   = st.number_input("Construction period (years)", 1, 20, 2)
        prj.expected_delay_years  = st.number_input("Expected delay (years)", 0.0, 10.0, 0.0, 0.1)
        prj.inflation_rate        = st.number_input("Inflation % p.a.", 0.0, 100.0, 5.0, 0.5)
        prj.tax_rate              = st.number_input("Tax rate %", 0.0, 100.0, 25.0, 1.0)
        prj.discount_rate         = st.number_input("Discount rate (WACC) %", 0.0, 100.0, 12.0, 0.5)

    st.subheader("Capital Expenditure")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        prj.construction_cost = st.number_input("Construction cost", 0.0, step=100_000.0, format="%.0f")
        prj.construction_currency = st.selectbox("Currency", _ccy_opts(),
                                                  index=_ccy_opts().index(prj.reporting_currency))
        prj.construction_fx_pa = st.number_input("Expected FX move % p.a.", -30.0, 30.0, 0.0, 0.5)
        prj.construction_supplier_country = st.selectbox("Supplier country", [""] + COUNTRY_LIST)
    with c2:
        prj.equipment_cost = st.number_input("Equipment cost", 0.0, step=100_000.0, format="%.0f")
        prj.equipment_currency = st.selectbox("Currency##eq", _ccy_opts(),
                                               index=_ccy_opts().index(prj.reporting_currency))
        prj.equipment_fx_pa = st.number_input("Expected FX move % p.a.##eq", -30.0, 30.0, 0.0, 0.5)
        prj.equipment_supplier_country = st.selectbox("Supplier country##eq", [""] + COUNTRY_LIST)
    with c3:
        prj.land_building_cost = st.number_input("Land & buildings cost", 0.0, step=100_000.0, format="%.0f")
        prj.land_currency = st.selectbox("Currency##land", _ccy_opts(),
                                          index=_ccy_opts().index(prj.reporting_currency))
        prj.land_fx_pa = st.number_input("Expected FX move % p.a.##land", -30.0, 30.0, 0.0, 0.5)
        prj.land_supplier_country = st.selectbox("Supplier country##land", [""] + COUNTRY_LIST)
    with c4:
        prj.working_capital  = st.number_input("Working capital", 0.0, step=100_000.0, format="%.0f")
        prj.salvage_value    = st.number_input("Salvage value", 0.0, step=100_000.0, format="%.0f")
        prj.debt_ratio       = st.number_input("Debt ratio %", 0.0, 100.0, 40.0, 1.0)
        prj.loan_interest_rate = st.number_input("Loan interest %", 0.0, 100.0, 9.0, 0.5)
        prj.loan_term        = st.number_input("Loan term (years)", 1, 30, 8)

    st.subheader("Revenue & Operating Costs")
    c1, c2, c3 = st.columns(3)
    with c1:
        prj.annual_revenue     = st.number_input("Annual revenue *", 0.0, step=100_000.0, format="%.0f")
        prj.revenue_growth     = st.number_input("Revenue growth % p.a.", -20.0, 50.0, 3.0, 0.5)
        prj.revenue_currency   = st.selectbox("Revenue currency", _ccy_opts(),
                                               index=_ccy_opts().index(prj.reporting_currency))
        prj.revenue_fx_pa      = st.number_input("Revenue FX move % p.a.", -30.0, 30.0, 0.0, 0.5)
    with c2:
        prj.annual_opex        = st.number_input("Annual operating costs *", 0.0, step=100_000.0, format="%.0f")
        prj.opex_growth        = st.number_input("Opex growth % p.a.", -20.0, 50.0, 2.0, 0.5)
        prj.opex_currency      = st.selectbox("Opex currency", _ccy_opts(),
                                               index=_ccy_opts().index(prj.reporting_currency))
        prj.opex_fx_pa         = st.number_input("Opex FX move % p.a.", -30.0, 30.0, 0.0, 0.5)
    with c3:
        prj.annual_maintenance = st.number_input("Annual maintenance", 0.0, step=10_000.0, format="%.0f")
        prj.maintenance_currency = st.selectbox("Maint. currency", _ccy_opts(),
                                                 index=_ccy_opts().index(prj.reporting_currency))
        prj.maintenance_fx_pa  = st.number_input("Maint. FX move % p.a.", -30.0, 30.0, 0.0, 0.5)
        prj.materials_supplier_country = st.selectbox("Materials supplier country", [""] + COUNTRY_LIST)
        prj.materials_currency = st.selectbox("Materials currency", _ccy_opts(),
                                               index=_ccy_opts().index(prj.reporting_currency))
        prj.materials_fx_pa    = st.number_input("Materials FX move % p.a.", -30.0, 30.0, 0.0, 0.5)

    st.subheader("Import / Landed-Cost %")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    prj.import_transport_pct  = c1.number_input("Transport %", 0.0, 100.0, 0.0, 0.5)
    prj.import_insurance_pct  = c2.number_input("Insurance %", 0.0, 100.0, 0.0, 0.5)
    prj.import_duty_pct       = c3.number_input("Duty %", 0.0, 100.0, 0.0, 0.5)
    prj.import_taxes_pct      = c4.number_input("Taxes %", 0.0, 100.0, 0.0, 0.5)
    prj.conversion_cost_pct   = c5.number_input("Conversion %", 0.0, 100.0, 0.0, 0.5)
    prj.import_financing_pct  = c6.number_input("Financing/FX %", 0.0, 100.0, 0.0, 0.5)

    st.subheader("Risk Profile (0 = Low · 1 = Moderate · 2 = High)")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    prj.market_risk         = c1.selectbox("Market", _risk_opts(), index=1)
    prj.construction_risk   = c2.selectbox("Construction", _risk_opts(), index=1)
    prj.operating_risk      = c3.selectbox("Operating", _risk_opts(), index=1)
    prj.country_risk        = c4.selectbox("Country", _risk_opts(), index=1)
    prj.currency_volatility = c5.selectbox("FX volatility", _risk_opts(), index=1)
    prj.supplier_risk       = c6.selectbox("Supplier", _risk_opts(), index=1)

    return prj


# ═══════════════════════════════════════════════════════════════════════════════
# EXCHANGE RATE BOARD
# ═══════════════════════════════════════════════════════════════════════════════
def fx_board_page():
    st.subheader("💱 Exchange Rate Board")
    st.info("Rates below are used by the engine for multi-currency cash flows. "
            "Rates are expressed as units of each currency per 1 USD.")

    c1, c2 = st.columns([1, 3])
    with c1:
        use_live = st.checkbox("Attempt live fetch (public API)", value=True)
        rep_ccy  = st.selectbox("Reporting currency", _ccy_opts(),
                                 index=_ccy_opts().index("USD"))

    live_info = None
    if use_live:
        with st.spinner("Fetching live rates..."):
            live_info = try_fetch_live_rates()
        if live_info:
            st.session_state.fx_live_info = live_info
            st.session_state.fx_user_rates.update(live_info["rates"])
            st.success(f"Live rates updated · Source: {live_info['source']} · "
                       f"{live_info['timestamp']}")
        else:
            st.warning("Live market data unavailable — using user-provided rates.")

    board = build_exchange_rate_board(
        st.session_state.fx_user_rates, source="LIVE / USER-PROVIDED",
        timestamp=live_info["timestamp"] if live_info else _now_utc(),
        reporting=rep_ccy,
        live=live_info is not None)

    if board:
        df = pd.DataFrame(board)
        st.dataframe(df, width="stretch", hide_index=True)

    st.markdown("---")
    st.markdown("#### Edit rates (units of currency per 1 USD)")
    rate_cols = st.columns(6)
    updated_rates = dict(st.session_state.fx_user_rates)
    rate_keys = [k for k in updated_rates.keys() if k != "USD"]
    for i, ccy in enumerate(rate_keys):
        col = rate_cols[i % 6]
        val = col.number_input(ccy, value=float(updated_rates.get(ccy, 1.0)),
                                step=0.01, format="%.4f", key=f"fx_{ccy}_{rep_ccy}")
        updated_rates[ccy] = val
    updated_rates["USD"] = 1.0
    st.session_state.fx_user_rates = updated_rates

    board2 = build_exchange_rate_board(updated_rates, "USER-PROVIDED",
                                       _now_utc(), rep_ccy, live=False)
    st.markdown("**Updated user-provided board:**")
    st.dataframe(pd.DataFrame(board2), width="stretch", hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PROJECT ANALYSIS PAGE
# ═══════════════════════════════════════════════════════════════════════════════
def project_analysis_page():
    st.subheader("📊 Project Analysis")

    # each data-entry tab writes its drafted project into session state so a
    # later tab can never silently overwrite another tab's input
    tab_manual, tab_upload, tab_demo = st.tabs([
        "MANUAL INPUT", "UPLOAD DATA", "LOAD GZU DEMO"])

    with tab_demo:
        if st.button("🚀 LOAD GZU MASHAVA INNOVATION HUB", type="primary",
                      width="stretch"):
            st.session_state["_pf_demo"] = demo_project()
            st.session_state["pf_source"] = "GZU demo"
            st.success("Demo project loaded — select it below and Run Analysis.")
        st.markdown(DEMO_DISCLAIMER)

    with tab_upload:
        st.markdown("Upload a CSV or Excel file matching the CAPEXX template.")
        st.download_button("📥 Download CSV template", data=make_template_csv_bytes(),
                           file_name="capexx_template.csv")
        uploaded = st.file_uploader("Upload file", type=["csv", "xlsx", "pdf"],
                                     key="pf_upload")
        if uploaded:
            res = parse_upload(uploaded.read(), uploaded.name)
            if res["ok"]:
                st.session_state["_pf_upload"] = res["input"]
                st.session_state["pf_source"] = "Uploaded file"
                st.success(f"File loaded: {res['input'].project_name}")
                st.dataframe(res["table"], width="stretch", hide_index=True)
            else:
                for e in res["errors"]:
                    st.error(e)
            for w in res.get("warnings", []):
                st.warning(w)

    with tab_manual:
        st.session_state.setdefault("_pf_manual", ProjectInput())
        st.session_state["_pf_manual"] = manual_input_form()

    # ── choose which drafted project to analyse ──────────────────────────
    sources: list[tuple[str, ProjectInput]] = [
        ("Manual entry", st.session_state.get("_pf_manual", ProjectInput()))]
    default_idx = 0
    if st.session_state.get("_pf_upload") is not None:
        sources.append(("Uploaded file", st.session_state["_pf_upload"]))
        default_idx = len(sources) - 1
    if st.session_state.get("_pf_demo") is not None:
        sources.append(("GZU demo", st.session_state["_pf_demo"]))
        default_idx = len(sources) - 1
    src_labels = [s[0] for s in sources]
    if "pf_source" not in st.session_state or \
            st.session_state.get("pf_source") not in src_labels:
        st.session_state["pf_source"] = src_labels[default_idx]
    src_label = st.radio("Project source", src_labels, horizontal=True,
                         key="pf_source")
    prj = next(s[1] for s in sources if s[0] == src_label)

    # ── RUN ANALYSIS ─────────────────────────────────────────────────────
    st.markdown("---")

    prj.fx_rates = dict(st.session_state.fx_user_rates)
    prj.fx_rate_source = ("LIVE" if st.session_state.fx_live_info
                          else "USER-PROVIDED / DEMONSTRATION")
    if st.session_state.fx_live_info:
        prj.fx_rate_timestamp = st.session_state.fx_live_info["timestamp"]

    errs = prj.validation_errors()
    if errs:
        for e in errs:
            st.error(e)
        return

    if st.button("🚀 RUN ANALYSIS", type="primary", width="stretch",
                  key="run_analysis"):
        steps = [
            "Building cash-flow model…", "Computing NPV / IRR / MIRR…",
            "Running sensitivity analysis…", "Running scenario engine…",
            "Running stress tests…", "Aggregating FX exposure…",
            "Generating currency strategy…", "Running risk engine…",
            "Running decision engine…", "Building AI interpretation…",
            "Generating narration script…", "Analyse complete ✓"]
        prog = st.progress(0, text=steps[0])
        for i, step in enumerate(steps):
            time.sleep(0.18)
            prog.progress((i + 1) / len(steps), text=step)
        bundle = run_analysis(prj)
        st.session_state.bundle = bundle
        prog.empty()

    bundle: AnalysisBundle | None = st.session_state.bundle
    if bundle is None or bundle.project_input.project_name != prj.project_name:
        st.info("Click RUN ANALYSIS to evaluate this project.")
        return

    p  = bundle.project_input
    m  = bundle.metrics
    fx = bundle.fx

    # ── Status-light overlay ─────────────────────────────────────────────
    status_light_overlay(bundle.decision.status)

    # ── Top KPI row ──────────────────────────────────────────────────────
    st.markdown("### Key Results")
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("NPV", fmt_ccy(m.npv, p.reporting_currency),
              f"WACC {m.wacc:.1f}%")
    k2.metric("IRR", fmt_pct(m.irr))
    k3.metric("MIRR", fmt_pct(m.mirr))
    k4.metric("Payback", f"{m.payback:.1f} y" if m.payback else "n/a",
              f"Discounted {m.discounted_payback:.1f} y" if m.discounted_payback else "")
    k5.metric("Profitability Index", f"{m.pi:.2f}" if m.pi else "n/a")
    k6.metric("Risk", bundle.risk.level,
              f"Score {bundle.risk.score:.0f}/100")

    # ── Decision banner + 10s overlay already above ──────────────────────
    decision_banner(bundle.decision.status, bundle.decision.score,
                    bundle.decision.grade, bundle.decision.reasons)

    # ── Metric formula/result/interpretation table ───────────────────────
    with st.expander("Full metrics — formula → result → interpretation", expanded=True):
        metric_rows = [
            ("NPV", f"Σ FCFt/(1+r)^t − C0",
             fmt_ccy(m.npv, p.reporting_currency),
             "Positive NPV means value created above the required return."),
            ("IRR", "Rate that makes NPV = 0",
             fmt_pct(m.irr),
             f"{'Above' if (m.irr or 0)>=m.wacc else 'Below'} WACC ({m.wacc:.1f}%)."),
            ("MIRR", "Finance rate + reinvestment rate",
             fmt_pct(m.mirr),
             "Single unambiguous return metric."),
            ("Payback", "Year cumulative FCF turns positive",
             f"{m.payback:.1f} y" if m.payback else "n/a",
             f"{'Within' if m.payback and m.payback<=p.project_life*0.5 else 'Outside'} half the project life."),
            ("PI", "PV inflows / PV outflows",
             f"{m.pi:.2f}" if m.pi else "n/a",
             "PI > 1.0 signals value creation per unit invested."),
            ("ARR", "Avg profit / avg investment",
             fmt_pct(m.arr) if m.arr else "n/a",
             "Accounting-based return check."),
            ("EAA", "NPV / annuity factor",
             fmt_ccy(m.eaa, p.reporting_currency) if m.eaa else "n/a",
             "Annualised value across the operating life."),
            ("Break-even", "Revenue multiplier driving NPV to 0",
             f"{m.break_even_multiplier:.2f}x" if m.break_even_multiplier is not None else "n/a",
             "Revenue must stay above this multiple of the base case."),
        ]
        st.dataframe(pd.DataFrame(metric_rows, columns=["Metric", "Formula", "Result", "Interpretation"]),
                      width="stretch", hide_index=True)

    # ── Cash flow table & charts ─────────────────────────────────────────
    with st.expander("Cash-flow schedule and charts"):
        cf_df = bundle.model.to_dataframe()
        st.dataframe(cf_df.style.format("{:,.0f}", subset=cf_df.columns[2:]),
                      width="stretch", hide_index=True)
        fig = go.Figure()
        fig.add_trace(go.Bar(x=cf_df["Year"], y=cf_df["Revenue"], name="Revenue",
                             marker_color="#22c55e"))
        fig.add_trace(go.Bar(x=cf_df["Year"], y=cf_df["Operating Costs"], name="Opex",
                             marker_color="#ef4444"))
        fig.add_trace(go.Bar(x=cf_df["Year"], y=cf_df["Maintenance"], name="Maintenance",
                             marker_color="#f59e0b"))
        fig.add_trace(go.Scatter(x=cf_df["Year"], y=cf_df["Free Cash Flow"], name="FCF",
                                 mode="lines+markers", line=dict(color="#1547A0", width=3)))
        fig.update_layout(barmode="group", title="Revenue / Costs / FCF by Year",
                          xaxis_title="Year", yaxis_title="Amount",
                          template="plotly_white")
        st.plotly_chart(fig, width="stretch")

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=cf_df["Year"], y=cf_df["Cumulative FCF"],
                                  name="Cumulative FCF", fill="tozeroy",
                                  line=dict(color="#1547A0")))
        fig2.add_trace(go.Scatter(x=cf_df["Year"], y=cf_df["Cumulative Disc. FCF"],
                                  name="Cumulative Discounted FCF", fill="tozeroy",
                                  line=dict(color="#93C5FD")))
        fig2.update_layout(title="Cumulative Cash Flows", template="plotly_white")
        st.plotly_chart(fig2, width="stretch")

    # ── Risk radar ───────────────────────────────────────────────────────
    with st.expander("Risk analysis"):
        rd = bundle.risk
        categories = (["Cost", "Delay", "Revenue", "Opex", "Inflation",
                        "FX", "Interest", "Cash-flow", "Completion", "Supplier"])
        values = [rd.cost_risk, rd.delay_risk, rd.revenue_risk, rd.opex_risk,
                  rd.inflation_risk, rd.fx_risk, rd.interest_rate_risk,
                  rd.cashflow_risk, rd.completion_risk, rd.supplier_risk]
        fig_r = go.Figure(go.Scatterpolar(
            r=values + [values[0]], theta=categories + [categories[0]],
            fill="toself", line_color="#1547A0", opacity=0.75))
        fig_r.update_layout(title=f"Risk Radar — Overall {rd.level} ({rd.score:.0f}/100)",
                            template="plotly_white", polar=dict(radialaxis=dict(range=[0, 100])))
        st.plotly_chart(fig_r, width="stretch")
        for d in rd.detail:
            st.markdown(f"• {d}")

    # ── FX / Currency ────────────────────────────────────────────────────
    with st.expander("Currency / FX analysis"):
        st.markdown(f"**FX risk level:** {fx['fx_risk_level']} · "
                    f"Score {fx['fx_risk_score']:.0f}/100 · "
                    f"{fx['mismatch_count']} mismatched stream(s)")
        if fx["exposures"]:
            st.dataframe(pd.DataFrame(fx["exposures"]),
                          width="stretch", hide_index=True)
        for n in fx["hedging_notes"]:
            st.markdown(f"• {n}")
        st.markdown("**Transaction-level currency strategy:**")
        for x in bundle.fx_strategy:
            st.markdown(f"**{x['transaction']}** — invoice {x['invoice_currency']} "
                        f"(supplier: {x['supplier_country'] or 'n/a'})")
            st.markdown(f"  *Analysis:* {x['analysis']}")
            st.markdown(f"  *Recommendation:* {x['recommendation']}")

        if bundle.landed_cost:
            st.markdown("**Landed-cost build-up (imported equipment):**")
            lc = bundle.landed_cost
            st.dataframe(pd.DataFrame([lc]).T.rename(columns={0: "Amount"}).style.format("{:,.0f}"),
                          width="stretch")

    # ── Scenarios ────────────────────────────────────────────────────────
    with st.expander("Scenario analysis"):
        sc = bundle.scenarios
        for name, data in sc.items():
            tag = "🟢" if data["npv"] > 0 else "🔴"
            st.markdown(f"{tag} **{name}** — NPV {fmt_ccy(data['npv'], p.reporting_currency)} "
                        f"· IRR {fmt_pct(data['irr'])} · {data['description']}")
        fig_sc = go.Figure(go.Bar(
            x=list(sc.keys()), y=[d["npv"] for d in sc.values()],
            marker_color=["#22c55e" if d["npv"] > 0 else "#ef4444" for d in sc.values()]))
        fig_sc.update_layout(title="Scenario NPVs", yaxis_title=f"NPV ({p.reporting_currency})",
                             template="plotly_white")
        st.plotly_chart(fig_sc, width="stretch")

    # ── Stress tests ─────────────────────────────────────────────────────
    with st.expander("Stress testing"):
        st.dataframe(pd.DataFrame(bundle.stress).rename(columns={
            "impact": "NPV impact", "impact_pct": "Impact %"}),
            width="stretch", hide_index=True)
        fig_st = go.Figure(go.Bar(
            x=[s["name"] for s in bundle.stress],
            y=[s["impact"] for s in bundle.stress],
            marker_color=["#22c55e" if s["impact"] >= 0 else "#ef4444" for s in bundle.stress]))
        fig_st.update_layout(title="Stress NPV Impact vs Base",
                             yaxis_title=f"NPV Change ({p.reporting_currency})",
                             template="plotly_white")
        st.plotly_chart(fig_st, width="stretch")

    # ── Sensitivity ──────────────────────────────────────────────────────
    with st.expander("Sensitivity (tornado)"):
        for s in bundle.sensitivity:
            st.markdown(f"**{s['driver']}** — low NPV {fmt_ccy(s['npv_low'], p.reporting_currency)} "
                        f"/ high NPV {fmt_ccy(s['npv_high'], p.reporting_currency)} "
                        f"/ delta {fmt_ccy(s['delta'], p.reporting_currency)}")

    # ── AI Interpretation ────────────────────────────────────────────────
    with st.expander("AI Interpretation — WHAT / WHY / SO WHAT / WHAT IF / MONITOR"):
        interp = generate_ai_interpretation(bundle)
        st.markdown(interp)
        if st.button("Why did CAPEXX reach this result?"):
            st.markdown(why_did_capexx_text(bundle))

    # ── Robot voice panel (post-analysis) ────────────────────────────────
    robot_panel(bundle)

    # ── Download report ──────────────────────────────────────────────────
    from capexx_engine import build_report_markdown, build_email
    md_report = build_report_markdown(bundle)
    rpt_path = OUTPUT_RPT / f"{p.project_name.replace(' ','_')}_CAPEXX_Report.md"
    rpt_path.write_text(md_report, encoding="utf-8")
    st.download_button("📥 Download Report (Markdown)", data=md_report,
                        file_name=rpt_path.name, mime="text/markdown")
    csv_buf = io.StringIO()
    bundle.model.to_dataframe().to_csv(csv_buf, index=False)
    st.download_button("📥 Download Cash Flows (CSV)", data=csv_buf.getvalue(),
                        file_name=f"{p.project_name.replace(' ','_')}_cashflows.csv",
                        mime="text/csv")


def make_template_csv_bytes() -> bytes:
    from capexx_engine import make_template_csv
    return make_template_csv().encode("utf-8")


# ═══════════════════════════════════════════════════════════════════════════════
# PORTFOLIO PAGE
# ═══════════════════════════════════════════════════════════════════════════════
def portfolio_page():
    st.subheader("📁 Portfolio — Compare Projects")
    st.info("Add projects to the portfolio for side-by-side comparison. "
            "CAPEXX does not rank or recommend between projects.")

    bundle = st.session_state.bundle
    if bundle:
        if st.button("➕ Add current project to portfolio"):
            existing = [b.project_input.project_name for b in st.session_state.portfolio]
            if bundle.project_input.project_name not in existing:
                st.session_state.portfolio.append(bundle)
                st.success(f"Added: {bundle.project_input.project_name}")
            else:
                st.info("Already in portfolio.")
    else:
        st.info("Run a project analysis first to add it to the portfolio.")

    if not st.session_state.portfolio:
        st.info("No projects in portfolio.")
        return

    rows = []
    for b in st.session_state.portfolio:
        p_ = b.project_input; m_ = b.metrics
        rows.append({
            "Project": p_.project_name,
            "Country": p_.country,
            "Type": p_.project_type,
            "NPV": f"{m_.npv:,.0f}",
            "IRR": fmt_pct(m_.irr),
            "MIRR": fmt_pct(m_.mirr),
            "Payback": f"{m_.payback:.1f}y" if m_.payback else "n/a",
            "Risk": b.risk.level,
            "Decision": b.decision.status,
        })
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    st.markdown("#### Capital allocation illustration")
    names  = [r["Project"] for r in rows]
    npvs   = [float(r["NPV"].replace(",","")) for r in rows]
    fig = go.Figure(go.Pie(labels=names, values=[abs(v) for v in npvs],
                            hole=0.45,
                            marker=dict(colors=["#1547A0", "#3B82F6", "#93C5FD",
                                                 "#22c55e", "#f59e0b", "#ef4444"])))
    fig.update_layout(title="Relative project NPV sizes (illustrative, not a ranking)",
                      template="plotly_white")
    st.plotly_chart(fig, width="stretch")


# ═══════════════════════════════════════════════════════════════════════════════
# COMPARE & SELECT PAGE  (side-by-side · scorecard · budget · A/B what-if)
# ═══════════════════════════════════════════════════════════════════════════════

VARIANT_PRESETS = {
    "Capital cost +30%":      lambda p: (_scale(p, ["construction_cost",
                            "equipment_cost", "land_building_cost"], 1.3),
                            "Capital cost raised 30% (overrun stress)."),
    "Revenue −20%":           lambda p: (setattr(p, "annual_revenue",
                            p.annual_revenue * 0.8), "Demand shock: revenue −20%."),
    "Operating costs +20%":   lambda p: (_scale(p, ["annual_opex",
                            "annual_maintenance"], 1.2),
                            "Opex and maintenance raised 20%."),
    "WACC +3 percentage points": lambda p: (setattr(p, "discount_rate",
                            min(100.0, p.discount_rate + 3)),
                            "Discount rate raised by 3 pp."),
    "Construction delay +1 year": lambda p: (setattr(p, "expected_delay_years",
                            p.expected_delay_years + 1),
                            "Expected completion delay of +1 year."),
    "Revenue growth +5pp":    lambda p: (setattr(p, "revenue_growth",
                            p.revenue_growth + 5),
                            "Annual revenue growth raised 5 pp."),
    "FX depreciation −15%":   lambda p: (_fx_shock(p, 0.15),
                            "Non-reporting-currency cash flows weaken 15% "
                            "against the reporting currency."),
}
VARIANT_LABELS = list(VARIANT_PRESETS.keys())


def _scale(p: ProjectInput, fields: list[str], factor: float) -> None:
    for f in fields:
        setattr(p, f, getattr(p, f) * factor)


def _fx_shock(p: ProjectInput, shock: float) -> None:
    """Scale down non-reporting-currency cash flows by `shock` (FX shock)."""
    for f in ("annual_revenue", "annual_opex", "annual_maintenance",
              "construction_cost", "equipment_cost", "land_building_cost"):
        cur = getattr(p, f.replace("cost", "currency") if f.endswith("cost")
                      else f + "_currency", p.reporting_currency)
        if cur != p.reporting_currency:
            setattr(p, f, getattr(p, f) * (1.0 - shock))
    p.currency_volatility = min(5, p.currency_volatility + 1)


def _variant_project(p: ProjectInput, preset: str) -> tuple[ProjectInput, str]:
    p2 = copy.deepcopy(p)
    note = VARIANT_PRESETS[preset](p2)[1]
    p2.project_name = f"{p.project_name}  [{preset}]"
    return p2, note


def _available_bundles() -> list[AnalysisBundle]:
    """All analyses available to Compare: current + portfolio + compare store."""
    seen: dict[str, AnalysisBundle] = {}
    for b in list(st.session_state.compare_analyses.values()):
        seen[b.project_input.project_name] = b
    for b in st.session_state.portfolio:
        seen[b.project_input.project_name] = b
    b = st.session_state.bundle
    if b:
        seen[b.project_input.project_name] = b
    return list(seen.values())


def _ensure_analysis(prj: ProjectInput) -> AnalysisBundle:
    pname = prj.project_name
    if pname in st.session_state.compare_analyses:
        return st.session_state.compare_analyses[pname]
    b = run_analysis(prj)
    st.session_state.compare_analyses[pname] = b
    return b


def _compare_sidebar() -> None:
    with st.sidebar:
        st.divider()
        st.caption("⚖️ Compare sources")
        if st.button("➕ Add current analysis", width="stretch"):
            if st.session_state.bundle:
                _ensure_analysis(st.session_state.bundle.project_input)
                st.toast("Current analysis added to Compare")
            else:
                st.toast("No current analysis — run one on Project Analysis")
        if st.button("🏫 Add GZU demo project", width="stretch"):
            _ensure_analysis(demo_project())
            st.toast("GZU demo added to Compare")
        if st.button("🌍 Add sample portfolio (5 projects)", width="stretch"):
            for prj in sample_portfolio_projects():
                _ensure_analysis(prj)
            st.toast("5 sample projects added")


def _show_comparison_metrics(bundles: list[AnalysisBundle]) -> None:
    m_rows = {
        "Reporting currency": [b.project_input.reporting_currency for b in bundles],
        "Country":            [b.project_input.country for b in bundles],
        "Project type":       [b.project_input.project_type for b in bundles],
        "Life (years)":       [str(b.project_input.project_life) for b in bundles],
        "NPV":                [fmt_ccy(b.metrics.npv, b.project_input.reporting_currency) for b in bundles],
        "IRR":                [fmt_pct(b.metrics.irr) for b in bundles],
        "MIRR":               [fmt_pct(b.metrics.mirr) for b in bundles],
        "Payback (years)":    [f"{b.metrics.payback:.1f}" if b.metrics.payback else "n/a" for b in bundles],
        "PI (Profitability)": [f"{b.metrics.pi:.2f}" if b.metrics.pi else "n/a" for b in bundles],
        "EAA":                [fmt_ccy(b.metrics.eaa, b.project_input.reporting_currency) for b in bundles],
        "Total investment":   [fmt_ccy(b.metrics.total_investment, b.project_input.reporting_currency) for b in bundles],
        "Risk score":         [f"{b.risk.score:.0f}/100" for b in bundles],
        "FX mismatches":      [f"{b.fx['mismatch_count']}" for b in bundles],
        "Decision score":     [f"{b.decision.score:.0f}/100" for b in bundles],
    }
    df = pd.DataFrame(m_rows, index=[b.project_input.project_name for b in bundles]).T
    st.dataframe(df, width="stretch")


def _show_radar_overlay(bundles: list[AnalysisBundle]) -> None:
    categories = (["Cost", "Delay", "Revenue", "Opex", "Inflation", "FX",
                   "Interest", "Cash-flow", "Completion", "Supplier"])
    colors = ["#1547A0", "#f59e0b", "#22c55e", "#ef4444", "#8b5cf6", "#06b6d4"]
    fig = go.Figure()
    for i, b in enumerate(bundles):
        rd = b.risk
        values = [rd.cost_risk, rd.delay_risk, rd.revenue_risk, rd.opex_risk,
                  rd.inflation_risk, rd.fx_risk, rd.interest_rate_risk,
                  rd.cashflow_risk, rd.completion_risk, rd.supplier_risk]
        fig.add_trace(go.Scatterpolar(
            r=values + [values[0]], theta=categories + [categories[0]],
            fill="toself", name=b.project_input.project_name,
            line_color=colors[i % len(colors)], opacity=0.45))
    fig.update_layout(title="Risk radar overlay (higher = higher risk)",
                      template="plotly_white",
                      polar=dict(radialaxis=dict(range=[0, 100])))
    st.plotly_chart(fig, width="stretch")


def compare_page():
    st.subheader("⚖️ Compare & Select — choose what to fund")
    st.markdown(
        "Decision-support for banks, government and DFIs: rank candidate "
        "projects objectively, select the best set within a capital budget, "
        "and stress one project against a what-if variant. **Nothing here is "
        "financial advice** — final governance decisions remain with the "
        "human decision-maker.")
    _compare_sidebar()

    tab_note, tab_sbs, tab_score, tab_budget, tab_ab = st.tabs(
        ["ℹ️ How to use", "⬅️➡️ Side-by-side", "🎯 Funding scorecard",
         "💵 Budget selection", "🅰️🅱️ A/B what-if"])

    with tab_note:
        st.markdown("""
**What each tab does**

| Tab | Use it when… |
|---|---|
| **Side-by-side** | You want 2+ projects compared line-by-line (NPV, IRR, risk radar, FX). |
| **Funding scorecard** | 20+ proposals and you need an objective priority order ("which 10 of 20 do we fund first"). |
| **Budget selection** | You have a fixed capital ceiling (e.g. $100M or $500M) and want the best set that fits it. |
| **A/B what-if** | You want to see one project under a different assumption (costs +30%, revenue −20%, delay, FX shock…). |

**Where do projects come from?** Run analyses on **Project Analysis** (portfolio adds them),
or click the *Compare sources* buttons in the sidebar to batch-load the GZU demo and the
5-country sample portfolio.
""")

    bundles = _available_bundles()
    names = [b.project_input.project_name for b in bundles]

    # ── Side-by-side ───────────────────────────────────────────────────────
    with tab_sbs:
        if not bundles:
            st.info("No analyses yet. Run a project on Project Analysis, or add "
                    "demo/sample projects from the sidebar 'Compare sources'.")
        else:
            sel = st.multiselect("Projects to compare", names, default=names,
                                 key="cmp_sbs_sel")
            chosen = [b for b in bundles if b.project_input.project_name in sel]
            if chosen:
                st.markdown("#### Decision banners")
                cols = st.columns(len(chosen))
                for col, b in zip(cols, chosen):
                    with col:
                        st.markdown(f"**{b.project_input.project_name}**")
                        metric_card(b.decision.status, f"{b.decision.score:.0f}/100",
                                    f"{b.decision.grade} · {b.risk.level} risk")
                st.markdown("#### Metrics")
                _show_comparison_metrics(chosen)
                st.markdown("#### Risk radar overlay")
                _show_radar_overlay(chosen)
                with st.expander("Currency / FX exposure by project"):
                    for b in chosen:
                        st.markdown(f"**{b.project_input.project_name}** — "
                                    f"{b.fx['fx_risk_level']} "
                                    f"({b.fx['fx_risk_score']:.0f}/100, "
                                    f"{b.fx['mismatch_count']} mismatches)")
                        if b.fx["exposures"]:
                            st.dataframe(pd.DataFrame(b.fx["exposures"]),
                                         width="stretch", hide_index=True)
                with st.expander("Scenario comparison"):
                    sc_rows = {}
                    for b in chosen:
                        sc_rows[b.project_input.project_name] = {
                            "Base":        fmt_ccy(b.scenarios["BASE CASE"]["npv"],
                                                   b.project_input.reporting_currency),
                            "Optimistic":  fmt_ccy(b.scenarios["OPTIMISTIC"]["npv"],
                                                   b.project_input.reporting_currency),
                            "Pessimistic": fmt_ccy(b.scenarios["PESSIMISTIC"]["npv"],
                                                   b.project_input.reporting_currency),
                        }
                    st.dataframe(pd.DataFrame(sc_rows).T, width="stretch")

    # ── Funding scorecard ──────────────────────────────────────────────────
    with tab_score:
        st.markdown(
            "Objective **funding-priority score (0–100)** = 40 points value "
            "(NPV/PI) + 30 risk + 15 resilience + 15 Vision-2030 alignment. "
            "Use it to *justify* a capital plan, never to replace governance.")
        if not bundles:
            st.info("No projects available to score.")
        else:
            vision = st.slider("Vision 2030 alignment weight input (0–10)",
                               0.0, 10.0, 7.0, 0.5, key="cmp_vision")
            sel2 = st.multiselect("Projects to score", names, default=names,
                                  key="cmp_score_sel")
            wanted = [b for b in bundles if b.project_input.project_name in sel2]
            if wanted:
                rows = []
                for b in wanted:
                    f = funding_priority(b, vision_score=vision)
                    rows.append({
                        "Rank": 0, "Project": f["project_name"],
                        "Score": f["score"], "Grade": f["grade"],
                        "Value (40)": f["components"]["value"],
                        "Risk (30)": f["components"]["risk"],
                        "Resilience (15)": f["components"]["resilience"],
                        "Vision (15)": f["components"]["vision"],
                        "NPV": fmt_ccy(f["npv"], f["reporting_currency"]),
                        "Investment": fmt_ccy(f["investment"], f["reporting_currency"]),
                        "Decision": f["decision"],
                    })
                rows.sort(key=lambda r: r["Score"], reverse=True)
                for i, r in enumerate(rows, start=1):
                    r["Rank"] = i
                st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
                st.markdown("#### First-fund / defer / avoid")
                st.markdown("• **High priority (≥75):** shortlist for funding.")
                st.markdown("• **Moderate (50–74):** fund only with the flagged "
                            "conditions met.")
                st.markdown("• **Low (<50):** defer or drop unless a strategic "
                            "override applies.")
                with st.expander("Automatic conditions & flags"):
                    for b in wanted:
                        f = funding_priority(b, vision_score=vision)
                        st.markdown(f"**{f['project_name']}** "
                                    f"({f['grade']}, {f['score']})")
                        for flag in f["flags"]:
                            st.markdown(f"• {flag}")

    # ── Budget selection ───────────────────────────────────────────────────
    with tab_budget:
        st.markdown(
            "Enter a capital budget and CAPEXX uses a knapsack optimisation "
            "to pick the **highest total priority-score set that fits**.")
        if not bundles:
            st.info("No projects available to budget-select.")
        else:
            sel3 = st.multiselect("Candidate projects", names, default=names,
                                  key="cmp_budget_sel")
            cands = [b for b in bundles if b.project_input.project_name in sel3]
            if cands:
                vision_b = st.slider("Vision 2030 alignment input (budget tab)",
                                     0.0, 10.0, 7.0, 0.5, key="cmp_budget_vision")
                total = sum(b.metrics.total_investment for b in cands)
                budget = st.number_input(
                    "Capital budget (reporting currency of each project)",
                    min_value=1.0, value=max(1.0, total),
                    step=100000.0, key="cmp_budget_amt")
                force_in = st.multiselect("Force-include (even if over budget)",
                                          [b.project_input.project_name for b in cands],
                                          key="cmp_budget_force")
                if st.button("💵 Run budget selection", key="cmp_budget_run"):
                    costs = [b.metrics.total_investment for b in cands]
                    vals = [funding_priority(b, vision_score=vision_b)["score"]
                            for b in cands]
                    forced_names = set(force_in)
                    forced_cost = sum(b.metrics.total_investment for b in cands
                                      if b.project_input.project_name in forced_names)
                    remainder = budget - forced_cost
                    nf_indices = [i for i, b in enumerate(cands)
                                  if b.project_input.project_name not in forced_names]
                    picked: set[int] = set()
                    if remainder > 0 and nf_indices:
                        res = knapsack_select([costs[i] for i in nf_indices],
                                              [vals[i] for i in nf_indices],
                                              remainder)
                        picked = {nf_indices[pos] for pos in res["selected"]}
                    else:
                        res = {"unspent": max(0.0, remainder), "total_cost": 0.0,
                               "total_value": 0.0}
                    selected, deferred = [], []
                    for j, b in enumerate(cands):
                        if b.project_input.project_name in forced_names:
                            selected.append((j, b, "forced"))
                        elif j in picked:
                            selected.append((j, b, "optimised"))
                        else:
                            deferred.append((j, b))
                    s_rows = [{
                        "Project": b.project_input.project_name,
                        "Investment": fmt_ccy(b.metrics.total_investment,
                                              b.project_input.reporting_currency),
                        "Priority": vals[j],
                        "Basis": "forced" if basis == "forced" else "optimised",
                    } for j, b, basis in sorted(selected,
                                                key=lambda t: -vals[t[0]])]
                    total_used = sum(b.metrics.total_investment
                                     for _, b, _ in selected)
                    if forced_cost > budget:
                        st.warning(f"Forced projects alone need "
                                   f"{forced_cost:,.0f}, which exceeds "
                                   f"the budget {budget:,.0f}.")
                    st.success(
                        f"**Selected {len(selected)} of {len(cands)} "
                        f"projects.** Total required **{total_used:,.0f}** "
                        f"within budget **{budget:,.0f}** — surplus "
                        f"{budget - total_used:,.0f}. Combined priority "
                        f"{sum(vals[j] for j, *_ in selected):.0f} pts.")
                    st.dataframe(pd.DataFrame(s_rows), width="stretch",
                                 hide_index=True)
                    if deferred:
                        st.markdown("#### Deferred / not funded")
                        st.dataframe(pd.DataFrame([{
                            "Project": b.project_input.project_name,
                            "Investment": fmt_ccy(b.metrics.total_investment,
                                                  b.project_input.reporting_currency),
                            "Priority": vals[j],
                        } for j, b in deferred]), width="stretch", hide_index=True)
                    # ─ sensitivity: budget −10% ─
                    tight = budget * 0.9
                    tight_rem = tight - forced_cost
                    tight_picked: set[int] = set()
                    if tight_rem > 0 and nf_indices:
                        res_t = knapsack_select([costs[i] for i in nf_indices],
                                                [vals[i] for i in nf_indices],
                                                tight_rem)
                        tight_picked = {nf_indices[pos]
                                        for pos in res_t["selected"]}
                    kept_t = {j for j, b in enumerate(cands)
                              if b.project_input.project_name in forced_names} | tight_picked
                    dropped = [(j, b) for j, b in enumerate(cands)
                               if j not in kept_t]
                    with st.expander(f"📉 Sensitivity: budget cut 10% "
                                     f"({tight:,.0f})"):
                        if dropped:
                            st.markdown("**These projects fall out of the "
                                        "funded set under a 10% cut:**")
                            for j, b in dropped:
                                st.markdown(f"• **{b.project_input.project_name}**"
                                            f" — priority {vals[j]:.0f}, "
                                            f"investment {fmt_ccy(b.metrics.total_investment, b.project_input.reporting_currency)}")
                        else:
                            st.markdown("The funded set is unchanged under a "
                                        "10% budget cut (there was spare "
                                        "capacity).")
                    with st.expander("Conditions attached to selected projects"):
                        for j, b, basis in selected:
                            f = funding_priority(b, vision_score=vision_b)
                            st.markdown(f"**{f['project_name']}** "
                                        f"({f['grade']}, {f['score']})")
                            for flag in f["flags"]:
                                st.markdown(f"• {flag}")

    # ── A/B what-if ────────────────────────────────────────────────────────
    with tab_ab:
        st.markdown(
            "Run one project under a what-if variant and compare base vs "
            "variant side-by-side (overruns, demand shock, delay, FX…).")
        if not bundles:
            st.info("No projects available for what-if analysis.")
        else:
            base_name = st.selectbox("Baseline project",
                                     [b.project_input.project_name for b in bundles],
                                     key="cmp_ab_base")
            base = next(b for b in bundles
                        if b.project_input.project_name == base_name)
            preset = st.selectbox("What-if variant", VARIANT_LABELS,
                                  key="cmp_ab_preset")
            if st.button("🅰️🅱️ Run A/B what-if", key="cmp_ab_run"):
                p2, note = _variant_project(base.project_input, preset)
                v_bundle = run_analysis(p2)
                st.info(note)
                m = base.metrics
                vm = v_bundle.metrics
                delta = lambda a, b: (b - a)
                ab_rows = [
                    ("NPV",        fmt_ccy(m.npv, base.project_input.reporting_currency),
                     fmt_ccy(vm.npv, p2.reporting_currency)),
                    ("IRR",        fmt_pct(m.irr), fmt_pct(vm.irr)),
                    ("MIRR",       fmt_pct(m.mirr), fmt_pct(vm.mirr)),
                    ("Payback (y)", f"{m.payback:.1f}" if m.payback else "n/a",
                     f"{vm.payback:.1f}" if vm.payback else "n/a"),
                    ("PI",         f"{m.pi:.2f}" if m.pi else "n/a",
                     f"{vm.pi:.2f}" if vm.pi else "n/a"),
                    ("EAA",        fmt_ccy(m.eaa, base.project_input.reporting_currency),
                     fmt_ccy(vm.eaa, p2.reporting_currency)),
                    ("Decision",   f"{base.decision.status} "
                                   f"{base.decision.score:.0f}/100",
                     f"{v_bundle.decision.status} {v_bundle.decision.score:.0f}/100"),
                    ("Risk",       f"{base.risk.level} {base.risk.score:.0f}",
                     f"{v_bundle.risk.level} {v_bundle.risk.score:.0f}"),
                    ("FX risk",    f"{base.fx['fx_risk_level']} "
                                   f"{base.fx['fx_risk_score']:.0f}",
                     f"{v_bundle.fx['fx_risk_level']} "
                                   f"{v_bundle.fx['fx_risk_score']:.0f}"),
                ]
                dfab = pd.DataFrame(ab_rows, columns=["Metric", "Baseline", "Variant"])
                st.dataframe(dfab, width="stretch", hide_index=True)
                st.markdown("#### Decision banners")
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Baseline**")
                    metric_card(base.decision.status,
                                f"{base.decision.score:.0f}/100",
                                base.decision.grade)
                with c2:
                    st.markdown("**Variant — " + p2.project_name.split("  [")[0]
                                + "**")
                    metric_card(v_bundle.decision.status,
                                f"{v_bundle.decision.score:.0f}/100",
                                v_bundle.decision.grade)
                if st.button("➕ Add variant to portfolio", key="cmp_ab_add"):
                    st.session_state.portfolio.append(v_bundle)
                    st.success("Variant added to Portfolio page.")


# ═══════════════════════════════════════════════════════════════════════════════
# OVERRUN AI & EXPLAINABILITY PAGE (Version 2.0)
# ═══════════════════════════════════════════════════════════════════════════════

def _ml_callout(title: str, body: str):
    st.markdown(
        f'<div style="background:#EEF2FF;border-left:4px solid #1547A0;'
        f'border-radius:6px;padding:.6rem .9rem;margin:.4rem 0;">'
        f'<strong style="color:#1547A0;">{title}</strong><br>'
        f'<span style="color:#334155;font-size:.9rem;">{body}</span></div>',
        unsafe_allow_html=True)


def overrun_ai_page():
    st.subheader("🧠 Overrun AI & Explainability")
    st.markdown(
        "Predicts the chance a capital project will **overrun its budget by "
        f"{OVERRUN_THRESHOLD:.0f}% or more**, using machine learning trained on "
        "**historical project data**. Every chart and every number comes with a "
        "plain-English explanation. The model only ever learns from data you "
        "provide — or from the bundled **invented DEMONSTRATION dataset**.")

    tab_data, tab_train, tab_trends, tab_predict = st.tabs(
        ["📥 History data", "🎯 Train & validate", "📈 Trends & sectors",
         "🕵️ Predict a project"])

    # ── Tab 1: history data ─────────────────────────────────────────────────
    with tab_data:
        c1, c2 = st.columns(2)
        with c1:
            if st.button("🌍 Load DEMONSTRATION dataset (invented)",
                         width="stretch"):
                with st.spinner("Generating demonstration project history…"):
                    st.session_state["ml_history"] = demo_history_dataframe()
                st.session_state["ml_source"] = "DEMONSTRATION"
                st.session_state["ml_art"] = None
                st.rerun()
            st.download_button("⬇️ Download demo dataset CSV",
                               data=demo_history_csv_bytes(),
                               file_name="overrun_history_demo.csv",
                               width="stretch")
        with c2:
            st.download_button("🧾 Download history template CSV",
                               data=history_template_csv_bytes(),
                               file_name="overrun_history_template.csv",
                               width="stretch")
            up = st.file_uploader("Or upload YOUR historical projects",
                                  type=["csv", "xlsx"], key="ml_upload")
            if up is not None:
                try:
                    df = (pd.read_csv(up) if up.name.endswith(".csv")
                          else pd.read_excel(up))
                    ok, errs, warns = validate_history_frame(df)
                    if ok:
                        st.session_state["ml_history"] = df
                        st.session_state["ml_source"] = "USER DATA"
                        st.session_state["ml_art"] = None
                        st.success(f"Loaded {len(df)} rows of project history.")
                    else:
                        for e in errs:
                            st.error(e)
                    for w in warns:
                        st.warning(w)
                except Exception as ex:
                    st.error(f"Could not read file: {ex}")

        hist = st.session_state.get("ml_history")
        if hist is not None:
            if st.session_state["ml_source"] == "DEMONSTRATION":
                st.warning(ML_DEMO_DISCLAIMER)
            risky = float((hist[TARGET] >= OVERRUN_THRESHOLD).mean())
            st.markdown(
                f"**_Source:_** {st.session_state['ml_source']} · "
                f"{len(hist)} projects · "
                f"{risky:.0%} had overruns ≥ {OVERRUN_THRESHOLD:.0f}% · "
                f"mean overrun {hist[TARGET].mean():.1f}% · "
                f"{hist['sector'].nunique()} sectors")
            st.dataframe(hist.head(15), width="stretch", hide_index=True)
        else:
            st.info("Load the demo dataset or upload your own project history "
                    "to enable training and validation.")

    # ── Tab 2: train & validate ─────────────────────────────────────────────
    with tab_train:
        hist = st.session_state.get("ml_history")
        if hist is None:
            st.info("Load or upload project history first (first tab).")
        else:
            art = st.session_state.get("ml_art")
            if art is None:
                if st.button("🎯 TRAIN OVERRUN MODELS", type="primary",
                             key="ml_train"):
                    with st.spinner("Training Random Forest, XGBoost, Logistic "
                                    "& Neural Network on the history… "
                                    "(~20–40 seconds)"):
                        try:
                            art = train_overrun_models(
                                hist, demo_fitted=(st.session_state["ml_source"]
                                                   == "DEMONSTRATION"))
                            st.session_state["ml_art"] = art
                        except Exception as ex:
                            st.error(f"Training failed: {ex}")
            else:
                st.button("↻ Re-train with same data", key="ml_retrain",
                          on_click=lambda: (st.session_state.__setitem__(
                              "ml_art", None)))
                m = art.metrics
                # model comparison
                st.markdown("#### Choosing the best model")
                cmp_df = pd.DataFrame(art.model_compare).rename(
                    columns={"roc_auc": "ROC-AUC", "accuracy": "Accuracy"})
                cmp_df["ROC-AUC"] = cmp_df["ROC-AUC"].map(lambda v: f"{v:.1%}")
                cmp_df["Accuracy"] = cmp_df["Accuracy"].map(lambda v: f"{v:.1%}")
                st.dataframe(cmp_df, width="stretch", hide_index=True)
                fig_cmp = go.Figure(go.Bar(
                    x=[r["model"] for r in art.model_compare],
                    y=[r["roc_auc"] for r in art.model_compare],
                    marker_color=["#93C5FD", "#1547A0", "#f59e0b", "#8b5cf6"]))
                fig_cmp.add_hline(y=0.5, line_dash="dot", line_color="#94a3b8")
                fig_cmp.update_layout(
                    title="How each model separates risky from safe projects",
                    yaxis_title="ROC-AUC (0.5 = coin flip)",
                    template="plotly_white")
                st.plotly_chart(fig_cmp, width="stretch")
                _ml_callout("Why this model?",
                            f"**{m['best_model']}** was selected as the deployed "
                            "model — the most robust model whose accuracy was "
                            "within 2 points of the best. Its strong AUC means "
                            "it catches risky projects far better than human "
                            "guesswork, and unlike a black box we can show you "
                            "exactly which signals it weighs.")

                # risk distribution
                st.markdown("#### Overall risk picture of the test projects")
                fig_risk = go.Figure(go.Bar(
                    x=RISK_LABELS, y=m["risk_counts"],
                    marker_color=["#22c55e", "#f59e0b", "#ef4444"],
                    text=m["risk_counts"], textposition="outside"))
                fig_risk.update_layout(
                    title="How many held-out projects the model placed in each "
                          "predicted-risk bucket",
                    yaxis_title="Number of projects", template="plotly_white")
                st.plotly_chart(fig_risk, width="stretch")
                _ml_callout(
                    "Reading this chart",
                    f"The **green** bar (LOW, under 30% predicted overrun "
                    f"chance) shows the safest candidates — {m['risk_counts'][0]} "
                    f"project(s). The **yellow** bar (MEDIUM, 30–60%) needs "
                    f"extra safeguards — {m['risk_counts'][1]} project(s). The "
                    f"**red** bar (HIGH, over 60%) is capital at risk — "
                    f"{m['risk_counts'][2]} project(s) require contingency of "
                    f"65%+ or rejection unless strategically essential.")

                # ROC
                st.markdown("#### ROC curve — how well the model separates "
                            "risky from safe")
                roc = m["roc"]
                fig_roc = go.Figure()
                fig_roc.add_trace(go.Scatter(
                    x=roc["fpr"], y=roc["tpr"], mode="lines",
                    name=f"{m['best_model']}",
                    line=dict(color="#1547A0", width=3)))
                fig_roc.add_trace(go.Scatter(
                    x=[0, 1], y=[0, 1], mode="lines", name="Random guess",
                    line=dict(color="#94a3b8", dash="dot")))
                fig_roc.update_layout(
                    xaxis_title="False alarm rate",
                    yaxis_title="Share of risky projects caught",
                    template="plotly_white")
                st.plotly_chart(fig_roc, width="stretch")
                _ml_callout("What the number means", explain_roc(m["roc_auc"]))

                # confusion matrix
                st.markdown("#### Confusion matrix — exact hits and misses")
                cm = m["confusion_matrix"]
                cm_df = pd.DataFrame(
                    [["Caught", cm["tp"], cm["fp"]],
                     ["Missed", cm["fn"], cm["tn"]]],
                    columns=["Outcome", "Actually risky",
                             "Actually safe"]).set_index("Outcome")
                st.dataframe(cm_df, width="stretch")
                _ml_callout("Good or bad?", explain_confusion(m))

                # feature importance
                st.markdown("#### Top warning signs the model weighs")
                fig_imp = go.Figure(go.Bar(
                    x=[f["share"] for f in art.importances][::-1],
                    y=[f["feature"] for f in art.importances][::-1],
                    orientation="h",
                    marker_color="#1547A0"))
                fig_imp.update_layout(
                    xaxis_title="Share of the model's signal (%)",
                    template="plotly_white")
                st.plotly_chart(fig_imp, width="stretch")
                _ml_callout("Why these matter",
                            explain_feature_importance(art.importances))

                # residuals
                st.markdown("#### Residual plot — how big are the mistakes?")
                rp = art.residual_stats
                fig_res = go.Figure()
                fig_res.add_trace(go.Scatter(
                    x=rp["actual_vs_pred"]["actual"],
                    y=rp["actual_vs_pred"]["predicted"], mode="markers",
                    name="predicted vs actual",
                    marker=dict(color="#1547A0", size=7)))
                mx = max(max(rp["actual_vs_pred"]["actual"] or [0]), 1)
                fig_res.add_trace(go.Scatter(
                    x=[0, mx], y=[0, mx], mode="lines", name="perfect line",
                    line=dict(color="#94a3b8", dash="dot")))
                fig_res.update_layout(
                    xaxis_title="Actual overrun (%)",
                    yaxis_title="Predicted overrun (%)",
                    template="plotly_white")
                st.plotly_chart(fig_res, width="stretch")
                _ml_callout("Accuracy", explain_residuals(rp))

                # calibration
                st.markdown("#### Calibration — can '60% risk' be trusted?")
                cal = m["calibration"]
                fig_cal = go.Figure()
                fig_cal.add_trace(go.Scatter(
                    x=cal["mean_predicted"], y=cal["fraction_positive"],
                    mode="lines+markers", name="model",
                    line=dict(color="#f59e0b", width=3)))
                fig_cal.add_trace(go.Scatter(
                    x=[0, 1], y=[0, 1], mode="lines", name="perfect calibration",
                    line=dict(color="#94a3b8", dash="dot")))
                fig_cal.update_layout(
                    xaxis_title="Model's predicted overrun chance",
                    yaxis_title="Share that actually overran",
                    template="plotly_white")
                st.plotly_chart(fig_cal, width="stretch")
                _ml_callout("Can you trust the probabilities?",
                            explain_calibration(m))

                # statistical tests
                st.markdown("#### Statistical health checks")
                st1, st2 = st.columns(2)
                st3, st4 = st.columns(2)
                with st1:
                    sw = art.stats["shapiro_wilk"]
                    st.markdown(f"**Shapiro–Wilk** — p = {sw['pvalue']:.4f}")
                    _ml_callout("Bell-curve check", explain_shapiro(sw))
                with st2:
                    bp = art.stats["breusch_pagan"]
                    st.markdown(f"**Breusch–Pagan** — p = "
                                f"{bp['pvalue']:.2f}"
                                if not np.isnan(bp["pvalue"]) else
                                "**Breusch–Pagan** — not computed")
                    _ml_callout("Error consistency", explain_breusch_pagan(bp))
                with st3:
                    dw = art.stats["durbin_watson"]
                    st.markdown(f"**Durbin–Watson** — {dw:.2f}")
                    _ml_callout("Random or patterned mistakes?",
                                explain_durbin_watson(dw))
                with st4:
                    st.markdown("**Variance Inflation Factor (VIF)**")
                    vif_df = pd.DataFrame(
                        [{"Factor": k, "VIF": (round(v, 1) if not np.isnan(v)
                                               else None)}
                         for k, v in art.stats["vif"].items()])
                    st.dataframe(vif_df, width="stretch", hide_index=True)
                    _ml_callout("Redundant inputs?", explain_vif(art.stats["vif"]))

    # ── Tab 3: trends & sectors ─────────────────────────────────────────────
    with tab_trends:
        art = st.session_state.get("ml_art")
        hist = st.session_state.get("ml_history")
        if art is None or hist is None:
            st.info("Train the model first (second tab).")
        else:
            t = art.trends
            _ml_callout("What the history as a whole says",
                        explain_trends(t))

            st.markdown("#### Average cost overrun by approval year")
            fig_y = go.Figure()
            fig_y.add_trace(go.Scatter(
                x=t["by_year"]["years"], y=t["by_year"]["overruns"],
                mode="lines+markers", line=dict(color="#1547A0", width=3),
                fill="tozeroy"))
            fig_y.update_layout(xaxis_title="Year",
                                yaxis_title="Average overrun (%)",
                                template="plotly_white")
            st.plotly_chart(fig_y, width="stretch")

            st.markdown("#### Average overrun by sector")
            fig_s = go.Figure(go.Bar(
                x=t["by_sector"]["sectors"], y=t["by_sector"]["overruns"],
                marker_color=["#22c55e" if v < 25 else
                              "#f59e0b" if v < 40 else "#ef4444"
                              for v in t["by_sector"]["overruns"]]))
            fig_s.update_layout(xaxis_title="Sector",
                                yaxis_title="Average overrun (%)",
                                template="plotly_white")
            st.plotly_chart(fig_s, width="stretch")

            st.markdown("#### Does size mean more risk? "
                        "(overruns vs project size)")
            corr = t["size_corr"]
            fig_sc = go.Figure(go.Scatter(
                x=np.log1p(hist["budget_usd"].clip(lower=1.0)),
                y=hist[TARGET], mode="markers", name="projects",
                marker=dict(color="#1547A0", size=6, opacity=0.6)))
            z = np.polyfit(np.log1p(hist["budget_usd"].clip(lower=1.0)),
                           hist[TARGET], 1)
            xs = np.linspace(hist["budget_usd"].clip(lower=1).apply(np.log1p).min(),
                             hist["budget_usd"].clip(lower=1).apply(np.log1p).max(),
                             50)
            fig_sc.add_trace(go.Scatter(x=xs, y=np.polyval(z, xs), mode="lines",
                                        name="trend",
                                        line=dict(color="#ef4444", width=2)))
            fig_sc.update_layout(xaxis_title="Ln(project size, USD)",
                                 yaxis_title="Actual overrun (%)",
                                 template="plotly_white")
            st.plotly_chart(fig_sc, width="stretch")
            if abs(corr) > 0.05:
                _ml_callout("Size insight",
                            f"Correlation {corr:+.2f}: "
                            f"{'larger projects overran more' if corr > 0 else 'larger projects overran less'}"
                            " in this history. Consider phasing very big builds.")
            else:
                _ml_callout("Size insight",
                            "Project size makes little difference in this "
                            "history — risk is driven by other factors.")

            st.markdown("#### Actual overruns inside each predicted-risk bucket")
            buckets = m["risk_buckets"] if (m := art.metrics) else {}
            fig_b = go.Figure()
            for lbl in RISK_LABELS:
                b = buckets[lbl]
                fig_b.add_trace(go.Box(
                    name=f"{lbl} predicted risk (n={b['n']})",
                    q1=[b["q25"]], median=[b["median"]], q3=[b["q75"]],
                    lowerfence=[b["whisker_lo"]], upperfence=[b["whisker_hi"]],
                    mean=[b["median"]], boxpoints="all",
                    pointpos=0, jitter=0.3))
            fig_b.update_layout(yaxis_title="Actual overrun (%)",
                                template="plotly_white")
            st.plotly_chart(fig_b, width="stretch")
            _ml_callout(
                "Box plot guide",
                "Each box is the middle half of projects, the line inside is the "
                "median. In this history, projects the model called HIGH risk "
                f"ended with a median overrun of "
                f"**{buckets['HIGH']['median']:.0f}%** vs "
                f"**{buckets['LOW']['median']:.0f}%** for LOW-risk ones — the "
                "risk labels separate real outcomes.")

    # ── Tab 4: predict a project ────────────────────────────────────────────
    with tab_predict:
        art = st.session_state.get("ml_art")
        if art is None:
            st.info("Train the model first (second tab).")
        else:
            with st.expander("⚡ Pre-fill from the current CAPEXX analysis",
                             expanded=False):
                bnd = st.session_state.bundle
                if bnd:
                    if st.button("Use current project's inputs", key="ml_use"):
                        p = bnd.project_input
                        st.session_state["_ml_form"] = {
                            "sector": p.project_type,
                            "budget_usd": float(bnd.metrics.total_investment),
                            "complexity": round(min(5.0, 1.0 + p.construction_risk
                                                    * 1.2), 1),
                            "design_completeness_pct": 60.0,
                            "procurement_delay_days": 30.0,
                            "change_orders": 2.0,
                            "inflation_at_award": 50.0,
                            "fx_rate_at_award": float(
                                p.fx_rates.get(p.reporting_currency, 1.0)
                                if p.reporting_currency != "USD" else 40.0),
                            "year": 2026,
                        }
                        st.rerun()
                else:
                    st.info("Run a project on Project Analysis to pre-fill it here.")

            f = st.session_state.get("_ml_form", {})
            sectors = sorted(hist["sector"].astype(str).unique() if
                             (hist := st.session_state.get("ml_history")) is not None
                             and "sector" in hist else ["Building"])
            c1, c2 = st.columns(2)
            with c1:
                sector = st.selectbox("Sector", sectors,
                                      index=sectors.index(f["sector"]) if
                                      f.get("sector") in sectors else 0,
                                      key="ml_p_sector")
                budget = st.number_input("Project size (USD)",
                                         min_value=1000.0,
                                         value=float(f.get("budget_usd", 10_000_000)),
                                         step=100_000.0, key="ml_p_budget")
                complexity = st.slider("Complexity (1 = simple, 5 = very complex)",
                                       1.0, 5.0, float(f.get("complexity", 3.0)),
                                       0.1, key="ml_p_complex")
                design = st.slider("Design completeness before construction (%)",
                                   0, 100, int(f.get("design_completeness_pct", 60)),
                                   key="ml_p_design")
            with c2:
                delay = st.slider("Procurement delay at start (days)",
                                  0, 300, int(f.get("procurement_delay_days", 30)),
                                  key="ml_p_delay")
                changes = st.slider("Change orders during construction",
                                    0, 12, int(f.get("change_orders", 2)),
                                    key="ml_p_changes")
                infl = st.slider("Inflation at award (%)", 0, 600,
                                 int(f.get("inflation_at_award", 50)),
                                 key="ml_p_infl")
                fxr = st.slider("Exchange rate at award (units per USD)",
                                0.0, 600.0, float(f.get("fx_rate_at_award", 40)),
                                1.0, key="ml_p_fx")
            row = {"sector": sector, "budget_usd": budget, "complexity": complexity,
                   "design_completeness_pct": float(design),
                   "procurement_delay_days": float(delay),
                   "change_orders": float(changes),
                   "inflation_at_award": float(infl),
                   "fx_rate_at_award": float(fxr),
                   "year": int(f.get("year", 2026))}
            if st.button("🕵️ PREDICT OVERRUN RISK", type="primary",
                         key="ml_predict"):
                pred = predict_overrun(art, row)
                p = pred["probability"]
                colour = ("#ef4444" if p >= 0.6 else
                          "#f59e0b" if p >= 0.3 else "#22c55e")
                st.markdown(
                    f'<div style="background:{colour};color:#fff;border-radius:12px;'
                    f'padding:1.2rem 1.5rem;">'
                    f'<div style="font-size:.85rem;opacity:.9;">Probability of a '
                    f'cost overrun ≥ {OVERRUN_THRESHOLD:.0f}%</div>'
                    f'<div style="font-size:2.4rem;font-weight:800;">{p:.0%}</div>'
                    f'<div style="font-size:1rem;">{pred["prediction"]} · '
                    f'confidence: {pred["confidence"]}</div></div>',
                    unsafe_allow_html=True)
                _ml_callout("Confidence", (
                    "**HIGH** — the warning signs (or green flags) match patterns "
                    "the model has seen many times." if pred["confidence"] == "HIGH"
                    else "**MEDIUM** — the model has a reasonable read but this "
                    "project is unusual; combine with expert judgement."
                    if pred["confidence"] == "MEDIUM" else
                    "**LOW** — this project looks unlike history, so lean on "
                    "expert judgement and extra due diligence."))
                st.markdown("#### Why this result?")
                for f_ in pred["top_factors"]:
                    st.markdown(f"• **{f_['factor']}** — {f_['detail']}")
                if pred["risk_flags"]:
                    st.markdown("#### Warning flags raised")
                    for fl in pred["risk_flags"]:
                        st.markdown(f"• **{fl['factor']}** ({fl['level']}): "
                                    f"{fl['why']}")
                st.markdown("#### What to do about it")
                for a in pred["recommended_actions"]:
                    st.markdown(f"**1.** {a}" if pred["recommended_actions"].index(a) == 0 else f"• {a}")
                st.markdown("#### What would reduce the risk")
                for dr in pred["data_reductions"]:
                    st.markdown(f"• **{dr['action']}** — {dr['impact']}")

            source = st.session_state["ml_source"]
            st.caption(
                ("This prediction is made on the DEMONSTRATION dataset — "
                 "entirely invented numbers, NOT real evidence." if
                 source == "DEMONSTRATION" else
                 "Prediction based on the history you uploaded — limited by "
                 "how representative and how complete that data is."))


# ═══════════════════════════════════════════════════════════════════════════════
# ASK CAPEXX AI PAGE
# ═══════════════════════════════════════════════════════════════════════════════
def ask_ai_page():
    st.subheader("🤖 ASK CAPEXX AI — Global Investment Research")
    st.markdown(
        "Ask any capital-budgeting, risk, currency, macro, country, "
        "company or financial-market question. CAPEXX grounds every answer "
        "in verifiable sources and never fabricates data.")

    with st.expander("💡 Try these example questions"):
        for ex in EXAMPLE_QUESTIONS:
            if st.button(ex, key=f"ex_{ex[:25]}"):
                st.session_state["_ask_q"] = ex

    with st.expander("🔎 Search modes", expanded=False):
        for mode in SEARCH_MODES:
            st.markdown(f"• {mode}")

    q = st.text_input("Your question", value=st.session_state.get("_ask_q", ""),
                       key="ask_input",
                       placeholder="e.g. Analyse the investment climate in Zimbabwe")
    mode = st.selectbox("Mode", SEARCH_MODES, index=0)

    if q and st.button("ASK CAPEXX AI", type="primary"):
        ctx = {}
        if st.session_state.bundle:
            ctx["project_results"] = st.session_state.bundle
        with st.spinner("Researching…"):
            result = research(q, mode=mode.lower().split()[0], context=ctx if ctx else None)
        st.session_state.ask_history.append(result)
        st.markdown(result["markdown"])
        if result.get("status_line"):
            st.caption(result["status_line"])
        st.caption(f"Status: {result['status']}")
        if result.get("references"):
            with st.expander("References"):
                for r in result["references"]:
                    st.markdown(f"• {r}")
        if st.button("🔊 Read answer aloud", key="ask_tts"):
            res = read_aloud(result["spoken"])
            if res["ok"]:
                st.audio(res["path"])
            else:
                st.info(f"⚠️ VOICE UNAVAILABLE — {res['error']}")

    if st.session_state.ask_history:
        st.markdown("---")
        st.markdown("#### Session history")
        for i, h in enumerate(reversed(st.session_state.ask_history[-10:])):
            with st.expander(f"Q: {h['question'][:70]}…  [{h['status']}]"):
                st.markdown(h["markdown"])


# ═══════════════════════════════════════════════════════════════════════════════
# REPORTS & DELIVERY PAGE
# ═══════════════════════════════════════════════════════════════════════════════
def reports_page():
    st.subheader("📤 Reports & Delivery")
    bundle = st.session_state.bundle
    if not bundle:
        st.info("Run a project analysis first.")
        return

    from capexx_engine import build_report_markdown, build_email, try_send_email

    md_report = build_report_markdown(bundle)
    st.download_button("📥 Download Full Report (Markdown)",
                        data=md_report,
                        file_name=f"{bundle.project_input.project_name.replace(' ','_')}_CAPEXX_Report.md",
                        mime="text/markdown")

    email = build_email(bundle)
    st.markdown("**Email preview:**")
    st.markdown(f"**Subject:** {email['subject']}")
    st.text_area("Body", email["body"], height=260)

    recipient = st.text_input("Recipient email", value="decision-maker@example.com")
    smtp_host = st.text_input("SMTP host", value="smtp.gmail.com")
    smtp_port = st.text_input("SMTP port", value="587")
    smtp_user = st.text_input("SMTP user", value="")
    smtp_pass = st.text_input("SMTP password", type="password", value="")

    if st.button("📧 SEND EMAIL", type="primary"):
        cfg = {"smtp_host": smtp_host, "smtp_port": smtp_port,
               "smtp_user": smtp_user, "smtp_password": smtp_pass,
               "smtp_from": smtp_user}
        res = try_send_email(email["subject"], email["body"], recipient, config=cfg)
        if res["confirmed"]:
            st.success(res["message"])
        else:
            st.warning(res["message"])

    st.markdown("---")
    st.markdown("#### Audio narration")
    narr = build_narration_text(bundle)
    au_path = str(OUTPUT_AU / f"{bundle.project_input.project_name.replace(' ','_')}_narration.mp3")
    if st.button("🎙️ Generate narration audio"):
        res = synthesize(narr, out_path=au_path)
        if res["ok"]:
            st.audio(res["path"])
            with open(res["path"], "rb") as f:
                st.download_button("📥 Download audio", data=f.read(),
                                    file_name=os.path.basename(res["path"]),
                                    mime="audio/mpeg")
        else:
            st.warning(f"⚠️ {res['error']}")


# ═══════════════════════════════════════════════════════════════════════════════
# ABOUT PAGE
# ═══════════════════════════════════════════════════════════════════════════════
def about_page():
    st.subheader("ℹ️ About CAPEXX AI AGENT")
    robot_img = ASSETS / "robot.png"
    if robot_img.exists():
        st.image(str(robot_img), width=120)
    st.markdown("""
**CAPEXX AI AGENT** is an AI-powered capital-project decision intelligence platform.

### How it works
CAPEXX takes your project proposal and runs it through **one central cash-flow
engine** that produces a full capital-budgeting analysis, risk profile, currency
(FX) analysis, scenario comparison, stress tests, and a transparent
rule-based decision — all from a single source of truth.

### Who it is for
Any organisation making long-term capital investment decisions —
infrastructure, hospitals, universities, innovation hubs, energy, mining,
technology, manufacturing, agriculture, transport, real estate, tourism,
telecommunications.

### Transparency
- Every decision rule is shown explicitly (financial, risk, scenario pillars).
- All figures trace to Input → Calculation → Evidence → Interpretation.
- External data points carry Source | Date | Status labels.
- Market-simulation prices are labelled **DEMONSTRATION ONLY**.
- This is a decision-support system; it does not replace professional advice.

### Data sources
- Exchange rates: public keyless API (open.er-api.com) at run time.
- Country statistics: World Bank Open Data API.
- News: Google News RSS.
- Knowledge base: standard published finance textbooks and references.
""", unsafe_allow_html=False)

    st.markdown("### Documentation")
    docs_dir = ROOT / "docs"
    for md_file in sorted(docs_dir.glob("*.md")):
        if st.button(f"📄 {md_file.stem.replace('_',' ').title()}", key=f"doc_{md_file.name}"):
            st.markdown(md_file.read_text(encoding="utf-8"))


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    inject_css()
    if not st.session_state.logged_in:
        login_page()
        return

    page = sidebar()
    if   page == PAGES[0]: home_page()
    elif page == PAGES[1]: project_analysis_page()
    elif page == PAGES[2]: portfolio_page()
    elif page == PAGES[3]: compare_page()
    elif page == PAGES[4]: overrun_ai_page()
    elif page == PAGES[5]: fx_board_page()
    elif page == PAGES[6]: ask_ai_page()
    elif page == PAGES[7]: reports_page()
    elif page == PAGES[8]: about_page()


if __name__ == "__main__":
    main()
