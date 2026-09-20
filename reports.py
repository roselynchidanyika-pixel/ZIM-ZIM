"""RFC Securities — downloadable reports (HTML), plus CSV and JSON exports."""
from __future__ import annotations

import html
import io
import json
from datetime import datetime, timezone

import numpy as np

from .ai.analyst import build_narrative
from .config import fmt_currency, fmt_num, GOLD, DEEP_GREEN, GREEN, WHITE
from .engine import AnalysisState


def _esc(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:,.2f}"
    return html.escape(str(v))


def build_html_report(state: AnalysisState, username: str = "") -> str:
    """Build a complete, polished HTML report from the analysis state."""
    p = state.profitability
    cf = state.cashflow
    bc = state.business_case
    risk = state.risk
    opt = state.optimization
    ccy = state.company.currency
    narr = build_narrative(state)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    def metric(label, value, sub=""):
        return (f'<div class="metric"><div class="m-label">{label}</div>'
                f'<div class="m-value">{value}</div>'
                f'<div class="m-sub">{sub}</div></div>')

    rows_profit = zip(p.years, p.revenue, p.gross_profit, p.ebitda, p.ebit,
                      p.net_income)
    profit_tr = "".join(
        f"<tr><td>{y}</td><td>{fmt_currency(r, ccy)}</td>"
        f"<td>{fmt_currency(g, ccy)}</td><td>{fmt_currency(e, ccy)}</td>"
        f"<td>{fmt_currency(b, ccy)}</td><td>{fmt_currency(n, ccy)}</td></tr>"
        for y, r, g, e, b, n in rows_profit)

    rows_cf = zip(p.years, cf.net_income, cf.change_nwc, cf.operating_cash_flow,
                  cf.capex, cf.fcf, cf.cash_balance)
    cf_tr = "".join(
        f"<tr><td>{y}</td><td>{fmt_currency(ni, ccy)}</td>"
        f"<td>{fmt_currency(d, ccy)}</td><td>{fmt_currency(o, ccy)}</td>"
        f"<td>{fmt_currency(c, ccy)}</td><td>{fmt_currency(f, ccy)}</td>"
        f"<td>{fmt_currency(cb, ccy)}</td></tr>"
        for y, ni, d, o, c, f, cb in rows_cf)

    project_rows = ""
    if state.project_metrics:
        trs = []
        for m in state.project_metrics:
            trs.append(
                f"<tr><td>{html.escape(m.name)}</td>"
                f"<td>{fmt_currency(m.npv_, ccy)}</td>"
                f"<td>{_esc(m.irr_ * 100 if m.irr_ is not None else None)}%</td>"
                f"<td>{_esc(m.mirr_ * 100 if m.mirr_ is not None else None)}%</td>"
                f"<td>{_esc(m.pi_)}</td>"
                f"<td>{_esc(m.payback_)} y</td>"
                f"<td>{_esc(m.eaa_ and fmt_currency(m.eaa_, ccy))}</td></tr>")
        project_rows = "".join(trs)

    scenario_rows = "".join(
        f"<tr><td>{html.escape(s['name'])}</td>"
        f"<td>{html.escape(s['description'])}</td>"
        f"<td style='color:{'#157347' if s['npv'] >= 0 else '#C0392B'};"
        f"font-weight:700'>{fmt_currency(s['npv'], ccy)}</td></tr>"
        for s in risk.scenarios)

    stress_rows = "".join(
        f"<tr><td>{html.escape(s['name'])}</td>"
        f"<td>{fmt_currency(s['npv'], ccy)}</td>"
        f"<td>{s['impact_pct']:+.1f}%</td></tr>" for s in risk.stress)

    opt_html = ""
    if opt and opt.selection:
        sel = f"""<div class="slice">
<h3>Project selection optimisation</h3>
<p>Budget: <b>{fmt_currency(opt.selection.budget, ccy)}</b> · Used:
<b>{fmt_currency(opt.selection.used_budget, ccy)}</b> · Selected:
<b>{', '.join(opt.selection.selected) or 'none'}</b></p>
<p>Optimised NPV: <b>{fmt_currency(opt.selection.total_npv, ccy)}</b>
vs {fmt_currency(opt.selection.base_npv, ccy)} if all projects were taken.</p>
</div>"""
        opt_html += sel
    if opt and opt.pricing:
        pr = opt.pricing
        opt_html += f"""<div class="slice">
<h3>Pricing optimisation</h3>
<p>Recommended price: <b>{fmt_currency(pr.optimal_price, ccy, 2)}</b>
({pr.price_change_pct:+.1f}% vs current {fmt_currency(pr.current_price, ccy, 2)})
at elasticity {pr.elasticity:.1f}.</p>
<p>Contribution uplift:
<b>{fmt_currency(pr.uplift_contribution, ccy)}</b> ({pr.uplift_pct:+.1f}%).</p>
</div>"""

    be_line = ""
    if p.breakeven_revenue is not None:
        be_line = (f"Break-even revenue {fmt_currency(p.breakeven_revenue, ccy)}"
                   f" · margin of safety {p.margin_of_safety_pct:.0f}%"
                   if p.margin_of_safety_pct is not None else
                   f"Break-even revenue {fmt_currency(p.breakeven_revenue, ccy)}")

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>RFC Securities — Financial Modelling Report</title>
<style>
 body{{font-family:Segoe UI,Calibri,Arial,sans-serif;color:#0F172A;background:#fff;margin:0;}}
 .hero{{background:linear-gradient(135deg,#0B3D2E,#157347);color:#fff;padding:1.6rem 2rem;}}
 .hero h1{{margin:0;color:#F3E5B8;letter-spacing:.04em;}}
 .hero .tag{{color:#F3E5B8;letter-spacing:.22em;font-size:.8rem;}}
 .hero .meta{{margin-top:.4rem;font-size:.8rem;color:#B8D8C8;}}
 .wrap{{padding:1.4rem 2rem;}}
 h2{{color:#0B3D2E;border-bottom:3px solid #C9A227;padding-bottom:.3rem;margin-top:1.6rem;}}
 h3{{color:#157347;}}
 .kpis{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:.8rem;margin-top:1rem;}}
 .metric{{background:#F6F8F4;border-left:5px solid #C9A227;border-radius:8px;padding:.7rem .9rem;}}
 .m-label{{font-size:.72rem;color:#5B7266;text-transform:uppercase;letter-spacing:.05em;}}
 .m-value{{font-size:1.25rem;font-weight:700;color:#0B3D2E;}}
 .m-sub{{font-size:.7rem;color:#5B7266;}}
 table{{border-collapse:collapse;width:100%;margin:.6rem 0;font-size:.82rem;}}
 th{{background:#0B3D2E;color:#F3E5B8;text-align:left;padding:.45rem .6rem;}}
 td{{padding:.4rem .6rem;border-bottom:1px solid #E3EAE5;}}
 tr:nth-child(even) td{{background:#F6F8F4;}}
 .slice{{background:#fff;border:1px solid #E3EAE5;border-radius:10px;padding:1rem;margin:.6rem 0;}}
 .narr{{background:#F6F8F4;border-left:4px solid #C9A227;padding:.8rem 1rem;border-radius:6px;line-height:1.55;font-size:.88rem;}}
 .foot{{margin-top:2rem;padding-top:.8rem;border-top:2px solid #E3EAE5;font-size:.72rem;color:#5B7266;text-align:center;}}
 ul{{margin:.3rem 0;padding-left:1.1rem;}}
</style></head><body>
<div class="hero">
  <div class="tag">RFC SECURITIES</div>
  <h1>TURN UNCERTAINTY INTO PROFITABILITY</h1>
  <div class="meta">MODEL. PREDICT. OPTIMIZE. — Financial Modelling Report ·
  {html.escape(state.company.company_name)} · generated {now} · {html.escape(username) or "RFC Securities User"}</div>
</div>
<div class="wrap">
  <div class="kpis">
    {metric("NPV (business case)", fmt_currency(bc.npv_, ccy), f"WACC {bc.wacc * 100:.1f}%")}
    {metric("IRR", f"{bc.irr_ * 100:.1f}%" if bc.irr_ is not None else "n/a", f"MIRR {bc.mirr_ * 100 if bc.mirr_ else 0:.1f}%")}
    {metric("Payback", f"{bc.payback_:.1f} y", f"Disc. {bc.discounted_payback_:.1f} y")}
    {metric("Total FCF", fmt_currency(cf.total_fcf, ccy), f"{state.company.periods}-year plan")}
    {metric("Net margin (Y1)", f"{p.net_margin:.1f}%", be_line)}
    {metric("P5 NPV (MC)", fmt_currency(risk.mc_p5, ccy) if risk.mc_p5 is not None else "n/a",
            f"P(neg) {risk.mc_prob_negative:.0f}%" if risk.mc_prob_negative is not None else "")}
  </div>

  <h2>1 · AI Analyst Summary</h2>
  <div class="narr">{html.escape(narr['overview'])}</div>

  <h2>2 · Profitability Model</h2>
  <div class="narr">{html.escape(narr['profitability'])}</div>
  <div class="slice">
  <table><tr><th>Year</th><th>Revenue</th><th>Gross profit</th><th>EBITDA</th>
  <th>EBIT</th><th>Net income</th></tr>{profit_tr}</table></div>

  <h2>3 · Cash-Flow Forecasting Model</h2>
  <div class="narr">{html.escape(narr['cashflow'])}</div>
  <div class="slice">
  <table><tr><th>Year</th><th>Net income</th><th>Δ WC</th><th>Operating CF</th>
  <th>Capex</th><th>Free CF</th><th>Cash balance</th></tr>{cf_tr}</table></div>

  <h2>4 · Investment & Capital Budgeting</h2>
  <div class="narr">{html.escape(narr['investment'])}</div>
  <div class="slice">
  <h3>Project metrics</h3>
  <table><tr><th>Project</th><th>NPV</th><th>IRR %</th><th>MIRR %</th><th>PI</th>
  <th>Payback y</th><th>EAA</th></tr>{project_rows or '<tr><td colspan=7>No project catalogue loaded.</td></tr>'}</table></div>

  <h2>5 · Risk & Stress Testing</h2>
  <div class="narr">{html.escape(narr['risk'])}</div>
  <div class="slice">
  <h3>Scenario analysis</h3>
  <table><tr><th>Scenario</th><th>Description</th><th>NPV</th></tr>{scenario_rows}</table>
  <h3>Stress tests (NPV impact vs base)</h3>
  <table><tr><th>Stress</th><th>NPV</th><th>Impact %</th></tr>{stress_rows}</table></div>

  {('<h2>6 · Optimization</h2><div>' + opt_html + '</div>') if opt_html else ''}

  <h2>7 · Model Inputs & Assumptions</h2>
  <div class="slice">
  <ul>
    <li>Currency: <b>{html.escape(state.company.currency)}</b></li>
    <li>Periods modelled: <b>{state.company.periods} years</b></li>
    <li>Tax rate: <b>{state.company.tax_rate * 100:.0f}%</b></li>
    <li>WACC discount rate: <b>{state.company.wacc * 100:.1f}%</b></li>
    <li>Inflation: <b>{state.company.inflation * 100:.1f}%</b></li>
    <li>Working capital days (AR/AP/INV): <b>{state.company.receivable_days:.0f} /
        {state.company.payable_days:.0f} / {state.company.inventory_days:.0f}</b></li>
    <li>Debt ratio / interest: <b>{state.company.debt_ratio * 100:.0f}% /
        {state.company.debt_interest_rate * 100:.1f}%</b></li>
    <li>Monte-Carlo paths: <b>{risk.mc_iterations}</b></li>
  </ul></div>

  <div class="foot">RFC Securities · TURN UNCERTAINTY INTO PROFITABILITY ·
  MODEL. PREDICT. OPTIMIZE.<br>Generated by the RFC Securities AI Financial
  Modelling Platform. Decision-support only — not financial advice.</div>
</div></body></html>"""


def build_json_export(state: AnalysisState) -> str:
    """JSON export of every headline number."""
    p, cf, bc, risk = state.profitability, state.cashflow, state.business_case,\
        state.risk
    payload = {
        "platform": "RFC Securities",
        "company": state.company.company_name,
        "currency": state.company.currency,
        "periods": state.company.periods,
        "profitability": {
            "gross_margin_pct": p.gross_margin,
            "operating_margin_pct": p.operating_margin,
            "net_margin_pct": p.net_margin,
            "break_even_revenue": (float(p.breakeven_revenue)
                                   if p.breakeven_revenue is not None else None),
            "fixed_cost": p.fixed_cost,
        },
        "cashflow": {
            "total_fcf": cf.total_fcf,
            "closing_cash": float(cf.cash_balance[-1]),
            "current_ratio_y1": float(cf.current_ratio[0]),
        },
        "investment": {
            "npv": bc.npv_,
            "irr": bc.irr_,
            "mirr": bc.mirr_,
            "pi": bc.pi_,
            "payback": bc.payback_,
            "discounted_payback": bc.discounted_payback_,
            "eaa": bc.eaa_,
            "break_even_performance": bc.break_even_performance_,
            "initial_investment": bc.initial_investment,
            "wacc": bc.wacc,
        },
        "risk": {
            "scenarios": [{"name": s["name"], "npv": s["npv"]}
                          for s in risk.scenarios],
            "stress": [{"name": s["name"], "npv": s["npv"],
                        "impact_pct": s["impact_pct"]} for s in risk.stress],
            "mc_p5": risk.mc_p5,
            "mc_p50": risk.mc_p50,
            "mc_p95": risk.mc_p95,
            "mc_prob_negative_pct": risk.mc_prob_negative,
            "mc_iterations": risk.mc_iterations,
        },
        "optimization": (
            {"selection": {"budget": state.optimization.selection.budget,
                           "used_budget": state.optimization.selection.used_budget,
                           "selected": state.optimization.selection.selected,
                           "total_npv": state.optimization.selection.total_npv}}
            if state.optimization and state.optimization.selection else {}),
    }
    return json.dumps(payload, indent=2, default=str)


def csv_export_for(state: AnalysisState, kind: str) -> bytes:
    """CSV export of a model table as bytes (utf-8)."""
    import pandas as pd
    p, cf = state.profitability, state.cashflow
    if kind == "income":
        buf = io.StringIO()
        p.statement_df().to_csv(buf, index=False)
        return buf.getvalue().encode("utf-8")
    if kind == "cashflow":
        buf = io.StringIO()
        cf.statement_df().to_csv(buf, index=False)
        return buf.getvalue().encode("utf-8")
    if kind == "projects":
        buf = io.StringIO()
        rows = [m.as_dict() for m in state.project_metrics]
        pd.DataFrame(rows).to_csv(buf, index=False)
        return buf.getvalue().encode("utf-8")
    return b""