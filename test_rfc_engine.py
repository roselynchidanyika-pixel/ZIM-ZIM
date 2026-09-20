"""Tests for the RFC Securities financial-modelling engines.

These verify that the calculations are mathematically correct and that the
five models interconnect properly through the central engine.
"""
from __future__ import annotations

import numpy as np
import pytest

from rfc.data.samples import sample_company, sample_projects, financials_csv_bytes
from rfc.data.loader import parse_upload
from rfc.engine import run_full_analysis, _npv_of
from rfc.models import investment as inv
from rfc.models import optimization as opt
from rfc.models import risk as rsk
from rfc.models.investment import ProjectSpec, evaluate_project
from rfc.models.optimization import select_projects, optimise_price

CF1000 = np.array([-1000.0, 500.0, 500.0, 500.0])
CF1100 = np.array([-1000.0, 1100.0])


def approx(a, b, tol=1e-6):
    return abs(a - b) <= tol * max(1.0, abs(b))


# ── pure metric functions ────────────────────────────────────────────────
def test_npv_known_value():
    assert approx(inv.npv(CF1000, 0.10), 243.4259, 1e-2)


def test_irr_single_period():
    assert approx(inv.irr(CF1100), 0.10, 1e-6)


def test_mirr_single_period():
    # single inflow at maturity: MIRR = IRR (no reinvestment effect)
    assert approx(inv.mirr(CF1100, 0.10, 0.10), 0.10, 1e-9)


def test_mirr_reinvests_interim_cash():
    cf = np.array([-1000.0, 500.0, 500.0, 500.0])
    # FV = 500*1.1^2 + 500*1.1 + 500 = 1655; MIRR = (1655/1000)^(1/3) - 1
    expect = (1655.0 / 1000.0) ** (1.0 / 3.0) - 1.0
    assert approx(inv.mirr(cf, 0.10, 0.10), expect, 1e-9)


def test_payback():
    assert approx(inv.payback(CF1000), 2.0, 1e-9)


def test_discounted_payback_button_edge():
    # never recovers discounted
    assert inv.discounted_payback(np.array([-100.0, 10.0, 10.0]), 0.20) is None


def test_pi():
    pv_in = inv.pv(np.array([0.0, 500.0, 500.0, 500.0]), 0.10)
    assert approx(inv.pi(CF1000, 0.10), pv_in / 1000.0, 1e-9)


def test_eaa():
    r = 0.10
    npv_ = inv.npv(CF1000, r)
    af = (1 - (1 + r) ** -3) / r
    assert approx(inv.eaa(CF1000, r), npv_ / af, 1e-9)


def test_break_even_performance():
    pv_in = inv.pv(np.array([0.0, 500.0, 500.0, 500.0]), 0.10)
    assert approx(inv.break_even_performance(CF1000, 0.10), 1000.0 / pv_in, 1e-9)


def test_arr():
    assert approx(inv.arr(CF1100), 1100.0 / 500.0, 1e-9)


# ── full-chain connectivity ──────────────────────────────────────────────
def test_full_chain_links_income_statement():
    state = run_full_analysis(sample_company(), projects=sample_projects())
    p = state.profitability
    assert np.allclose(p.gross_profit, p.revenue - p.cogs)
    assert np.allclose(p.ebitda, p.gross_profit - p.opex)
    assert np.allclose(p.ebit, p.ebitda - p.d_and_a)
    assert np.allclose(p.ebt, p.ebit - p.interest)
    assert np.allclose(p.income_tax, np.maximum(p.ebt, 0) * 0.25)
    assert np.allclose(p.net_income, p.ebt - p.income_tax)


def test_full_chain_links_cashflow():
    state = run_full_analysis(sample_company(), projects=sample_projects())
    cf = state.cashflow
    assert np.allclose(cf.operating_cash_flow,
                       cf.net_income + cf.d_and_a - cf.change_nwc)
    assert np.allclose(cf.fcf, cf.operating_cash_flow + cf.investing_cash_flow)
    # cash balance arithmetic
    expect = np.zeros_like(cf.cash_balance)
    expect[0] = state.company.initial_cash + cf.net_change[0]
    for t in range(1, len(expect)):
        expect[t] = expect[t - 1] + cf.net_change[t]
    assert np.allclose(expect, cf.cash_balance)
    assert np.allclose(cf.net_change, cf.fcf + cf.financing_cash_flow)


def test_business_case_uses_investment_buffer():
    state = run_full_analysis(sample_company(), projects=sample_projects())
    bc = state.business_case
    capex0 = state.company.capex[0]
    nwc0 = state.cashflow.initial_nwc
    assert approx(bc.initial_investment, capex0 + nwc0, 1e-6)
    assert bc.npv_ == inv.npv(bc.full_timeline, state.company.wacc)


def test_sample_solution_sensible():
    state = run_full_analysis(sample_company(), projects=sample_projects())
    bc = state.business_case
    assert bc.irr_ is not None and bc.irr_ > 0
    assert bc.pi_ is not None and bc.pi_ > 0
    assert state.profitability.breakeven_units is not None
    assert state.profitability.breakeven_units < state.profitability.units[0]


# ── risk model ───────────────────────────────────────────────────────────
def test_pessimistic_worse_than_base():
    state = run_full_analysis(sample_company(), projects=sample_projects())
    risk = state.risk
    base = next(s["npv"] for s in risk.scenarios if s["name"] == "Base")
    pess = next(s["npv"] for s in risk.scenarios if s["name"] == "Pessimistic")
    opt_ = next(s["npv"] for s in risk.scenarios if s["name"] == "Optimistic")
    assert pess < base < opt_
    assert len(risk.stress) == len(rsk.STRESS_PRESETS)


def test_monte_carlo_distribution_stats():
    state = run_full_analysis(sample_company(), projects=sample_projects(),
                              risk_config={"mc_iterations": 200, "seed": 1})
    assert state.risk.mc_p5 <= state.risk.mc_p50 <= state.risk.mc_p95
    assert len(state.risk.mc_samples) == 200


def test_custom_scenario_appears():
    custom = {"name": "Test shock", "rev_mult": 1.2, "var_mult": 1.0,
              "fixed_mult": 1.0, "wacc_pp": 0.0, "delay": 0.0}
    state = run_full_analysis(sample_company(), scenario_custom=custom)
    names = [s["name"] for s in state.risk.scenarios]
    assert "Test shock" in names


# ── optimization ─────────────────────────────────────────────────────────
def test_optimine_selection_brute_force():
    projects = [
        ProjectSpec("P1", 100, np.array([0, 90, 90])),
        ProjectSpec("P2", 100, np.array([0, 60, 60])),
        ProjectSpec("P3", 200, np.array([0, 140, 140])),
    ]
    budget = 300
    metrics = [evaluate_project(p, 0.10) for p in projects]
    res = select_projects(metrics, budget)
    # brute force best over 2^3 combos
    best_val, best_set = -1e18, None
    bits = len(projects)
    for mask in range(1 << bits):
        cost = sum(m.full_timeline[0] for i, m in enumerate(metrics) if mask & (1 << i))
        if cost <= budget + 1e-9:
            v = sum(m.npv_ for i, m in enumerate(metrics) if mask & (1 << i))
            if v > best_val:
                best_val, best_set = v, mask
    assert approx(res.total_npv, best_val, 1e-6)
    selected_names = [m.name for i, m in enumerate(metrics) if best_set & (1 << i)]
    assert sorted(res.selected) == sorted(selected_names)


def test_pricing_closed_form():
    res = optimise_price(price=45.0, vc=24.0, volume=400_000, elasticity=-2.5)
    expect = 24.0 * -2.5 / (-2.5 + 1.0)          # 40.0
    assert approx(res.optimal_price, expect, 1e-9)
    assert res.uplift_pct > 0


def test_allocate_capital_respects_budget():
    res = opt.allocate_capital(["A", "B"], [20.0, 5.0], 1000.0, [500.0, 1000.0])
    assert approx(res.used, 1000.0, 1e-6)
    assert approx(res.allocation[0], 500.0, 1e-9)  # high-return unit capped first
    assert res.allocation[0] >= res.allocation[1]


# ── data layer ───────────────────────────────────────────────────────────
def test_parse_financials_csv():
    raw = financials_csv_bytes()
    res = parse_upload(raw, "financials.csv")
    assert res["ok"]
    assert res["kind"] == "financials"
    assert res["ci"].periods == 8
    assert np.isclose(res["ci"].revenue[0], 400_000 * 45.0)
    assert _npv_of(res["ci"]) != 0.0 or res["ci"].opex[0] < 1e9


def test_parse_projects_csv():
    from rfc.data.samples import projects_csv_bytes
    raw = projects_csv_bytes()
    res = parse_upload(raw, "projects.csv")
    assert res["ok"]
    assert res["kind"] == "projects"
    assert len(res["projects"]) == 6


# ── auth ─────────────────────────────────────────────────────────────────
def test_auth_roundtrip():
    from rfc.data.auth import verify, register
    res = verify("rfc", "rfc2024")
    assert res["ok"] and res["role"] == "Admin"
    assert not verify("rfc", "wrong").get("ok")
    assert verify("", "").get("ok") is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])