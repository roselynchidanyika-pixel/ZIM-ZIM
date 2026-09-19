"""Tests for the CAPEXX global currency / FX engine."""
import math

from capexx_engine import (ProjectInput, run_analysis, currency_exposures,
                           aggregate_fx_risk, currency_strategy_lines,
                           build_exchange_rate_board, compute_landed_cost,
                           fx_convert)


def _multi_currency_input() -> ProjectInput:
    p = ProjectInput(
        project_name="Multi-Currency Hospital", organisation="HealthCo",
        country="Zambia", project_type="Hospital / Healthcare",
        reporting_currency="USD",
        construction_cost=9_000_000, construction_currency="ZMW",
        construction_fx_pa=6.0, construction_supplier_country="Zambia",
        equipment_cost=6_000_000, equipment_currency="EUR",
        equipment_fx_pa=1.0, equipment_supplier_country="Germany",
        land_building_cost=800_000, land_currency="USD",
        working_capital=500_000, salvage_value=1_500_000,
        annual_revenue=3_000_000, revenue_currency="USD",
        annual_opex=1_000_000, opex_currency="ZMW", opex_fx_pa=6.0,
        annual_maintenance=300_000, maintenance_currency="ZAR",
        maintenance_fx_pa=3.0,
        materials_supplier_country="South Africa",
        materials_currency="ZAR",
        project_life=15, construction_period=2, discount_rate=12.0,
        tax_rate=25.0,
        fx_rates={"USD": 1.0, "EUR": 0.9, "ZMW": 26.0, "ZAR": 18.0},
    )
    return p


def _same_currency_input() -> ProjectInput:
    p = ProjectInput(
        project_name="Single-Currency Depo", organisation="LogCo",
        country="United States", project_type="Transport / Logistics",
        reporting_currency="USD",
        construction_cost=5_000_000, construction_currency="USD",
        equipment_cost=3_000_000, equipment_currency="USD",
        land_building_cost=1_000_000, land_currency="USD",
        annual_revenue=2_500_000, annual_opex=900_000,
        project_life=10, construction_period=1, discount_rate=10.0,
        tax_rate=21.0,
    )
    return p


def test_exposure_detection():
    p = _multi_currency_input()
    ex = currency_exposures(p)
    mismatched = [e for e in ex if e["mismatch"]]
    assert len(mismatched) >= 3
    ccy_set = {e["currency"] for e in ex}
    assert {"EUR", "ZMW", "ZAR"} <= ccy_set


def test_single_currency_no_mismatch():
    p = _same_currency_input()
    fx = aggregate_fx_risk(p)
    assert fx["mismatch_count"] == 0
    assert fx["fx_risk_level"] in ("LOW", "MODERATE", "HIGH")


def test_currency_strategy_lines_generated():
    p = _multi_currency_input()
    lines = currency_strategy_lines(p)
    assert len(lines) > 0
    for line in lines:
        assert line["invoice_currency"]
        assert line["analysis"]
        assert line["recommendation"]


def test_exchange_rate_board():
    board = build_exchange_rate_board(
        {"USD": 1.0, "EUR": 0.9, "ZAR": 18.0}, "TEST", "now", "USD", live=False)
    assert any(b["pair"] == "USD/EUR" for b in board)
    assert all(b["status"] == "DEMONSTRATION / USER-PROVIDED" for b in board)


def test_landed_cost_build_up():
    r = compute_landed_cost(
        unit_cost=100_000, qty=2,
        transport_pct=10, insurance_pct=2, duty_pct=15, taxes_pct=15,
        conversion_pct=2, financing_pct=2, fx_effect_pct=5)
    purchase = 200_000
    assert r["purchase"] == purchase
    assert r["landed_total"] > purchase
    # every component is present and positive
    for k in ("transport", "insurance", "duty", "taxes", "conversion",
              "financing_fx"):
        assert r[k] > 0


def test_run_analysis_multi_currency_works():
    b = run_analysis(_multi_currency_input())
    assert b.metrics.npv is not None
    assert b.fx["mismatch_count"] >= 3
    assert len(b.fx_strategy) > 0