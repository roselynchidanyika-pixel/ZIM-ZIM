"""Tests for the CAPEXX central financial / capital-budgeting engine."""
import math

import pytest
import numpy as np
import numpy_financial as npf

from capexx_engine import (ProjectInput, build_cash_flow_model, compute_metrics,
                           npv_of, run_analysis, demo_project, fmt_ccy, fmt_pct,
                           fx_convert)


def _base_input() -> ProjectInput:
    p = ProjectInput(
        project_name="Test Plant", organisation="TestCo", country="Kenya",
        project_type="Manufacturing Plant",
        construction_cost=8_000_000, equipment_cost=4_000_000,
        land_building_cost=1_000_000, working_capital=500_000,
        salvage_value=1_000_000,
        annual_revenue=5_000_000, revenue_growth=3.0,
        annual_opex=1_500_000, annual_maintenance=200_000,
        project_life=12, construction_period=2, tax_rate=25.0,
        discount_rate=11.0, debt_ratio=40.0, loan_interest_rate=8.0,
        loan_term=6, reporting_currency="USD",
    )
    return p


def test_model_shapes():
    p = _base_input()
    m = build_cash_flow_model(p)
    assert len(m.years) == 2 + 12
    assert sum(1 for l in m.labels if l == "Operation") == 12
    assert sum(m.is_construction) == 2
    # capex occurs only in construction years
    for i, c in enumerate(m.is_construction):
        if not c:
            assert m.capex[i] == 0.0
        else:
            assert m.capex[i] > 0.0


def test_npv_definition_matches_manual():
    p = _base_input()
    m = build_cash_flow_model(p)
    npv_manual = sum(f / (1 + 0.11) ** i for i, f in enumerate(m.free_cash_flow))
    assert math.isclose(npv_of(m, 11.0), npv_manual, rel_tol=1e-9)


def test_metrics_returned():
    p = _base_input()
    b = run_analysis(p)
    m = b.metrics
    assert m.npv is not None
    assert m.irr is not None and 0 < m.irr < 100
    assert m.mirr is not None and 0 < m.mirr < 100
    assert m.pi is not None and m.pi > 0
    assert m.payback is not None and m.payback > 0
    assert m.eaa is not None
    assert b.decision.status in ("ACCEPT", "REVIEW", "REJECT")


def test_irr_consistency_with_npf():
    p = _base_input()
    m = build_cash_flow_model(p)
    expected = npf.irr(np.array(m.free_cash_flow, dtype=float)) * 100.0
    met = compute_metrics(m, p)
    assert math.isclose(met.irr, expected, rel_tol=1e-6)


def test_percent_formatting():
    assert fmt_pct(12.34) == "12.34%"
    assert fmt_pct(None) == "n/a"
    assert fmt_ccy(1234.5, "USD").startswith("USD")


def test_fx_convert_consistency():
    rates = {"USD": 1.0, "EUR": 0.9, "ZAR": 18.0}
    assert math.isclose(fx_convert(100, "EUR", "USD", rates), 111.111, rel_tol=1e-3)
    assert math.isclose(fx_convert(18, "ZAR", "USD", rates), 1.0, rel_tol=1e-9)
    assert fx_convert(50, "USD", "USD", rates) == 50.0


def test_validation_catches_missing_fields():
    p = ProjectInput(project_name="X", organisation="Y", country="Z")
    errs = p.validation_errors()
    joined = " ".join(errs).lower()
    assert "capital expenditure" in joined
    assert "revenue" in joined