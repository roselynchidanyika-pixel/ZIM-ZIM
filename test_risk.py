"""Tests for the CAPEXX risk engine."""
from capexx_engine import ProjectInput, run_analysis, run_risk_engine


def _risky_input() -> ProjectInput:
    p = ProjectInput(
        project_name="Risky Tunnel", organisation="GovCo", country="South Africa",
        project_type="Infrastructure",
        construction_cost=20_000_000, equipment_cost=5_000_000,
        land_building_cost=1_000_000, working_capital=1_000_000,
        salvage_value=2_000_000,
        annual_revenue=6_000_000, revenue_growth=2.0,
        annual_opex=2_000_000, annual_maintenance=300_000,
        project_life=20, construction_period=4, expected_delay_years=2.0,
        tax_rate=30.0, discount_rate=13.0, debt_ratio=60.0,
        loan_interest_rate=12.0, loan_term=10,
        construction_risk=2, market_risk=2, country_risk=2,
        currency_volatility=2, supplier_risk=2, operating_risk=2,
        inflation_rate=12.0, reporting_currency="USD",
    )
    return p


def _safe_input() -> ProjectInput:
    p = ProjectInput(
        project_name="Safe Hub", organisation="SmallCo", country="Germany",
        project_type="Technology / Data Centre",
        construction_cost=3_000_000, equipment_cost=2_000_000,
        land_building_cost=1_000_000, working_capital=200_000,
        salvage_value=300_000,
        annual_revenue=1_500_000, revenue_growth=2.0,
        annual_opex=400_000, annual_maintenance=50_000,
        project_life=8, construction_period=1, expected_delay_years=0.0,
        tax_rate=25.0, discount_rate=8.0, debt_ratio=20.0,
        loan_interest_rate=5.0, loan_term=4, inflation_rate=2.0,
        reporting_currency="EUR",
    )
    return p


def test_risk_levels_differentiate():
    r_risky = run_analysis(_risky_input()).risk
    r_safe = run_analysis(_safe_input()).risk
    assert r_risky.score > r_safe.score
    assert 0 <= r_risky.score <= 100
    assert r_risky.level in ("LOW", "MODERATE", "HIGH")


def test_risk_detail_present():
    b = run_analysis(_safe_input())
    assert len(b.risk.detail) >= 5
    assert any("risk" in d.lower() for d in b.risk.detail)


def test_all_category_scores_in_range():
    b = run_analysis(_safe_input())
    r = b.risk
    for val in (r.cost_risk, r.delay_risk, r.revenue_risk, r.opex_risk,
                r.inflation_risk, r.fx_risk, r.interest_rate_risk,
                r.cashflow_risk, r.completion_risk, r.supplier_risk):
        assert 0 <= val <= 100