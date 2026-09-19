"""Tests for the CAPEXX transparent decision engine."""
import math

from capexx_engine import ProjectInput, run_analysis, run_decision_engine


def _strong_project() -> ProjectInput:
    p = ProjectInput(
        project_name="Strong Project", organisation="BigCo", country="Germany",
        project_type="Technology / Data Centre", reporting_currency="EUR",
        construction_cost=4_000_000, equipment_cost=3_000_000,
        land_building_cost=500_000, working_capital=300_000,
        salvage_value=800_000,
        annual_revenue=3_000_000, revenue_growth=5.0,
        annual_opex=800_000, annual_maintenance=80_000,
        project_life=12, construction_period=1, expected_delay_years=0.0,
        tax_rate=25.0, discount_rate=9.0, debt_ratio=30.0,
        loan_interest_rate=6.0, loan_term=5, inflation_rate=2.0,
    )
    return p


def _weak_project() -> ProjectInput:
    p = ProjectInput(
        project_name="Weak Project", organisation="SmallCo", country="Zambia",
        project_type="Mining / Extraction", reporting_currency="USD",
        construction_cost=40_000_000, equipment_cost=25_000_000,
        land_building_cost=4_000_000, working_capital=5_000_000,
        salvage_value=3_000_000,
        annual_revenue=9_000_000, revenue_growth=1.0,
        annual_opex=6_000_000, annual_maintenance=2_000_000,
        project_life=10, construction_period=3, expected_delay_years=2.0,
        tax_rate=35.0, discount_rate=16.0, debt_ratio=70.0,
        loan_interest_rate=14.0, loan_term=6, inflation_rate=14.0,
        market_risk=2, construction_risk=2, country_risk=2,
        currency_volatility=2, supplier_risk=2, operating_risk=2,
    )
    return p


def test_decision_statuses_are_valid():
    for p in (_strong_project(), _weak_project()):
        b = run_analysis(p)
        assert b.decision.status in ("ACCEPT", "REVIEW", "REJECT")


def test_strong_project_accepted():
    b = run_analysis(_strong_project())
    assert b.metrics.npv > 0
    assert b.decision.status == "ACCEPT"
    assert b.decision.score >= 75


def test_weak_project_not_accepted():
    b = run_analysis(_weak_project())
    assert b.decision.status in ("REVIEW", "REJECT")
    assert b.decision.score < 50 or b.decision.status == "REJECT"


def test_decision_transparency_rules_exposed():
    b = run_analysis(_strong_project())
    assert len(b.decision.rules) >= 8
    assert all("rule" in r and "points" in r and "earned" in r
               for r in b.decision.rules)
    reasons = " ".join(b.decision.reasons).lower()
    assert "npv" in reasons or "risk" in reasons or "payback" in reasons


def test_monitoring_points_nonempty():
    from capexx_engine import monitoring_points
    b = run_analysis(_strong_project())
    assert len(monitoring_points(b)) >= 3