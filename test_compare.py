"""Tests for the CAPEXX Compare & Select features:
funding priority scorecard, knapsack budget selection, sample portfolio."""
from capexx_engine import (run_analysis, funding_priority, knapsack_select,
                           sample_portfolio_projects, vision_pillar,
                           demo_project, ProjectInput)


def test_funding_priority_components_sum_to_score():
    b = run_analysis(demo_project())
    f = funding_priority(b, vision_score=7.0)
    comp = f["components"]
    assert abs(sum(comp.values()) - f["score"]) <= 0.5
    assert 0 <= f["score"] <= 100
    assert {"value", "risk", "resilience", "vision"} <= set(comp.keys())
    for k, v in comp.items():
        assert 0 <= v <= {"value": 40, "risk": 30,
                          "resilience": 15, "vision": 15}[k]
    assert len(f["flags"]) >= 1
    assert f["grade"] in ("HIGH PRIORITY", "MODERATE PRIORITY", "LOW PRIORITY")


def test_funding_priority_demo_beats_obviously_weak():
    strong = funding_priority(run_analysis(demo_project()))
    weak = ProjectInput(
        project_name="Graveyard Mall", organisation="X",
        country="Zimbabwe", project_type="Real Estate",
        reporting_currency="USD",
        construction_cost=90_000_000, equipment_cost=60_000_000,
        land_building_cost=20_000_000,
        annual_revenue=40_000_000, annual_opex=55_000_000,
        annual_maintenance=10_000_000,
        project_life=10, construction_period=3, discount_rate=18.0,
        tax_rate=30.0, inflation_rate=15.0, debt_ratio=70.0,
        loan_interest_rate=16.0, market_risk=2, construction_risk=2,
        country_risk=2, currency_volatility=2, supplier_risk=2,
    )
    fweak = funding_priority(run_analysis(weak))
    assert strong["score"] > fweak["score"]
    assert fweak["npv"] < 0 or fweak["decision"] in ("REVIEW", "REJECT")


def test_funding_priority_vision_boundaries():
    b = run_analysis(demo_project())
    lo = funding_priority(b, vision_score=0.0)["components"]["vision"]
    hi = funding_priority(b, vision_score=10.0)["components"]["vision"]
    assert lo == 0.0 and hi == 15.0


def test_knapsack_chooses_optimal_subset():
    res = knapsack_select(costs=[10, 20, 30], values=[60, 100, 120],
                          budget=50)
    assert sorted(res["selected"]) == [1, 2]      # 20+30=50, value 220
    assert res["total_cost"] == 50
    assert abs(res["total_value"] - 220) < 1e-6
    assert res["unspent"] == 0


def test_knapsack_respects_budget():
    res = knapsack_select(costs=[100, 200, 300], values=[50, 80, 60],
                          budget=250)
    assert res["total_cost"] <= 250
    assert res["selected"]  # at least one item fits


def test_knapsack_empty_and_zero_budget():
    assert knapsack_select([], [], 100)["selected"] == []
    z = knapsack_select([10, 20], [5, 9], 0)
    assert z["selected"] == [] and z["total_cost"] == 0


def test_sample_portfolio_all_analyse_cleanly():
    projects = sample_portfolio_projects()
    assert len(projects) == 5
    countries = {p.country for p in projects}
    assert countries == {"Zimbabwe", "Botswana", "Uganda", "Namibia", "Germany"}
    for p in projects:
        b = run_analysis(p)
        assert b.metrics.npv is not None
        assert b.decision.status in ("ACCEPT", "REVIEW", "REJECT")
        assert not p.validation_errors()


def test_knapsack_handles_billion_scale_budgets():
    # UGX-billion-scale investments must not blow up the DP grid.
    res = knapsack_select(costs=[45_000_000_000, 17_000_000, 120_000_000],
                          values=[60.0, 85.0, 50.0],
                          budget=45_000_000_000 + 10_000_000)
    assert res["total_cost"] <= 45_010_000_000
    assert len(res["selected"]) >= 2
    assert res["unspent"] >= 0


def test_vision_pillar_mapping():
    assert "Infrastructure & Utilities" in vision_pillar("Energy / Power")
    assert "Human Capital Development" in vision_pillar("University / Education")
    assert vision_pillar("Real Estate") != []