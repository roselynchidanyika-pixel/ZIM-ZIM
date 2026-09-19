"""Tests for the CAPEXX scenario / stress / sensitivity engine."""
from capexx_engine import (ProjectInput, run_scenarios, run_stress_tests,
                           run_sensitivity, demo_project)


def _input() -> ProjectInput:
    p = ProjectInput(
        project_name="Senerio Test", organisation="EnCo", country="Kenya",
        project_type="Energy / Power", reporting_currency="USD",
        construction_cost=10_000_000, equipment_cost=6_000_000,
        land_building_cost=1_000_000, working_capital=600_000,
        salvage_value=1_200_000,
        annual_revenue=5_000_000, revenue_growth=3.0,
        annual_opex=1_400_000, annual_maintenance=150_000,
        project_life=15, construction_period=2, expected_delay_years=0.5,
        tax_rate=25.0, discount_rate=12.0, inflation_rate=5.0,
    )
    return p


def test_scenarios_ordering():
    s = run_scenarios(_input())
    assert set(s.keys()) == {"BASE CASE", "OPTIMISTIC", "PESSIMISTIC",
                             "EXTREME STRESS"}
    base_npv = s["BASE CASE"]["npv"]
    opt_npv = s["OPTIMISTIC"]["npv"]
    pes_npv = s["PESSIMISTIC"]["npv"]
    stress_npv = s["EXTREME STRESS"]["npv"]
    assert opt_npv >= base_npv >= pes_npv
    assert stress_npv <= pes_npv


def test_stress_tests_present():
    st = run_stress_tests(_input())
    assert len(st) >= 10
    for row in st:
        assert row["npv"] is not None
        assert row["name"]
        assert "impact" in row


def test_stress_shock_moves_npv():
    st = run_stress_tests(_input())
    cap30 = next(s for s in st if s["name"] == "CAPEX +30%")
    rev30 = next(s for s in st if s["name"] == "Revenue -30%")
    assert cap30["npv"] < 0 or rev30["npv"] < 0 or abs(cap30["npv"] - rev30["npv"]) > 1


def test_sensitivity_drivers():
    sens = run_sensitivity(_input())
    assert len(sens) >= 4
    drivers = [s["driver"] for s in sens]
    assert "Capex" in drivers and "Revenue" in drivers
    # sorted by absolute impact descending
    deltas = [s["delta"] for s in sens]
    assert deltas == sorted(deltas, reverse=True)


def test_demo_project_runs_e2e():
    b = run_scenarios(demo_project())
    assert "BASE CASE" in b
    stress = run_stress_tests(demo_project())
    assert len(stress) > 0