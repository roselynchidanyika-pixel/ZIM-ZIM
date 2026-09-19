"""Tests for CAPEXX email generation and delivery reporting."""
from capexx_engine import (ProjectInput, run_analysis, build_email,
                           try_send_email)


def _input() -> ProjectInput:
    return ProjectInput(
        project_name="Email Test Project", organisation="TestCo",
        country="Kenya", project_type="Agriculture / Agro-Processing",
        reporting_currency="USD",
        construction_cost=2_000_000, equipment_cost=1_500_000,
        land_building_cost=300_000, working_capital=200_000,
        salvage_value=400_000,
        annual_revenue=1_200_000, revenue_growth=4.0,
        annual_opex=350_000, annual_maintenance=40_000,
        project_life=10, construction_period=1, tax_rate=25.0,
        discount_rate=11.0,
    )


def test_email_generation_shape():
    b = run_analysis(_input())
    email = build_email(b)
    assert email["subject"].startswith("CAPEXX AI Capital Project Analysis")
    assert b.project_input.project_name in email["subject"]
    assert "NPV" in email["body"]
    assert "IRR" in email["body"]
    assert "MIRR" in email["body"]
    assert "Payback" in email["body"]
    assert "monitor" in email["body"].lower()


def test_email_never_claims_sent_without_config():
    b = run_analysis(_input())
    email = build_email(b)
    res = try_send_email(email["subject"], email["body"], "x@example.com",
                         config={})
    assert res["sent"] is False
    assert res["confirmed"] is False
    assert "NOT CONFIGURED" in res["message"]


def test_email_send_error_reported_not_faked():
    b = run_analysis(_input())
    email = build_email(b)
    cfg = {"smtp_host": "smtp.invalid.invalid", "smtp_port": "25",
           "smtp_user": "u", "smtp_password": "p", "smtp_from": "a@b.c"}
    res = try_send_email(email["subject"], email["body"], "x@example.com",
                         config=cfg)
    # Either it failed (reported honestly) or delivered; never falsely 'sent'
    # without a confirmation. In this offline fixture it must not claim success.
    assert res["confirmed"] is False