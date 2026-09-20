"""Realistic, deterministic sample datasets for RFC Securities.

Two datasources are provided:

1. A full eight-year financial model for a manufacturing company
   (``sample_company`` + ``financials_dataframe``).
2. A six-project capital-budgeting catalogue (``sample_projects``).

Both can be exported to CSV or Excel so users can immediately test every part
of the platform.
"""
from __future__ import annotations

import io

import numpy as np
import pandas as pd

from ..engine import CompanyInput
from ..models.investment import ProjectSpec

SAMPLE_MC_SEED = 2026


def sample_company() -> CompanyInput:
    n = 8
    yrs = np.arange(n)
    units = 400_000 * 1.055 ** yrs
    price = 45.00 * 1.025 ** yrs
    uvc = 24.00 * 1.020 ** yrs
    revenue = units * price
    cogs = units * uvc
    opex = 4_600_000 * 1.03 ** yrs
    d_a = np.full(n, 2_000_000.0)
    capex = np.array([2.4e6, 2.3e6, 2.2e6, 2.1e6, 2.0e6, 2.0e6, 2.0e6, 2.0e6])
    return CompanyInput(
        company_name="Falcon Manufacturing Ltd",
        currency="USD",
        periods=n,
        revenue=revenue,
        cogs=cogs,
        opex=opex,
        d_and_a=d_a,
        capex=capex,
        units=units,
        price=price,
        unit_var_cost=uvc,
        tax_rate=0.25,
        wacc=0.12,
        inflation=0.03,
        receivable_days=45.0,
        payable_days=35.0,
        inventory_days=30.0,
        initial_cash=2_000_000.0,
        debt_ratio=0.40,
        debt_interest_rate=0.09,
    )


def sample_projects() -> list[ProjectSpec]:
    def cf(year1: float, growth: float, life: int = 6) -> np.ndarray:
        return np.array([year1 * (1 + growth) ** t for t in range(life)])

    return [
        ProjectSpec("Phoenix Assembly Line", 4_500_000,
                    cf(1_350_000, 0.08), "USD",
                    "New assembly line, 6-year life, mid growth."),
        ProjectSpec("Meridian Service Expansion", 3_000_000,
                    cf(950_000, 0.06), "USD",
                    "After-sales service expansion."),
        ProjectSpec("Titan Packaging Plant", 5_500_000,
                    cf(1_450_000, 0.09), "USD",
                    "Packaging plant with higher growth."),
        ProjectSpec("Aurora E-commerce Platform", 2_200_000,
                    cf(800_000, 0.12), "USD",
                    "Digital channel launch with the fastest growth."),
        ProjectSpec("Vanguard Automation", 3_800_000,
                    cf(1_100_000, 0.05), "USD",
                    "Automation upgrade, lower growth, safer cash flows."),
        ProjectSpec("Nexus Logistics Hub", 6_200_000,
                    cf(1_600_000, 0.07), "USD",
                    "Logistics hub, largest outlay."),
    ]


def sample_projects_dicts() -> list[dict]:
    return [
        {"name": p.name, "cost": p.cost, "cashflows": p.cashflows,
         "currency": p.currency, "description": p.description}
        for p in sample_projects()
    ]


# ── dataframes for download ──────────────────────────────────────────────
def financials_dataframe(ci: CompanyInput | None = None) -> pd.DataFrame:
    ci = ci or sample_company()
    if ci.units is not None and ci.price is not None and ci.unit_var_cost is not None:
        df = pd.DataFrame({
            "Period": range(1, ci.periods + 1),
            "Revenue": ci.revenue.round(0),
            "COGS": ci.cogs.round(0),
            "Opex": np.round(ci.opex, 0),
            "Depreciation": np.round(ci.d_and_a, 0),
            "Capex": np.round(ci.capex, 0),
            "Units": ci.units.round(0).astype(int),
            "Price": np.round(ci.price, 2),
            "Unit_Variable_Cost": np.round(ci.unit_var_cost, 2),
        })
    else:
        df = pd.DataFrame({
            "Period": range(1, ci.periods + 1),
            "Revenue": np.round(ci.revenue, 0),
            "COGS": np.round(ci.cogs, 0),
            "Opex": np.round(ci.opex, 0),
            "Depreciation": np.round(ci.d_and_a, 0),
            "Capex": np.round(ci.capex, 0),
        })
    return df


def projects_dataframe(projects: list[ProjectSpec] | None = None) -> pd.DataFrame:
    projects = projects or sample_projects()
    rows = []
    life = max(len(p.cashflows) for p in projects)
    for p in projects:
        row = {"Project": p.name, "Initial_Investment": p.cost,
               "Description": p.description}
        for t in range(life):
            val = p.cashflows[t] if t < len(p.cashflows) else 0.0
            row[f"Year{t + 1}"] = round(float(val), 0)
        rows.append(row)
    return pd.DataFrame(rows).fillna(0.0)


# ── export helpers ───────────────────────────────────────────────────────
def financials_csv_bytes(ci: CompanyInput | None = None) -> bytes:
    return financials_dataframe(ci).to_csv(index=False).encode("utf-8")


def financials_xlsx_bytes(ci: CompanyInput | None = None) -> bytes:
    buf = io.BytesIO()
    financials_dataframe(ci).to_excel(buf, index=False, sheet_name="Financials")
    return buf.getvalue()


def projects_csv_bytes(projects: list[ProjectSpec] | None = None) -> bytes:
    return projects_dataframe(projects).to_csv(index=False).encode("utf-8")


def projects_xlsx_bytes(projects: list[ProjectSpec] | None = None) -> bytes:
    buf = io.BytesIO()
    projects_dataframe(projects).to_excel(buf, index=False, sheet_name="Projects")
    return buf.getvalue()


def assumptions_dataframe(ci: CompanyInput) -> pd.DataFrame:
    return pd.DataFrame([
        {"Group": "Capital structure", "Item": "Debt ratio",
         "Value": f"{ci.debt_ratio * 100:.0f}%"},
        {"Group": "Capital structure", "Item": "Debt interest rate",
         "Value": f"{ci.debt_interest_rate * 100:.1f}%"},
        {"Group": "Valuation", "Item": "Discount rate (WACC)",
         "Value": f"{ci.wacc * 100:.1f}%"},
        {"Group": "Valuation", "Item": "Terminal growth",
         "Value": f"{ci.terminal_growth * 100:.1f}%"},
        {"Group": "Taxation", "Item": "Tax rate",
         "Value": f"{ci.tax_rate * 100:.0f}%"},
        {"Group": "Working capital", "Item": "Receivables days",
         "Value": f"{ci.receivable_days:.0f}"},
        {"Group": "Working capital", "Item": "Payables days",
         "Value": f"{ci.payable_days:.0f}"},
        {"Group": "Working capital", "Item": "Inventory days",
         "Value": f"{ci.inventory_days:.0f}"},
        {"Group": "Liquidity", "Item": "Opening cash",
         "Value": f"{ci.initial_cash:,.0f}"},
        {"Group": "Macro", "Item": "Inflation",
         "Value": f"{ci.inflation * 100:.1f}%"},
        {"Group": "Risk", "Item": "Monte Carlo seed",
         "Value": f"{SAMPLE_MC_SEED}"},
    ])