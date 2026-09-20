"""PROFITABILITY MODEL — revenue, costs, margins, contribution, break-even
and profitability drivers.

Inputs are per-period revenue, cost of goods sold (variable), fixed operating
costs, depreciation & amortisation plus the tax rate. Per-unit price, volume
and unit variable cost are optional and enable contribution / break-even in
physical units.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class ProfitabilityResult:
    company_name: str
    currency: str
    periods: int
    years: list[int]
    revenue: np.ndarray
    cogs: np.ndarray
    gross_profit: np.ndarray
    opex: np.ndarray
    ebitda: np.ndarray
    d_and_a: np.ndarray
    ebit: np.ndarray
    interest: np.ndarray
    ebt: np.ndarray
    income_tax: np.ndarray
    net_income: np.ndarray

    units: np.ndarray | None = None
    price: np.ndarray | None = None
    unit_var_cost: np.ndarray | None = None

    gross_margin: float = 0.0
    ebitda_margin: float = 0.0
    operating_margin: float = 0.0
    net_margin: float = 0.0

    cm_per_unit: float | None = None
    cm_ratio: float | None = None
    fixed_cost: float = 0.0
    breakeven_units: float | None = None
    breakeven_revenue: float | None = None
    margin_of_safety_pct: float | None = None
    operating_leverage: float | None = None

    assumptions: dict = field(default_factory=dict)
    drivers: list[dict] = field(default_factory=list)

    def statement_df(self) -> pd.DataFrame:
        df = pd.DataFrame({
            "Period": self.years,
            "Revenue": self.revenue,
            "COGS (variable)": self.cogs,
            "Gross profit": self.gross_profit,
            "Fixed opex": self.opex,
            "EBITDA": self.ebitda,
            "D&A": self.d_and_a,
            "EBIT": self.ebit,
            "Interest": self.interest,
            "EBT": self.ebt,
            "Tax": self.income_tax,
            "Net income": self.net_income,
        })
        return df

    def margin_df(self) -> pd.DataFrame:
        return pd.DataFrame({
            "Period": self.years,
            "Gross margin %": self.gross_profit / self.revenue * 100,
            "EBITDA margin %": self.ebitda / self.revenue * 100,
            "Operating margin %": self.ebit / self.revenue * 100,
            "Net margin %": self.net_income / self.revenue * 100,
        })


def compute(ci) -> ProfitabilityResult:
    """Build the full income-statement model for a :class:`CompanyInput`."""
    rev = np.asarray(ci.revenue, dtype=float)
    cogs = np.asarray(ci.cogs, dtype=float)
    opex = np.asarray(ci.opex, dtype=float)
    d_a = np.asarray(ci.d_and_a, dtype=float)
    n = ci.periods
    years = list(range(1, n + 1))

    gross = rev - cogs
    ebitda = gross - opex
    ebit = ebitda - d_a

    # ── interest schedule (target capital structure on net book value) ──
    capex = np.asarray(ci.capex, dtype=float)
    cum_capex = np.cumsum(capex)
    nbv = np.maximum(cum_capex - np.cumsum(d_a), 0.0)
    debt = np.clip(ci.debt_ratio * nbv, 0.0, None)
    interest = np.zeros(n)
    interest[0] = 0.0
    for t in range(1, n):
        interest[t] = debt[t - 1] * ci.debt_interest_rate

    ebt = ebit - interest
    loss_carry = np.minimum(ebt, 0.0)
    tax_base = np.maximum(ebt, 0.0)
    income_tax = tax_base * ci.tax_rate
    net_income = ebt - income_tax

    first = max(rev[0], 1e-9)
    gross_margin = float(gross[0] / first * 100)
    ebitda_margin = float(ebitda[0] / first * 100)
    operating_margin = float(ebit[0] / first * 100)
    net_margin = float(net_income[0] / first * 100)

    # ── contribution & break-even ───────────────────────────────────────
    units = np.asarray(ci.units, dtype=float) if ci.units is not None else None
    price = np.asarray(ci.price, dtype=float) if ci.price is not None else None
    uvc = (np.asarray(ci.unit_var_cost, dtype=float)
           if ci.unit_var_cost is not None else None)

    fixed_cost = float(opex[0] + d_a[0])
    cm_per_unit = None
    cm_ratio = None
    breakeven_units = None
    breakeven_revenue = None
    margin_of_safety_pct = None
    operating_leverage = None

    if units is not None and price is not None and uvc is not None:
        cm_per_unit = float(price[0] - uvc[0])
        cm_ratio = float(cm_per_unit / price[0] * 100)
        if cm_per_unit > 0:
            breakeven_units = fixed_cost / cm_per_unit
            breakeven_revenue = fixed_cost / (cm_per_unit / price[0])
            if units[0] > 0 and cm_per_unit > 0:
                margin_of_safety_pct = (units[0] - breakeven_units) / units[0] * 100
    else:
        cm_ratio = float(gross[0] / first * 100)
        if cm_ratio > 0:
            breakeven_revenue = fixed_cost / (cm_ratio / 100.0)
            breakeven_units = breakeven_revenue / (first / max(units[0], 1e-9)) if units is not None else None

    if cm_ratio and cm_ratio > 0 and ebit[0] != 0:
        # operating leverage = contribution / EBIT (approximation of DOL)
        contribution = float(gross[0]) if cm_per_unit is None else cm_per_unit * units[0]
        if ebit[0] != 0:
            operating_leverage = float(contribution / ebit[0])

    # ── profitability drivers (period 1 vs final period) ────────────────
    drivers = [
        {"driver": "Unit volume",
         "base": (units[0] if units is not None else None),
         "final": (units[-1] if units is not None else None),
         "effect": "Volume \u2192 revenue and contribution."},
        {"driver": "Price",
         "base": (price[0] if price is not None else None),
         "final": (price[-1] if price is not None else None),
         "effect": "Price \u2192 revenue and unit contribution."},
        {"driver": "Unit variable cost",
         "base": (uvc[0] if uvc is not None else None),
         "final": (uvc[-1] if uvc is not None else None),
         "effect": "Variable cost \u2192 gross margin."},
        {"driver": "Fixed opex",
         "base": float(opex[0]),
         "final": float(opex[-1]),
         "effect": "Fixed cost \u2192 break-even point and operating leverage."},
        {"driver": "Net margin",
         "base": float(net_income[0] / first * 100),
         "final": float(net_income[-1] / max(rev[-1], 1e-9) * 100),
         "effect": "Bottom-line profitability per unit of revenue."},
    ]

    assumptions = {
        "tax_rate_pct": ci.tax_rate * 100,
        "fixed_cost_used_pct": "opex + D&A (period 1)",
        "variable_cost_used_pct": "COGS",
        "per_unit_data": units is not None and price is not None and uvc is not None,
        "operating_leverage_def": "Contribution / EBIT (period 1, approximation)",
    }

    return ProfitabilityResult(
        company_name=ci.company_name, currency=ci.currency, periods=n,
        years=years, revenue=rev, cogs=cogs, gross_profit=gross, opex=opex,
        ebitda=ebitda, d_and_a=d_a, ebit=ebit, interest=interest, ebt=ebt,
        income_tax=income_tax, net_income=net_income, units=units, price=price,
        unit_var_cost=uvc, gross_margin=gross_margin, ebitda_margin=ebitda_margin,
        operating_margin=operating_margin, net_margin=net_margin,
        cm_per_unit=cm_per_unit, cm_ratio=cm_ratio, fixed_cost=fixed_cost,
        breakeven_units=breakeven_units, breakeven_revenue=breakeven_revenue,
        margin_of_safety_pct=margin_of_safety_pct,
        operating_leverage=operating_leverage, assumptions=assumptions,
        drivers=drivers,
    )