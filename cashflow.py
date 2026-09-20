"""CASH-FLOW FORECASTING MODEL — cash inflows/outflows, working capital,
liquidity and the projected cash balance.

The cash-flow statement is built indirectly from the profitability model
outputs: operating cash flow = net income + depreciation & amortisation
minus the change in working capital.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class CashFlowResult:
    company_name: str
    currency: str
    periods: int
    years: list[int]
    revenue: np.ndarray
    net_income: np.ndarray
    d_and_a: np.ndarray
    receivables: np.ndarray
    inventory: np.ndarray
    payables: np.ndarray
    net_working_capital: np.ndarray
    change_nwc: np.ndarray
    operating_cash_flow: np.ndarray
    capex: np.ndarray
    investing_cash_flow: np.ndarray
    fcf: np.ndarray                       # operating - investing
    debt: np.ndarray
    financing_cash_flow: np.ndarray
    net_change: np.ndarray
    cash_balance: np.ndarray
    current_ratio: np.ndarray
    quick_ratio: np.ndarray

    initial_nwc: float = 0.0
    initial_cash: float = 0.0
    total_fcf: float = 0.0
    cumulative_fcf: np.ndarray = field(default_factory=lambda: np.array([]))
    assumptions: dict = field(default_factory=dict)

    def statement_df(self) -> pd.DataFrame:
        return pd.DataFrame({
            "Period": self.years,
            "Net income": self.net_income,
            "D&A": self.d_and_a,
            "\u0394 Working capital": self.change_nwc,
            "Operating CF": self.operating_cash_flow,
            "Capex": -self.capex,
            "Free cash flow": self.fcf,
            "Financing": self.financing_cash_flow,
            "Cash balance": self.cash_balance,
        })

    def liquidity_df(self) -> pd.DataFrame:
        return pd.DataFrame({
            "Period": self.years,
            "Current ratio": self.current_ratio,
            "Quick ratio": self.quick_ratio,
            "Cash balance": self.cash_balance,
        })


def compute(ci, prof) -> CashFlowResult:
    """Build the indirect cash-flow statement from :class:`CompanyInput`
    and a :class:`ProfitabilityResult`."""
    n = ci.periods
    years = list(range(1, n + 1))
    rev = np.asarray(prof.revenue, dtype=float)
    cogs = np.asarray(prof.cogs, dtype=float)
    ni = np.asarray(prof.net_income, dtype=float)
    d_a = np.asarray(prof.d_and_a, dtype=float)
    capex = np.asarray(ci.capex, dtype=float)

    # ── working capital via day conventions ─────────────────────────────
    receivable = rev * ci.receivable_days / 365.0
    inventory = cogs * ci.inventory_days / 365.0
    payable = cogs * ci.payable_days / 365.0
    nwc = receivable + inventory - payable
    nwc_start = nwc[0]                       # working capital at the start of plan
    change_nwc = np.zeros(n)
    change_nwc[0] = nwc[0]                   # initial build-up funded at t=0
    change_nwc[1:] = nwc[1:] - nwc[:-1]

    ocf = ni + d_a - change_nwc
    icf = -capex
    fcf = ocf + icf                          # free cash flow to the firm

    # ── financing to keep the capital structure on target ───────────────
    cum_capex = np.cumsum(capex)
    nbv = np.maximum(cum_capex - np.cumsum(d_a), 0.0)
    debt = np.clip(ci.debt_ratio * nbv, 0.0, None)
    financing = np.zeros(n)
    financing[1:] = -(debt[1:] - debt[:-1])  # +inflow if debt grows, -outflow to repay

    net_change = fcf + financing
    cash = np.zeros(n)
    cash[0] = ci.initial_cash + net_change[0]
    for t in range(1, n):
        cash[t] = cash[t - 1] + net_change[t]

    # ── liquidity ratios ────────────────────────────────────────────────
    current_assets = cash + receivable + inventory
    current_liab = payable + debt
    current_ratio = np.divide(current_assets, current_liab,
                              out=np.full(n, np.nan), where=current_liab > 0)
    quick_ratio = np.divide(current_assets - inventory, current_liab,
                            out=np.full(n, np.nan), where=current_liab > 0)

    cum_fcf = np.cumsum(fcf)
    total_fcf = float(fcf.sum())

    assumptions = {
        "receivable_days": ci.receivable_days,
        "payable_days": ci.payable_days,
        "inventory_days": ci.inventory_days,
        "initial_cash": ci.initial_cash,
        "initial_nwc": float(nwc_start),
        "debt_ratio_pct": ci.debt_ratio * 100,
        "debt_interest_pct": ci.debt_interest_rate * 100,
        "method": "Indirect method: NI + D&A \u2212 \u0394NWC",
    }

    return CashFlowResult(
        company_name=ci.company_name, currency=ci.currency, periods=n,
        years=years, revenue=rev, net_income=ni, d_and_a=d_a,
        receivables=receivable, inventory=inventory, payables=payable,
        net_working_capital=nwc, change_nwc=change_nwc, operating_cash_flow=ocf,
        capex=capex, investing_cash_flow=icf, fcf=fcf, debt=debt,
        financing_cash_flow=financing, net_change=net_change, cash_balance=cash,
        current_ratio=current_ratio, quick_ratio=quick_ratio,
        initial_nwc=float(nwc_start), initial_cash=ci.initial_cash,
        total_fcf=total_fcf, cumulative_fcf=cum_fcf, assumptions=assumptions,
    )