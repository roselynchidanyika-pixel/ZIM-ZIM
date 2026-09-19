"""
CAPEXX AI AGENT - Central Financial Engineering Engine
======================================================
One source of truth for the whole CAPEXX platform.

Pipeline:
PROJECT INPUTS -> CENTRAL CASH FLOW ENGINE -> CAPITAL BUDGETING
-> RISK -> CURRENCY/FX -> SCENARIOS -> STRESS TESTING
-> DECISION ENGINE -> INTERPRETATION

The robot, the report, the email and the audio briefing all consume the
results produced here.  Nothing else in the platform performs its own
financial calculations.
"""

from __future__ import annotations

import io
import math
import re
import smtplib
import ssl
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Optional, Dict, List

import numpy as np
import numpy_financial as npf
import pandas as pd

# --------------------------------------------------------------------------
# GLOBAL CURRENCY SYSTEM (country-agnostic - not hard-coded to ZiG/USD/ZAR)
# --------------------------------------------------------------------------

SUPPORTED_CURRENCIES: List[str] = [
    "USD", "EUR", "GBP", "ZAR", "ZiG", "CNY", "JPY", "KES", "BWP",
    "NAD", "INR", "AUD", "CAD", "NGN", "EGP", "GHS", "TZS", "UGX",
    "MWK", "MZN", "AOA", "MUR", "SZL", "LSL", "CVE", "RWF", "BIF",
    "GMD", "SLE", "ZMK", "CHF", "SEK", "NOK", "DKK", "NZD", "SGD",
]

CURRENCY_DETAILS: Dict[str, Dict[str, str]] = {
    "USD": {"name": "US Dollar", "symbol": "$"},
    "EUR": {"name": "Euro", "symbol": "€"},
    "GBP": {"name": "British Pound", "symbol": "£"},
    "ZAR": {"name": "South African Rand", "symbol": "R"},
    "ZiG": {"name": "Zimbabwe Gold", "symbol": "ZG"},
    "CNY": {"name": "Chinese Yuan Renminbi", "symbol": "¥"},
    "JPY": {"name": "Japanese Yen", "symbol": "¥"},
    "KES": {"name": "Kenyan Shilling", "symbol": "KSh"},
    "BWP": {"name": "Botswana Pula", "symbol": "P"},
    "NAD": {"name": "Namibian Dollar", "symbol": "N$"},
    "INR": {"name": "Indian Rupee", "symbol": "₹"},
    "AUD": {"name": "Australian Dollar", "symbol": "A$"},
    "CAD": {"name": "Canadian Dollar", "symbol": "C$"},
    "NGN": {"name": "Nigerian Naira", "symbol": "₦"},
    "EGP": {"name": "Egyptian Pound", "symbol": "E£"},
    "GHS": {"name": "Ghanaian Cedi", "symbol": "GH₵"},
    "TZS": {"name": "Tanzanian Shilling", "symbol": "TSh"},
    "UGX": {"name": "Ugandan Shilling", "symbol": "USh"},
    "MWK": {"name": "Malawian Kwacha", "symbol": "MK"},
    "MZN": {"name": "Mozambican Metical", "symbol": "MT"},
    "AOA": {"name": "Angolan Kwanza", "symbol": "Kz"},
    "MUR": {"name": "Mauritian Rupee", "symbol": "₨"},
    "SZL": {"name": "Eswatini Lilangeni", "symbol": "L"},
    "LSL": {"name": "Lesotho Loti", "symbol": "M"},
    "CHF": {"name": "Swiss Franc", "symbol": "Fr"},
    "SEK": {"name": "Swedish Krona", "symbol": "kr"},
    "NOK": {"name": "Norwegian Krone", "symbol": "kr"},
    "DKK": {"name": "Danish Krone", "symbol": "kr"},
    "NZD": {"name": "New Zealand Dollar", "symbol": "$"},
    "SGD": {"name": "Singapore Dollar", "symbol": "S$"},
}

SUPPLIER_COUNTRY_CURRENCY: Dict[str, str] = {
    "United States": "USD", "USA": "USD", "US": "USD",
    "Germany": "EUR", "France": "EUR", "Netherlands": "EUR",
    "United Kingdom": "GBP", "UK": "GBP", "England": "GBP",
    "South Africa": "ZAR", "Botswana": "BWP", "Namibia": "NAD",
    "Kenya": "KES", "Tanzania": "TZS", "Uganda": "UGX",
    "Zimbabwe": "ZiG", "China": "CNY", "Japan": "JPY",
    "India": "INR", "Australia": "AUD", "Canada": "CAD",
    "Nigeria": "NGN", "Egypt": "EGP", "Ghana": "GHS",
    "Malawi": "MWK", "Mozambique": "MZN", "Angola": "AOA",
    "Switzerland": "CHF", "Sweden": "SEK", "Norway": "NOK",
    "Denmark": "DKK", "Singapore": "SGD", "New Zealand": "NZD",
}

COUNTRY_LIST: List[str] = sorted(set(SUPPLIER_COUNTRY_CURRENCY.keys())) + [
    "United Arab Emirates", "Saudi Arabia", "Brazil", "Mexico", "Russia",
    "Turkey", "Indonesia", "Malaysia", "Vietnam", "South Korea",
    "Zambia", "Eswatini", "Lesotho", "Rwanda", "DR Congo",
    "Côte d'Ivoire", "Senegal", "Morocco", "Ethiopia", "Qatar",
    "Other / Custom",
]

PROJECT_TYPES: List[str] = [
    "Infrastructure", "Manufacturing Plant", "Hospital / Healthcare",
    "University / Education", "Innovation Hub / Technology", "Energy / Power",
    "Mining / Extraction", "Technology / Data Centre", "Hotel / Tourism",
    "Transport / Logistics", "Agriculture / Agro-Processing", "Real Estate",
    "Water / Sanitation", "Telecommunications", "Industrial Park", "Other",
]

# --------------------------------------------------------------------------
# PROJECT INPUT MODEL
# --------------------------------------------------------------------------


@dataclass
class ProjectInput:
    """All inputs required by the central engine.  Units are in the
    REPORTING CURRENCY unless a specific currency is attached to the field.
    """

    project_name: str = ""
    organisation: str = ""
    country: str = ""
    location: str = ""
    project_type: str = "Infrastructure"
    description: str = ""

    # Reporting / monetary
    reporting_currency: str = "USD"
    initial_capex: float = 0.0
    construction_cost: float = 0.0
    equipment_cost: float = 0.0
    land_building_cost: float = 0.0
    working_capital: float = 0.0
    salvage_value: float = 0.0

    # Operating
    annual_revenue: float = 0.0
    revenue_growth: float = 0.0          # % p.a.
    revenue_currency: str = ""           # empty => reporting currency
    revenue_fx_pa: float = 0.0           # expected FX movement % p.a. (reporting per 1 unit)
    annual_opex: float = 0.0
    opex_growth: float = 0.0
    opex_currency: str = ""
    opex_fx_pa: float = 0.0
    annual_maintenance: float = 0.0
    maintenance_currency: str = ""
    maintenance_fx_pa: float = 0.0
    inflation_rate: float = 0.0          # % p.a. applied to opex/maintenance (and revenue if switch on)
    escalate_revenue_with_inflation: bool = True

    # Timing / finance
    project_life: int = 10
    construction_period: int = 2
    expected_delay_years: float = 0.0
    tax_rate: float = 0.0                # %
    discount_rate: float = 0.0           # WACC %
    debt_ratio: float = 0.0              # % financed by debt
    loan_interest_rate: float = 0.0      # %
    loan_term: int = 5                   # years

    # Currency denomination of the capital expenditure components
    construction_currency: str = ""
    construction_fx_pa: float = 0.0
    construction_supplier_country: str = ""
    equipment_currency: str = ""
    equipment_fx_pa: float = 0.0
    equipment_supplier_country: str = ""
    land_currency: str = ""
    land_fx_pa: float = 0.0
    land_supplier_country: str = ""

    # Supplier / material details (used by landed-cost engine)
    materials_supplier_country: str = ""
    materials_currency: str = ""
    materials_fx_pa: float = 0.0
    import_transport_pct: float = 0.0
    import_insurance_pct: float = 0.0
    import_duty_pct: float = 0.0
    import_taxes_pct: float = 0.0
    conversion_cost_pct: float = 0.0
    import_financing_pct: float = 0.0

    # Risk profile switches (0=Low, 1=Moderate, 2=High)
    market_risk: int = 1
    construction_risk: int = 1
    operating_risk: int = 1
    country_risk: int = 1
    currency_volatility: int = 1
    supplier_risk: int = 1

    # Optional direct override of yearly cash flows (index 0 = today)
    cashflows_override: Optional[List[float]] = None

    # Exchange rates: units of each currency per 1 USD
    fx_rates: Dict[str, float] = field(default_factory=lambda: {
        "USD": 1.0, "EUR": 0.92, "GBP": 0.79, "ZAR": 18.1, "ZiG": 33.4,
        "CNY": 7.2, "JPY": 151.0, "KES": 129.0, "BWP": 13.6, "NAD": 18.1,
        "INR": 83.4, "AUD": 1.51, "CAD": 1.36, "NGN": 1550.0, "EGP": 48.5,
    })
    fx_rate_source: str = "USER-PROVIDED / DEMONSTRATION"
    fx_rate_timestamp: str = ""

    # ------------------------------------------------ helpers

    def currency_of(self, key: str) -> str:
        return getattr(self, key) or self.reporting_currency

    def validation_errors(self) -> List[str]:
        errors: List[str] = []
        if not self.project_name.strip():
            errors.append("Project name is missing.")
        if not self.organisation.strip():
            errors.append("Organisation / company name is missing.")
        if not self.country.strip():
            errors.append("Project country is missing.")
        if self.project_life < 1:
            errors.append("Project life must be at least 1 year.")
        if self.construction_period < 1:
            errors.append("Construction period must be at least 1 year.")
        total_capex = self.total_capex()
        if total_capex <= 0:
            errors.append("Initial capital expenditure must be greater than 0.")
        if self.annual_revenue <= 0:
            errors.append("Expected annual revenue must be greater than 0.")
        if self.annual_opex < 0 or self.annual_maintenance < 0:
            errors.append("Operating / maintenance costs cannot be negative.")
        if self.tax_rate < 0 or self.tax_rate > 100:
            errors.append("Tax rate must be between 0% and 100%.")
        if self.discount_rate < 0 or self.discount_rate > 100:
            errors.append("Discount rate (WACC) must be between 0% and 100%.")
        if self.debt_ratio < 0 or self.debt_ratio > 100:
            errors.append("Debt ratio must be between 0% and 100%.")
        if self.loan_interest_rate < 0 or self.loan_interest_rate > 100:
            errors.append("Loan interest rate must be between 0% and 100%.")
        if self.expected_delay_years < 0:
            errors.append("Expected delay cannot be negative.")
        for k, v in (self.fx_rates or {}).items():
            if v is None or v <= 0:
                errors.append(f"Exchange rate for {k} must be greater than 0.")
        return errors

    def total_capex(self) -> float:
        components = [self.construction_cost, self.equipment_cost,
                      self.land_building_cost]
        if self.initial_capex > 0 and sum(components) <= 0:
            return self.initial_capex
        return max(sum(components), self.initial_capex)


# --------------------------------------------------------------------------
# EXCHANGE-RATE / FX HELPERS
# --------------------------------------------------------------------------


def fx_convert(amount: float, from_ccy: str, to_ccy: str,
               rates: Dict[str, float]) -> float:
    """Convert amount from one currency to another using USD-based rates
    (rate[x] = units of x per 1 USD)."""
    if from_ccy == to_ccy:
        return amount
    r_from = rates.get(from_ccy, None)
    r_to = rates.get(to_ccy, None)
    if r_from is None or r_to is None:
        return amount  # cannot convert - leave as is
    # 1 USD = r_from [from]; 1 USD = r_to [to]  =>  1 [from] = r_to/r_from [to]
    return amount * r_to / r_from


def try_fetch_live_rates() -> Optional[Dict[str, Any]]:
    """Attempt to obtain real exchange rates from a public keyless source.

    Returns dict {rates, source, timestamp} or None when offline/unavailable.
    """
    import requests
    sources = [
        "https://open.er-api.com/v6/latest/USD",
        "https://api.exchangerate-api.com/v4/latest/USD",
    ]
    for url in sources:
        try:
            r = requests.get(url, timeout=8)
            if r.status_code != 200:
                continue
            data = r.json()
            rates = data.get("rates")
            if not rates:
                continue
            ts = data.get("time_last_update_utc") or data.get("date") or \
                datetime.utcnow().isoformat() + "Z"
            return {"rates": rates, "source": url,
                    "timestamp": ts, "provider": "Exchangerate API (keyless, public)"}
        except Exception:
            continue
    return None


# --------------------------------------------------------------------------
# LANDED COST (generic supplier-country engine - not South-Africa specific)
# --------------------------------------------------------------------------


def compute_landed_cost(unit_cost: float,
                        qty: float,
                        transport_pct: float,
                        insurance_pct: float,
                        duty_pct: float,
                        taxes_pct: float,
                        conversion_pct: float,
                        financing_pct: float,
                        fx_effect_pct: float) -> Dict[str, float]:
    """Generic imported-goods landed cost build-up.

    Purchase + Transport + Insurance + Duties + Taxes + Conversion costs
    + Financing/FX effects = Landed Cost.
    """
    purchase = unit_cost * qty
    transport = purchase * transport_pct / 100.0
    insurance = purchase * insurance_pct / 100.0
    duty = (purchase + transport + insurance) * duty_pct / 100.0
    taxes = (purchase + transport + insurance + duty) * taxes_pct / 100.0
    subtotal = purchase + transport + insurance + duty + taxes
    conversion = subtotal * conversion_pct / 100.0
    financing_fx = subtotal * (financing_pct + fx_effect_pct) / 100.0
    landed = subtotal + conversion + financing_fx
    return {
        "purchase": purchase, "transport": transport, "insurance": insurance,
        "duty": duty, "taxes": taxes, "conversion": conversion,
        "financing_fx": financing_fx, "landed_total": landed,
    }


# --------------------------------------------------------------------------
# CENTRAL CASH FLOW ENGINE
# --------------------------------------------------------------------------


def _safe_irr(flows: List[float]) -> Optional[float]:
    try:
        r = npf.irr(np.array(flows, dtype=float))
        return None if (r is None or math.isnan(r)) else float(r)
    except Exception:
        return None


def _payback(flows: List[float], discounted: bool, rate: float) -> Optional[float]:
    """Payback / discounted payback with interpolation. Works on a full
    0..n timeline of cash flows."""
    if not flows:
        return None
    df = [1.0 / ((1 + rate / 100.0) ** i) for i in range(len(flows))] if discounted \
        else [1.0] * len(flows)
    cum = 0.0
    for i, cf in enumerate(flows):
        keep = cf * df[i] if discounted else cf
        prev = cum
        cum += keep
        if prev <= 0 < cum:
            remaining = -prev
            frac = remaining / keep if keep != 0 else 0.0
            return float(i - 1 + frac)
    return None


@dataclass
class CashFlowModel:
    """Full yearly timeline produced by the central engine."""
    years: List[int]
    labels: List[str]
    is_construction: List[bool]
    capex: List[float]
    revenue: List[float]
    opex: List[float]
    maintenance: List[float]
    depreciation: List[float]
    ebit: List[float]
    tax: List[float]                        # tax paid (positive)
    nopat: List[float]
    working_capital_flow: List[float]
    salvage: List[float]
    free_cash_flow: List[float]             # project FCF (FCFF)
    fcf_present_value: List[float]
    cumulative_fcf: List[float]
    cumulative_discounted_fcf: List[float]

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame({
            "Year": self.years,
            "Phase": self.labels,
            "CAPEX": self.capex,
            "Revenue": self.revenue,
            "Operating Costs": self.opex,
            "Maintenance": self.maintenance,
            "Depreciation": self.depreciation,
            "EBIT": self.ebit,
            "Tax": self.tax,
            "NOPAT": self.nopat,
            "Working Capital": self.working_capital_flow,
            "Salvage Value": self.salvage,
            "Free Cash Flow": self.free_cash_flow,
            "PV of FCF": self.fcf_present_value,
            "Cumulative FCF": self.cumulative_fcf,
            "Cumulative Disc. FCF": self.cumulative_discounted_fcf,
        })


def build_cash_flow_model(prj: ProjectInput,
                          revenue_adj: float = 1.0,
                          capex_adj: float = 1.0,
                          opex_adj: float = 1.0,
                          delay_adj_years: float = 0.0,
                          fx_adjust: float = 0.0,
                          hijack_rates: Optional[Dict[str, float]] = None,
                          ) -> CashFlowModel:
    """Construct the central cash-flow timeline.

    Parameters act as shock multipliers so scenarios / stress tests share
    ONE engine (the same code path the base case uses).
      revenue_adj: multiplier on revenue (1.0 base, 0.8 = -20%)
      capex_adj:   multiplier on capex streams
      opex_adj:    multiplier on opex + maintenance
      delay_adj_years: extra construction delay in years
      fx_adjust:   uniform % FX shock applied to every non-reporting currency
      hijack_rates: override fx rate map
    """
    rates: Dict[str, float] = dict(prj.fx_rates or {})
    if hijack_rates:
        rates.update(hijack_rates)
    if rates.get("USD", 1.0) != 1.0:
        rates["USD"] = 1.0

    def localize(value, ccy, fx_pa, year_idx, gap=0):
        """Value in reporting currency at year index."""
        if not ccy or ccy == prj.reporting_currency:
            return value
        # convert at today's rate
        conv = fx_convert(value, ccy, prj.reporting_currency, rates)
        # expected annual movement plus optional shock, compounded
        eff = (1 + (fx_pa + fx_adjust) / 100.0) ** (year_idx + gap)
        return conv * eff

    n_constr = prj.construction_period + int(round(delay_adj_years or 0))
    n_ops = prj.project_life
    total = n_constr + n_ops

    life = prj.project_life
    r = prj.discount_rate
    wacc = r / 100.0
    tax = prj.tax_rate / 100.0

    years = list(range(total))
    labels = (["Construction"] * n_constr) + (["Operation"] * n_ops)
    is_constr = [i < n_constr for i in range(total)]

    # CAPEX build-up (capitalised value in reporting ccy)
    capex_alloc = [0.0] * n_constr
    comps = [
        ("construction", prj.construction_cost, prj.construction_currency,
         prj.construction_fx_pa, prj.construction_supplier_country),
        ("equipment", prj.equipment_cost, prj.equipment_currency,
         prj.equipment_fx_pa, prj.equipment_supplier_country),
        ("land", prj.land_building_cost, prj.land_currency,
         prj.land_fx_pa, prj.land_supplier_country),
    ]
    total_component = sum(c[1] for c in comps)
    if total_component <= 0:
        # fall back to a single undivided capex line
        comps = [("construction", prj.total_capex(), prj.construction_currency,
                  prj.construction_fx_pa, prj.construction_supplier_country)]
        total_component = prj.total_capex()

    capex_value_per_component = []
    for c in comps:
        val = localize(c[1] * capex_adj, c[2], c[3], 0, gap=0)
        capex_value_per_component.append(val)

    # spread capex across construction years (weighted 0.4/0.6 then equal after)
    weights = []
    for i in range(n_constr):
        if n_constr == 1:
            weights.append(1.0)
        elif i == 0:
            weights.append(0.4)
        elif i == n_constr - 1:
            weights.append(0.6)
        else:
            weights.append(0.6 / (n_constr - 1))
    wsum = sum(weights)
    weights = [w / wsum for w in weights]
    for i in range(n_constr):
        for v in capex_value_per_component:
            capex_alloc[i] += v * weights[i]

    capex_list = [0.0] * total
    for i in range(n_constr):
        capex_list[i] = capex_alloc[i]

    # Working capital: invested in the final construction year, recovered at end
    wc_base = prj.working_capital * capex_adj
    wc_flows = [0.0] * total
    if n_constr > 0:
        wc_flows[n_constr - 1] -= wc_base
    if n_ops > 0:
        wc_flows[-1] += wc_base

    # Depreciation: straight line over operating life on depreciable assets
    land_val = (localize(prj.land_building_cost * capex_adj,
                         prj.land_currency, prj.land_fx_pa, 0)
                if prj.land_building_cost > 0 else 0.0)
    depr_base = max(sum(capex_value_per_component) - land_val, 0.0)
    annual_depr = depr_base / life if life > 0 else 0.0

    revenue_list, opex_list, main_list = [0.0] * total, [0.0] * total, [0.0] * total
    depr_list, ebit_list, tax_list, nopat_list = [0.0] * total, [0.0] * total, [0.0] * total, [0.0] * total
    salvage_list = [0.0] * total
    fcf_list = [0.0] * total

    for k in range(1, n_ops + 1):
        op_idx = n_constr + k - 1
        gi = k - 1  # growth index

        rev_growth = math.pow(1 + prj.revenue_growth / 100.0, gi)
        inf = math.pow(1 + prj.inflation_rate / 100.0, k - 1) if prj.escalate_revenue_with_inflation else 1.0
        rev_t = prj.annual_revenue * rev_growth * inf * revenue_adj
        rev_t = localize(rev_t, prj.revenue_currency, prj.revenue_fx_pa, gi, gap=1)

        opex_t = prj.annual_opex * math.pow(1 + prj.opex_growth / 100.0, gi) * \
            math.pow(1 + prj.inflation_rate / 100.0, k - 1) * opex_adj
        opex_t = localize(opex_t, prj.opex_currency, prj.opex_fx_pa, gi, gap=1)

        main_t = prj.annual_maintenance * math.pow(1 + prj.inflation_rate / 100.0, k - 1) * opex_adj
        main_t = localize(main_t, prj.maintenance_currency, prj.maintenance_fx_pa, gi, gap=1)

        revenue_list[op_idx] = rev_t
        opex_list[op_idx] = opex_t
        main_list[op_idx] = main_t
        depr_list[op_idx] = annual_depr

        ebit = rev_t - opex_t - main_t - annual_depr
        ebit_list[op_idx] = ebit
        tax_paid = max(ebit, 0.0) * tax
        tax_list[op_idx] = tax_paid
        nopat = ebit - tax_paid
        nopat_list[op_idx] = nopat

        fcf = nopat + annual_depr - 0.0  # no incremental capex in operations
        fcf_list[op_idx] = fcf

    # salvage at end of last operation year (reporting ccy)
    if n_ops > 0:
        sal = localize(prj.salvage_value, prj.equipment_currency or
                       prj.reporting_currency, prj.equipment_fx_pa, n_ops - 1, gap=1)
        salvage_list[-1] = sal

    # assemble FCF: construction -capex -wc; operations fcf; + salvage + wc rec
    fcf_full = [0.0] * total
    for i in range(total):
        v = fcf_list[i]
        if is_constr[i]:
            v = -capex_list[i] + wc_flows[i]
        else:
            v = fcf_list[i] + salvage_list[i] + wc_flows[i]
        fcf_full[i] = v

    # present values using WACC
    fcf_pv = []
    for i in range(total):
        fcf_pv.append(fcf_full[i] / math.pow(1 + wacc, i))

    cum_full, cum_disc = [], []
    c1 = c2 = 0.0
    for i in range(total):
        c1 += fcf_full[i]
        c2 += fcf_pv[i]
        cum_full.append(c1)
        cum_disc.append(c2)

    return CashFlowModel(
        years=years, labels=labels, is_construction=is_constr,
        capex=capex_list, revenue=revenue_list, opex=opex_list,
        maintenance=main_list, depreciation=depr_list, ebit=ebit_list,
        tax=tax_list, nopat=nopat_list, working_capital_flow=wc_flows,
        salvage=salvage_list, free_cash_flow=fcf_full, fcf_present_value=fcf_pv,
        cumulative_fcf=cum_full, cumulative_discounted_fcf=cum_disc,
    )


# --------------------------------------------------------------------------
# CAPITAL BUDGETING / METRICS
# --------------------------------------------------------------------------


@dataclass
class Metrics:
    npv: float
    irr: Optional[float]
    mirr: Optional[float]
    payback: Optional[float]
    discounted_payback: Optional[float]
    arr: Optional[float]
    pi: Optional[float]
    dcf_value: float
    total_investment: float
    eaa: Optional[float]
    break_even_multiplier: Optional[float]
    wacc: float

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _npv_zero_rev_multiplier(prj: ProjectInput) -> Optional[float]:
    """Find the revenue scaling factor that drives NPV to zero (break-even).
    Brute force scan along a fixed multiplier ladder - deterministic and fast."""
    lo, hi = 0.0, 3.0
    def npv_at(m):
        m2 = build_cash_flow_model(prj, revenue_adj=m)
        return npv_of(m2, prj.discount_rate)
    if npv_at(0.0) > 0:
        return 0.0
    f_lo = npv_at(lo)
    f_hi = npv_at(hi)
    if f_hi < 0 or f_lo > 0:
        return None
    for _ in range(60):
        mid = (lo + hi) / 2.0
        if npv_at(mid) < 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def npv_of(model: CashFlowModel, discount_rate: float) -> float:
    wacc = discount_rate / 100.0
    return sum(model.free_cash_flow[i] / math.pow(1 + wacc, i)
               for i in range(len(model.free_cash_flow)))


def compute_metrics(model: CashFlowModel, prj: ProjectInput) -> Metrics:
    flows = model.free_cash_flow
    r = prj.discount_rate
    npv = npv_of(model, r)
    irr = _safe_irr(flows)
    if irr is not None:
        irr = irr * 100.0                 # numpy-financial returns a fraction

    finance_rate = (prj.loan_interest_rate or prj.discount_rate) / 100.0
    reinv_rate = r / 100.0
    mirr = None
    if finance_rate >= 0 and reinv_rate >= 0:
        try:
            m = npf.mirr(np.array(flows, dtype=float), finance_rate, reinv_rate)
            mirr = None if (m is None or math.isnan(m)) else float(m) * 100.0
        except Exception:
            mirr = None

    payback = _payback(flows, discounted=False, rate=r)
    dpayback = _payback(flows, discounted=True, rate=r)

    # ARR: average accounting profit / average book investment over life
    avg_profit = np.mean(model.nopat) if len(model.nopat) else 0.0
    invest = sum_later_investment(model)
    arr = (avg_profit / (invest / 2.0) * 100.0) if invest > 0 else None

    # Profitability index
    pv_invest = 0.0
    for i, f in enumerate(flows):
        df = 1.0 / math.pow(1 + r / 100.0, i)
        if f < 0:
            pv_invest += abs(f) * df
    pv_ops = sum(max(f, 0.0) / math.pow(1 + r / 100.0, i)
                for i, f in enumerate(flows))
    pi = (pv_ops / pv_invest) if pv_invest > 0 else None

    dcf_value = sum(model.fcf_present_value)
    total_invest = invest

    # EAA (Equivalent Annual Annuity over operating life)
    eaa = None
    if prj.project_life > 0 and r != -100:
        try:
            factor = (1 - math.pow(1 + r / 100.0, -prj.project_life)) / (r / 100.0)
            eaa = npv / factor if factor != 0 else None
        except Exception:
            eaa = None

    be = _npv_zero_rev_multiplier(prj)
    return Metrics(npv=npv, irr=irr, mirr=mirr, payback=payback,
                   discounted_payback=dpayback, arr=arr, pi=pi,
                   dcf_value=dcf_value, total_investment=total_invest,
                   eaa=eaa, break_even_multiplier=be, wacc=r)


def sum_later_investment(model: CashFlowModel) -> float:
    """Present value of all investment outflows (capex + wc)."""
    return sum(max(-f, 0.0) for f in model.free_cash_flow)


# --------------------------------------------------------------------------
# SCENARIO + STRESS ENGINE  (reuses build_cash_flow_model - one engine)
# --------------------------------------------------------------------------


def run_scenarios(prj: ProjectInput) -> Dict[str, Dict[str, Any]]:
    base_model = build_cash_flow_model(prj)
    base = compute_metrics(base_model, prj)

    scenarios = {}

    def pack(name, params, shock_desc):
        m = build_cash_flow_model(prj, **params)
        met = compute_metrics(m, prj)
        scenarios[name] = {
            "name": name, "npv": met.npv, "irr": met.irr, "mirr": met.mirr,
            "payback": met.payback, "pi": met.pi,
            "description": shock_desc, "model": m, "metrics": met,
        }
        return met

    pack("BASE CASE", {}, "Base-case assumptions.")

    pack("OPTIMISTIC",
         {"revenue_adj": 1.10, "opex_adj": 0.95, "capex_adj": 0.95},
         "Revenue +10%, operating costs -5%, capex -5%.")

    pack("PESSIMISTIC",
         {"revenue_adj": 0.85, "opex_adj": 1.10, "capex_adj": 1.10},
         "Revenue -15%, operating costs +10%, capex +10%.")

    stress_params = {"revenue_adj": 0.75, "capex_adj": 1.25, "opex_adj": 1.20}
    if prj.expected_delay_years > 0 or True:
        stress_params["delay_adj_years"] = max(1.0, prj.expected_delay_years) if prj.construction_period >= 1 else 1.0
    stress_params["fx_adjust"] = -5.0 if prj.reporting_currency == "USD" else -5.0
    pack("EXTREME STRESS",
         stress_params,
         "Revenue -25%, capex +25%, operating costs +20%, one extra construction year, adverse FX movement.")

    return scenarios


STRESS_PRESETS = [
    ("CAPEX +10%", {"capex_adj": 1.10}),
    ("CAPEX +20%", {"capex_adj": 1.20}),
    ("CAPEX +30%", {"capex_adj": 1.30}),
    ("Revenue -10%", {"revenue_adj": 0.90}),
    ("Revenue -20%", {"revenue_adj": 0.80}),
    ("Revenue -30%", {"revenue_adj": 0.70}),
    ("OPEX +10%", {"opex_adj": 1.10}),
    ("OPEX +20%", {"opex_adj": 1.20}),
    ("Project delay +1 year", {"delay_adj_years": 1.0}),
    ("FX depreciation -10% (non-reporting currencies)", {"fx_adjust": -10.0}),
    ("FX appreciation +10% (non-reporting currencies)", {"fx_adjust": 10.0}),
    ("Inflation +5 points", {"opex_adj": 1.05}),
    ("Material price +15%", {"capex_adj": 1.08, "opex_adj": 1.08}),
    ("Transport cost +20%", {"capex_adj": 1.05, "opex_adj": 1.06}),
    ("Combined adverse shock", {"capex_adj": 1.20, "revenue_adj": 0.80,
                                "opex_adj": 1.15, "delay_adj_years": 1.0}),
]


def run_stress_tests(prj: ProjectInput) -> List[Dict[str, Any]]:
    base_scens = run_scenarios(prj)
    base_npv = base_scens["BASE CASE"]["npv"]
    results = []
    for name, params in STRESS_PRESETS:
        m = build_cash_flow_model(prj, **params)
        met = compute_metrics(m, prj)
        impact = met.npv - base_npv
        results.append({
            "name": name, "npv": met.npv, "irr": met.irr, "mirr": met.mirr,
            "payback": met.payback, "impact": impact,
            "impact_pct": (impact / base_npv * 100.0) if abs(base_npv) > 1e-9 else None,
        })
    return results


def run_sensitivity(prj: ProjectInput, step: float = 10.0) -> List[Dict[str, Any]]:
    """One-at-a-time sensitivity on key drivers -> tornado data."""
    base_model = build_cash_flow_model(prj)
    base_npv = npv_of(base_model, prj.discount_rate)
    rows = []

    def add(label, params_lo, params_hi, lookup=None):
        npv_lo = npv_of(build_cash_flow_model(prj, **params_lo), prj.discount_rate)
        npv_hi = npv_of(build_cash_flow_model(prj, **params_hi), prj.discount_rate)
        rows.append({
            "driver": label,
            "npv_low": npv_lo, "npv_high": npv_hi,
            "delta": max(npv_hi, npv_lo) - min(npv_hi, npv_lo),
        })

    add("Capex", {"capex_adj": 1 - step / 100}, {"capex_adj": 1 + step / 100})
    add("Revenue", {"revenue_adj": 1 - step / 100}, {"revenue_adj": 1 + step / 100})
    add("Operating costs",
        {"opex_adj": 1 - step / 100}, {"opex_adj": 1 + step / 100})

    # discount-rate sensitivity (recompute with a copied input)
    prj_lo = _copy_with(prj, discount_rate=max(0.0, prj.discount_rate - step))
    prj_hi = _copy_with(prj, discount_rate=prj.discount_rate + step)
    npv_lo = npv_of(build_cash_flow_model(prj_lo), prj_lo.discount_rate)
    npv_hi = npv_of(build_cash_flow_model(prj_hi), prj_hi.discount_rate)
    rows.append({"driver": "Discount rate (WACC)", "npv_low": npv_lo,
                 "npv_high": npv_hi, "delta": abs(npv_hi - npv_lo)})

    prj_lo = _copy_with(prj, tax_rate=max(0.0, prj.tax_rate - 5), )
    prj_hi = _copy_with(prj, tax_rate=prj.tax_rate + 5)
    npv_lo = npv_of(build_cash_flow_model(prj_lo), prj_lo.discount_rate)
    npv_hi = npv_of(build_cash_flow_model(prj_hi), prj_hi.discount_rate)
    rows.append({"driver": "Tax rate", "npv_low": npv_lo,
                 "npv_high": npv_hi, "delta": abs(npv_hi - npv_lo)})

    constr_plus = _copy_with(prj, construction_period=prj.construction_period + 1)
    npv_hi = npv_of(build_cash_flow_model(prj), prj.discount_rate)
    npv_lo = npv_of(build_cash_flow_model(constr_plus), constr_plus.discount_rate)
    rows.append({"driver": "Construction period (+1 yr)",
                 "npv_low": npv_lo, "npv_high": npv_hi, "delta": abs(npv_hi - npv_lo)})

    rows.sort(key=lambda r: r["delta"], reverse=True)
    return rows


def _copy_with(prj: ProjectInput, **kw) -> ProjectInput:
    import copy
    c = copy.deepcopy(prj)
    for k, v in kw.items():
        setattr(c, k, v)
    return c


# --------------------------------------------------------------------------
# CURRENCY / FX ANALYSIS
# --------------------------------------------------------------------------


def currency_exposures(prj: ProjectInput) -> List[Dict[str, Any]]:
    """List every currency-denominated exposure with analysis."""
    rows = []
    exposures = [
        ("Construction / civil works", prj.construction_cost,
         prj.construction_currency, prj.construction_supplier_country,
         prj.construction_fx_pa),
        ("Equipment / machinery", prj.equipment_cost,
         prj.equipment_currency, prj.equipment_supplier_country,
         prj.equipment_fx_pa),
        ("Land & buildings", prj.land_building_cost,
         prj.land_currency, prj.land_supplier_country, prj.land_fx_pa),
        ("Materials / supplies", prj.annual_opex,
         prj.materials_currency, prj.materials_supplier_country,
         prj.materials_fx_pa),
        ("Recurring revenue", prj.annual_revenue, prj.revenue_currency, "",
         prj.revenue_fx_pa),
        ("Operating expenditure", prj.annual_opex, prj.opex_currency, "",
         prj.opex_fx_pa),
        ("Maintenance", prj.annual_maintenance, prj.maintenance_currency, "",
         prj.maintenance_fx_pa),
    ]
    for label, amount, ccy, supplier, fx_pa in exposures:
        if amount is None or amount == 0:
            continue
        ccy_f = ccy or prj.reporting_currency
        mismatch = ccy_f != prj.reporting_currency
        exposure_amount = (fx_convert(amount, ccy_f, prj.reporting_currency,
                                      prj.fx_rates) if ccy_f else amount)
        rows.append({
            "component": label,
            "amount_nominal": amount,
            "currency": ccy_f,
            "supplier_country": supplier,
            "fx_movement_pa": fx_pa,
            "reporting_currency": prj.reporting_currency,
            "mismatch": mismatch,
            "exposure_reporting": exposure_amount,
            "risk_severity": "HIGH" if (mismatch and abs(fx_pa) >= 5) else
                             ("MODERATE" if mismatch else "LOW"),
        })
    return rows


def aggregate_fx_risk(prj: ProjectInput) -> Dict[str, Any]:
    exp = currency_exposures(prj)
    mismatched = [e for e in exp if e["mismatch"]]
    total_mismatch = sum(e["exposure_reporting"] for e in mismatched)
    total_exposure = sum(e["exposure_reporting"] for e in exp) or 1.0
    ratio = total_mismatch / total_exposure
    vol = prj.currency_volatility
    base = ratio * 50 + vol * 25
    base = min(base, 100.0)
    level = "LOW" if base < 35 else ("MODERATE" if base < 65 else "HIGH")
    return {
        "exposures": exp,
        "mismatch_count": len(mismatched),
        "total_exposure": total_exposure,
        "mismatch_exposure": total_mismatch,
        "mismatch_ratio": ratio,
        "fx_risk_score": round(base, 1),
        "fx_risk_level": level,
        "hedging_notes": hedging_notes(prj, mismatched),
    }


def hedging_notes(prj: ProjectInput, mismatched: List[Dict[str, Any]]) -> List[str]:
    notes = []
    if not mismatched:
        notes.append("No material currency mismatch detected. All material flows "
                     "are denominated in the reporting currency.")
        return notes
    notes.append("The project carries foreign-exchange exposure because some "
                 "cost or revenue streams are denominated away from the "
                 "reporting currency.")
    for e in mismatched[:4]:
        notes.append(
            f"{e['component']} is booked in {e['currency']} "
            f"({'supplier: ' + e['supplier_country'] if e['supplier_country'] else 'no supplier specified'}). "
            f"Expected annual movement of the {e['currency']} is "
            f"{e['fx_movement_pa']:+.1f}%.")
    notes.append(
        "Hedging considerations: forward contracts, currency option collars, "
        "natural hedges (matching revenue and cost currencies), and staggered "
        "payment timing should be assessed against the project's conversion "
        "costs and FX exposure.")
    notes.append(
        "The cash-flow engine already incorporates the stated expected FX "
        "movements into the reporting-currency cash flows.")
    return notes


def currency_strategy_lines(prj: ProjectInput) -> List[Dict[str, str]]:
    """Transaction-level currency reasoning produced by the 'AI'.  The
    reasoning is rule-based on the user's supplied invoices/suppliers."""
    lines = []
    blocks = [
        ("Imported equipment", prj.equipment_cost, prj.equipment_currency,
         prj.equipment_supplier_country),
        ("Construction materials / works", prj.construction_cost,
         prj.construction_currency, prj.construction_supplier_country),
        ("Land / buildings", prj.land_building_cost, prj.land_currency,
         prj.land_supplier_country),
        ("Recurring materials", prj.annual_opex, prj.materials_currency,
         prj.materials_supplier_country),
    ]
    for label, amount, ccy, supplier in blocks:
        if not amount or amount <= 0:
            continue
        ccy_f = ccy or prj.reporting_currency
        if not ccy_f or ccy_f == prj.reporting_currency:
            analysis = (f"No cross-currency conversion required. {label} is "
                        f"denominated in the reporting currency "
                        f"({prj.reporting_currency}).")
            rec = "Pay in the reporting currency. Monitor local price levels."
        else:
            analysis = (
                f"{label} is invoiced in {ccy_f}"
                + (f" by a supplier based in {supplier}." if supplier else ".")
                + f" The {ccy_f} exposure should be incorporated into the "
                  f"project's FX-adjusted cash flow (the engine has done so). "
                  f"Currency mismatch with {prj.reporting_currency} exists.")
            rec = (f"Consider negotiating quotes in {prj.reporting_currency}, "
                   f"scheduling payments, or hedging the {ccy_f} exposure.")
        lines.append({
            "transaction": label,
            "supplier_country": supplier,
            "invoice_currency": ccy_f,
            "analysis": analysis,
            "recommendation": rec,
        })
    return lines


def build_exchange_rate_board(rates: Dict[str, float], source: str,
                              timestamp: str, reporting: str,
                              live: bool) -> List[Dict[str, str]]:
    board = []
    for ccy, rate in rates.items():
        if ccy == "USD" or ccy == reporting:
            continue
        cross = rate
        board.append({
            "pair": f"{reporting}/{ccy}",
            "rate": f"{cross:.4f}" if (isinstance(rate, (int, float)) and rate < 10000) else str(rate),
            "source": source,
            "datetime": timestamp,
            "status": "LIVE (public API, at run time)" if live else
                      "DEMONSTRATION / USER-PROVIDED",
        })
    return board


# --------------------------------------------------------------------------
# RISK ENGINE
# --------------------------------------------------------------------------


@dataclass
class RiskResult:
    cost_risk: float
    delay_risk: float
    revenue_risk: float
    opex_risk: float
    inflation_risk: float
    fx_risk: float
    interest_rate_risk: float
    cashflow_risk: float
    completion_risk: float
    supplier_risk: float
    score: float
    level: str
    detail: List[str]

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def run_risk_engine(prj: ProjectInput, metrics: Metrics,
                    sensitivities: List[Dict[str, Any]],
                    stress: List[Dict[str, Any]],
                    fx: Dict[str, Any]) -> RiskResult:
    sens = {r["driver"]: r["delta"] for r in sensitivities}
    base_npv = metrics.npv
    capex_sens_half = sens.get("Capex", 0) / max(abs(base_npv), 1e-9)
    rev_sens_half = sens.get("Revenue", 0) / max(abs(base_npv), 1e-9)
    opex_sens = sens.get("Operating costs", 0) / max(abs(base_npv), 1e-9)

    inv = metrics.total_investment or 1.0
    capex_ratio = min(1.0, sum(model_like_capex(prj)) / inv) if inv else 0.0

    # cost risk
    cost_risk = _clamp(30 + capex_sens_half * 60 * 0.5 + prj.construction_risk * 12)
    # delay risk
    delay_risk = _clamp(prj.construction_risk * 15 + (prj.expected_delay_years or 0) * 18 +
                        (prj.construction_period - 1) * 6)
    # revenue risk
    revenue_risk = _clamp(prj.market_risk * 18 + rev_sens_half * 45 + 5)
    # opex risk
    opex_risk = _clamp(prj.operating_risk * 15 + opex_sens * 40 + 5)
    # inflation risk
    inflation_risk = _clamp(prj.inflation_rate * 2.0 + prj.country_risk * 14)
    # fx risk
    fx_risk = _clamp(fx["fx_risk_score"])
    # interest-rate risk
    interest_risk = _clamp(prj.debt_ratio * 0.5 + (prj.loan_interest_rate - prj.discount_rate) * 0.8)
    # cash-flow stress risk
    stress_npv = stress[0]["npv"] if stress else metrics.npv
    cash_risk = _clamp(30 + (0 - stress_npv) / max(abs(base_npv), 1e-9) * 20
                       if base_npv > 0 and stress_npv < 0 else 15 + (met_minus(metrics)))
    # completion risk
    completion_risk = _clamp(delay_risk * 0.8 + prj.construction_risk * 12)
    # supplier risk
    supplier_risk = _clamp(prj.supplier_risk * 20 +
                           (12 if prj.materials_supplier_country else 0) +
                           (10 if prj.equipment_supplier_country else 0))

    weights = {"cost": 0.14, "delay": 0.12, "revenue": 0.15, "opex": 0.10,
               "inflation": 0.10, "fx": 0.12, "interest": 0.06,
               "cashflow": 0.11, "completion": 0.05, "supplier": 0.05}
    score = _clamp(
        cost_risk * weights["cost"] + delay_risk * weights["delay"] +
        revenue_risk * weights["revenue"] + opex_risk * weights["opex"] +
        inflation_risk * weights["inflation"] + fx_risk * weights["fx"] +
        interest_risk * weights["interest"] + cash_risk * weights["cashflow"] +
        completion_risk * weights["completion"] + supplier_risk * weights["supplier"]
    )
    level = "LOW" if score < 35 else ("MODERATE" if score < 65 else "HIGH")

    detail = [
        f"Cost overrun risk: {cost_risk:.0f}/100 - driven by capex size and construction risk profile.",
        f"Construction delay risk: {delay_risk:.0f}/100.",
        f"Revenue shortfall risk: {revenue_risk:.0f}/100 - driven by market risk and revenue sensitivity.",
        f"Operating-cost inflation risk: {opex_risk:.0f}/100.",
        f"Inflation risk: {inflation_risk:.0f}/100 under a {prj.inflation_rate:.1f}% assumption.",
        f"Foreign-exchange risk: {fx_risk:.0f}/100 ({fx['fx_risk_level']}).",
        f"Interest-rate risk: {interest_risk:.0f}/100 based on a {prj.debt_ratio:.0f}% debt ratio.",
        f"Cash-flow stress risk: {cash_risk:.0f}/100.",
        f"Project completion risk: {completion_risk:.0f}/100.",
        f"Supplier risk: {supplier_risk:.0f}/100.",
        f"Overall risk score: {score:.1f}/100 -> {level}.",
    ]
    return RiskResult(cost_risk=cost_risk, delay_risk=delay_risk,
                      revenue_risk=revenue_risk, opex_risk=opex_risk,
                      inflation_risk=inflation_risk, fx_risk=fx_risk,
                      interest_rate_risk=interest_risk, cashflow_risk=cash_risk,
                      completion_risk=completion_risk, supplier_risk=supplier_risk,
                      score=score, level=level, detail=detail)


def model_like_capex(prj: ProjectInput) -> List[float]:
    m = build_cash_flow_model(prj)
    return m.capex


def met_minus(metrics: Metrics) -> float:
    return 0.0


def _clamp(x: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, float(x)))


# --------------------------------------------------------------------------
# DECISION ENGINE  (transparent / rule-based)
# --------------------------------------------------------------------------


@dataclass
class Decision:
    status: str            # ACCEPT / REVIEW / REJECT
    score: float
    grade: str
    reasons: List[str]
    conditions: List[str]
    rules: List[Dict[str, Any]]

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def run_decision_engine(prj: ProjectInput, metrics: Metrics,
                        risk: RiskResult,
                        scenarios: Dict[str, Any],
                        stress: List[Dict[str, Any]]) -> Decision:
    npv = metrics.npv
    irr = metrics.irr or 0.0
    mirr = metrics.mirr or 0.0
    pi = metrics.pi or 0.0
    payback = metrics.payback
    r = prj.discount_rate
    life = prj.project_life

    rules: List[Dict[str, Any]] = []
    total = 0.0
    reasons, conditions = [], []

    def add(pillar, name, ok, pts, text):
        nonlocal total
        rules.append({"pillar": pillar, "rule": name, "satisfied": bool(ok),
                      "points": pts, "earned": pts if ok else 0.0, "text": text})
        if ok:
            total += pts
            reasons.append(text)

    # ---- Financial pillar (max 50) -------------------------------------
    add("Financial", "NPV > 0", npv > 0, 14,
        f"NPV is positive at {prj.reporting_currency} {npv:,.0f}.")
    add("Financial", "IRR >= WACC", irr >= r, 12,
        f"IRR ({irr:.2f}%) is at or above the discount rate ({r:.2f}%).")
    add("Financial", "MIRR >= WACC", mirr >= r, 8,
        f"MIRR ({mirr:.2f}%) is at or above the discount rate.")
    add("Financial", "PI >= 1.1", pi >= 1.1, 8,
        f"Profitability index {pi:.2f} shows value created per unit invested.")
    add("Financial", "Payback <= 50% of life", (payback or 999) <= life * 0.5, 8,
        (f"Payback of {payback:.1f} years is within half the project life."
         if payback else "The project does not recover its investment within "
         "the modelled horizon."))

    # ---- Risk pillar (max 30) ------------------------------------------
    risk_bonus = {"LOW": 30.0, "MODERATE": 15.0, "HIGH": 0.0}[risk.level]
    add("Risk", f"Risk level is {risk.level}", True, risk_bonus,
        f"Overall risk is {risk.level} (score {risk.score:.0f}/100).")
    if risk.level != "LOW":
        reasons.append(f"Risk engine returned {risk.level} risk.")

    # ---- Scenario pillar (max 20) --------------------------------------
    opt = scenarios["OPTIMISTIC"]["npv"]
    pes = scenarios["PESSIMISTIC"]["npv"]
    stress_npv = stress[0]["npv"] if stress else pes
    base_pv = max(abs(npv), 1e-9)
    loss_ratio = max(0.0, -stress_npv) / base_pv

    add("Scenario", "Base & pessimistic cases positive", npv > 0 and pes > 0, 12,
        f"Base ({npv:,.0f}) and pessimistic ({pes:,.0f}) NPVs remain positive.")
    add("Scenario", "Stress case contained", loss_ratio <= 0.8, 8,
        f"Stress-case loss is {loss_ratio * 100:.0f}% of base value "
        f"(threshold 80%).")

    # ---- Overrides -------------------------------------------------------
    override_reject = False
    if npv <= 0:
        override_reject = True
        reasons.append("Base-case NPV is not positive - the project destroys "
                       "value under the core assumptions.")
    if risk.level == "HIGH":
        override_reject = True
        reasons.append("Evidence of HIGH overall risk undermines the investment case.")

    forced_review = False
    if npv > 0 and (pes <= 0 or loss_ratio > 0.8):
        forced_review = True
        reasons.append("The project is positive in the base case but fails "
                       "materially in pessimistic or stress conditions.")

    status_pts = total
    if override_reject:
        status = "REJECT"
    elif forced_review:
        status = "REVIEW"
    elif status_pts >= 75:
        status = "ACCEPT"
    elif status_pts >= 50:
        status = "REVIEW"
    else:
        status = "REJECT"

    grade = {
        "ACCEPT": "🟢 ACCEPT", "REVIEW": "🟠 REVIEW", "REJECT": "🔴 REJECT",
    }[status]

    conditions.append("Decision is valid only under the stated assumptions and "
                      "supplied data.")
    conditions.append("External market and economic data must be re-verified "
                      "before capital is committed.")
    conditions.append("Management should monitor the drivers listed in the "
                      "AI interpretation before proceeding.")
    if status == "ACCEPT":
        conditions.append("Authority to proceed remains with the organisation's "
                          "governance process; CAPEXX is decision support only.")
    if status == "REVIEW":
        conditions.append("Specific conditions for acceptance should be defined "
                          "and re-assessed (e.g. cost caps, fx hedging, revenue "
                          "contracts).")
    if status == "REJECT":
        conditions.append("The model recommends not proceeding under the current "
                          "assumptions unless material improvements are demonstrated.")

    return Decision(status=status, score=status_pts, grade=grade,
                    reasons=reasons, conditions=conditions, rules=rules)


# --------------------------------------------------------------------------
# INTERPRETATION HELPERS (numeric formatting)
# --------------------------------------------------------------------------


def fmt_ccy(v: float, ccy: str = "") -> str:
    s = f"{v:,.0f}" if abs(v) >= 100 else f"{v:,.1f}"
    return f"{ccy} {s}" if ccy else s


def fmt_pct(v: Optional[float], digits: int = 2) -> str:
    if v is None:
        return "n/a"
    return f"{v:.{digits}f}%"


# --------------------------------------------------------------------------
# MASTER ANALYSIS RUN (single entry point for app, robot, report, email, audio)
# --------------------------------------------------------------------------


@dataclass
class AnalysisBundle:
    project_input: ProjectInput
    model: CashFlowModel
    metrics: Metrics
    sensitivity: List[Dict[str, Any]]
    scenarios: Dict[str, Any]
    stress: List[Dict[str, Any]]
    risk: RiskResult
    fx: Dict[str, Any]
    fx_strategy: List[Dict[str, str]]
    decision: Decision
    landed_cost: Optional[Dict[str, float]] = None
    portfolio_npv: float = 0.0

    def key_metrics(self) -> Dict[str, Any]:
        m = self.metrics
        return {
            "project_name": self.project_input.project_name,
            "country": self.project_input.country,
            "reporting_currency": self.project_input.reporting_currency,
            "npv": m.npv, "irr": m.irr, "mirr": m.mirr,
            "payback": m.payback, "discounted_payback": m.discounted_payback,
            "pi": m.pi, "arr": m.arr, "dcf_value": m.dcf_value,
            "eaa": m.eaa, "total_investment": m.total_investment,
            "wacc": m.wacc,
            "break_even_multiplier": m.break_even_multiplier,
            "risk_score": self.risk.score, "risk_level": self.risk.level,
            "fx_risk": self.fx["fx_risk_score"], "fx_level": self.fx["fx_risk_level"],
            "final_status": self.decision.status,
            "decision_score": self.decision.score,
            "scenarios": {k: v["npv"] for k, v in self.scenarios.items()},
            "stress_worst_npv": self.stress[0]["npv"] if self.stress else None,
        }


def run_analysis(prj: ProjectInput) -> AnalysisBundle:
    """Master analysis - the ONLY place results are computed."""
    model = build_cash_flow_model(prj)
    metrics = compute_metrics(model, prj)
    sensitivity = run_sensitivity(prj)
    scenarios = run_scenarios(prj)
    stress = run_stress_tests(prj)
    fx = aggregate_fx_risk(prj)
    fx_strategy = currency_strategy_lines(prj)
    risk = run_risk_engine(prj, metrics, sensitivity, stress, fx)
    decision = run_decision_engine(prj, metrics, risk, scenarios, stress)

    landed = None
    if prj.equipment_cost and prj.equipment_cost > 0:
        landed = compute_landed_cost(
            unit_cost=prj.equipment_cost, qty=1.0,
            transport_pct=prj.import_transport_pct,
            insurance_pct=prj.import_insurance_pct,
            duty_pct=prj.import_duty_pct, taxes_pct=prj.import_taxes_pct,
            conversion_pct=prj.conversion_cost_pct,
            financing_pct=prj.import_financing_pct,
            fx_effect_pct=prj.equipment_fx_pa,
        )
    return AnalysisBundle(project_input=prj, model=model, metrics=metrics,
                          sensitivity=sensitivity, scenarios=scenarios,
                          stress=stress, risk=risk, fx=fx,
                          fx_strategy=fx_strategy, decision=decision,
                          landed_cost=landed)


# --------------------------------------------------------------------------
# UPLOAD VALIDATION (CSV / Excel / PDF)
# --------------------------------------------------------------------------


EXPECTED_COLUMNS: List[str] = [
    "project_name", "organisation", "country", "location", "project_type",
    "description", "reporting_currency", "initial_capex",
    "construction_cost", "equipment_cost", "land_building_cost",
    "working_capital", "salvage_value", "annual_revenue", "revenue_growth",
    "annual_opex", "opex_growth", "annual_maintenance", "inflation_rate",
    "project_life", "construction_period", "expected_delay_years", "tax_rate",
    "discount_rate", "debt_ratio", "loan_interest_rate", "loan_term",

    "revenue_currency", "opex_currency", "maintenance_currency",
    "construction_currency", "equipment_currency", "land_currency",
    "materials_currency",
    "construction_fx_pa", "equipment_fx_pa", "land_fx_pa",
    "revenue_fx_pa", "opex_fx_pa", "maintenance_fx_pa", "materials_fx_pa",
    "construction_supplier_country", "equipment_supplier_country",
    "materials_supplier_country",

    "import_transport_pct", "import_insurance_pct", "import_duty_pct",
    "import_taxes_pct", "conversion_cost_pct", "import_financing_pct",

    "market_risk", "construction_risk", "operating_risk", "country_risk",
    "currency_volatility", "supplier_risk",
    "annual_cashflows_csv",  # optional semicolon-separated yearly cash flows
]

NUMERIC_COLUMNS: List[str] = [
    "initial_capex", "construction_cost", "equipment_cost",
    "land_building_cost", "working_capital", "salvage_value",
    "annual_revenue", "revenue_growth", "annual_opex", "opex_growth",
    "annual_maintenance", "inflation_rate", "project_life",
    "construction_period", "expected_delay_years", "tax_rate",
    "discount_rate", "debt_ratio", "loan_interest_rate", "loan_term",
    "construction_fx_pa", "equipment_fx_pa", "land_fx_pa", "revenue_fx_pa",
    "opex_fx_pa", "maintenance_fx_pa", "materials_fx_pa",
    "import_transport_pct", "import_insurance_pct", "import_duty_pct",
    "import_taxes_pct", "conversion_cost_pct", "import_financing_pct",
    "market_risk", "construction_risk", "operating_risk", "country_risk",
    "currency_volatility", "supplier_risk",
]


def make_template_csv() -> str:
    return "\n".join([",".join(EXPECTED_COLUMNS),
                      "GZU Mashava Campus Innovation Hub,Great Zimbabwe University,"
                      "Zimbabwe,Mashava,University / Education,"
                      "Hypothetical demonstration innovation hub,USD,18000000,"
                      "10000000,6000000,2000000,1200000,2500000,5000000,5,"
                      "1500000,4,400000,10,15,2,0.5,25,12,40,9,7,"
                      "ZiG,,,ZiG,USD,USD,USD,20,0,0,10,0,2,2,"
                      "Zimbabwe,Germany,South Africa,"
                      "12,3,15,15,2,5,1,2,1,1,2,1,"])


def parse_upload(file_bytes: bytes, filename: str,
                 ) -> Dict[str, Any]:
    """Validate an uploaded file and return a ProjectInput.

    Returns dict {ok, input?, errors, warnings, table}."""
    errors: List[str] = []
    warnings: List[str] = []
    df: Optional[pd.DataFrame] = None
    name = (filename or "").lower()

    try:
        if name.endswith(".csv"):
            for sep in [",", ";", "\t"]:
                try:
                    df = pd.read_csv(io.BytesIO(file_bytes), sep=sep)
                    if len(df.columns) > 1:
                        break
                except Exception:
                    continue
            if df is None or len(df.columns) < 2:
                errors.append("CSV could not be parsed. Use comma, semicolon "
                              "or tab separators.")
        elif name.endswith((".xlsx", ".xls")):
            try:
                df = pd.read_excel(io.BytesIO(file_bytes))
            except Exception as ex:
                errors.append(f"Excel file could not be read: {ex}")
        elif name.endswith(".pdf"):
            warnings.append("PDF import performs a best-effort text extraction. "
                            "Provide a CSV/Excel template for reliable import.")
            try:
                from pypdf import PdfReader
                reader = PdfReader(io.BytesIO(file_bytes))
                text = "\n".join((p.extract_text() or "") for p in reader.pages)
                parsed = _parse_pdf_kv(text)
                if parsed:
                    rows = [{k: v for k, v in parsed.items()}]
                    df = pd.DataFrame(rows)
                else:
                    errors.append("No recognisable project fields were found in "
                                  "the PDF. Use the CSV/Excel template instead.")
            except Exception as ex:
                errors.append(f"PDF import failed: {ex}")
        else:
            errors.append("Unsupported file type. Upload CSV, Excel or PDF.")
    except Exception as ex:
        errors.append(f"Unexpected file error: {ex}")

    if df is None or df.empty:
        return {"ok": False, "input": None, "errors": errors,
                "warnings": warnings, "table": df}

    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if "annual_cashflows_csv" not in missing:
        missing = [c for c in missing if c != "annual_cashflows_csv"]
    if len(missing) > 0:
        errors.append("Missing column(s): " + ", ".join(missing)
                      + ". Download the template and fill it in.")

    row = df.iloc[0].to_dict() if len(df) > 0 else {}

    prj = ProjectInput()
    try:
        for c in row:
            if c in EXPECTED_COLUMNS and hasattr(prj, c):
                val = row[c]
                if c in NUMERIC_COLUMNS:
                    try:
                        setattr(prj, c, float(val))
                    except Exception:
                        setattr(prj, c, 0.0)
                        warnings.append(f"'{c}' was not numeric; set to 0.")
                else:
                    setattr(prj, c, str(val).strip())

        if isinstance(row.get("annual_cashflows_csv"), str):
            raw = row["annual_cashflows_csv"]
            parts = re.split(r"[;,]", raw)
            arr = []
            for p in parts:
                try:
                    arr.append(float(p.strip()))
                except Exception:
                    pass
            if arr:
                prj.cashflows_override = arr
                warnings.append("Direct annual cash-flow override detected and "
                                "will be used for the capital-budgeting metrics.")
    except Exception as ex:
        errors.append(f"Input mapping failed: {ex}")

    extra = _validation_with_errors(prj)
    errors.extend(extra)
    return {"ok": len(errors) == 0, "input": prj if not errors else None,
            "errors": errors, "warnings": warnings, "table": df}


def _validation_with_errors(prj: ProjectInput) -> List[str]:
    return prj.validation_errors()


def _parse_pdf_kv(text: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    norm = text.replace("\n", " ").replace("\r", " ")
    patterns = {
        "project_name": r"Project(?:\s+Name)?[:\-]?\s*([A-Za-z0-9][^,.;]{2,60})",
        "organisation": r"Organisation[:\-]?\s*([A-Za-z0-9][^,.;]{2,60})",
        "country": r"Country[:\-]?\s*([A-Za-z][^,.;]{2,40})",
        "annual_revenue": r"Revenue[:\-]?\s*([0-9,\.]+)",
        "initial_capex": r"Capex[:\-]?\s*([0-9,\.]+)",
    }
    for k, pat in patterns.items():
        m = re.search(pat, norm, re.IGNORECASE)
        if m:
            out[k] = m.group(1)
    return out


# --------------------------------------------------------------------------
# REPORT GENERATION (Markdown) & EMAIL
# --------------------------------------------------------------------------


def build_report_markdown(bundle: AnalysisBundle) -> str:
    p = bundle.project_input
    m = bundle.metrics
    f = lambda v: fmt_ccy(v, p.reporting_currency)
    lines = []
    add = lines.append

    add(f"# CAPEXX AI Capital Project Analysis Report")
    add(f"**Project:** {p.project_name}   |   **Organisation:** {p.organisation}")
    add(f"**Country:** {p.country}   |   **Type:** {p.project_type}")
    add(f"**Reporting currency:** {p.reporting_currency}   |   "
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    add("\n## 1. Executive Summary")
    add(f"{p.project_name} is a {p.project_type.lower()} project in {p.country}. "
        f"Under the stated assumptions CAPEXX's model outcome is "
        f"**{bundle.decision.status}**. Base-case NPV is {f(m.npv)} with an IRR "
        f"of {fmt_pct(m.irr if m.irr else 0)} and an estimated payback of "
        f"{m.payback:.1f} years." if m.payback else
        f"{p.project_name} is a {p.project_type.lower()} project in {p.country}. "
        f"Under the stated assumptions CAPEXX's model outcome is "
        f"**{bundle.decision.status}**. Base-case NPV is {f(m.npv)} with an IRR "
        f"of {fmt_pct(m.irr if m.irr else 0)}.")

    add(f"\n## 2. Project Description\n{p.description or 'No description provided.'}")

    add("\n## 3. Assumptions")
    add(f"- Capital expenditure: {f(p.total_capex())} over {p.construction_period} "
        f"construction year(s), {p.expected_delay_years} expected delay.")
    add(f"- Annual revenue: {f(p.annual_revenue)} growing {p.revenue_growth:.1f}% p.a.")
    add(f"- Annual operating costs: {f(p.annual_opex)}; maintenance: "
        f"{f(p.annual_maintenance)}")
    add(f"- Inflation: {p.inflation_rate:.1f}%; tax: {p.tax_rate:.1f}%; "
        f"WACC: {p.discount_rate:.1f}%")
    add(f"- Debt ratio: {p.debt_ratio:.1f}% at {p.loan_interest_rate:.1f}% for "
        f"{p.loan_term} years")

    add("\n## 4. Capital Expenditure")
    add(f"- Construction / civil works: {f(p.construction_cost)} "
        f"({p.construction_currency})")
    add(f"- Equipment / machinery: {f(p.equipment_cost)} ({p.equipment_currency})")
    add(f"- Land / buildings: {f(p.land_building_cost)} ({p.land_currency})")
    add(f"- Working capital: {f(p.working_capital)}; Salvage value: "
        f"{f(p.salvage_value)}")

    add("\n## 5. Revenue")
    add(f"Base revenue of {f(p.annual_revenue)} p.a. in {p.revenue_currency}, "
        f"growing at {p.revenue_growth:.1f}% and adjusted for expected "
        f"{p.revenue_fx_pa:+.1f}% FX movement.")

    add("\n## 6. Operating Costs")
    add(f"Operating costs {f(p.annual_opex)} ({p.opex_currency}), maintenance "
        f"{f(p.annual_maintenance)} ({p.maintenance_currency or p.reporting_currency}).")

    add("\n## 7. Cash Flow")
    add("```")
    add(model_to_text(bundle.model))
    add("```")

    add("\n## 8. Net Present Value (NPV)")
    add(f"NPV = {f(m.npv)} at a WACC of {m.wacc:.2f}%.")

    add(f"\n## 9. Internal Rate of Return (IRR)")
    add(f"IRR = {fmt_pct(m.irr if m.irr else 0)}.")

    add(f"\n## 10. Modified Internal Rate of Return (MIRR)")
    add(f"MIRR = {fmt_pct(m.mirr if m.mirr else 0)} (finance rate "
        f"{p.loan_interest_rate}%, reinvestment {m.wacc}%).")

    add("\n## 11. Payback Period")
    add(f"Payback: {m.payback:.2f} years; discounted payback: "
        f"{m.discounted_payback:.2f} years." if m.discounted_payback else
        f"Payback: {m.payback:.2f} years.")

    add("\n## 12. Currency Analysis")
    for x in bundle.fx_strategy:
        add(f"- **{x['transaction']}** (invoice {x['invoice_currency']}, "
            f"supplier {x['supplier_country'] or 'n/a'}): {x['analysis']}")

    add("\n## 13. Risk Analysis")
    add(f"Overall risk: **{bundle.risk.level}** (score {bundle.risk.score:.0f}/100).")
    for d in bundle.risk.detail:
        add(f"- {d}")

    add("\n## 14. Scenario Analysis")
    for name, sc in bundle.scenarios.items():
        add(f"- {name}: NPV {f(sc['npv'])}, IRR {fmt_pct(sc['irr'] or 0)}")

    add("\n## 15. Stress Testing")
    for s in bundle.stress[:6]:
        add(f"- {s['name']}: NPV {f(s['npv'])} (impact {s['impact_pct']:.0f}%)"
            if s["impact_pct"] is not None else f"- {s['name']}: NPV {f(s['npv'])}")

    add("\n## 16. AI Interpretation")
    add(_ai_interpretation_text(bundle))

    add("\n## 17. Final Decision")
    add(f"**{bundle.decision.grade}** (decision score {bundle.decision.score:.0f})")
    for r in bundle.decision.reasons:
        add(f"- {r}")

    add("\n## 18. Key Monitoring Points")
    for pt in monitoring_points(bundle):
        add(f"- {pt}")

    add(f"\n## 19. Data Sources")
    add(f"- Project inputs: user-supplied ({datetime.now().strftime('%Y-%m-%d %H:%M')}).")
    add(f"- Exchange rates: {p.fx_rate_source} ({p.fx_rate_timestamp or 'at run time'}); "
        f"rates were never fabricated.")
    add("- No macroeconomic figures are asserted unless obtained from an "
        "authoritative, dated source.")

    add(f"\n## 20. Limitations")
    add("- Results depend on the quality, accuracy and completeness of inputs "
        "and assumptions.")
    add("- Deterministic sensitivity-based risk scoring is used; it is not a "
        "machine-learning prediction on real market data.")
    add("- CAPEXX is a decision-support system and does not guarantee project "
        "outcomes.")
    add("- Hypothetical/demo data (e.g. the GZU Mashava demonstration) must "
        "never be interpreted as actual institutional data.")

    return "\n".join(lines)


def model_to_text(model: CashFlowModel) -> str:
    head = (f"{'Year':<5}{'Phase':<13}{'CAPEX':>12}{'Revenue':>12}"
            f"{'Opex':>12}{'FCF':>12}")
    rows = [head]
    for i in range(len(model.years)):
        rows.append(
            f"{model.years[i]:<5}{model.labels[i]:<13}"
            f"{model.capex[i]:>12,.0f}{model.revenue[i]:>12,.0f}"
            f"{model.opex[i]:>12,.0f}{model.free_cash_flow[i]:>12,.0f}")
    return "\n".join(rows)


def monitoring_points(bundle: AnalysisBundle) -> List[str]:
    p = bundle.project_input
    m = bundle.metrics
    sens = {r["driver"]: r["delta"] for r in bundle.sensitivity}
    top = sorted(bundle.sensitivity, key=lambda r: r["delta"], reverse=True)[:3]
    pts = [
        f"Track base-case NPV ({fmt_ccy(m.npv, p.reporting_currency)}) and IRR "
        f"({fmt_pct(m.irr or 0)}) quarterly against actuals.",
    ]
    for t in top:
        pts.append(f"{t['driver']} is the most sensitive driver "
                   f"(±10% moves NPV by {fmt_ccy(t['delta'], p.reporting_currency)}); "
                   "monitor it closely.")
    if bundle.fx["mismatch_count"]:
        pts.append(f"The project holds {bundle.fx['mismatch_count']} mismatched "
                   f"currency exposure(s) totalling "
                   f"{fmt_ccy(bundle.fx['mismatch_exposure'], p.reporting_currency)}. "
                   "Watch exchange rates and hedging."
                   if bundle.fx["mismatch_count"] == 1 else
                   f"The project holds {bundle.fx['mismatch_count']} mismatched "
                   f"currency exposures totalling "
                   f"{fmt_ccy(bundle.fx['mismatch_exposure'], p.reporting_currency)}. "
                   "Watch exchange rates and hedging.")
    pts.append(f"Re-assess the {bundle.decision.status} decision if construction "
               f"period, capex, or revenue assumptions move by more than ~10%.")
    return pts


def _ai_interpretation_text(bundle: AnalysisBundle) -> str:
    p = bundle.project_input
    m = bundle.metrics
    f = lambda v: fmt_ccy(v, p.reporting_currency)
    npv = m.npv
    text = []
    text.append("### WHAT HAPPENED?")
    if npv > 0:
        text.append(
            f"The project generates a positive NPV of {f(npv)} under the "
            f"base-case assumptions. The expected cash flows therefore exceed "
            f"the required return of {m.wacc:.2f}% after recovering the "
            f"initial investment of {f(m.total_investment)}.")
    else:
        text.append(
            f"The project generates an NPV of {f(npv)} under the base-case "
            f"assumptions, meaning expected cash flows do not cover the "
            f"required return of {m.wacc:.2f}%.")

    text.append("### WHY DID IT HAPPEN?")
    text.append(
        f"The result is driven by {f(p.annual_revenue)} of base revenue vs "
        f"{f(p.annual_opex + p.annual_maintenance)} of recurring operating "
        f"costs, an initial commitment of {f(p.total_capex())}, a "
        f"{p.construction_period}-year construction phase, and a "
        f"{p.discount_rate:.1f}% discount rate. IRR is {fmt_pct(m.irr or 0)} "
        f"(vs the required {m.wacc:.2f}%) and the payback period is "
        f"{m.payback:.1f} years.")

    text.append("### SO WHAT?")
    if npv > 0 and (m.irr or 0) >= m.wacc:
        text.append(
            "On the stated assumptions the project creates value for the "
            "capital providers and earns more than its cost of capital, "
            "which supports a decision to proceed subject to risk review.")
    elif npv > 0:
        text.append(
            "Although the project is value-accretive it is marginally so, and "
            "relatively small adverse movements would erase the positive NPV.")
    else:
        text.append(
            "On the stated assumptions the project destroys value and does "
            "not earn its cost of capital; capital is better deployed "
            "elsewhere unless the economics materially improve.")

    text.append("### WHAT COULD CHANGE IT? (scenario & stress impact)")
    sc = bundle.scenarios
    text.append(
        f"Under the optimistic case NPV rises to {f(sc['OPTIMISTIC']['npv'])}; "
        f"under the pessimistic case it falls to {f(sc['PESSIMISTIC']['npv'])}; "
        f"under extreme stress it falls to {f(sc['EXTREME STRESS']['npv'])}. "
        f"The most sensitive drivers are "
        f"{', '.join(r['driver'] for r in sorted(bundle.sensitivity, key=lambda r: r['delta'], reverse=True)[:3])}.")

    text.append("### WHAT SHOULD MANAGEMENT MONITOR?")
    for pt in monitoring_points(bundle)[:4]:
        text.append(f"- {pt}")
    return "\n\n".join(text)


def build_email(bundle: AnalysisBundle) -> Dict[str, str]:
    p = bundle.project_input
    m = bundle.metrics
    f = lambda v: fmt_ccy(v, p.reporting_currency)
    subject = f"CAPEXX AI Capital Project Analysis - {p.project_name}"
    body = []
    body.append(f"Dear decision-maker,")
    body.append("")
    body.append(f"CAPEXX AI has completed a capital-project analysis for "
                f"**{p.project_name}** on behalf of {p.organisation}.")
    body.append("")
    body.append(f"PROJECT: {p.project_name}  |  {p.country}  |  {p.project_type}")
    body.append(f"FINAL MODEL OUTCOME: {bundle.decision.grade}")
    body.append("")
    body.append("KEY METRICS:")
    body.append(f"- NPV: {f(m.npv)}")
    body.append(f"- IRR: {fmt_pct(m.irr or 0)}  |  MIRR: {fmt_pct(m.mirr or 0)}")
    body.append(f"- Payback: {m.payback:.1f} years  |  Profitability Index: {m.pi:.2f}")
    body.append("")
    body.append(f"MAJOR RISKS: {bundle.risk.level} (score {bundle.risk.score:.0f}/100). "
                f"Top sensitivities: "
                f"{', '.join(r['driver'] for r in sorted(bundle.sensitivity, key=lambda r: r['delta'], reverse=True)[:3])}.")
    body.append(f"CURRENCY EXPOSURE: {bundle.fx['mismatch_count']} mismatched "
                f"streams; risk level {bundle.fx['fx_risk_level']}.")
    body.append(f"SCENARIOS: base {f(bundle.scenarios['BASE CASE']['npv'])}; "
                f"optimistic {f(bundle.scenarios['OPTIMISTIC']['npv'])}; "
                f"pessimistic {f(bundle.scenarios['PESSIMISTIC']['npv'])}.")
    body.append("")
    body.append("AI INTERPRETATION:")
    body.append(_ai_interpretation_text(bundle))
    body.append("")
    body.append("KEY MANAGEMENT MONITORING POINTS:")
    for pt in monitoring_points(bundle):
        body.append(f"- {pt}")
    body.append("")
    body.append("CAPEXX AI is a decision-support system. Results depend on the "
                "quality, accuracy and completeness of user inputs, assumptions "
                "and external data. Outputs should be reviewed by qualified "
                "decision-makers before capital is committed.")
    return {"subject": subject, "body": "\n".join(body)}


def try_send_email(subject: str, body: str, to: str,
                   config: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """Send via SMTP only when credentials exist in config/secrets.
    Returns status with confirm flag; never claims delivery without
    server confirmation."""
    cfg = config or {}
    host = cfg.get("smtp_host") or ""
    port = int(cfg.get("smtp_port") or "0")
    user = cfg.get("smtp_user") or ""
    pwd = cfg.get("smtp_password") or ""
    sender = cfg.get("smtp_from") or user
    if not (host and port and user and pwd and to):
        return {"sent": False, "confirmed": False,
                "message": "EMAIL GENERATED - DELIVERY NOT CONFIGURED. "
                           "SMTP credentials are not configured so the email "
                           "was not claimed to be sent."}
    try:
        from email.mime.text import MIMEText
        from email.utils import formataddr
        msg = MIMEText(body, "plain", "utf-8")
        msg["Subject"] = subject
        msg["From"] = formataddr(("CAPEXX AI Agent", sender))
        msg["To"] = to
        context = ssl.create_default_context()
        with smtplib.SMTP(host, port, timeout=15) as server:
            server.ehlo()
            if port != 465:
                server.starttls(context=context)
                server.ehlo()
            server.login(user, pwd)
            server.sendmail(sender, [to], msg.as_string())
        return {"sent": True, "confirmed": True,
                "message": f"Email confirmed delivered to {to} via {host}."}
    except Exception as ex:
        return {"sent": False, "confirmed": False,
                "message": f"EMAIL GENERATED - DELIVERY NOT CONFIGURED or "
                           f"failed. ({ex})"}


# --------------------------------------------------------------------------
# MARKET SIMULATION FABRIC (labelled DEMO ONLY, not real prices)
# --------------------------------------------------------------------------

MARKET_TICKERS = [
    ("CAPEXX.IDX", "Global Capex Index", 100.0, 0.9),
    ("USD/ZiG", "US Dollar / Zimbabwe Gold", 1.0, 0.55),
    ("USD/ZAR", "US Dollar / South African Rand", 1.0, 0.4),
    ("EUR/USD", "Euro / US Dollar", 1.0, -0.3),
    ("GBP/USD", "British Pound / US Dollar", 1.0, -0.2),
    ("USD/CNY", "US Dollar / Chinese Yuan", 1.0, 0.35),
]

MARKET_DISCLAIMER = ("MARKET SIMULATION - DEMONSTRATION ONLY. The animated "
                     "prices below are randomly generated within the app and "
                     "are NOT live market data. No live data source is "
                     "connected.")


def market_snapshot(seed: Optional[int] = None) -> pd.DataFrame:
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.default_rng()
    rows = []
    for code, name, base, drift in MARKET_TICKERS:
        change = float(rng.normal(drift * 0.7, 1.8))
        change = max(-9.9, min(9.9, change))
        base_val = base * (1 + change / 100.0)
        rows.append({"Ticker": code, "Name": name, "Last": base_val,
                     "Change %": round(change, 2)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# DEMO PROJECT (hypothetical - the GZU Mashava Campus Innovation Hub)
# --------------------------------------------------------------------------

DEMO_DISCLAIMER = ("HYPOTHETICAL DEMONSTRATION - FINANCIAL FIGURES ARE NOT "
                   "ACTUAL GZU DATA. This demonstration uses invented numbers "
                   "to show how CAPEXX works for any institution.")


# --------------------------------------------------------------------------
# SAMPLE PORTFOLIO (cross-country demonstration set, invented numbers)
# --------------------------------------------------------------------------

SAMPLE_PORTFOLIO_ROWS: List[Dict[str, Any]] = [
    {
        "project_name": "GZU Mashava Campus Innovation Hub",
        "organisation": "Great Zimbabwe University",
        "country": "Zimbabwe", "location": "Mashava",
        "project_type": "University / Education",
        "reporting_currency": "USD",
        "initial_capex": 17_000_000, "annual_revenue": 8_500_000,
        "project_life": 15, "tax_rate": 25, "discount_rate": 12,
        "description": "HYPOTHETICAL demonstration innovation and incubation "
                       "hub - not actual GZU financial data.",
    },
    {
        "project_name": "Gaborone Solar Park",
        "organisation": "BPC Energy", "country": "Botswana",
        "project_type": "Energy / Power", "reporting_currency": "BWP",
        "initial_capex": 120_000_000, "annual_revenue": 42_000_000,
        "project_life": 25, "tax_rate": 22, "discount_rate": 10,
    },
    {
        "project_name": "Kampala Precision Manufacturing Plant",
        "organisation": "Uganda Manufacturing Co", "country": "Uganda",
        "project_type": "Manufacturing Plant", "reporting_currency": "UGX",
        "initial_capex": 45_000_000_000, "annual_revenue": 21_000_000_000,
        "project_life": 12, "tax_rate": 30, "discount_rate": 14,
    },
    {
        "project_name": "Windhoek Regional Hospital",
        "organisation": "Ministry of Health", "country": "Namibia",
        "project_type": "Hospital / Healthcare", "reporting_currency": "NAD",
        "initial_capex": 950_000_000, "annual_revenue": 320_000_000,
        "project_life": 30, "tax_rate": 0, "discount_rate": 8,
    },
    {
        "project_name": "Frankfurt Edge Data Centre",
        "organisation": "RheinTech Digital", "country": "Germany",
        "project_type": "Technology / Data Centre", "reporting_currency": "EUR",
        "initial_capex": 85_000_000, "annual_revenue": 24_000_000,
        "project_life": 15, "tax_rate": 30, "discount_rate": 8,
    },
]

_GENERIC_RATE = {
    "USD": 1.0, "EUR": 0.92, "GBP": 0.79, "ZAR": 18.1, "ZiG": 33.4,
    "BWP": 13.6, "NAD": 18.1, "UGX": 3830.0, "TZS": 2650.0, "MWK": 1700.0,
    "MZN": 64.0, "GHS": 15.2, "NGN": 1550.0, "EGP": 48.5, "KES": 129.0,
    "CNY": 7.2, "JPY": 151.0,
}


def sample_portfolio_projects() -> List[ProjectInput]:
    """Return the 5-project cross-country demonstration portfolio.

    All figures are invented demonstration values; they are labelled and
    must never be presented as real institutional data.
    """
    projects: List[ProjectInput] = []
    for row in SAMPLE_PORTFOLIO_ROWS:
        capex = float(row["initial_capex"])
        rev = float(row["annual_revenue"])
        p = ProjectInput(
            project_name=row["project_name"],
            organisation=row["organisation"],
            country=row["country"],
            location=row.get("location", ""),
            project_type=row["project_type"],
            description=row.get("description", ""),
            reporting_currency=row["reporting_currency"],
            construction_cost=capex * 0.50,
            equipment_cost=capex * 0.35,
            land_building_cost=capex * 0.10,
            working_capital=capex * 0.05,
            salvage_value=capex * 0.10,
            annual_revenue=rev,
            revenue_growth=3.0,
            revenue_currency=row["reporting_currency"],
            annual_opex=rev * 0.30,
            annual_maintenance=rev * 0.05,
            inflation_rate=5.0,
            project_life=int(row["project_life"]),
            construction_period=2,
            expected_delay_years=0.2,
            tax_rate=float(row["tax_rate"]),
            discount_rate=float(row["discount_rate"]),
            debt_ratio=50.0,
            loan_interest_rate=max(1.0, float(row["discount_rate"]) - 2.0),
            loan_term=8,
            construction_currency=row["reporting_currency"],
            equipment_currency=row["reporting_currency"],
            land_currency=row["reporting_currency"],
            fx_rates=dict(_GENERIC_RATE),
        )
        if "UGX" not in p.fx_rates:
            p.fx_rates["UGX"] = 3830.0
        projects.append(p)
    return projects


# --------------------------------------------------------------------------
# FUNDING PRIORITY SCORE + BUDGET-CONSTRAINED SELECTION (decision support)
# --------------------------------------------------------------------------

VISION_2030_PILLARS: Dict[str, List[str]] = {
    "Economic Growth & Prosperity": ["Infrastructure", "Industrial Park",
                                     "Manufacturing Plant", "Real Estate",
                                     "Agriculture / Agro-Processing", "Other"],
    "Infrastructure & Utilities": ["Infrastructure", "Energy / Power",
                                   "Water / Sanitation",
                                   "Transport / Logistics",
                                   "Telecommunications"],
    "Human Capital Development": ["University / Education", "Hospital / Healthcare",
                                  "Innovation Hub / Technology"],
    "Science, Technology & Innovation": ["Innovation Hub / Technology",
                                        "Technology / Data Centre"],
    "Green Growth & Environment": ["Energy / Power", "Water / Sanitation",
                                   "Agriculture / Agro-Processing"],
}


def vision_pillar(project_type: str) -> List[str]:
    """Vision-2030 pillars a project type most directly supports."""
    out = []
    for pillar, types in VISION_2030_PILLARS.items():
        if project_type in types:
            out.append(pillar)
    if not out:
        out = ["Economic Growth & Prosperity"]  # fallback for unknown types
    return out


def funding_priority(bundle: AnalysisBundle,
                     vision_score: float = 7.0) -> Dict[str, Any]:
    """Objective funding-priority score 0-100 (decision support, not a vote).

    Components:
      value  (0-40)  NPV/PI based   - value created per unit of capital
      risk   (0-30)  risk level     - LOW 30 / MODERATE 18 / HIGH 0
      resil  (0-15)  scenario & stress resilience
      vision (0-15)  Vision-2030 alignment supplied by the user (0-10)
    """
    p = bundle.project_input
    m = bundle.metrics
    npv = m.npv
    pi = m.pi or 0.0

    # value
    if npv <= 0:
        value = 0.0
    else:
        value = min(40.0, 20.0 + max(0.0, (pi - 1.0)) * 20.0)

    # risk
    risk_pts = {"LOW": 30.0, "MODERATE": 18.0, "HIGH": 0.0}[bundle.risk.level]

    # resilience
    pes = bundle.scenarios["PESSIMISTIC"]["npv"]
    worst = min(bundle.stress, key=lambda s: s["npv"])["npv"]
    loss_ratio = max(0.0, -worst) / max(abs(npv), 1e-9)
    resil = 0.0
    if pes > 0:
        resil += 8.0
    if loss_ratio < 0.4:
        resil += 5.0
    elif loss_ratio < 0.8:
        resil += 2.0
    if m.break_even_multiplier is not None and m.break_even_multiplier >= 0.95:
        resil += 2.0
    resil = min(15.0, resil)

    # vision
    vision = min(15.0, 15.0 * max(0.0, min(10.0, vision_score)) / 10.0)

    score = round(value + risk_pts + resil + vision, 1)
    grade = ("HIGH PRIORITY" if score >= 75 else
             "MODERATE PRIORITY" if score >= 50 else "LOW PRIORITY")

    # flags
    flags: List[str] = []
    if bundle.risk.level == "HIGH":
        flags.append("High overrun risk - include a 25%+ contingency and "
                     "design-hold fund-release conditions.")
    if pes <= 0:
        flags.append("Fails the pessimistic case - require revenue guarantees "
                     "or offtake contracts before funds are released.")
    if loss_ratio > 0.8:
        flags.append(f"Stress loss exceeds 80% of base value ({loss_ratio:.0%}) "
                     "- cap the award size and require an added equity tranche.")
    if bundle.fx["fx_risk_level"] == "HIGH":
        flags.append("High foreign-exchange exposure - require an approved "
                     "hedging plan.")
    if m.payback and m.payback > p.project_life * 0.75:
        flags.append(f"Long payback ({m.payback:.1f}y) - use a longer loan "
                     "tenor and milestone-linked drawdowns.")
    if not flags:
        flags.append("No automatic risk flags - normal governance conditions "
                     "still apply.")

    return {
        "project_name": p.project_name,
        "reporting_currency": p.reporting_currency,
        "components": {"value": round(value, 1), "risk": round(risk_pts, 1),
                       "resilience": round(resil, 1), "vision": round(vision, 1)},
        "pillars": vision_pillar(p.project_type),
        "vision_score": vision_score,
        "score": score,
        "grade": grade,
        "flags": flags,
        "npv": npv,
        "investment": m.total_investment,
        "decision": bundle.decision.status,
    }


def knapsack_select(costs: List[float], values: List[float],
                    budget: float) -> Dict[str, Any]:
    """0/1 knapsack: pick items maximising total value within budget.

    Costs and the budget are scaled down to a bounded integer grid so the
    dynamic program stays fast even for very large amounts (e.g. UGX
    billions). The small rounding error is acceptable for decision support.
    Returns the selection indices, total cost and total value.
    """
    MAX_CAP = 200_000  # dp grid width cap
    n = len(costs)
    c0 = [int(round(x)) for x in costs]
    v = [float(x) for x in values]
    cap0 = int(round(budget))
    if n == 0 or cap0 <= 0:
        return {"selected": [], "total_cost": 0.0, "total_value": 0.0,
                "capacity": budget, "unspent": budget}
    scale = max(1, int(math.ceil(cap0 / MAX_CAP)))
    cap = max(1, int(round(cap0 / scale)))
    c = [max(1, int(round(x / scale))) for x in c0]
    dp = [[0.0] * (cap + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for w in range(cap + 1):
            if c[i - 1] <= w:
                dp[i][w] = max(dp[i - 1][w], dp[i - 1][w - c[i - 1]] + v[i - 1])
            else:
                dp[i][w] = dp[i - 1][w]
    sel: List[int] = []
    w = cap
    for i in range(n, 0, -1):
        if dp[i][w] != dp[i - 1][w]:
            sel.append(i - 1)
            w -= c[i - 1]
    sel.reverse()
    used = sum(c[i] for i in sel) * scale
    return {
        "selected": sel,
        "total_cost": used,
        "total_value": sum(values[i] for i in sel),
        "capacity": budget,
        "unspent": max(0.0, budget - used),
    }


def demo_project() -> ProjectInput:
    p = ProjectInput(
        project_name="GZU Mashava Campus Innovation Hub",
        organisation="Great Zimbabwe University",
        country="Zimbabwe",
        location="Mashava",
        project_type="University / Education",
        description=("Hypothetical mixed-use innovation and incubation hub at "
                     "the Great Zimbabwe University Mashava campus, combining "
                     "teaching, research and SME incubation space. DEMO / "
                     "HYPOTHETICAL PROJECT - not actual GZU financial data."),

        reporting_currency="USD",
        construction_cost=9_000_000,
        equipment_cost=6_000_000,
        land_building_cost=2_000_000,
        working_capital=1_000_000,
        salvage_value=2_500_000,

        annual_revenue=8_500_000,
        revenue_growth=5.0,
        revenue_currency="USD",
        revenue_fx_pa=0.0,
        annual_opex=60_000_000,        # local expenses denominated in ZiG
        opex_growth=4.0,
        opex_currency="ZiG",
        opex_fx_pa=8.0,
        annual_maintenance=9_000_000,  # SA-maintained equipment / maintenance in ZAR
        maintenance_currency="ZAR",
        maintenance_fx_pa=3.0,
        inflation_rate=6.0,

        project_life=15,
        construction_period=2,
        expected_delay_years=0.3,
        tax_rate=25.0,
        discount_rate=12.0,
        debt_ratio=40.0,
        loan_interest_rate=9.0,
        loan_term=8,

        construction_currency="USD",
        construction_fx_pa=0.0,
        construction_supplier_country="Zimbabwe",
        equipment_currency="EUR",
        equipment_fx_pa=1.0,
        equipment_supplier_country="Germany",
        land_currency="USD",
        land_fx_pa=0.0,
        land_supplier_country="Zimbabwe",

        materials_supplier_country="South Africa",
        materials_currency="ZAR",
        materials_fx_pa=3.0,
        import_transport_pct=12.0,
        import_insurance_pct=3.0,
        import_duty_pct=15.0,
        import_taxes_pct=15.0,
        conversion_cost_pct=2.0,
        import_financing_pct=5.0,

        market_risk=1,
        construction_risk=1,
        operating_risk=1,
        country_risk=2,
        currency_volatility=2,
        supplier_risk=1,
    )
    return p


def demo_csv_bytes() -> bytes:
    """CSV build of the demo project for users who want to inspect/upload it."""
    p = demo_project()
    df = pd.DataFrame([{k: v for k, v in p.__dict__.items()
                        if k != "fx_rates" and k != "fx_rate_timestamp"}])
    for c in EXPECTED_COLUMNS:
        if c not in df.columns:
            df[c] = ""
    return df[EXPECTED_COLUMNS].to_csv(index=False).encode("utf-8")