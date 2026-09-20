"""RISK & STRESS-TESTING MODEL — base/optimistic/pessimistic/custom scenarios,
configurable shocks (inflation, FX, interest rates, revenue, costs, delays),
tornado sensitivity and Monte Carlo simulation.

The risk engine re-runs the *entire* interconnected chain through the central
engine's `solve_npv` callback, so every scenario produces a fully recomputed
business-case NPV rather than a crude linear approximation.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import profitability, cashflow, investment

STRESS_PRESETS = [
    {"name": "Revenue shock \u221220%", "rev": 0.80, "desc": "Demand collapses 20% below plan."},
    {"name": "Variable costs +20%", "var": 1.20, "desc": "Materials and variable costs rise 20%."},
    {"name": "Fixed opex +15%", "fixed": 1.15, "desc": "Overheads rise 15% above plan."},
    {"name": "Price cut \u221210%", "rev": 0.90, "desc": "Selling price falls 10% (COGS unchanged)."},
    {"name": "Volume \u221210%", "vol": 0.90, "desc": "Units sold fall 10% (variable costs fall too)."},
    {"name": "Inflation +3pp", "fixed": 1.03, "var": 1.03, "desc": "Cost inflation 3 percentage points higher."},
    {"name": "Interest rate +3pp (WACC)", "wacc_pp": 3.0, "desc": "Discount rate rises 3 percentage points."},
    {"name": "FX / currency \u221215%", "rev": 0.85, "var": 1.15, "desc": "Reporting currency strengthens 15% vs revenue currencies; imports cost 15% more."},
    {"name": "Capex +25%", "capex": 1.25, "desc": "Initial capital spend overruns by 25%."},
    {"name": "Project delay 1 year", "delay": 1.0, "desc": "Returns arrive one full year late."},
]

SCENARIO_PRESETS = {
    "Base": {"rev": 1.00, "var": 1.00, "fixed": 1.00,
             "desc": "Everything proceeds exactly as planned."},
    "Optimistic": {"rev": 1.15, "var": 0.92, "fixed": 0.92,
                   "desc": "Revenue 15% above plan, variable costs 8% lower, fixed costs 8% lower."},
    "Pessimistic": {"rev": 0.85, "var": 1.12, "fixed": 1.10,
                    "desc": "Revenue 15% below plan, variable costs 12% higher, fixed costs 10% higher."},
}

TORNADO_DRIVERS = [
    {"driver": "Revenue", "low": {"rev": 0.88}, "high": {"rev": 1.12}},
    {"driver": "Variable costs", "low": {"var": 0.88}, "high": {"var": 1.12}},
    {"driver": "Fixed opex", "low": {"fixed": 0.88}, "high": {"fixed": 1.12}},
    {"driver": "Volume", "low": {"vol": 0.88}, "high": {"vol": 1.12}},
    {"driver": "WACC", "low": {"wacc_pp": -2.0}, "high": {"wacc_pp": 2.0}},
    {"driver": "Project delay", "low": {"delay": 0.0}, "high": {"delay": 1.0}},
]


def apply_multipliers(ci, *, rev=1.0, var=1.0, fixed=1.0, vol=1.0,
                      wacc_pp=0.0, delay=0.0, capex=1.0):
    """Return a modified copy of ``ci`` with the requested shocks applied."""
    import copy
    m = copy.deepcopy(ci)
    m.revenue = np.asarray(ci.revenue, dtype=float) * rev * vol
    m.cogs = np.asarray(ci.cogs, dtype=float) * var * vol
    m.opex = np.asarray(ci.opex, dtype=float) * fixed
    m.capex = np.asarray(ci.capex, dtype=float) * capex
    m.wacc = float(ci.wacc + wacc_pp / 100.0)
    m.delay_years = float((ci.delay_years or 0.0) + delay)
    if ci.units is not None:
        m.units = np.asarray(ci.units, dtype=float) * vol
    if ci.price is not None:
        m.price = np.asarray(ci.price, dtype=float) * rev
    if ci.unit_var_cost is not None:
        m.unit_var_cost = np.asarray(ci.unit_var_cost, dtype=float) * var
    return m


def _npv_with_delay(ci, npv_like):
    """Business-case NPV for a (possibly delay-shifted) model.

    ``npv_like`` must compute the business NPV for a company input given a
    **shift applied to the operating return timeline**.
    """
    return npv_like(ci)


@dataclass
class RiskResult:
    company_name: str
    currency: str
    base_npv: float
    base_irr: float | None
    scenarios: list[dict] = field(default_factory=list)
    stress: list[dict] = field(default_factory=list)
    tornado: list[dict] = field(default_factory=list)
    mc_mean: float | None = None
    mc_p5: float | None = None
    mc_p50: float | None = None
    mc_p95: float | None = None
    mc_prob_negative: float | None = None
    mc_samples: np.ndarray = field(default_factory=lambda: np.array([]))
    mc_iterations: int = 0
    assumptions: dict = field(default_factory=dict)
    stress_tolerance_message: str = ""


def compute(ci, npv_like, *, custom: dict | None = None,
            mc_iterations: int = 1000, seed: int = 2026) -> RiskResult:
    """Run scenario, stress, tornado and Monte-Carlo analysis.

    ``npv_like(ci)`` is a callable provided by the central engine that returns
    the business-case NPV for a modified :class:`CompanyInput` by re-running
    the full interconnected chain.
    """
    base_npv = float(npv_like(ci))
    base_irr = None

    scenarios = []
    for name, cfg in list(SCENARIO_PRESETS.items()):
        m = apply_multipliers(ci, rev=cfg["rev"], var=cfg["var"],
                              fixed=cfg["fixed"])
        scenarios.append({
            "name": name,
            "description": cfg["desc"],
            "npv": float(npv_like(m)),
            "revenue_mult": cfg["rev"],
            "cost_var_mult": cfg["var"],
        })

    # ── custom user scenario ─────────────────────────────────────────────
    if custom:
        m = apply_multipliers(ci,
                              rev=custom.get("rev_mult", 1.0),
                              var=custom.get("var_mult", 1.0),
                              fixed=custom.get("fixed_mult", 1.0),
                              vol=custom.get("vol_mult", 1.0),
                              wacc_pp=custom.get("wacc_pp", 0.0),
                              delay=custom.get("delay", 0.0))
        name = custom.get("name", "Custom scenario")
        scenarios.append({
            "name": name,
            "description": custom.get("desc", "User-defined scenario."),
            "npv": float(npv_like(m)),
            "revenue_mult": custom.get("rev_mult", 1.0),
            "cost_var_mult": custom.get("var_mult", 1.0),
        })

    # ── stress tests (one driver shocked at a time) ──────────────────────
    stress = []
    for s in STRESS_PRESETS:
        m = apply_multipliers(ci, rev=s.get("rev", 1.0), var=s.get("var", 1.0),
                              fixed=s.get("fixed", 1.0), vol=s.get("vol", 1.0),
                              wacc_pp=s.get("wacc_pp", 0.0),
                              delay=s.get("delay", 0.0),
                              capex=s.get("capex", 1.0))
        npv_s = float(npv_like(m))
        stress.append({
            "name": s["name"],
            "description": s["desc"],
            "impact": npv_s - base_npv,
            "impact_pct": (npv_s - base_npv) / base_npv * 100 if base_npv != 0 else 0.0,
            "npv": npv_s,
        })

    # ── tornado sensitivity ──────────────────────────────────────────────
    tornado = []
    for d in TORNADO_DRIVERS:
        ml = apply_multipliers(ci, **d["low"])
        mh = apply_multipliers(ci, **d["high"])
        nv_low = float(npv_like(ml))
        nv_high = float(npv_like(mh))
        tornado.append({
            "driver": d["driver"],
            "npv_low": nv_low,
            "npv_high": nv_high,
            "delta": nv_high - nv_low,
            "worst_index": "low" if nv_low < nv_high else "high",
        })
    tornado.sort(key=lambda t: -abs(t["delta"]))

    # ── Monte Carlo ──────────────────────────────────────────────────────
    mc_mean = mc_p5 = mc_p50 = mc_p95 = mc_neg = None
    samples = np.array([])
    if mc_iterations and mc_iterations > 0:
        rng = np.random.default_rng(seed)
        samples = np.empty(mc_iterations)
        for i in range(mc_iterations):
            rev_m = 1.0 + rng.normal(0.0, 0.08)
            var_m = 1.0 + rng.normal(0.0, 0.06)
            fixed_m = 1.0 + rng.normal(0.0, 0.05)
            wacc_pp = rng.normal(0.0, 1.5)
            m = apply_multipliers(ci, rev=rev_m, var=var_m, fixed=fixed_m,
                                  wacc_pp=wacc_pp)
            samples[i] = float(npv_like(m))
        if len(samples):
            samples = np.sort(samples)
            mc_mean = float(samples.mean())
            mc_p5 = float(np.percentile(samples, 5))
            mc_p50 = float(np.percentile(samples, 50))
            mc_p95 = float(np.percentile(samples, 95))
            mc_neg = float((samples < 0).mean() * 100)

    # ── plain-language risk posture ──────────────────────────────────────
    worst_stress = min(stress, key=lambda s: s["npv"]) if stress else None
    if worst_stress and worst_stress["npv"] < 0:
        tolerance = (f"The worst stress case (\u201c{worst_stress['name']}\u201d) "
                     f"turns the plan negative (NPV "
                     f"{worst_stress['npv']:,.0f}). Review contingency buffers.")
    elif mc_p5 is not None and mc_p5 < 0:
        tolerance = ("Monte Carlo shows a 5th-percentile outcome below zero. "
                     "The plan carries meaningful downside risk.")
    else:
        tolerance = "The plan withstands the tested stress cases at the 5th percentile."

    assumptions = {
        "tornado_range_pct": "\u00B112% for revenue/costs/volume; \u00B12pp WACC",
        "monte_carlo_iterations": mc_iterations,
        "mc_shocks": "revenue \u03C3=8%, variable costs \u03C3=6%, fixed \u03C3=5%, WACC \u03C3=1.5pp",
        "stress_scope": "single-driver shocks on the full model",
        "base_npv": base_npv,
    }

    return RiskResult(
        company_name=ci.company_name, currency=ci.currency,
        base_npv=base_npv, base_irr=base_irr, scenarios=scenarios, stress=stress,
        tornado=tornado, mc_mean=mc_mean, mc_p5=mc_p5, mc_p50=mc_p50,
        mc_p95=mc_p95, mc_prob_negative=mc_neg, mc_samples=samples,
        mc_iterations=mc_iterations, assumptions=assumptions,
        stress_tolerance_message=tolerance,
    )


def recompute_metrics(ci) -> dict:
    """Quick full-chain recompute used by risk scenarios via the engine."""
    prof = profitability.compute(ci)
    cf = cashflow.compute(ci, prof)
    bc = investment.compute_business_case(ci, prof, cf)
    return {"bc": bc, "npv": bc.npv_, "irr": bc.irr_,
            "net_income": prof.net_income, "fcf": cf.fcf,
            "cash_balance": cf.cash_balance}