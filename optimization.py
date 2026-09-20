"""OPTIMIZATION MODEL — optimise investment selection, pricing, costs and
capital allocation subject to user-defined constraints.

* Project selection  — binary MILP maximising total NPV under a capital budget.
* Price optimisation — isoelastic demand, closed-form profit-maximising price.
* Capital allocation — linear programming across business units with caps.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .investment import ProjectMetrics

GREEN = "#157347"
GOLD = "#C9A227"


@dataclass
class SelectionResult:
    budget: float
    used_budget: float
    idling: float
    selected: list[str]
    total_npv: float
    base_npv: float
    uplift_npv: float


@dataclass
class PricingResult:
    current_price: float
    optimal_price: float
    price_change_pct: float
    elasticity: float
    current_contribution: float
    optimal_contribution: float
    uplift_contribution: float
    uplift_pct: float
    current_volume: float
    projected_volume: float


@dataclass
class AllocationResult:
    budget: float
    used: float
    units: list[str]
    allocation: list[float]
    returns_pct: list[float]
    total_return: float
    caps: list[float]


@dataclass
class OptimizationResult:
    selection: SelectionResult | None = None
    pricing: PricingResult | None = None
    allocation: AllocationResult | None = None
    products: list[dict] = field(default_factory=list)


# ── project selection (binary MILP) ──────────────────────────────────────
def select_projects(metrics: list[ProjectMetrics], budget: float) -> SelectionResult:
    """Maximise total NPV of selected projects subject to the budget."""
    budget = max(budget, 0.0)
    base_npv = sum(max(m.npv_, 0.0) for m in metrics)

    if not metrics or budget <= 0:
        return SelectionResult(budget=budget, used_budget=0.0, idling=0.0,
                               selected=[], total_npv=0.0, base_npv=base_npv,
                               uplift_npv=-base_npv)

    costs = np.array([abs(m.full_timeline[0]) if len(m.full_timeline) else 0.0
                      for m in metrics])
    npvs = np.array([m.npv_ for m in metrics])
    n = len(metrics)

    # drop projects that cannot fit individually
    feasible = costs <= budget + 1e-9
    if not np.any(feasible):
        return SelectionResult(budget=budget, used_budget=0.0, idling=budget,
                               selected=[], total_npv=0.0, base_npv=base_npv,
                               uplift_npv=-base_npv)

    c = np.where(feasible, -npvs, 0.0)                  # ignore non-fitting
    from scipy.optimize import milp, Bounds, LinearConstraint
    A = np.vstack([costs * feasible]).reshape(1, -1)
    lp = LinearConstraint(A, ub=np.array([budget]))
    bounds = Bounds(np.zeros(n), np.ones(n))
    integrality = np.ones(n)
    res = milp(c=c, constraints=lp, integrality=integrality, bounds=bounds)

    sel = np.zeros(n, dtype=bool)
    if res.success:
        sel = res.x.round() > 0.5

    selected = [m.name for m, s in zip(metrics, sel) if s]
    used = float(np.sum(costs[sel]))
    total_npv = float(np.sum(npvs[sel]))
    return SelectionResult(budget=budget, used_budget=used, idling=budget - used,
                           selected=selected, total_npv=total_npv,
                           base_npv=base_npv,
                           uplift_npv=total_npv - base_npv)


# ── price optimisation (isoelastic demand) ───────────────────────────────
def optimal_price(vc: float, elasticity: float) -> float:
    """Profit-maximising price for isoelastic demand Q = Q0 (p/p0)^e, e < -1.

    FOC gives  p* = e * vc / (e + 1).
    """
    if elasticity <= -1.0:
        return vc * elasticity / (elasticity + 1.0)
    return vc


def optimise_price(price: float, vc: float, volume: float,
                   elasticity: float) -> PricingResult:
    """Optimise selling price; contribution = (p - vc) Q(p)."""
    e = elasticity if elasticity < -1.0 else -2.0
    best = optimal_price(vc, e)
    cur_q = volume
    opt_q = cur_q * (best / price) ** e if price > 0 else cur_q
    cur_c = (price - vc) * cur_q
    opt_c = (best - vc) * opt_q
    return PricingResult(
        current_price=price, optimal_price=best,
        price_change_pct=(best / price - 1.0) * 100 if price > 0 else 0.0,
        elasticity=e, current_contribution=cur_c, optimal_contribution=opt_c,
        uplift_contribution=opt_c - cur_c,
        uplift_pct=(opt_c / cur_c - 1.0) * 100 if cur_c > 0 else 0.0,
        current_volume=cur_q, projected_volume=max(opt_q, 0.0),
    )


# ── capital allocation (LP) ──────────────────────────────────────────────
def allocate_capital(units: list[str], returns_pct: list[float],
                     budget: float, caps: list[float]) -> AllocationResult:
    """Maximise return \u03A3 r_i a_i subject to \u03A3 a_i \u2264 budget and 0 \u2264 a_i \u2264 cap_i."""
    n = len(units)
    if n == 0 or budget <= 0:
        return AllocationResult(budget=budget, used=0.0, units=units,
                                allocation=[0.0] * n, returns_pct=returns_pct,
                                total_return=0.0, caps=caps)
    rets = np.array(returns_pct, dtype=float)
    cap = np.array(caps, dtype=float)
    c = -rets
    A_ub = np.ones((1, n))
    b_ub = np.array([budget])
    bounds = [(0.0, min(caps[i], budget)) for i in range(n)]
    from scipy.optimize import linprog
    res = linprog(c=c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    alloc = np.zeros(n)
    if res.success:
        alloc = res.x
    tot = float(np.sum(np.asarray(alloc) * rets))
    return AllocationResult(budget=budget, used=float(np.sum(alloc)),
                            units=units, allocation=[float(a) for a in alloc],
                            returns_pct=returns_pct, total_return=tot, caps=caps)


def compute(ci, proj_metrics: list[ProjectMetrics] | None,
            *, selection_budget: float | None = None,
            price_params: dict | None = None,
            allocation_params: dict | None = None) -> OptimizationResult:
    """Run the optimisation bundle."""
    res = OptimizationResult()

    if proj_metrics:
        budget = selection_budget if selection_budget is not None else \
            float(np.sum([abs(m.full_timeline[0]) if len(m.full_timeline) else 0
                          for m in proj_metrics])) * 0.60
        res.selection = select_projects(proj_metrics, budget)

    if price_params is None and ci.price is not None and ci.unit_var_cost is not None:
        price_params = {"price": float(ci.price[0]),
                        "vc": float(ci.unit_var_cost[0]),
                        "volume": (float(ci.units[0]) if ci.units is not None else 1.0),
                        "elasticity": -2.5}
    if price_params and ci.price is not None and ci.unit_var_cost is not None:
        p = price_params
        price = p.get("price", float(ci.price[0]))
        vc = p.get("vc", float(ci.unit_var_cost[0]))
        vol = p.get("volume", float(ci.units[0]) if ci.units is not None else 1.0)
        e = p.get("elasticity", -2.5)
        res.pricing = optimise_price(price, vc, vol, e)

    if allocation_params:
        ap = allocation_params
        res.allocation = allocate_capital(
            ap.get("units", ["Division A", "Division B", "Division C"]),
            ap.get("returns_pct", [18.0, 12.0, 9.0]),
            ap.get("budget", 5_000_000.0),
            ap.get("caps", [2_000_000.0, 2_000_000.0, 2_000_000.0]))

    return res