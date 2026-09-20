"""INVESTMENT & CAPITAL BUDGETING MODEL — NPV, IRR, MIRR, PI, ARR, payback,
discounted payback, DCF, EAA and break-even analysis.

Two layers:

* :func:`compute_business_case` evaluates the company's own investment plan:
  the upfront capital is the first-year capex plus the initial working
  capital build-up; the return stream is the projected operating cash flow.
  (All capital is treated as spent at t=0; all future capex is avoided by
  construction, keeping the valuation unambiguous.)
* :func:`evaluate_project` / :func:`evaluate_projects` evaluate explicit
  projects from the projects table (upfront cost + per-period cash flows).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .cashflow import CashFlowResult
from .profitability import ProfitabilityResult


# ── pure metric functions ────────────────────────────────────────────────
def pv(cashflows: np.ndarray, rate: float) -> float:
    """Present value of a cash-flow timeline at a constant discount rate."""
    if rate < -0.9999:
        return 0.0
    years = np.arange(len(cashflows), dtype=float)
    return float(np.sum(cashflows / (1.0 + rate) ** years))


def npv(cashflows: np.ndarray, rate: float) -> float:
    """NPV with cashflows[0] treated as the t=0 flow (usually the outlay)."""
    return pv(cashflows, rate)


def irr(cashflows: np.ndarray) -> float | None:
    """Internal rate of return (per unit, e.g. 0.12 = 12%)."""
    cf = np.asarray(cashflows, dtype=float)
    crossings: list[float] = []
    r = 1e-6
    f0 = pv(cf, r)
    for _ in range(80):
        nr = r * 1.6
        f1 = pv(cf, nr)
        if (f0 >= 0) != (f1 >= 0):
            try:
                from scipy.optimize import brentq
                sol = brentq(lambda x: pv(cf, x), r, nr,
                             xtol=1e-10, maxiter=200)
                crossings.append(float(sol))
            except Exception:
                pass
        r, f0 = nr, f1
        if r > 1e7:
            break
    if crossings:
        positives = [c for c in crossings if c > -0.9999]
        if positives:
            return float(min(positives))
        return float(min(crossings, key=abs))
    # fallback: polynomial roots
    try:
        roots = np.roots(cf[::-1])
        reals = np.real(roots[np.isclose(roots.imag, 0, atol=1e-9)])
        reals = reals[reals > -0.9999]
        if len(reals) == 0:
            return None
        reals = reals[np.argsort(np.abs(reals))]
        return float(reals[0])
    except Exception:
        return None


def mirr(cashflows: np.ndarray, finance_rate: float,
         reinvest_rate: float) -> float | None:
    """Modified internal rate of return."""
    cf = np.asarray(cashflows, dtype=float)
    n = len(cf) - 1
    if n <= 0 or finance_rate < -0.9999 or reinvest_rate < -0.9999:
        return None
    pv_out = sum(-cf[i] / (1 + finance_rate) ** i for i in range(len(cf)) if cf[i] < 0)
    fv_in = sum(cf[i] * (1 + reinvest_rate) ** (n - i) for i in range(len(cf)) if cf[i] > 0)
    if pv_out <= 0 or fv_in <= 0:
        return None
    return (fv_in / pv_out) ** (1.0 / n) - 1.0


def payback(cashflows: np.ndarray) -> float | None:
    """Payback period in years (linear interpolation)."""
    cf = np.asarray(cashflows, dtype=float)
    cum = np.cumsum(cf)
    if cum[-1] < 0:
        return None
    for i in range(1, len(cum)):
        if cum[i] >= 0 and cum[i - 1] < 0:
            frac = cum[i - 1] / (cum[i - 1] - cum[i])
            return float(i - 1 + frac)
    return float(len(cf) - 1)


def discounted_payback(cashflows: np.ndarray, rate: float) -> float | None:
    cf = np.asarray(cashflows, dtype=float)
    years = np.arange(len(cf))
    disc = cf / (1 + rate) ** years
    cum = np.cumsum(disc)
    if cum[-1] < 0:
        return None
    for i in range(1, len(cum)):
        if cum[i] >= 0 and cum[i - 1] < 0:
            frac = cum[i - 1] / (cum[i - 1] - cum[i])
            return float(i - 1 + frac)
    return float(len(cf) - 1)


def pi(cashflows: np.ndarray, rate: float) -> float | None:
    """Profitability index: PV of inflows / |initial outlay|."""
    cf = np.asarray(cashflows, dtype=float)
    if len(cf) == 0 or cf[0] >= 0:
        return None
    inflows = np.where(cf > 0, cf, 0.0)
    pv_in = pv(inflows, rate)
    return pv_in / abs(cf[0]) if cf[0] != 0 else None


def arr(cashflows: np.ndarray) -> float | None:
    """Approximate accounting rate of return (avg return / avg investment)."""
    cf = np.asarray(cashflows, dtype=float)
    if len(cf) < 2 or cf[0] >= 0:
        return None
    income = cf[1:].mean()
    avg_inv = abs(cf[0]) / 2.0
    if avg_inv == 0:
        return None
    return income / avg_inv


def eaa(cashflows: np.ndarray, rate: float) -> float | None:
    """Equivalent annual annuity from NPV."""
    n = len(cashflows) - 1
    if n <= 0 or rate <= -0.9999:
        return None
    npv_ = npv(cashflows, rate)
    if abs(rate) < 1e-12:
        af = float(n)
    else:
        af = (1.0 - (1.0 + rate) ** -n) / rate
    return npv_ / af if af != 0 else None


def dcf_value(cashflows: np.ndarray, rate: float) -> float:
    """Gross discounted cash-flow value (returns only, excluding outflows)."""
    return pv(np.where(np.asarray(cashflows) > 0, cashflows, 0.0), rate)


def break_even_performance(cashflows: np.ndarray, rate: float) -> float | None:
    """Multiplier m on the return stream that drives NPV to zero.

    NPV(m) = m * PV(returns) - |C0| = 0  =>  m = |C0| / PV(returns).
    """
    cf = np.asarray(cashflows, dtype=float)
    if len(cf) < 2 or cf[0] >= 0:
        return None
    pv_rets = pv(np.where(cf > 0, cf, 0.0), rate)
    if pv_rets <= 0:
        return None
    return abs(cf[0]) / pv_rets


# ── project / business-case wrappers ─────────────────────────────────────
@dataclass
class ProjectSpec:
    name: str
    cost: float
    cashflows: np.ndarray      # inflows per period (years 1..life)
    currency: str = "USD"
    description: str = ""


@dataclass
class ProjectMetrics:
    name: str
    currency: str
    full_timeline: np.ndarray  # [cost_neg, cf_1, ..., cf_life]
    life: int
    npv_: float
    irr_: float | None
    mirr_: float | None
    pi_: float | None
    arr_: float | None
    payback_: float | None
    discounted_payback_: float | None
    dcf_value_: float
    eaa_: float | None
    break_even_performance_: float | None

    def as_dict(self) -> dict:
        return {
            "Project": self.name,
            "NPV": self.npv_,
            "IRR %": (self.irr_ * 100) if self.irr_ is not None else None,
            "MIRR %": (self.mirr_ * 100) if self.mirr_ is not None else None,
            "PI": (self.pi_ if self.pi_ is not None else None),
            "ARR %": (self.arr_ * 100) if self.arr_ is not None else None,
            "Payback y": self.payback_,
            "Disc. payback y": self.discounted_payback_,
            "DCF value": self.dcf_value_,
            "EAA": self.eaa_,
            "Break-even performance x": self.break_even_performance_,
        }


def evaluate_project(proj: ProjectSpec, discount_rate: float,
                     finance_rate: float | None = None,
                     reinvest_rate: float | None = None) -> ProjectMetrics:
    """Compute full investment metrics for one project."""
    fr = finance_rate if finance_rate is not None else discount_rate
    rr = reinvest_rate if reinvest_rate is not None else discount_rate
    cf = np.asarray(proj.cashflows, dtype=float)
    timeline = np.concatenate(([-proj.cost], cf))
    return ProjectMetrics(
        name=proj.name,
        currency=proj.currency,
        full_timeline=timeline,
        life=len(cf),
        npv_=npv(timeline, discount_rate),
        irr_=irr(timeline),
        mirr_=mirr(timeline, fr, rr),
        pi_=pi(timeline, discount_rate),
        arr_=arr(timeline),
        payback_=payback(timeline),
        discounted_payback_=discounted_payback(timeline, discount_rate),
        dcf_value_=dcf_value(timeline, discount_rate),
        eaa_=eaa(timeline, discount_rate),
        break_even_performance_=break_even_performance(timeline, discount_rate),
    )


def evaluate_projects(projects: list[ProjectSpec], discount_rate: float,
                      finance_rate: float | None = None,
                      reinvest_rate: float | None = None) -> list[ProjectMetrics]:
    return [evaluate_project(p, discount_rate, finance_rate, reinvest_rate)
            for p in projects]


# ── business case (company plan) ─────────────────────────────────────────
@dataclass
class BusinessCaseMetrics(ProjectMetrics):
    initial_investment: float = 0.0
    return_stream: np.ndarray = field(default_factory=lambda: np.array([]))
    wacc: float = 0.0
    gross_returns_pv: float = 0.0


def compute_business_case(ci, prof: ProfitabilityResult,
                          cf: CashFlowResult,
                          discount_rate: float | None = None) -> BusinessCaseMetrics:
    """Evaluate the company's own investment plan as a single project.

    Timeline: CF[0] = -(first-year capex + initial working capital),
    CF[t]   = operating cash flow in year t (NI + D&A - \u0394NWC with the
    initial NWC build-up treated as part of CF[0]).
    """
    n = ci.periods
    rate = discount_rate if discount_rate is not None else ci.wacc
    capex = np.asarray(ci.capex, dtype=float)
    nwc = np.asarray(cf.net_working_capital, dtype=float)
    cred = np.maximum(capex[0], 0.0) + max(cf.initial_nwc, 0.0)   # C0

    # returns: NI_t + D&A_t - (nwc_t - nwc_{t-1}); year 1 change excluded
    returns = np.zeros(n)
    returns[0] = prof.net_income[0] + prof.d_and_a[0]
    for t in range(1, n):
        returns[t] = (prof.net_income[t] + prof.d_and_a[t] -
                      (nwc[t] - nwc[t - 1]))

    timeline = np.concatenate(([-cred], returns))
    fr = rate
    rr = rate
    return BusinessCaseMetrics(
        name=f"{ci.company_name} — business plan",
        currency=ci.currency,
        full_timeline=timeline,
        life=n,
        npv_=npv(timeline, rate),
        irr_=irr(timeline),
        mirr_=mirr(timeline, fr, rr),
        pi_=pi(timeline, rate),
        arr_=arr(timeline),
        payback_=payback(timeline),
        discounted_payback_=discounted_payback(timeline, rate),
        dcf_value_=dcf_value(timeline, rate),
        eaa_=eaa(timeline, rate),
        break_even_performance_=break_even_performance(timeline, rate),
        initial_investment=cred,
        return_stream=returns,
        wacc=rate,
        gross_returns_pv=pv(np.where(returns > 0, returns, 0.0), rate),
    )