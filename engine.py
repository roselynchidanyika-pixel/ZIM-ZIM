"""CENTRAL FINANCIAL MODELLING ENGINE — the single entry point that wires all
five model engines together.

Outputs from one model feed the next automatically:

    Profitability -> Cash Flow -> Investment -> Risk -> Optimization

The central engine owns the :class:`CompanyInput` data model, knows how to
recompute the full chain for arbitrary shock-modified inputs (used by the risk
engine so every scenario is a true full-model recompute), and produces a single
:class:`AnalysisState` consumed by every dashboard page.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .models import profitability as _profitability
from .models import cashflow as _cashflow
from .models import investment as _investment
from .models import risk as _risk
from .models import optimization as _optimization
from .models.investment import ProjectSpec

DEFAULT_TERMINAL_GROWTH = 0.0


# ── the unified input data model ─────────────────────────────────────────
@dataclass
class CompanyInput:
    company_name: str = "My Company"
    currency: str = "USD"
    periods: int = 8
    revenue: np.ndarray = field(default_factory=lambda: np.array([]))
    cogs: np.ndarray = field(default_factory=lambda: np.array([]))
    opex: np.ndarray = field(default_factory=lambda: np.array([]))
    d_and_a: np.ndarray = field(default_factory=lambda: np.array([]))
    capex: np.ndarray = field(default_factory=lambda: np.array([]))
    units: np.ndarray | None = None
    price: np.ndarray | None = None
    unit_var_cost: np.ndarray | None = None
    tax_rate: float = 0.25
    wacc: float = 0.12
    inflation: float = 0.03
    receivable_days: float = 45.0
    payable_days: float = 35.0
    inventory_days: float = 30.0
    initial_cash: float = 2_000_000.0
    debt_ratio: float = 0.40
    debt_interest_rate: float = 0.09
    delay_years: float = 0.0
    terminal_growth: float = DEFAULT_TERMINAL_GROWTH

    def validate(self, project_input: bool = False) -> list[str]:
        errs: list[str] = []
        n = self.periods
        for label, arr, required in (("Revenue", self.revenue, True),
                                     ("COGS", self.cogs, True),
                                     ("Fixed opex", self.opex, True),
                                     ("Depreciation & amortisation", self.d_and_a, False),
                                     ("Capex", self.capex, True)):
            if arr is None or len(arr) == 0:
                if required:
                    errs.append(f"{label} is missing or empty.")
                continue
            a = np.asarray(arr, dtype=float)
            if len(a) != n:
                errs.append(f"{label} must have exactly {n} values "
                            f"(found {len(a)}).")
            if np.any(np.isnan(a)):
                errs.append(f"{label} contains blank/NaN values.")
            if label == "Revenue" and np.any(a <= 0):
                errs.append("Revenue must be positive in every period.")
        if not (0 <= self.tax_rate <= 1):
            errs.append("Tax rate must be between 0% and 100%.")
        if not (0 < self.wacc < 1):
            errs.append("Discount rate (WACC) must be between 0% and 100%.")
        if self.periods < 2:
            errs.append("At least 2 periods are required.")
        return errs


def default_company_input(**overrides) -> CompanyInput:
    """A coherent default company spanning :func:`sample_company` values."""
    from .data.samples import sample_company
    ci = sample_company()
    for k, v in overrides.items():
        setattr(ci, k, v)
    return ci


# ── project catalogue helpers ────────────────────────────────────────────
def projects_as_specs(projects: list[dict]) -> list[ProjectSpec]:
    specs = []
    for p in projects:
        cf = np.asarray(p.get("cashflows", []), dtype=float)
        if len(cf) == 0:
            continue
        specs.append(ProjectSpec(
            name=p.get("name", "Project"),
            cost=float(p.get("cost", 0.0)),
            cashflows=cf,
            currency=p.get("currency", "USD"),
            description=p.get("description", ""),
        ))
    return specs


# ── the full analysis state (single gateway object) ──────────────────────
@dataclass
class AnalysisState:
    company: CompanyInput
    profitability: _profitability.ProfitabilityResult
    cashflow: _cashflow.CashFlowResult
    business_case: _investment.BusinessCaseMetrics
    projects: list[ProjectSpec] = field(default_factory=list)
    project_metrics: list[_investment.ProjectMetrics] = field(default_factory=list)
    risk: _risk.RiskResult | None = None
    optimization: _optimization.OptimizationResult | None = None
    scenario_custom: dict = field(default_factory=dict)
    risk_config: dict = field(default_factory=dict)
    opt_config: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    computed_at: str = ""

    # ── interconnected recompute used by the risk engine ────────────────
    def solve_npv(self, ci) -> float:
        return _npv_of(ci)

    def recompute(self, ci) -> "AnalysisState":
        return run_full_analysis(ci, projects=self.projects,
                                 scenario_custom=self.scenario_custom,
                                 risk_config=self.risk_config,
                                 opt_config=self.opt_config)

    def run_risk(self, custom=None, mc_iterations=1000, seed=2026) -> AnalysisState:
        cfg = dict(self.risk_config)
        cfg["mc_iterations"] = mc_iterations
        st = run_full_analysis(self.company, projects=self.projects,
                               scenario_custom=custom or self.scenario_custom,
                               risk_config=cfg, opt_config=self.opt_config)
        return st or self


def _npv_of(ci) -> float:
    """Standalone full-chain NPV for a company input (used by risk engine)."""
    prof = _profitability.compute(ci)
    cf = _cashflow.compute(ci, prof)
    bc = _investment.compute_business_case(ci, prof, cf)
    return bc.npv_


def run_full_analysis(company: CompanyInput,
                      projects: list[ProjectSpec] | None = None,
                      scenario_custom: dict | None = None,
                      risk_config: dict | None = None,
                      opt_config: dict | None = None,
                      computed_at: str = "") -> AnalysisState:
    """Run ALL five interconnected models and return one :class:`AnalysisState`."""
    prof = _profitability.compute(company)
    cf = _cashflow.compute(company, prof)
    bc = _investment.compute_business_case(company, prof, cf)

    specs = list(projects) if projects else []
    p_metrics = _investment.evaluate_projects(specs, company.wacc)

    rc = dict(risk_config or {})
    mc_iter = int(rc.get("mc_iterations", 1000))
    seed = int(rc.get("seed", 2026))
    risk = _risk.compute(company, _npv_of,
                         custom=scenario_custom or None,
                         mc_iterations=mc_iter,
                         seed=seed)

    ocfg = dict(opt_config or {})
    opt = _optimization.compute(company, p_metrics,
                                selection_budget=ocfg.get("selection_budget"),
                                price_params=ocfg.get("price_params"),
                                allocation_params=ocfg.get("allocation_params"))

    warnings = []
    if company.terminal_growth and abs(company.terminal_growth) > 0:
        warnings.append("Terminal growth is set but not applied inside project "
                        "metrics; it only informs the DCF narrative.")

    return AnalysisState(
        company=company, profitability=prof, cashflow=cf, business_case=bc,
        projects=specs, project_metrics=p_metrics, risk=risk,
        optimization=opt, scenario_custom=scenario_custom or {},
        risk_config=rc, opt_config=ocfg, warnings=warnings,
        computed_at=computed_at,
    )