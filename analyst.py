"""AI FINANCIAL ANALYST — turns numeric results into plain-language narrative,
answers questions about the five models, and produces read-aloud scripts.

The analyst is rule-based over the computed :class:`AnalysisState` (no network
call required), so every answer is mathematically consistent with the engine.
"""
from __future__ import annotations

from ..config import fmt_currency, fmt_pct, fmt_num
from ..engine import AnalysisState

ROBOT_WELCOME = "Welcome to RFC Securities. How may I help you?"

EXAMPLE_QUESTIONS = [
    "What is my net margin and break-even point?",
    "How much free cash flow will I generate?",
    "Is the investment NPV positive?",
    "What would happen in a pessimistic scenario?",
    "Which projects should I select under my budget?",
    "What is my recommended selling price?",
    "What does WACC mean and why do you use it?",
    "How is IRR different from MIRR?",
]


def _m(state: AnalysisState, v: float | None) -> str:
    return fmt_currency(v or 0.0, state.company.currency)


def _be_perf_text(state: AnalysisState) -> str:
    bc = state.business_case
    m = bc.break_even_performance_
    if m is None:
        return "Break-even performance could not be computed (returns are non-positive)."
    return (f"The business plan must deliver at least "
            f"{m * 100:.0f}% of the projected operating cash flows for the NPV "
            f"to stay above zero ({_m(state, bc.npv_)} today).")


def build_narrative(state: AnalysisState) -> dict[str, str]:
    """Return section-wise plain-language explanations, ready for TTS."""
    p = state.profitability
    cf = state.cashflow
    bc = state.business_case
    risk = state.risk
    opt = state.optimization
    ccy = state.company.currency

    rec = f"{p.revenue[-1]:,.0f}" if len(p.revenue) else "n/a"
    year1_rev = p.revenue[0]
    yN = p.years[-1] if p.years else 1

    overview = (
        f"{state.company.company_name} is modelled over {state.company.periods} "
        f"years in {state.company.currency}. Revenue starts at "
        f"{_m(state, year1_rev)} in Year 1 and reaches {_m(state, p.revenue[-1])} "
        f"by Year {yN}. The plan produces cumulative free cash flow of "
        f"{_m(state, cf.total_fcf)} with a closing cash balance of "
        f"{_m(state, cf.cash_balance[-1])}. The initial capital of "
        f"{_m(state, bc.initial_investment)} generates a business-case NPV of "
        f"{_m(state, bc.npv_)} at a {bc.wacc * 100:.1f}% discount rate, "
        f"which is {'attractive' if bc.npv_ >= 0 else 'not yet attractive'} "
        f"against the required return.")

    profitability = (
        f"Gross margin is {p.gross_margin:.1f} percent, operating margin "
        f"{p.operating_margin:.1f} percent and net margin {p.net_margin:.1f} "
        f"percent of revenue. Fixed costs of {_m(state, p.fixed_cost)} "
        f"per year require ")
    if p.breakeven_revenue is not None:
        profitability += (f"break-even revenue of {_m(state, p.breakeven_revenue)} "
                          f"(you are {p.margin_of_safety_pct:.0f} percent above it "
                          f"if the per-unit data is accurate). ")
    else:
        profitability += "break-even revenue that could not be computed because variable-cost data is incomplete. "
    if p.operating_leverage and p.operating_leverage > 1:
        profitability += f"Operating leverage of {p.operating_leverage:.1f}x means a 10 percent revenue swing moves operating profit by about {p.operating_leverage * 10:.0f} percent."

    cashflow = (
        f"Operating cash flow builds from {_m(state, cf.operating_cash_flow[0])} "
        f"in Year 1 to {_m(state, cf.operating_cash_flow[-1])} in Year "
        f"{yN}. Free cash flow totals {_m(state, cf.total_fcf)} over the plan. "
        f"Liquidity stays "
        f"{'healthy' if cf.current_ratio[0] >= 1.5 else 'tight'} with a Year 1 "
        f"current ratio of {cf.current_ratio[0]:.2f} and quick ratio of "
        f"{cf.quick_ratio[0]:.2f}. The closing cash balance is "
        f"{_m(state, cf.cash_balance[-1])}.")

    investment = (
        f"On the base business plan, NPV is {_m(state, bc.npv_)}, IRR is "
        f"{fmt_pct(bc.irr_ * 100) if bc.irr_ is not None else 'n/a'} versus a "
        f"{bc.wacc * 100:.1f} percent cost of capital, so the margin is "
        f"{'positive' if (bc.irr_ or 0) >= bc.wacc else 'negative'} "
        f"({((bc.irr_ or 0) - bc.wacc) * 100:+.1f} percentage points). "
        f"Profitability index is {bc.pi_:.2f} and payback is "
        f"{bc.payback_:.1f} years ({bc.discounted_payback_:.1f} years "
        f"discounted). ")
    investment += _be_perf_text(state)

    risk_text = ""
    if risk:
        risk_text = (
            f"Under the optimistic scenario NPV rises to "
            f"{_m(state, next(s['npv'] for s in risk.scenarios if s['name'] == 'Optimistic'))}; "
            f"under the pessimistic scenario it falls to "
            f"{_m(state, next(s['npv'] for s in risk.scenarios if s['name'] == 'Pessimistic'))}. "
            f"Monte Carlo across {risk.mc_iterations} paths gives a median NPV of "
            f"{_m(state, risk.mc_p50)} with a 5th-percentile outcome of "
            f"{_m(state, risk.mc_p5)} and a {risk.mc_prob_negative:.0f} percent "
            f"probability of a negative NPV. {risk.stress_tolerance_message}")

    opt_text = ""
    if opt and opt.selection:
        opt_text = (f"Optimising the project selection under a "
                    f"{_m(state, opt.selection.budget)} budget selects "
                    f"{len(opt.selection.selected)} project(s) — "
                    f"{', '.join(opt.selection.selected) or 'none'} — for "
                    f"{_m(state, opt.selection.total_npv)} of NPV against "
                    f"{_m(state, opt.selection.base_npv)} if all were taken.")

    return {"overview": overview, "profitability": profitability,
            "cashflow": cashflow, "investment": investment,
            "risk": risk_text, "optimization": opt_text}


def full_narrative(state: AnalysisState) -> str:
    parts = build_narrative(state)
    return " ".join(parts.values())


def section_summaries(state: AnalysisState) -> dict[str, dict]:
    """Structured metric headlines per dashboard section."""
    p, cf, bc = state.profitability, state.cashflow, state.business_case
    return {
        "Profitability": {
            "Net margin": fmt_pct(p.net_margin),
            "Gross margin": fmt_pct(p.gross_margin),
            "EBITDA margin": fmt_pct(p.ebitda_margin),
            "Break-even revenue": _m(state, p.breakeven_revenue),
            "Margin of safety": (f"{p.margin_of_safety_pct:.0f}%"
                                 if p.margin_of_safety_pct is not None else "n/a"),
        },
        "Cash Flow": {
            "Total FCF": _m(state, cf.total_fcf),
            "Closing cash": _m(state, cf.cash_balance[-1]),
            "Current ratio (Y1)": f"{cf.current_ratio[0]:.2f}",
            "Quick ratio (Y1)": f"{cf.quick_ratio[0]:.2f}",
        },
        "Investment": {
            "NPV": _m(state, bc.npv_),
            "IRR": fmt_pct(bc.irr_ * 100) if bc.irr_ is not None else "n/a",
            "MIRR": fmt_pct(bc.mirr_ * 100) if bc.mirr_ is not None else "n/a",
            "Payback": f"{bc.payback_:.1f} y",
            "PI": f"{bc.pi_:.2f}",
        },
        "Risk": {
            "Optimistic NPV": _m(state, next(s["npv"] for s in state.risk.scenarios
                                if s["name"] == "Optimistic")),
            "Pessimistic NPV": _m(state, next(s["npv"] for s in state.risk.scenarios
                                 if s["name"] == "Pessimistic")),
            "P5 NPV": _m(state, state.risk.mc_p5),
            "P(NPV<0)": f"{state.risk.mc_prob_negative:.0f}%" if state.risk.mc_prob_negative is not None else "n/a",
        },
        "Optimization": (
            {"Selected NPV": _m(state, state.optimization.selection.total_npv),
             "Budget used": _m(state, state.optimization.selection.used_budget)}
            if state.optimization and state.optimization.selection
            else {"note": "Run optimisation to populate."}
        ),
    }


# ── question answering ───────────────────────────────────────────────────
def answer_question(q: str, state: AnalysisState) -> str:
    """Answer a free-text question using the computed model results."""
    text = (q or "").lower()
    p, cf, bc = state.profitability, state.cashflow, state.business_case
    risk = state.risk
    opt = state.optimization
    ccy = state.company.currency

    if any(k in text for k in ("break-even", "breakeven", "break even")):
        if p.breakeven_revenue is not None:
            vu = (f" at {p.breakeven_units:,.0f} units" if p.breakeven_units is not None else "")
            return (f"Your break-even is {_m(state, p.breakeven_revenue)} of revenue{vu}. "
                    f"Profitability drivers reach break-even when contribution "
                    f"({p.cm_ratio:.1f}% of revenue) covers fixed costs of "
                    f"{_m(state, p.fixed_cost)} per year. Revenue is "
                    f"currently {_m(state, p.revenue[0])}, giving a margin of "
                    f"safety of {p.margin_of_safety_pct:.0f}%.")
        return (f"Break-even requires variable-cost or per-unit data. We know "
                f"fixed costs are {_m(state, p.fixed_cost)} per year, so "
                f"break-even revenue = fixed cost \u00F7 contribution margin "
                f"of {p.cm_ratio:.1f}% \u2248 {_m(state, p.fixed_cost / (p.cm_ratio / 100))}.")

    if any(k in text for k in ("irr", "internal rate")):
        if bc.irr_ is not None:
            verdict = ("exceeds" if bc.irr_ >= bc.wacc else "falls below")
            return (f"IRR is {fmt_pct(bc.irr_ * 100)}, which {verdict} your "
                    f"cost of capital of {fmt_pct(bc.wacc * 100)}. IRR is the "
                    f"discount rate that makes NPV exactly zero. MIRR, at "
                    f"{fmt_pct(bc.mirr_ * 100) if bc.mirr_ else 'n/a'}, "
                    f"re-invests interim cash flows at the cost of capital, "
                    f"giving a more conservative single number.")
        return "IRR could not be computed from the current cash-flow pattern (multiple or negative returns)."

    if "npv" in text:
        verdict = ("positive — value is created above the required return"
                   if bc.npv_ >= 0 else "negative — the required return is not earned")
        return (f"NPV is {_m(state, bc.npv_)} at a {fmt_pct(bc.wacc * 100)} "
                f"discount rate. The verdict: the business plan NPV is {verdict}. "
                f"NPV = \u03A3 CF\u209C \u00F7 (1+r)\u1D5B minus the initial "
                f"investment of {_m(state, bc.initial_investment)}.")

    if any(k in text for k in ("free cash", "cash flow", "fcf", "liquidity", "cash_f")):
        return (f"Free cash flow totals {_m(state, cf.total_fcf)} over "
                f"{state.company.periods} years — operating cash flow minus "
                f"capital expenditure. It starts at {_m(state, cf.operating_cash_flow[0])} "
                f"and ends at {_m(state, cf.operating_cash_flow[-1])}. The cash "
                f"balance closes at {_m(state, cf.cash_balance[-1])} with a "
                f"current ratio of {cf.current_ratio[0]:.2f} in Year 1.")

    if any(k in text for k in ("margin", "profitab", "gross", "net_margin")):
        return (f"Your margins are: gross {fmt_pct(p.gross_margin)}, EBITDA "
                f"{fmt_pct(p.ebitda_margin)}, operating {fmt_pct(p.operating_margin)}, "
                f"net {fmt_pct(p.net_margin)}. Operating leverage is "
                f"{p.operating_leverage:.1f}x if available. Net margin in year "
                f"{p.years[-1]} is estimated at "
                f"{p.net_income[-1] / p.revenue[-1] * 100:.1f}%.")

    if any(k in text for k in ("payback", "pay back")):
        return (f"Payback is {bc.payback_:.1f} years, and "
                f"{bc.discounted_payback_:.1f} years when cash flows are "
                f"discounted at {fmt_pct(bc.wacc * 100)}. Payback is the time "
                f"until cumulative cash recovered the "
                f"{_m(state, bc.initial_investment)} initial investment.")

    if any(k in text for k in ("scenario", "optimistic", "pessimistic", "stress", "risk")):
        lines = ["Scenario names, stresses and NPV results:"]
        for s in risk.scenarios:
            tag = "positive" if s["npv"] >= 0 else "negative"
            lines.append(f"- {s['name']}: {_m(state, s['npv'])} ({tag}).")
        worst = min(risk.stress, key=lambda s: s["npv"])
        lines.append(f"The largest single stress is '{worst['name']}' — "
                     f"NPV {_m(state, worst['npv'])}, a "
                     f"{worst['impact_pct']:+.1f}% move vs base.")
        lines.append(risk.stress_tolerance_message)
        return "\n".join(lines)

    if any(k in text for k in ("optim", "select", "budget", "portfolio")):
        if opt and opt.selection:
            return (f"Under a {_m(state, opt.selection.budget)} budget, the "
                    f"optimiser selects: {', '.join(opt.selection.selected) or 'none'}, "
                    f"using {_m(state, opt.selection.used_budget)} to capture "
                    f"{_m(state, opt.selection.total_npv)} of net present value.")
        return "Load the project catalogue, then open the Optimisation section to run selection."

    if any(k in text for k in ("price", "pricing")):
        if opt and opt.pricing:
            pr = opt.pricing
            return (f"The profit-maximising price is {fmt_currency(pr.optimal_price, ccy, 2)} "
                    f"vs the current {fmt_currency(pr.current_price, ccy, 2)} "
                    f"({pr.price_change_pct:+.1f}%). At elasticity "
                    f"{pr.elasticity:.1f}, projected contribution rises "
                    f"{pr.uplift_pct:+.1f}% to {fmt_currency(pr.optimal_contribution, ccy)}.")
        return "Pricing inputs are not set. Run the Optimisation section with price and elasticity data."

    if any(k in text for k in ("wacc", "discount rate")):
        return (f"WACC (weighted average cost of capital, {fmt_pct(bc.wacc * 100)}) "
                f"is the blended return investors require. It is used to "
                f"discount future cash flows: one dollar of return in Year 5 is "
                f"worth {_m(state, 1 / (1 + bc.wacc) ** 4 * p.revenue[-1] * 0.001)} "
                f"per thousand today at this rate — this is why future value "
                f"is worth less today.")

    if any(k in text for k in ("hi", "hello", "help", "what can", "about the models", "methodol")):
        return (ROBOT_WELCOME + " I can explain profitability, cash flow, "
                "investment metrics (NPV, IRR, MIRR, payback, PI, EAA), risk "
                "and stress scenarios, Monte-Carlo outcomes, and optimisation "
                "results — all from your actual model data.")

    return None


def answer_or_fallback(q: str, state: AnalysisState) -> tuple[str, bool]:
    ans = answer_question(q, state)
    if ans:
        return ans, True
    return (f"I can answer questions grounded in your RFC Securities model: "
            f"break-even, margins, free cash flow, NPV, IRR, MIRR, payback, "
            f"scenarios, stress tests, project selection, pricing and WACC. "
            f"Try one of the suggested questions."), False