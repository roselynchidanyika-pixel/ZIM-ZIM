# CAPEXX Methodology

## 1. What CAPEXX does

CAPEXX is a decision-support platform that answers one management question:

> **"Should this capital project be accepted, reviewed, or rejected based on its
> expected financial performance, risks, uncertainty, currencies, scenarios and
> cash flows?"**

The platform works for any type of capital project in any country -
infrastructure, manufacturing, hospitals, universities, innovation hubs, energy,
mining, technology, hotels, transport, agriculture and real estate.

## 2. One central engine (single source of truth)

All modules - capital budgeting, risk, currency, scenarios, stress testing,
decision, and the robot/email/report/audio outputs - consume the results of a
single cash-flow model built from the project inputs.

```
PROJECT INPUTS
     ↓
CENTRAL CASH FLOW ENGINE
     ↓
CAPITAL BUDGETING  →  RISK  →  CURRENCY/FX  →  SCENARIOS  →  STRESS
     ↓
DECISION ENGINE
     ↓
AI INTERPRETATION
     ↓  ┌──────────────┬──────────────┐
     ↓  ↓              ↓              ↓
  REPORT          EMAIL          ROBOT VOICE (audio)
```

If a single assumption changes, every dependent output is recalculated from
the same engine. There are no disconnected calculators.

## 3. Cash-flow model construction

1. Capital expenditure is split across the construction years using fixed
   weights (or a straight split for longer programmes) and adjusted for the
   currency of each component.
2. Working capital is invested in the final construction year and recovered in
   the final operating year.
3. Operating years project:
   - Revenue = base revenue × (1 + growth)^(t-1) × inflation escalation × FX.
   - Opex/maintenance = base × inflation escalation × FX.
   - Straight-line depreciation on depreciable assets (land excluded).
   - EBIT = Revenue − Opex − Maintenance − Depreciation.
   - Tax on positive EBIT.
   - NOPAT = EBIT − Tax.
4. Free cash flow per year = NOPAT + Depreciation − capex − ΔWC, plus salvage
   and working-capital recovery flows.
5. All years are discounted to today at the WACC (after-tax weighted average).

## 4. Capital budgeting metrics

| Metric | Definition |
|---|---|
| NPV | Σ FCFt/(1+r)^t − C0 |
| IRR | rate making NPV = 0 (multiple roots avoided; use MIRR) |
| MIRR | reinvestment at WACC, financing at debt rate - a single unambiguous return |
| Payback | years until cumulative cash flow turns positive |
| Discounted payback | same, using present values |
| ARR | average accounting profit / average book investment |
| PI | PV of inflows / PV of outflows |
| DCF value | sum of discounted free cash flows |
| EAA | annuity equivalent of NPV over the project life |
| Break-even | revenue multiplier that drives NPV to zero |

Each metric is presented with its method, result and interpretation.

## 5. Risk engine (deterministic)

Risk categories are scored 0-100 from the input risk profile and the measured
sensitivity of NPV:

- Cost overrun, construction delay, revenue shortfall, operating-cost increase,
  inflation, FX, interest-rate, cash-flow stress, project completion, supplier.

The overall score is a weighted combination and is labelled LOW (<35),
MODERATE (35-65) or HIGH (>65).

**Honesty**: this is a deterministic sensitivity-based score. It is clearly
labelled as such and is never presented as a machine-learning prediction on
real market data. If a real-data machine-learning layer is later added, it will
be explicitly labelled "REAL DATA MODEL" and kept separate from any synthetic
demonstration data.

## 6. Decision engine (transparent rules)

Three pillars feed a decision score:

- **Financial (max 50):** NPV>0 (+14), IRR≥WACC (+12), MIRR≥WACC (+8),
  PI≥1.1 (+8), payback≤50% of life (+8).
- **Risk (max 30):** LOW +30, MODERATE +15, HIGH +0.
- **Scenario (max 20):** base & pessimistic cases positive (+12); stress loss
  contained ≤80% of base value (+8).

Overrides:
- Base NPV ≤ 0 → REJECT.
- Overall risk HIGH → REJECT.
- Positive base but failing pessimistic/stress conditions → REVIEW.

Outcome: `≥75 ACCEPT`, `50-74 REVIEW`, `<50 REJECT`. Every rule that was
evaluated is shown to the user - the logic is never hidden.

## 7. Currency / FX engine (country-agnostic)

- Any country, any supplier, any currency, any reporting currency.
- Exchange rates are expressed as units of each currency per 1 USD so all
  conversions are consistent and transparent.
- Rates may be (a) fetched live from a public keyless API at run time or
  (b) entered by the user. Neither option is ever presented as an invented
  constant.
- Each capex/opex/revenue component can carry its own currency, supplier
  country, expected annual FX movement, and payment context.
- The engine computes currency mismatches, exposure amounts in the reporting
  currency, an FX risk score/level, hedging notes and a transaction-level
  currency strategy.
- A generic landed-cost build-up is provided:
  Purchase + Transport + Insurance + Duties + Taxes + Conversion costs
  + Financing/FX effects = Landed cost.

## 8. Scenarios & stress

- BASE, OPTIMISTIC, PESSIMISTIC, EXTREME STRESS are computed with the same
  engine under explicit multiplier shocks.
- A pre-defined stress battery includes capex +10/20/30%, revenue −10/20/30%,
  opex +10/20%, project delay, FX depreciation/appreciation, inflation,
  material and transport cost increases, and a combined adverse case.
- Sensitivity (tornado) analysis varies each key driver one at a time.

## 9. AI interpretation & the robot

Interpretation follows five questions for every major result:

1. **WHAT HAPPENED?** - describe the result.
2. **WHY DID IT HAPPEN?** - identify the main drivers.
3. **SO WHAT?** - the financial meaning.
4. **WHAT COULD CHANGE IT?** - scenarios/stress that move the result.
5. **WHAT SHOULD MANAGEMENT MONITOR?** - factual monitoring points.

Every AI conclusion is traceable: Input → Calculation → Evidence → Interpretation.
The robot is a communication layer only - it never calculates; it narrates the
engine's actual numbers. Audio is generated with gTTS when available and the
interface degrades gracefully to a text script otherwise.

## 10. Data transparency

- External/market data points display: **Source | Date | Status**.
- Live prices are only shown when a verified live source delivered them during
  the session; otherwise the label is
  **MARKET SIMULATION - DEMONSTRATION ONLY** or
  **DATA STATUS: DEMONSTRATION / USER-PROVIDED**.
- No economic or market figure is ever fabricated.