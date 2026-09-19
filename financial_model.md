# Financial Model Documentation

## Purpose

`capexx_engine.py` contains the **only** financial arithmetic in CAPEXX.
Everything else - the dashboard, the robot, reports, email and audio - reads
from the results of `run_analysis()`.

## Central data structures

### `ProjectInput`
All inputs in one object. Monetary fields are expressed in the *reporting
currency* unless a currency field is attached to them (e.g. `equipment_currency`).

Key fields:

| Group | Fields |
|---|---|
| Identity | project_name, organisation, country, location, project_type, description |
| CAPEX | initial_capex, construction_cost, equipment_cost, land_building_cost, working_capital, salvage_value |
| Operating | annual_revenue, revenue_growth, annual_opex, opex_growth, annual_maintenance, inflation_rate |
| Timing | project_life, construction_period, expected_delay_years |
| Finance | tax_rate, discount_rate (WACC), debt_ratio, loan_interest_rate, loan_term |
| FX | `{construction|equipment|land|revenue|opex|maintenance|materials}_currency`, `..._fx_pa`, supplier countries |
| Imports | import_transport_pct, import_insurance_pct, import_duty_pct, import_taxes_pct, conversion_cost_pct, import_financing_pct |
| Risk profile | market_risk, construction_risk, operating_risk, country_risk, currency_volatility, supplier_risk (0/1/2) |
| Rates | fx_rates (units per USD), fx_rate_source, fx_rate_timestamp |

### `CashFlowModel`
A yearly timeline object containing construction capex, revenue, opex,
maintenance, depreciation, EBIT, tax, NOPAT, working-capital flows, salvage,
free cash flow, present values and cumulative lines.

### `AnalysisBundle`
Everything the UI needs: inputs, model, metrics, sensitivity, scenarios,
stress list, risk result, FX analysis, FX strategy lines, decision, and an
optional landed-cost build-up.

## Cash-flow construction (`build_cash_flow_model`)

Parameters are shock multipliers so that scenarios/stress reuse the very same
engine:

- `revenue_adj` - multiplier on revenue (1.0 = base)
- `capex_adj` - multiplier on capex components
- `opex_adj` - multiplier on opex + maintenance
- `delay_adj_years` - additional construction delay
- `fx_adjust` - uniform % FX shock on non-reporting currencies
- `hijack_rates` - override of the exchange-rate map

Timeline: `n_construction + n_operations` years. Discount engine applies
`1/(1+WACC)^year` to every year including construction years.

## Metrics (`compute_metrics`)

- NPV computed directly on the yearly free cash flows.
- IRR via `numpy_financial.irr` (converted from a fraction to percent).
- MIRR via `numpy_financial.mirr` with finance rate = loan rate, reinvestment
  rate = WACC (converted to percent).
- Payback / discounted payback with linear interpolation between years.
- ARR = average NOPAT / (investment / 2).
- PI = PV of positive flows / PV of investment outflows.
- DCF value = sum of discounted FCF.
- EAA = NPV / annuity factor over the operating life.
- Break-even = revenue multiplier driving NPV to zero (bisection).

## Demonstration project

`demo_project()` returns a hypothetical **GZU Mashava Campus Innovation Hub**
with multi-currency flows (ZiG local expenses, EUR equipment from Germany,
ZAR maintenance from South Africa). It is clearly labelled
**HYPOTHETICAL DEMONSTRATION - NOT ACTUAL GZU DATA**.