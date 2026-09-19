# Risk Methodology

## Approach

CAPEXX risk analysis is **deterministic and sensitivity-based**. It does not
pretend to be a machine-learning prediction unless real training data and a
trained model are explicitly connected. In the current build the scoring is a
transparent function of:

1. the user's stated risk profile (per category: Low / Moderate / High),
2. the measured NPV sensitivity to each driver,
3. structural parameters (construction length, delay expectation, debt ratio,
   currency exposure, inflation).

## Categories scored (0-100)

| Risk | Inputs used |
|---|---|
| Cost overrun | capex sensitivity, construction-risk profile, capex size |
| Construction delay | construction-risk profile, expected delay, length of construction |
| Revenue shortfall | market-risk profile, revenue sensitivity |
| Operating-cost increase | operating-risk profile, opex sensitivity |
| Inflation | inflation assumption, country-risk profile |
| Foreign exchange | currency-mismatch ratio × expected FX movement, currency volatility |
| Interest rate | debt ratio, gap between loan rate and WACC |
| Cash-flow stress | worst stress-case NPV vs base NPV |
| Project completion | delay risk + construction-risk profile |
| Supplier | supplier-risk profile, dependence on international suppliers |

## Aggregation

A weighted average produces an overall score:

- LOW      (< 35)
- MODERATE (35 - 65)
- HIGH     (> 65)

The weights are documented in `capexx_engine.run_risk_engine`.

## Stress testing

A fixed battery of shocks is applied through the one central cash-flow engine:

CAPEX +10/+20/+30%, Revenue −10/−20/−30%, OPEX +10/+20%, project delay +1yr,
FX depreciation −10%, FX appreciation +10%, inflation +5 points, material
price +15%, transport cost +20%, and a combined adverse case.

Each stress row reports the new NPV, IRR, MIRR, payback and the impact versus
base NPV, so management can see *which* shock hurts most.

## Real data vs synthetic data

- Deterministic scores = clearly labelled **"sensitivity-based risk scoring"**.
- Any future trained model = labelled **"REAL DATA MODEL"**.
- All demo figures = labelled **"SYNTHETIC DEMONSTRATION DATA"**.
- The two are never merged or presented as the same thing.