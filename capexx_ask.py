"""
CAPEXX AI AGENT - Global Investment Research Module ("ASK CAPEXX AI")
===================================================================
A professional investment research assistant, not a generic chatbot.

QUESTION -> UNDERSTAND -> SEARCH -> VERIFY -> ANALYSE -> EXPLAIN
         -> CITE -> REFERENCE -> (optionally speak)

Evidence rules:
- Every factual claim obtained from web research carries an inline citation
  to the real source.  No fabricated URLs / citations are ever produced.
- Current information is requested from public, dated sources at run time
  (exchange-rate APIs, World Bank country statistics, Google News RSS).
- When a source cannot be reached the module says so explicitly.
- Distinguishes CURRENT / RECENT / HISTORICAL / GENERAL knowledge.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from capexx_engine import COUNTRY_LIST, try_fetch_live_rates, SUPPORTED_CURRENCIES

# --------------------------------------------------------------------------
# KNOWLEDGE BASE - educational references (all genuine publications)
# --------------------------------------------------------------------------

CL = {
    "BrealeyMyers": ("Brealey, R., Myers, S., & Allen, F. (2020). "
                     "Principles of Corporate Finance. McGraw-Hill Education."),
    "Damodaran": ("Damodaran, A. (2012). Investment Valuation: Tools and "
                  "Techniques for Determining the Value of Any Asset. Wiley."),
    "Markowitz": ("Markowitz, H. (1952). Portfolio Selection. The Journal of "
                  "Finance, 7(1), 77-91."),
    "Sharpe": ("Sharpe, W. F. (1964). Capital Asset Prices: A Theory of Market "
               "Equilibrium under Conditions of Risk. The Journal of Finance, "
               "19(3), 425-442."),
    "MM": ("Modigliani, F., & Miller, M. H. (1958). The Cost of Capital, "
           "Corporation Finance and the Theory of Investment. American "
           "Economic Review, 48(3), 261-297."),
    "Graham": ("Graham, B., & Zweig, J. (2006). The Intelligent Investor. "
               "Collins Business."),
    "Bodie": ("Bodie, Z., Kane, A., & Marcus, A. J. (2018). Investments. "
              "McGraw-Hill Education."),
    "Hull": ("Hull, J. C. (2018). Options, Futures, and Other Derivatives. "
             "Pearson."),
    "DownsGoodman": ("Downes, J., & Goodman, J. E. (2010). Dictionary of "
                     "Finance and Investment Terms. Barron's."),
    "WorldBankGEP": ("World Bank (2026). Global Economic Prospects. "
                     "https://www.worldbank.org/en/publication/global-economic-prospects"),
    "IMF_WEO": ("International Monetary Fund (2026). World Economic Outlook. "
                "https://www.imf.org/en/Publications/WEO"),
    "BIS": ("Bank for International Settlements (n.d.). Statistical releases "
            "and annual economic reports. https://www.bis.org"),
    "OECD": ("OECD (n.d.). Economic Outlook and country surveys. "
             "https://www.oecd.org/economic-outlook"),
    "RBZ": ("Reserve Bank of Zimbabwe (n.d.). Monetary policy and financial "
            "stability publications. https://www.rbz.co.zw"),
    "ZIMSTAT": ("Zimbabwe National Statistics Agency (n.d.). Economic and "
                "social statistics. https://www.zimstat.co.zw"),
    "NYU_Damodaran": ("Damodaran, A. (n.d.). Damodaran Online. "
                      "https://pages.stern.nyu.edu/~adamodar"),
    "SEC": ("U.S. Securities and Exchange Commission (n.d.). EDGAR company "
            "filings database. https://www.sec.gov/edgar"),
    "Investope dia".replace(" ", ""): None,  # placeholder removed below
}
CL.pop("Investopedia")
CL["Investopedia"] = ("Investopedia (n.d.). Financial education and glossary "
                      "articles. https://www.investopedia.com")

# knowledge entries: keyword sets -> (markdown answer, ref keys)
KB: List[Tuple[List[str], str, List[str]]] = []


def kb_add(keywords, md, refs):
    KB.append((keywords, md, refs))


with_standard = None  # markdown continuously defined below

kb_add(
    ["npv", "net present value", "should this project be accepted", "accept or reject",
     "capital budgeting", "investment appraisal"],
    """
## 🔎 WHAT HAPPENED?
Net Present Value (NPV) is the difference between the present value of a project's
expected cash inflows and the present value of its cash outflows, discounted at
the required rate of return (typically the weighted average cost of capital, WACC).

## 🧠 WHY DOES IT MATTER?
NPV is the core acceptance rule in capital budgeting:
- NPV > 0  -> the project is expected to add value to the firm and should be accepted.
- NPV = 0  -> value-neutral (returns exactly the required return).
- NPV < 0  -> the project destroys value and should ordinarily be rejected.

## 📊 KEY DATA
Formula:  NPV = Σ Ct/(1+r)^t - C0
- Ct = net cash flow in year t
- r  = discount rate (WACC)
- C0 = initial investment

## 💰 INVESTMENT IMPLICATIONS
A positive NPV means the investment earns more than its cost of capital.
Because NPV captures the *timing* and *magnitude* of all cash flows, it is
usually preferred over IRR and payback in evaluating mutually exclusive projects.

## ⚠️ RISKS
NPV is only as good as the cash-flow forecasts, the discount rate, the tax and
inflation assumptions, and the currency conversion used. Garbage in, garbage out.

## 🌍 GLOBAL CONTEXT
Used identically for infrastructure, mining, energy, hospitals, universities and
technology projects in every country - only the cash flows, tax systems, currencies
and discount rates change.

## 📚 REFERENCES
""",
    ["BrealeyMyers", "Damodaran", "Investopedia"],
)


kb_add(
    ["irr", "internal rate of return"],
    """
## 🔎 WHAT HAPPENED?
The Internal Rate of Return (IRR) is the discount rate that makes a project's NPV
equal to zero - in other words, the rate at which the investment breaks even in
present-value terms.

## 🧠 WHY DOES IT MATTER?
Management compares IRR with the required return (WACC). IRR >= WACC signals that
the project meets its cost-of-capital hurdle.

## 📊 KEY DATA
Solve 0 = Σ Ct/(1+IRR)^t - C0 for IRR. Because the equation is non-linear, IRR can
be ambiguous (multiple roots) when cash flows change sign more than once. MIRR fixes
this by reinvesting at a stated rate.

## 💰 INVESTMENT IMPLICATIONS
Used as a quick percentage "hurdle" metric. NPV remains the more reliable decision
metric, especially for mutually exclusive projects of different sizes.

## ⚠️ RISKS
High IRRs can mask high risk; a large IRR from a small, risky project may be worse
than a modest IRR from a large, safe one. IRR also assumes reinvestment at the IRR.

## 🌍 GLOBAL CONTEXT
The same maths applies worldwide; the hurdle rate should reflect each country's
sovereign risk, currency risk and inflation expectations.

## 📚 REFERENCES
""",
    ["BrealeyMyers", "Damodaran", "NYU_Damodaran"],
)


kb_add(
    ["mirr", "modified internal rate of return"],
    """
## 🔎 WHAT HAPPENED?
MIRR is a modification of IRR that separates the financing cost (rate at which the
firm borrows capital) from the reinvestment rate (rate earned on reinvested cash
flows), so a project produces a single, unambiguous percentage return.

## 🧠 WHY DOES IT MATTER?
It corrects IRR's reinvestment-rate assumption and eliminates multiple-root problems.

## 📊 KEY DATA
MIRR = (FV of positive cash flows at reinvestment rate / -PV of outflows at finance
rate) ^ (1/n) - 1

## 💰 INVESTMENT IMPLICATIONS
Where IRR exceeds MIRR, the project's return is partly dependent on an aggressive
reinvestment assumption - management should prefer the more conservative MIRR for
decision-making.

## ⚠️ RISKS
MIRR still depends on the chosen finance and reinvestment rates - assumptions must
be stated explicitly.

## 📚 REFERENCES
""",
    ["BrealeyMyers", "Investopedia"],
)


kb_add(
    ["payback", "discounted payback", "payback period"],
    """
## 🔎 WHAT HAPPENED?
Payback period is the number of years until cumulative cash inflows recover the
initial investment. Discounted payback does the same using present values.

## 🧠 WHY DOES IT MATTER?
It is a simple liquidity/risk screen: the sooner capital is recovered, the less
time the investment is exposed to economic and currency shocks.

## 📊 KEY DATA
Payback = year in which cumulative (discounted) cash flow turns positive.

## 💰 INVESTMENT IMPLICATIONS
Useful as a secondary screen to NPV/IRR, especially in volatile or high-risk
environments, but it ignores cash flows after the payback date and the time value
of money (in the non-discounted version).

## 📚 REFERENCES
""",
    ["BrealeyMyers", "Investopedia"],
)


kb_add(
    ["wacc", "cost of capital", "discount rate", "hurdle rate"],
    """
## 🔎 WHAT HAPPENED?
WACC is the blended after-tax cost of debt and equity capital that a firm must earn
to satisfy its capital providers.

## 🧠 WHY DOES IT MATTER?
It is the standard discount rate for project cash flows. Using a WACC below the true
cost of capital over-states NPV; using one above it under-states NPV.

## 📊 KEY DATA
WACC = (E/V)·Re + (D/V)·Rd·(1 - Tc)
- Re = cost of equity (e.g. from CAPM: Rf + beta·equity risk premium)
- Rd = cost of debt; Tc = corporate tax rate
- E/V, D/V = market-value weights of equity and debt

## 💰 INVESTMENT IMPLICATIONS
In emerging markets, sovereign risk, currency risk and illiquidity raise WACC, which
reduces NPV for the same cash flows - a key reason promising projects can fail the
hurdle in high-risk countries.

## ⚠️ RISKS
WACC assumes a stable capital structure and tax regime; both can change materially
during a long construction project.

## 📚 REFERENCES
""",
    ["BrealeyMyers", "MM", "Sharpe", "NYU_Damodaran"],
)


kb_add(
    ["project finance", "project financing", "non-recourse", "spv", "special purpose vehicle"],
    """
## 🔎 WHAT HAPPENED?
Project finance finances a single project through a ring-fenced special purpose
vehicle (SPV). Lenders rely primarily on the project's future cash flows and assets
as security, not on the sponsors' balance sheets (non-recourse or limited-recourse).

## 🧠 WHY DOES IT MATTER?
It lets large infrastructure/energy/mining projects be built without putting the
sponsoring firm's whole balance sheet at risk, and it allows risk to be allocated
to the parties best able to manage it (construction, operation, fuel supply, offtake).

## 📊 KEY DATA
Typical parameters: 70-90% debt; long-tenor loans (10-25 years); construction-phase
grace periods; debt-service cover ratios (e.g. DSCR > 1.2x); a cash-flow waterfall.

## 💰 INVESTMENT IMPLICATIONS
Compare with corporate finance: corporate finance uses the firm's general credit,
incurs the project on the balance sheet, and typically has higher sponsor recourse.

## ⚠️ RISKS
High leverage magnifies downside risk; political, regulatory, currency and offtake
risks can be severe and are usually mitigated with contracts and guarantees.

## 🌍 GLOBAL CONTEXT
Widely used for power, transport, mining, telecoms and social infrastructure in
both developed and emerging markets.

## 📚 REFERENCES
""",
    ["BrealeyMyers", "Investopedia", "WorldBankGEP"],
)


kb_add(
    ["diversif", "asset allocation", "correlation", "portfolio theory",
     "risk and return", "concentration risk"],
    """
## 🔎 WHAT HAPPENED?
Diversification combines assets whose returns are imperfectly correlated so that
idiosyncratic (asset-specific) risk is reduced without proportionally reducing
expected return. Asset allocation decides the mix across asset classes.

## 🧠 WHY DOES IT MATTER?
Per Markowitz (1952), the risk of a portfolio is less than the weighted average of
its parts because unsystematic risk can be diversified away. Remaining risk is
systematic (market) risk, which must be priced.

## 📊 KEY DATA
Portfolio variance = ΣΣ wi·wj·σi·σj·ρij
- wi, wj = weights; σi, σj = standard deviations; ρij = correlation.
- Diversification benefit grows as ρ falls; assets with ρ < 0 add the most benefit.

## 💰 INVESTMENT IMPLICATIONS
Diversify across: asset classes (equities, bonds, cash, commodities, real assets),
geographies, currencies, sectors and maturities. Rebalance periodically because
weights drift with prices.

## ⚠️ RISKS
Diversification does not remove systemic crises (correlations rise in crashes);
over-diversification dilutes returns and adds cost.

## 🌍 GLOBAL CONTEXT
Currency diversification adds a genuine risk-return dimension in emerging markets,
where a single currency can depreciate sharply.

## 📚 REFERENCES
""",
    ["Markowitz", "Sharpe", "Bodie", "Investopedia"],
)


kb_add(
    ["duration", "bond duration", "macauley", "bond prices", "interest rate risk",
     "how does a rise in interest rates affect bond prices"],
    """
## 🔎 WHAT HAPPENED?
Bond duration measures the weighted average time to receive a bond's cash flows and
approximates its price sensitivity to interest-rate changes. Modified duration =
ΔPrice %/ΔYield.

## 🧠 WHY DOES IT MATTER?
When interest rates rise, bond prices fall; duration tells you by how much. A
duration of 5 means an approximate 5% price drop for a 1% rise in yield.

## 📊 KEY DATA
Macauley duration = Σ t·PV(Ct) / Price. Modified duration = Macauley / (1 + yield).
Convexity refines the estimate for large rate moves.

## 💰 INVESTMENT IMPLICATIONS
Longer-duration bonds (long maturities, low coupons) are more rate-sensitive.
In a rising-rate cycle, short-duration bonds, floating-rate notes and T-bills are
less vulnerable.

## ⚠️ RISKS
Duration assumes a parallel shift in the yield curve and constant cash flows (no
embedded options such as callability).

## 📚 REFERENCES
""",
    ["Bodie", "Investopedia"],
)


kb_add(
    ["treasury bill", "t-bill", "money market", "money market fund", "treasury bills"],
    """
## 🔎 WHAT HAPPENED?
Treasury bills are short-term (usually up to one year) government securities sold at
a discount to face value; the investor's return is the difference. Money-market
funds invest in T-bills, bank deposits, commercial paper and repos.

## 🧠 WHY DOES IT MATTER?
They provide capital preservation, liquidity and a benchmark "risk-free" rate that
anchors all other required returns in an economy.

## 📊 KEY DATA
Return ≈ (Face - Price)/Price · (365/days). Yields move with central-bank policy.

## 💰 INVESTMENT IMPLICATIONS
T-bills suit short-dated cash reserves and low-risk allocation; they are illiquid
alternatives rarely. Their yield is the risk-free anchor in CAPM.

## ⚠️ RISKS
Reinvestment risk and (in some countries) negative real returns when inflation
exceeds the yield. Sovereign default risk, though usually low, is real in stressed
economies.

## 📚 REFERENCES
""",
    ["Bodie", "Investopedia", "DownsGoodman"],
)


kb_add(
    ["equities", "stocks", "shares", "stock market", "capital markets", "stock exchange"],
    """
## 🔎 WHAT HAPPENED?
Equities (stocks/shares) are ownership claims on a company. Their return comes from
capital gains and dividends, and their value reflects expected future cash flows
discounted at the required return.

## 🧠 WHY DOES IT MATTER?
Equity provides a claim on residual earnings after debt is served - hence higher
expected return and higher risk than debt. Equity markets are primary venues for
capital raising and price discovery.

## 📊 KEY DATA
Valuation anchors: P/E, P/B, dividend yield, free-cash-flow yield, EV/EBITDA.
Macro drivers: growth, inflation, interest rates, risk appetite, earnings.

## 💰 INVESTMENT IMPLICATIONS
Equity investors earn the equity risk premium (ERP) over the risk-free rate. In
emerging markets, the ERP is higher but political, currency and liquidity risk are
larger.

## ⚠️ RISKS
Valuation, earnings, market-wide and currency risk; correlation across markets
increases in global stress periods.

## 📚 REFERENCES
""",
    ["Bodie", "Graham", "NYU_Damodaran", "Investopedia"],
)


kb_add(
    ["bonds", "bond market", "fixed income", "sovereign bond", "corporate bond"],
    """
## 🔎 WHAT HAPPENED?
A bond is a debt security paying regular coupons plus principal at maturity. Bond
prices move inversely to yields; the yield reflects the risk-free rate plus a credit
spread.

## 🧠 WHY DOES IT MATTER?
Bonds are the core of fixed-income allocation, funding governments and companies,
and their yields set benchmark rates for the whole economy.

## 📊 KEY DATA
Price = Σ Coupon/(1+y)^t + Face/(1+y)^n. Credit spreads widen with default risk and
for longer maturities (term premium).

## 💰 INVESTMENT IMPLICATIONS
Portfolio role: income, capital preservation, and a diversifier to equities. In
rising-rate regimes, investors shorten duration.

## ⚠️ RISKS
Interest-rate risk, credit/default risk, inflation risk, and in emerging markets,
currency depreciation risk on foreign-currency bonds.

## 📚 REFERENCES
""",
    ["Bodie", "Investopedia"],
)


kb_add(
    ["etf", "exchange traded fund", "mutual fund", "index fund"],
    """
## 🔎 WHAT HAPPENED?
ETFs and mutual funds pool investor capital into a diversified portfolio. ETFs trade
intraday on exchanges; open-end mutual funds are priced once daily at NAV.

## 🧠 WHY DOES IT MATTER?
They give small investors diversified, low-cost access to markets, sectors and asset
classes globally.

## 📊 KEY DATA
Key characteristics: expense ratio, tracking error (vs index), AUM, liquidity
(spread/volume), and dividend/income treatment.

## 💰 INVESTMENT IMPLICATIONS
Low-fee index funds capture market returns minus costs and are a baseline around
which more concentrated (and riskier) positions can be built.

## ⚠️ RISKS
Market risk, tracking error, and ETF premium/discount to NAV in stressed markets.

## 📚 REFERENCES
""",
    ["Bodie", "Investopedia"],
)


kb_add(
    ["commodities", "gold", "oil", "commodity", "precious metals"],
    """
## 🔎 WHAT HAPPENED?
Commodities are physical assets (energy, metals, agriculture) whose prices respond
to supply, demand, storage, weather, geopolitics and the US dollar.

## 🧠 WHY DOES IT MATTER?
Commodity exposure hedges inflation, diversifies equities, and is central to
energy/mining/agriculture economies.

## 📊 KEY DATA
Supply/demand balances, inventories (e.g. EIA petroleum stocks), real interest
rates (gold especially), and the USD index shape prices.

## 💰 INVESTMENT IMPLICATIONS
Commodities tend to have low correlation with financial assets but high volatility;
they are best sized as a diversifier, not a core "growth" holding.

## ⚠️ RISKS
Volatility, contango/backwardation in futures, and commodity-specific (counterparty,
storage, regulatory) risk.

## 📚 REFERENCES
""",
    ["Bodie", "Investopedia"],
)


kb_add(
    ["inflation", "consumer price", "cpi", "price level"],
    """
## 🔎 WHAT HAPPENED?
Inflation is the general rise in the price level, measured by indices such as the
CPI. It erodes the real value of nominal cash flows and assets.

## 🧠 WHY DOES IT MATTER?
For capital projects, inflation raises nominal revenues and costs; its net effect
depends on which escalates faster. Central banks use policy rates to steer inflation.

## 📊 KEY DATA
Real return ≈ (1 + nominal)/(1 + inflation) - 1. A sustained high-inflation or
hyperinflation environment changes the whole investment calculus, forcing hard-currency
indexation or multi-currency analysis.

## 💰 INVESTMENT IMPLICATIONS
Inflation during construction raises replacement cost and may exceed budgeted
contingencies; long-term revenue-linked inflation assumptions must be explicit.

## ⚠️ RISKS
Forecast error is large; oil prices, exchange rates and supply shocks drive near-term
inflation in ways models miss.

## 📚 REFERENCES
""",
    ["Bodie", "IMF_WEO", "WorldBankGEP"],
)


kb_add(
    ["interest rate", "policy rate", "central bank", "federal reserve", "monetary policy",
     "does the fed", "fed rate"],
    """
## 🔎 WHAT HAPPENED?
Central banks set short-term policy rates to steer inflation and employment.
When the Federal Reserve (or any major central bank) raises rates, funding costs
rise worldwide through bond yields and the US dollar.

## 🧠 WHY DOES IT MATTER?
Higher rates raise discount rates - lowering NPVs of long-duration projects,
depressing equity valuations and bond prices, and strengthening the currency
issuer's exchange rate. The Fed's global transmission works through capital flows,
trade and pricing of dollar assets.

## 📊 KEY DATA
Follow: policy rate decisions, forward guidance, balance-sheet/QT, core inflation,
labour-market prints. For Zimbabwe: the RBZ policy rate; for South Africa: the SARB.

## 💰 INVESTMENT IMPLICATIONS
In a rising-rate cycle, prefer shorter paybacks, floating-rate liabilities, and
hedging of rate exposure; revisit WACC assumptions frequently.

## ⚠️ RISKS
Central banks can be wrong or slow; abrupt "hawkish" shocks spike volatility.

## 📚 REFERENCES
""",
    ["BIS", "OECD", "Investopedia"],
)


kb_add(
    ["exchange rate", "currency depreciation", "depreciat", "fx", "foreign exchange",
     "how does currency depreciation affect", "hedging", "forward contract", "hedge"],
    """
## 🔎 WHAT HAPPENED?
An exchange rate is the price of one currency in another. Depreciation means a
currency buys less foreign currency; appreciation means it buys more.

## 🧠 WHY DOES IT MATTER?
For a foreign investor, depreciation of the local currency erodes the local-currency
return when converted back. For a project, FX affects imported equipment, supplier
invoices, offshore debt and repatriated profits.

## 📊 KEY DATA
Real rate = nominal rate adjusted for inflation differentials (purchasing power
parity). Hedging tools: forwards (fix a future rate), options/collars (cap the
movement), natural hedges (match revenue and cost currencies), and staggered payments.

## 💰 INVESTMENT IMPLICATIONS
Projects with revenue in a stable currency but costs in a depreciating one improve
in value, and vice-versa. Exchange-rate risk is priced into required returns.

## ⚠️ RISKS
Forecasting FX accurately is very hard; sudden large moves (e.g. 30-50% devaluations
in emerging markets) are the tail risk hedging protects against.

## 📚 REFERENCES
""",
    ["BrealeyMyers", "Investopedia", "BIS"],
)


kb_add(
    ["emerging markets", "frontier markets", "developing", "country risk", "political risk",
     "sovereign risk"],
    """
## 🔎 WHAT HAPPENED?
Emerging-market investment carries higher expected returns to compensate for
higher risk: political, regulatory, legal, currency, liquidity, inflation and
sovereign risk are all greater than in developed markets.

## 🧠 WHY DOES IT MATTER?
Country risk shifts the entire distribution of project or portfolio outcomes.
Analysts use sovereign credit ratings, CDS spreads, external-debt levels, current
account and political-institution quality to gauge it.

## 📊 KEY DATA
Typical screens: inflation, policy rate, exchange-rate volatility, reserves,
fiscal deficit, debt/GDP, governance indices, and ease-of-doing-business metrics.

## 💰 INVESTMENT IMPLICATIONS
Demand a higher hurdle rate, use conservative multi-currency scenarios, secure
hard-currency offtake and financing, and stress-test for capital controls.

## ⚠️ RISKS
Sudden regime or policy change, capital-control introduction, expropriation, and
FX collapse are the defining tail risks.

## 📚 REFERENCES
""",
    ["WorldBankGEP", "IMF_WEO", "OECD"],
)


kb_add(
    ["credit risk", "default risk", "counterparty", "liquidity risk", "market risk",
     "operational risk"],
    """
## 🔎 WHAT HAPPENED?
Perspective on risk classes:
- Credit risk: the chance a borrower/counterparty fails to pay.
- Liquidity risk: inability to sell an asset quickly at fair value, or meet cash
  needs.
- Market risk: losses from price, rate or FX moves.
- Operational risk: losses from failed processes, people or systems.

## 🧠 WHY DOES IT MATTER?
Each risk type requires a different treatment: credit via due diligence, covenants
and ratings; market via diversification and hedging; liquidity via reserves and
managing concentration; operational via controls and insurance.

## 📚 REFERENCES
""",
    ["Bodie", "Hull", "Investopedia"],
)


kb_add(
    ["derivatives", "options", "futures", "swaps", "collars", "financial engineering"],
    """
## 🔎 WHAT HAPPENED?
Derivatives are contracts whose value depends on an underlying asset, rate or index:
futures/forwards (obligation to transact), options (right, not obligation), and swaps
(exchanging flows, e.g. interest-rate or currency swaps).

## 🧠 WHY DOES IT MATTER?
They allow firms to transfer risk (hedging) and investors to gain exposure or
leverage - but without risk controls they amplify losses.

## 📊 KEY DATA
Option value = intrinsic value + time value; pricing per Black-Scholes or binomial
models. Futures require margin. Swaps exchange floating for fixed (and vice versa).

## 💰 INVESTMENT IMPLICATIONS
Projects with FX or commodity exposure hedge with these instruments; hedging costs
(forward points, premia) should be weighed against unhedged risk.

## ⚠️ RISKS
Counterparty risk, basis risk, leverage, model risk, and liquidity in stress.

## 📚 REFERENCES
""",
    ["Hull", "Bodie"],
)


kb_add(
    ["private equity", "venture capital", "startup", "early stage", "angel"],
    """
## 🔎 WHAT HAPPENED?
Venture capital funds early-stage, high-growth companies in exchange for equity,
typically bearing high failure risk for a minority that "wins big". Private equity
buys more mature companies, often with leverage, to improve operations and exit.

## 🧠 WHY DOES IT MATTER?
They fill the financing gap between banks and public markets, creating innovation
and restructuring value - important sources of growth capital in emerging markets.

## 📊 KEY DATA
VC economics: ownership diligence, milestone-based funding, liquidation preference,
exit via trade sale or IPO. PE drivers: EBITDA, leverage, cost-out, multiple
expansion.

## 💰 INVESTMENT IMPLICATIONS
Long holding periods and illiquidity demand a premium; carry/management fees
matter. Diversify across funds/vintages; treat most VC positions as losses.

## ⚠️ RISKS
Illiquidity, valuation subjectivity, key-person and founder risk, and cyclical exits.

## 📚 REFERENCES
""",
    ["Graham", "Investopedia", "WorldBankGEP"],
)


kb_add(
    ["real estate", "property", "reits", "real asset", "infrastructure investment",
     "energy", "mining", "technology investment", "hospital", "university"],
    """
## 🔎 WHAT HAPPENED?
Real-asset investing (real estate, infrastructure, energy, mining, tech) buys assets
with operating cash flows: rentals, tariffs, offtake contracts, usage fees.

## 🧠 WHY DOES IT MATTER?
These assets suit the CAPEXX evaluation framework: they need CAPEX committed today,
revenues arrive over many years, and value depends on discount rates, costs, tax,
inflation, FX and completion risk.

## 📊 KEY DATA
For each project type: unit economics (tenancy, throughput, capacity, resource grade,
node demand), capital intensity, operating leverage, regulatory/market structures.

## 💰 INVESTMENT IMPLICATIONS
Long-duration cash flows are very sensitive to the discount rate and to inflation
indexation; location/country risk and construction execution dominate outcomes.

## ⚠️ RISKS
Construction overruns, demand/tenancy risk, commodity prices, regulatory and
currency risk, and technological obsolescence.

## 📚 REFERENCES
""",
    ["BrealeyMyers", "Damodaran", "WorldBankGEP"],
)


kb_add(
    ["esg", "sustainable", "environmental", "social governance", "green investment"],
    """
## 🔎 WHAT HAPPENED?
ESG investing integrates environmental, social and governance factors into
investment analysis. Green finance channels capital to climate and sustainability
outcomes.

## 🧠 WHY DOES IT MATTER?
Regulators, lenders and investors increasingly price ESG factors; green bonds and
sustainability-linked loans can lower financing costs.

## 📊 KEY DATA
Assessment dimensions: emissions, resource use, community impact, labour, board
structure, transparency, and compliance with frameworks (e.g. TCFD/ISSB).

## 💰 INVESTMENT IMPLICATIONS
Integrate ESG as risk screening and opportunity identification - not as a
guaranteed return premium. Verify claims against actual disclosures.

## ⚠️ RISKS
Greenwashing, inconsistent ratings, and measurement difficulty.

## 📚 REFERENCES
""",
    ["BIS", "OECD", "Investopedia"],
)


kb_add(
    ["how to invest", "what should i invest in", "personal finance", "beginner"],
    """
## 🔎 WHAT HAPPENED?
Sound investing starts with goals, horizon, risk capacity and diversification -
not with forecasts. Build an emergency reserve, clear high-cost debt, then build a
diversified, low-cost portfolio aligned to your horizon.

## 🧠 WHY DOES IT MATTER?
The main determinant of outcomes is the asset mix and costs over time, plus
behavioural discipline (avoiding panic selling).

## 📊 KEY DATA
Asset classes: cash/T-bills (safety), bonds (income), equities (growth), real
assets (inflation hedge). Higher expected return = higher risk.

## 💰 INVESTMENT IMPLICATIONS
Match duration to horizon; rebalance on schedule; increase equity weight only as
horizon lengthens.

## ⚠️ RISKS
Concentration, leverage, chasing past performance, and ignoring fees/taxes.

## 🌍 GLOBAL CONTEXT
CAPEXX provides research and decision support - it does not provide personalised
investment advice.

## 📚 REFERENCES
""",
    ["Bodie", "Graham", "Investopedia"],
)


kb_add(
    ["infrastructure", "megaproject", "public private partnership", "ppp", "concession"],
    """
## 🔎 WHAT HAPPENED?
Infrastructure projects (transport, water, power, telecoms, social) are
capital-intensive, long-lived and usually regulated. PPPs/concessions bundle
design-build-finance-operate into one long contract.

## 🧠 WHY DOES IT MATTER?
Their cash flows are long-dated and often inflation-indexed, making valuation very
sensitive to discount rates and operational performance, and to FX when funding is
hard currency.

## 📊 KEY DATA
Key contract terms: availability payments, tariffs, volume guarantees, lifecycle
maintenance, termination clauses. Success hinges on cost forecasting and delay risk.

## 💰 INVESTMENT IMPLICATIONS
Use DCF with scenario and stress analysis (CAPEXX does this); monitor construction
completion and demand/usage assumptions.

## ⚠️ RISKS
Cost overruns, delays, demand shortfalls, regulatory/political change, FX and
interest-rate movements over long tenors.

## 📚 REFERENCES
""",
    ["WorldBankGEP", "BrealeyMyers", "IMF_WEO"],
)


kb_add(
    ["investing in germany", "investing in china", "investing in the united states",
     "investing in the us", "investing in south africa", "investing in botswana",
     "investing in kenya", "investing in india", "investing in zimbabwe",
     "investing in nigeria", "investing in the united kingdom", "investing in uk",
     "country investment", "tell me everything i need to know about investing"],
    """
## 🔎 WHAT HAPPENED?
A full country-investment review must cover: economic environment, financial
markets, stock and bond markets, interest rates, inflation, currency, industries,
opportunities, risks, tax/regulation, foreign-investor access and indicators to
monitor.

## 🧠 WHY DOES IT MATTER?
Country context changes the required return and the risk treatment of every
investment made there.

## 📊 KEY DATA
Structure your assessment around: GDP and growth, inflation, policy and long-term
rates, exchange-rate regime and volatility, fiscal position and public debt,
current account, FDI, market microstructure and regulation.

## 💰 INVESTMENT IMPLICATIONS
Use the CAPEXX project engine with the country's currencies, tax rates and discount
rates; stress-test FX and political/regulatory scenarios.

## ⚠️ RISKS
Never treat a country overview as a forecast. Verify each statistic from an
authoritative, dated source (central bank, IMF, World Bank, national statistics).

## 🌍 GLOBAL CONTEXT
CAPEXX is country-agnostic: Zimbabwe is only one demonstration context, never a
limitation.

## 📚 REFERENCES
""",
    ["IMF_WEO", "WorldBankGEP", "OECD", "RBZ", "ZIMSTAT"],
)


# --------------------------------------------------------------------------
# COUNTRY ISO MAP (used for World Bank calls)
# --------------------------------------------------------------------------

COUNTRY_ISO: Dict[str, str] = {
    "US": "US", "United States": "US", "USA": "US",
    "Germany": "DE", "France": "FR", "Netherlands": "NL",
    "United Kingdom": "GB", "UK": "GB", "England": "GB",
    "South Africa": "ZA", "Botswana": "BW", "Namibia": "NA",
    "Kenya": "KE", "Tanzania": "TZ", "Uganda": "UG",
    "Zimbabwe": "ZW", "China": "CN", "Japan": "JP",
    "India": "IN", "Australia": "AU", "Canada": "CA",
    "Nigeria": "NG", "Egypt": "EG", "Ghana": "GH",
    "Malawi": "MW", "Mozambique": "MZ", "Angola": "AO",
    "Zambia": "ZM", "Rwanda": "RW", "Botswana ": "BW",
    "Switzerland": "CH", "Sweden": "SE", "Norway": "NO",
    "Denmark": "DK", "Singapore": "SG", "New Zealand": "NZ",
    "Brazil": "BR", "Mexico": "MX", "Russia": "RU",
    "Turkey": "TR", "Indonesia": "ID", "Malaysia": "MY",
    "Vietnam": "VN", "South Korea": "KR", "Ethiopia": "ET",
    "Morocco": "MA", "Senegal": "SN", "Qatar": "QA",
    "Saudi Arabia": "SA", "United Arab Emirates": "AE",
}


# --------------------------------------------------------------------------
# LIVE DATA ACCESS (all keyless, public, dated - never fabricated)
# --------------------------------------------------------------------------


def live_fx_board() -> Dict[str, Any]:
    """Real exchange rates when the public API is reachable."""
    info = try_fetch_live_rates()
    if not info:
        return {"ok": False,
                "message": "⚠️ LIVE MARKET DATA UNAVAILABLE - no public rate "
                           "source could be reached. Use user-provided rates "
                           "in the Exchange Rate Board.",
                "rates": {}}
    rates = {c: info["rates"][c] for c in SUPPORTED_CURRENCIES
             if c in info["rates"]}
    return {"ok": True, "rates": rates, "source": "open.er-api.com/v6/latest/USD "
            "(public, keyless)", "timestamp": info["timestamp"], "info": info}


def worldbank_indicators(country_name: str) -> Dict[str, Any]:
    """Real World Bank statistics for a country (dated, authoritative)."""
    iso = COUNTRY_ISO.get(country_name.strip()) or COUNTRY_ISO.get(
        country_name.strip().title())
    if not iso:
        return {"ok": False,
                "message": f"No World Bank indicator mapping for "
                           f"'{country_name}'. Please verify the source manually.",
                "indicators": {}}
    indicators = {
        "GDP (current US$)": "NY.GDP.MKTP.CD",
        "GDP growth (annual %)": "NY.GDP.MKTP.KD.ZG",
        "Inflation, consumer prices (annual %)": "FP.CPI.TOTL.ZG",
        "Unemployment (% of labour force)": "SL.UEM.TOTL.ZS",
        "Population (total)": "SP.POP.TOTL",
        "FDI, net inflows (BoP, current US$)": "BX.KLT.DINV.CD.WD",
        "Current account balance (US$)": "BN.CAB.XOKA.CD",
    }
    out = {}
    try:
        import requests
    except Exception:
        return {"ok": False, "message": "requests unavailable.",
                "indicators": {}}
    for label, code in indicators.items():
        url = (f"https://api.worldbank.org/v2/country/{iso}/indicator/{code}"
               f"?format=json&per_page=10")
        try:
            r = requests.get(url, timeout=12)
            if r.status_code != 200:
                continue
            data = r.json()
            if not isinstance(data, list) or len(data) < 2:
                continue
            items = [x for x in data[1] if x.get("value") is not None]
            if not items:
                continue
            latest = items[0]
            out[label] = {"value": latest["value"],
                          "year": latest.get("date"),
                          "source": "World Bank Open Data",
                          "url": "https://data.worldbank.org"}
        except Exception:
            continue
    if not out:
        return {"ok": False,
                "message": f"World Bank data could not be retrieved for "
                           f"'{country_name}' (network or availability). No "
                           f"figures were invented.", "indicators": {}}
    return {"ok": True, "indicators": out,
            "source": "World Bank Open Data (API)",
            "country": country_name}


def news_research(query: str, max_items: int = 6) -> Dict[str, Any]:
    """Real recent headlines via Google News RSS."""
    import xml.etree.ElementTree as ET
    try:
        import requests
    except Exception:
        return {"ok": False, "items": []}
    url = ("https://news.google.com/rss/search?q=" +
           re.sub(r"\s+", "+", query.strip()) +
           "&hl=en-US&gl=US&ceid=US:en")
    try:
        r = requests.get(url, timeout=12)
        if r.status_code != 200:
            return {"ok": False, "items": []}
        root = ET.fromstring(r.content)
        items = []
        for it in root.iter("item"):
            title = (it.findtext("title") or "").strip()
            src = (it.findtext("source") or "").strip()
            pub = (it.findtext("pubDate") or "").strip()
            link = (it.findtext("link") or "").strip()
            if not title:
                continue
            m = re.match(r"^(.*?)\s*[-–]\s*([^-–]+)$", title)
            if m:
                headline, outlet = m.groups()
            else:
                headline, outlet = title, src
            items.append({
                "headline": headline.strip(),
                "outlet": outlet.strip(),
                "date": pub,
                "link": link,
            })
            if len(items) >= max_items:
                break
        if not items:
            return {"ok": False, "items": []}
        return {"ok": True, "items": items,
                "source": "Google News RSS (aggregation of publisher reports)"}
    except Exception:
        return {"ok": False, "items": []}


# --------------------------------------------------------------------------
# RESEARCH ROUTER
# --------------------------------------------------------------------------


def _detect_country(q: str) -> Optional[str]:
    for c in COUNTRY_LIST:
        if c.lower() in q:
            return c
    for c in COUNTRY_ISO:
        if c.lower() in q:
            return c
    return None


def _strip_md(text: str) -> str:
    text = re.sub(r"[*_`#>]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def research(question: str,
             mode: str = "deep",
             context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Main entry point for ASK CAPEXX AI.

    Returns a dict: {question, text, references, sources, status, spoken}
    """
    qraw = question.strip()
    q = qraw.lower()
    status = "GENERAL FINANCIAL KNOWLEDGE"
    ref_keys: List[str] = []
    answer_md = ""
    live_notes: List[str] = []
    status_line = ""

    ctx_project = (context or {}).get("project_results")  # AnalysisBundle or None

    # ---- contextual questions about the analysed project -----------------
    if ctx_project is not None:
        cx = _contextual_answer(q, ctx_project)
        if cx:
            answer_md, ref_keys, status_line = cx
            status = "BASED ON CURRENT PROJECT ANALYSIS"
            live_notes.append("Answer uses the results already calculated by the "
                              "CAPEXX engine for the current project.")
            return _build_response(qraw, answer_md, ref_keys, status,
                                   live_notes)

    # ---- country research ------------------------------------------------
    country = _detect_country(q)
    if country and any(w in q for w in ["invest", "investing", "investment",
                                        "analyse", "analyze", "econom",
                                        "conditions", "climate"]):
        wb = worldbank_indicators(country)
        if wb["ok"]:
            status = "COUNTRY RESEARCH (verifiable, dated)"
            rows = wb["indicators"]
            lines = [f"## 🌍 {country} - Country Investment Research",
                     ""]
            lines.append(f"Country overview structure: economic environment, "
                         f"financial markets, interest rates, inflation, currency, "
                         f"industries, opportunities, risks, regulation, "
                         f"foreign-investor access, and indicators to monitor.")
            lines.append("")
            lines.append("**World Bank indicators (the latest values the public "
                         "API returned - check the date column):**")
            lines.append("")
            lines.append("| Indicator | Latest value | Year |")
            lines.append("|-----------|-------------|------|")
            for label, info in rows.items():
                val = info["value"]
                try:
                    val = f"{float(val):,.1f}"
                except (TypeError, ValueError):
                    val = str(val)
                lines.append(f"| {label} | {val} | {info['year']} |")
            lines.append("")
            lines.append("[Source: World Bank Open Data API - "
                         "https://data.worldbank.org]")
            lines.append("")
            lines.append("Use the CAPEXX project engine with this country's "
                         "currencies, tax and discount rates. Verify every "
                         "figure with the original source before relying on it.")
            answer_md = "\n".join(lines)
            status_line = ("🔵 Data from World Bank Open Data API - country "
                           "statistics are typically published with a lag; "
                           "treat them as RECENT/HISTORICAL, and re-verify.")
            ref_keys = ["WorldBankGEP", "IMF_WEO", "OECD"]
        else:
            answer_md = KB_mini_country(country, wb["message"])
            status_line = "⚠️ Live country statistics unavailable; framework only, no figures invented."
            ref_keys = ["WorldBankGEP", "IMF_WEO", "OECD"]
        return _build_response(qraw, answer_md, ref_keys, status, live_notes +
                               [status_line])

    # ---- live market data request? ----------------------------------------
    if any(w in q for w in ["live", "current rate", "current exchange", "exchange rate today",
                            "how much is", "rate now"]):
        fx = live_fx_board()
        if fx["ok"]:
            lines = ["## 💱 EXCHANGE RATES (retrieved live at run time)",
                     "",
                     f"Source: {fx['source']}  |  Updated: "
                     f"{fx['info']['timestamp']}",
                     "",
                     "| Currency | Units per USD |",
                     "|----------|---------------|"]
            for c, r in list(fx["rates"].items())[:18]:
                lines.append(f"| {c} | {r:,.4f} |")
            lines.append("")
            lines.append("These were retrieved from a public API during this "
                         "session. Prices move continuously - re-verify before "
                         "acting.")
            answer_md = "\n".join(lines)
            status_line = ("🟢 LIVE data from a public API, dated at retrieval "
                           "time. This is not a guarantee of subsequent prices.")
            status = "LIVE MARKET DATA (retrieved at run time)"
            ref_keys = ["BIS", "Investopedia"]
        else:
            answer_md = fx["message"] + "\n\nUse the **Exchange Rate Board** to " \
                                        "enter your own verified rates."
            status_line = "⚠️ LIVE MARKET DATA UNAVAILABLE this session."
            status = "LIVE MARKET DATA UNAVAILABLE"
            ref_keys = ["BIS"]
        return _build_response(qraw, answer_md, ref_keys, status, live_notes +
                               [status_line])

    # ---- news / current developments --------------------------------------
    if any(w in q for w in ["what's happening", "what is happening", "news",
                            "latest", "recent developments", "currently",
                            "today"]):
        topic = q.replace("what's happening", " ").replace("what is happening",
                       " ").replace("news", "").replace("latest", "").replace(
                       "recent developments", "").replace("currently", "").replace(
                       "today", "").replace("?", "").strip().title()
        topic = topic or "global financial markets"
        nw = news_research(topic)
        if nw["ok"]:
            lines = [f"## 📰 GLOBAL INVESTMENT NEWS - {topic}",
                     "",
                     f"Aggregated source: {nw['source']}",
                     "",
                     "| Headline | Outlet | Date |",
                     "|----------|--------|------|"]
            for it in nw["items"]:
                lines.append(f"| {it['headline']} | {it['outlet']} | "
                             f"{it['date']} |")
            lines.append("")
            lines.append("News headlines are reports by publishers - CAPEXX "
                         "summarises links, it does not turn them into "
                         "investment recommendations. Open the original "
                         "publisher article to verify before acting.")
            answer_md = "\n".join(lines)
            status_line = ("🟢 Headlines aggregated live from Google News RSS "
                           "at run time; each carries its own publisher date.")
            status = "RECENT NEWS (aggregated at run time - verify)"
            ref_keys = ["Investopedia"]
        else:
            answer_md = ("📰 News aggregation was unavailable this session "
                         "(network or service). No headlines were fabricated. "
                         "Try again later or use your own verified sources.")
            status_line = "⚠️ News service unreachable."
            status = "NEWS UNAVAILABLE (no fabrication)"
            ref_keys = []
        return _build_response(qraw, answer_md, ref_keys, status, live_notes +
                               [status_line])

    # ---- company research ---------------------------------------------------
    comp = _detect_company(qraw)
    if comp:
        return _company_research(qraw, comp, q)

    # ---- concept knowledge base ----------------------------------------------
    for keys, md, refs in KB:
        if any(k in q for k in keys):
            answer_md = md
            ref_keys = refs
            if mode in ("quick",):
                # keep structured but trim the references block
                pass
            status = "GENERAL FINANCIAL KNOWLEDGE (educational)"
            status_line = ("🔵 Educational material grounded in standard "
                           "published finance theory; not current data.")
            return _build_response(qraw, answer_md, ref_keys, status,
                                   live_notes + [status_line])

    # ---- default: framework answer -------------------------------------------
    answer_md = _generic_research(qraw)
    status = "GENERAL FINANCIAL KNOWLEDGE"
    status_line = ("🔵 General educational response. For current figures, use "
                   "'current', 'live', 'news', or a country/company query - "
                   "results are then retrieved from dated public sources.")
    ref_keys = ["BrealeyMyers", "Damodaran", "Investopedia"]
    return _build_response(qraw, answer_md, ref_keys, status, live_notes +
                           [status_line])


def KB_mini_country(country: str, note: str) -> str:
    return (f"## 🌍 {country} - Country Investment Research\n\n{note}\n\n"
            "Structure a full review around: economic environment, financial "
            "markets, stock and bond markets, interest rates, inflation, "
            "currency, major industries, opportunities, risks, tax and "
            "regulation, foreign-investor access, market access, and key "
            "indicators to monitor.\n\n"
            "Every current statistic must carry its own source and date from "
            "an authoritative institution (central bank, national statistics, "
            "IMF, World Bank).")


_COMPANY_STOP = {"the", "a", "an", "company", "firm", "business", "how", "why",
                 "what", "which", "invest", "investing", "should", "about"}


def _detect_company(q: str) -> Optional[str]:
    m = re.search(r"\b(?:analyse|analyze|research|analysis|invest in)\s+"
                  r"(?:the\s+|about\s+|company\s+)?([A-Z][A-Za-z0-9\.&\-]{2,})",
                  q, re.IGNORECASE)
    if not m:
        m = re.search(r"\b(?:about|tell me about)\s+([A-Z][A-Za-z0-9\.&\-]{2,})",
                      q, re.IGNORECASE)
    if m:
        name = m.group(1).strip(".,;:")
        if name.lower() not in _COMPANY_STOP and len(name) >= 3:
            return name
    return None


def _company_research(qraw: str, name: str, q: str) -> Dict[str, Any]:
    nw = news_research(name)
    lines = [f"## 💼 Company Research - {name}",
             "",
             "(Reported facts are only included when verified from the company's "
             "official filings or regulatory disclosures.)",
             "",
             "**Analytical framework used: business model, revenue, profit, "
             "EPS, assets, liabilities, cash flow, debt, ROE, ROA, margins, "
             "valuation, dividends, strategy, industry, competitors, major "
             "risks, governance.**",
             ""]
    if nw["ok"]:
        lines.append("**Recent headlines located via Google News RSS (verify "
                     "each with the original publisher):**")
        lines.append("")
        lines.append("| Headline | Outlet | Date |")
        lines.append("|----------|--------|------|")
        for it in nw["items"][:6]:
            lines.append(f"| {it['headline']} | {it['outlet']} | {it['date']} |")
        lines.append("")
        lines.append("For financial statements use official filings: SEC EDGAR "
                     "(US), Companies House (UK), or the relevant national "
                     "regulator and the company's investor-relations page.")
    else:
        lines.append("No current headlines could be aggregated this session "
                     "(network/service). No financial figures have been "
                     "invented for this company.")
    lines.append("")
    lines.append("**Reported facts vs interpretation:** CAPEXX separates "
                 "verified disclosures from analytical interpretation. Any "
                 "figure asserted here must cite the actual filing it came from.")
    answer_md = "\n".join(lines)
    status = "COMPANY RESEARCH (framework + live news aggregation)"
    status_line = ("🟡 Live headline aggregation at run time; financial-statement "
                   "figures require the company's official filings - none "
                   "were fabricated.")
    return _build_response(qraw, answer_md, ["SEC", "Investopedia"], status,
                           [status_line])


def _generic_research(qraw: str) -> str:
    return (f"## 🔎 WHAT HAPPENED?\n"
            f"Your question - '{qraw}' - does not match a specialised module, "
            f"so CAPEXX provides a structured framework instead of guessing.\n\n"
            f"## 🧠 HOW TO GET A FULLER ANSWER\n"
            f"1. **Concepts** - ask about NPV, IRR, MIRR, payback, WACC, "
            f"duration, diversification, project finance, T-bills, bonds, "
            f"equities, commodities, inflation, interest rates, hedging, "
            f"emerging-market risk, derivatives.\n"
            f"2. **Countries** - 'analyse the investment climate in [country]'.\n"
            f"3. **Companies** - 'analyse company X' (uses live news "
            f"aggregation, never invented financials).\n"
            f"4. **Current data** - add 'current', 'live', 'news' or 'today' "
            f"to trigger dated public research.\n"
            f"5. **Your project** - after running a CAPEXX analysis you can "
            f"ask about your own project's results.\n\n"
            f"## 📚 REFERENCES\nClassic finance references that underpin the "
            f"CAPEXX methodology are listed below; they are genuine "
            f"publications.")


# ---- contextual answers about the CURRENT PROJECT --------------------------

def _contextual_answer(q: str, b) -> Optional[Tuple[str, List[str], str]]:
    ccy = b.project_input.reporting_currency
    m = b.metrics

    if any(w in q for w in ["stress", "why did my project fail", "fail the stress"]):
        worst = min(b.stress, key=lambda s: s["npv"])
        txt = (f"## 🔎 YOUR PROJECT - STRESS TEST\n"
               f"The worst stress case is **{worst['name']}** with an NPV of "
               f"**{ccy} {worst['npv']:,.0f}** versus a base NPV of "
               f"{ccy} {m.npv:,.0f}.\n\n"
               f"The most sensitive drivers are "
               f"{', '.join(r['driver'] for r in sorted(b.sensitivity, key=lambda r: r['delta'], reverse=True)[:3])}.\n\n"
               f"## 🧠 INTERPRETATION\nThe decision engine treated a stress "
               f"loss above 80% of base value as a review/reject signal. "
               f"Monitor the drivers above and re-run the stress screen after "
               f"any material change.")
        return txt, ["BrealeyMyers"], "BASED ON CURRENT PROJECT RESULTS"

    if any(w in q for w in ["currency", "exchange", "depreciat", "fx"]):
        fx = b.fx
        mism = [e for e in fx["exposures"] if e["mismatch"]]
        names = ", ".join(f"{e['component']} in {e['currency']}" for e in mism) or "none"
        txt = (f"## 🔎 YOUR PROJECT - CURRENCY\n"
               f"The project runs in {ccy} with {len(mism)} mismatched "
               f"currency stream(s): {names}.\n"
               f"FX risk level: {fx['fx_risk_level']} (score "
               f"{fx['fx_risk_score']:.0f}/100). Expected movements are already "
               f"inside the cash-flow engine.\n\n"
               f"## 🧠 WHY IT MATTERS\nIf the local currency depreciates more "
               f"than modelled, costs rise in reporting-currency terms and "
               f"NPV falls. CAPEXX's stress screen includes an FX depreciation "
               f"shock so you can see the impact directly.")
        return txt, ["BrealeyMyers", "BIS"], "BASED ON CURRENT PROJECT RESULTS"

    if any(w in q for w in ["monitor", "indicator", "watch", "track"]):
        pts = b.monitoring_points if hasattr(b, "monitoring_points") else []
        from capexx_engine import monitoring_points as _mp
        pts = _mp(b)
        txt = ("## 🔎 YOUR PROJECT - WHAT TO MONITOR\n" +
               "\n".join(f"- {p}" for p in pts))
        return txt, ["BrealeyMyers"], "BASED ON CURRENT PROJECT RESULTS"

    if any(w in q for w in ["what economic", "econom", "inflation affect", "macro"]):
        txt = (f"## 🔎 YOUR PROJECT - MACRO DRIVERS\n"
               f"Under the stated inflation of {b.project_input.inflation_rate:.1f}% "
               f"and WACC of {m.wacc:.1f}%, the base NPV is {ccy} {m.npv:,.0f}. "
               f"An inflation increase raises both revenues (if indexed) and "
               f"costs; the net effect depends on which dominates. The best "
               f"protection is matching currency and indexation of revenue "
               f"contracts to your hard costs.")
        return txt, ["BrealeyMyers", "IMF_WEO"], "BASED ON CURRENT PROJECT RESULTS"

    if any(w in q for w in ["what is the project", "overview", "summary", "status"]):
        dec = b.decision
        txt = (f"## 🔎 YOUR PROJECT STATUS\n"
               f"**{b.project_input.project_name}** ({b.project_input.country}, "
               f"{b.project_input.project_type}) - model outcome "
               f"**{dec.grade}**.\n"
               f"NPV {ccy} {m.npv:,.0f}; IRR {m.irr:.1f}%; MIRR "
               f"{m.mirr:.1f}% (if available); risk {b.risk.level}.")
        return txt, [], "BASED ON CURRENT PROJECT RESULTS"

    return None


def _build_response(qraw: str, answer_md: str, ref_keys: List[str],
                    status: str, notes: List[str]) -> Dict[str, Any]:
    refs = [CL[k] for k in ref_keys if k in CL]
    notes = [n for n in notes if n]
    text = answer_md
    if refs:
        text += "\n\n## 📚 REFERENCES\n" + "\n".join(f"- {r}" for r in refs)
    text += ("\n\n> **CAPEXX AI** provides investment research and "
             "decision-support information for educational and analytical "
             "purposes. Information may change and should be independently "
             "verified using the cited sources. CAPEXX AI does not guarantee "
             "investment returns and does not replace professional financial, "
             "legal, tax or investment advice.")
    return {
        "question": qraw,
        "text": text,
        "markdown": text,
        "spoken": _strip_md(text),
        "references": refs,
        "notes": notes,
        "status": status,
        "status_line": " \n".join(notes) if notes else "",
    }


EXAMPLE_QUESTIONS = [
    "How does a rise in interest rates affect bond prices?",
    "What are the major investment risks in emerging markets?",
    "Explain how currency depreciation affects foreign investors.",
    "Compare project finance and corporate finance.",
    "What is the difference between NPV and IRR?",
    "What are the current economic conditions in Zimbabwe?",
    "How does the US Federal Reserve affect global markets?",
    "What are the main risks of investing in mining projects?",
    "Explain portfolio diversification using a practical example.",
    "What is the current EUR/USD exchange rate (live)?",
    "Analyse company Microsoft.",
    "Analyse the investment climate in Botswana.",
]

SEARCH_MODES = ["QUICK ANSWER", "DEEP RESEARCH", "ACADEMIC RESEARCH",
                "MARKET RESEARCH", "COUNTRY RESEARCH", "COMPANY RESEARCH",
                "PROJECT RESEARCH"]