# Currency Methodology

## Global by design

CAPEXX supports **any country → any supplier → any currency → any reporting
currency**. Zimbabwe, ZiG, USD and ZAR appear only because the demonstration
project uses them - they are never hard-coded limitations.

Supported currencies include USD, EUR, GBP, ZAR, ZiG, CNY, JPY, KES, BWP, NAD,
INR, AUD, CAD, NGN, EGP, GHS, TZS, UGX, MWK, MZN, AOA and more, plus a
user-defined code.

## Rate convention

All exchange rates are stored as **units of that currency per 1 USD**.

Example: `ZAR = 18.1` means 1 USD buys 18.1 ZAR.

Conversion of an amount from currency X to currency Y:

```
value_Y = value_X × (rate_Y / rate_X)
```

Because USD is the reciprocal axis this is unambiguous and auditable.

## Where rates come from

1. **Live fetch (optional)** at run time from a public keyless API
   (`open.er-api.com/v6/latest/USD`), showing Source + Date/Time + Status.
   If the request fails, CAPEXX says **LIVE MARKET DATA UNAVAILABLE**.
2. **User-provided rates** in the Exchange Rate Board. The board is labelled
   **DATA STATUS: DEMONSTRATION / USER-PROVIDED**.

CAPEXX never invents or silently stores an exchange rate.

## Currency exposure model

Each component of the project can carry its own currency and expected annual
FX movement:

- Revenue currency
- Opex currency
- Maintenance currency
- Construction / civil-works currency + supplier country
- Equipment currency + supplier country
- Land/buildings currency + supplier country
- Materials currency + supplier country

The engine then flags **mismatches** (component currency ≠ reporting currency),
converts every exposure into reporting-currency amounts, and computes:

- an FX risk score and level (LOW / MODERATE / HIGH),
- conversion-cost context,
- hedging considerations (forwards, options/collars, natural hedges,
  staggered payment timing).

## Transaction-level strategy

For every distinct transaction the engine emits a line in the form:

```
TRANSACTION: Imported equipment
SUPPLIER:    Germany
INVOICE:     EUR
ANALYSIS:    EUR exposure should be incorporated into the project's
             FX-adjusted cash flow and hedged or scheduled accordingly.
```

The recommendation depends on the actual invoice currency, supplier country,
reporting currency and movement assumption - it never blindly recommends one
currency.

## Landed cost

A generic imported-goods build-up (not specific to South Africa):

```
Purchase + Transport + Insurance + Duties + Taxes + Conversion + Financing/FX
= Landed cost
```

Each percentage (transport, insurance, duty, taxes, conversion, financing/FX)
is user-editable so the engine works for any supplier country and currency.