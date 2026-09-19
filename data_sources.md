# Data Sources Policy

## Principles

1. **Never fabricate data.** CAPEXX does not invent macroeconomic figures,
   exchange rates, prices, statistics or citations.
2. **Never pretend a source exists.** References are only listed when the
   source was genuinely consulted.
3. **Every external data point shows: Source | Date | Status.**
4. **Current vs historical is distinguished.**
5. **Ambiguity is disclosed.** When reputable sources disagree, CAPEXX says so.

## What CAPEXX actually connects to (keyless, public, dated)

| Data | Source | Status label |
|---|---|---|
| Exchange rates | open.er-api.com/v6/latest/USD (public API) | LIVE (retrieved at run time) |
| GDP / growth / inflation / unemployment / FDI / current account | World Bank Open Data API | RECENT / HISTORICAL (tertiary lag) |
| Recent headlines | Google News RSS (publisher aggregation) | Each item carries its publisher date |

If any of these is unreachable at run time, the module says
**"UNAVAILABLE - no figures were invented"** and offers the user-provided-data
path instead.

## Acknowledge the lender of truth (no endorsement)

The platform does not guarantee the correctness of third-party data even when
live. Users must independently verify figures against the original
publication before acting.

## Recommended authoritative sources

### Global

- International Monetary Fund - World Economic Outlook
- World Bank - Global Economic Prospects / Open Data
- Bank for International Settlements
- OECD Economic Outlook
- United Nations / UNCTAD
- International Energy Agency (energy)

### Central banks

- US Federal Reserve, ECB, Bank of England, Bank of Japan
- South African Reserve Bank
- Reserve Bank of Zimbabwe (https://www.rbz.co.zw)
- Bank of Botswana, Central Bank of Kenya, Bank of Zambia
- any other relevant national central bank

### Statistics / government

- National statistics agencies (e.g. ZIMSTAT for Zimbabwe)
- Ministries of Finance

### Markets / companies

- Official exchange websites
- SEC EDGAR (US) and equivalent regulators
- Company investor-relations pages and audited annual reports

### Academic

- Peer-reviewed journals, SSRN, NBER, ScienceDirect, Springer, Wiley,
  Taylor & Francis, JSTOR.

## Photographic assets

CAPEXX ships a small set of real photographs used in the cinematic
introduction:

| Asset | Source | Licence |
|---|---|---|
| assets/rbz.jpg | Wikimedia Commons - Reserve Bank of Zimbabwe scene (Harare) | CC BY 2.0 |
| assets/construction.jpg | Unsplash contributor (construction site) | Unsplash Licence |
| assets/gzu_innovation_hub.jpg | Wikimedia Commons - Great Zimbabwe University reach out event | CC BY-SA 4.0 |
| assets/global_finance.jpg | Unsplash contributor (banknotes/global finance) | Unsplash Licence |
| assets/engineers.jpg | Unsplash contributor (engineers reviewing work) | Unsplash Licence |
| assets/robot.png | Wikimedia Commons - Cronos humanoid robot | CC BY-SA 3.0 |

Captions never claim a photograph is an "official project photograph" unless
the source confirms that. The RBZ image is a general Harare scene containing
the Reserve Bank - it is not an official RBZ photograph.

For the GZU Innovation Hub the platform always displays:

> **HYPOTHETICAL DEMONSTRATION — NOT ACTUAL GZU FINANCIAL DATA**

## Demonstration caveats

- The market-simulation screen is always labelled
  **MARKET SIMULATION — DEMONSTRATION ONLY**.
- The Exchange Rate Board defaults to
  **DATA STATUS: DEMONSTRATION / USER-PROVIDED**.
- Synthetic demo figures are never presented as real institutional data.