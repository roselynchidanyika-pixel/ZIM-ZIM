"""RFC Securities — brand identity, palette and formatting helpers."""
from __future__ import annotations

BRAND = "RFC Securities"
TAGLINE = "TURN UNCERTAINTY INTO PROFITABILITY."
SUBTAGLINE = "MODEL. PREDICT. OPTIMIZE."
MODEL_CHAIN = "PROFITABILITY \u2192 CASH FLOW \u2192 INVESTMENT \u2192 RISK \u2192 OPTIMIZATION"

# ── brand palette ────────────────────────────────────────────────────────
DEEP_GREEN = "#0B3D2E"     # dark / deep green (headers, sidebar, hero)
GREEN      = "#157347"     # primary green
GREEN_LIGHT= "#2E8B57"
GREEN_MINT = "#DCF3E6"     # light green background for cards
GOLD       = "#C9A227"     # gold / yellow accent
GOLD_LIGHT = "#F3E5B8"
GOLD_DEEP  = "#9C7C1E"
WHITE      = "#FFFFFF"
OFF_WHITE  = "#F6F8F4"
INK        = "#0F172A"
MUTED      = "#5B7266"
RED        = "#C0392B"
RED_BG     = "#FBE3E0"
AMBER      = "#B7791F"
AMBER_BG   = "#FDF3E0"

CURRENCY_SYMBOLS = {
    "USD": "$", "EUR": "\u20AC", "GBP": "\u00A3", "ZAR": "R",
    "NGN": "\u20A6", "KES": "KSh", "ZWL": "Z$", "AUD": "A$",
    "CAD": "C$", "JPY": "\u00A5", "CNY": "\u00A5", "INR": "\u20B9",
    "BRL": "R$", "CHF": "CHF", "AED": "AED", "EGP": "EGP",
}
SUPPORTED_CURRENCIES = list(CURRENCY_SYMBOLS)
DEFAULT_CURRENCY = "USD"


def ccy_symbol(ccy: str) -> str:
    return CURRENCY_SYMBOLS.get(ccy or DEFAULT_CURRENCY, (ccy or "") + " ")


def fmt_currency(value: float, ccy: str = DEFAULT_CURRENCY, digits: int = 0) -> str:
    """Format a money value as e.g. -$1,234,567 (0 dp) or $1,234.56 (2 dp)."""
    sym = ccy_symbol(ccy)
    sign = "-" if value < 0 else ""
    return f"{sign}{sym}{abs(value):,.{digits}f}"


def fmt_pct(value: float, digits: int = 1) -> str:
    return f"{value:.{digits}f}%"


def fmt_num(value: float, digits: int = 0) -> str:
    return f"{value:,.{digits}f}"


def fmt_mult(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}x"