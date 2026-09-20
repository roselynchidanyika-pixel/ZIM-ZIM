"""Upload parsing and data validation for RFC Securities.

Two file types are recognised automatically:

* **financials** — per-period income/cash-flow template (columns: Period,
  Revenue, COGS, Opex, Depreciation, Capex + optional Units/Price/
  Unit_Variable_Cost).
* **projects**  — a capital-budgeting catalogue (columns: Project,
  Initial_Investment, Year1..YearN).

Errors and warnings are returned instead of raised so the Streamlit UI can
display them professionally.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from ..engine import CompanyInput
from ..models.investment import ProjectSpec

FIN_COLUMNS_MAX = {
    "Revenue": "revenue", "COGS": "cogs", "COST_OF_GOODS_SOLD": "cogs",
    "OPEX": "opex", "OPERATING_COSTS": "opex", "FIXED_OPEX": "opex",
    "DEPRECIATION": "d_and_a", "DEPRECIATION_&_AMORTISATION": "d_and_a",
    "D&A": "d_and_a", "CAPEX": "capex",
}
FIN_COLUMNS_OPT = {
    "UNITS": "units", "UNITS_SOLD": "units", "PRICE": "price",
    "UNIT_VARIABLE_COST": "unit_var_cost", "UVC": "unit_var_cost",
}
PROJECT_COLUMNS = {
    "PROJECT": "name", "PROJECT_NAME": "name", "NAME": "name",
    "INITIAL_INVESTMENT": "cost", "INVESTMENT": "cost", "COST": "cost",
    "INITIAL_OUTLAY": "cost",
}


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9&]", "_", (s or "").lower().strip())


def _read_table(data: bytes, filename: str) -> tuple[bool, pd.DataFrame | None, list[str]]:
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    errors: list[str] = []
    import io
    try:
        if ext == "csv":
            df = pd.read_csv(io.BytesIO(data))
        elif ext in ("xlsx", "xls"):
            df = pd.read_excel(io.BytesIO(data))
        else:
            errors.append(f"Unsupported file type '.{ext}'. Use CSV or Excel.")
            return False, None, errors
    except Exception as e:
        errors.append(f"Could not read '{filename}': {e}")
        return False, None, errors
    if df is None or df.empty:
        errors.append("The uploaded file contains no rows.")
        return False, None, errors
    return True, df, errors


def io_to_bytes(data: bytes) -> bytes:
    return data


def _column_map(df: pd.DataFrame) -> dict[str, str]:
    return {_norm(str(c)): str(c) for c in df.columns}


def parse_upload(data: bytes, filename: str) -> dict:
    """Parse an uploaded financials or projects file into model objects.

    Returns ``{"ok", "kind", "ci"|"projects", "table", "errors", "warnings"}``.
    """
    ok, df, errors = _read_table(data, filename)
    if not ok:
        return {"ok": False, "kind": "unknown", "errors": errors,
                "warnings": []}
    cmap = _column_map(df)
    lk = {_norm(str(c)): c for c in df.columns}

    if any(k in cmap for k in ("initial_investment", "project", "project_name")):
        return _parse_projects(df, lk, filename)
    return _parse_financials(df, lk, filename)


def _parse_financials(df: pd.DataFrame, lk: dict, filename: str) -> dict:
    errors: list[str] = []
    warnings: list[str] = []
    missing = [k for k in FIN_COLUMNS_MAX if _norm(k) not in lk]
    if "revenue" in missing or len(df) < 2:
        return {"ok": False, "kind": "financials", "errors": [
            "Not a valid financials file. Required columns: Revenue, and at "
            "least one of COGS/Opex/Capex. Either use the template or upload a "
            "projects file instead."], "warnings": warnings}

    def col(name: str):
        return lk.get(name, "")

    try:
        n = len(df)
        revenue = _num(df, col("revenue"), errors, "Revenue")
        cogs = _num(df, col("cogs"), errors, "COGS") if "cogs" in lk else np.zeros(n)
        opex = _num(df, col("opex"), errors, "Opex") if "opex" in lk else np.zeros(n)
        d_a = _num(df, col("d_and_a"), errors, "Depreciation") if "d_and_a" in lk else np.zeros(n)
        capex = _num(df, col("capex"), errors, "Capex") if "capex" in lk else np.zeros(n)
        units = _num(df, col("units"), errors, "Units") if "units" in lk else None
        price = _num(df, col("price"), errors, "Price") if "price" in lk else None
        uvc = _num(df, col("unit_var_cost"), errors, "Unit variable cost") if "unit_var_cost" in lk else None
    except Exception as e:
        errors.append(f"Failed to convert numeric columns: {e}")
        return {"ok": False, "kind": "financials", "errors": errors,
                "warnings": warnings}

    if errors:
        return {"ok": False, "kind": "financials", "errors": errors,
                "warnings": warnings}
    if np.any(revenue <= 0):
        errors.append("Revenue must be positive in every period.")
    if np.any(cogs < 0) or np.any(opex < 0) or np.any(capex < 0):
        errors.append("Costs and capex cannot be negative.")
    if errors:
        return {"ok": False, "kind": "financials", "errors": errors,
                "warnings": warnings}

    ci = CompanyInput(
        company_name=filename.rsplit(".", 1)[0].replace("_", " ").title(),
        periods=n,
        revenue=revenue,
        cogs=cogs,
        opex=opex,
        d_and_a=d_a,
        capex=capex,
        units=units,
        price=price,
        unit_var_cost=uvc,
    )
    if "depreciation" not in lk:
        warnings.append("No Depreciation column found — treated as zero.")
    if "capex" not in lk:
        warnings.append("No Capex column found — the business case assumes no "
                        "initial capital spend.")

    errs2 = ci.validate()
    errors.extend(errs2)
    return {"ok": len(errors) == 0, "kind": "financials", "ci": ci,
            "table": df, "errors": errors, "warnings": warnings}


def _parse_projects(df: pd.DataFrame, lk: dict, filename: str) -> dict:
    errors: list[str] = []
    warnings: list[str] = []
    name_col = next((lk[_norm(pat)] for pat, value in PROJECT_COLUMNS.items()
                     if _norm(pat) in lk and value == "name"), None)
    cost_col = next((lk[_norm(pat)] for pat, value in PROJECT_COLUMNS.items()
                     if _norm(pat) in lk and value == "cost"),
                    lk.get("initial_investment"))
    if not name_col or not cost_col:
        errors.append("Projects file must contain 'Project' and "
                      "'Initial_Investment' columns.")
        return {"ok": False, "kind": "projects", "errors": errors,
                "warnings": warnings}

    year_cols = []
    for c in df.columns:
        m = re.match(r"^year\s*(\d+)$", _norm(str(c)))
        if m:
            year_cols.append((str(c), int(m.group(1))))
    year_cols.sort(key=lambda t: t[1])
    if not year_cols:
        errors.append("Projects file must contain Year1..YearN cash-flow columns.")
        return {"ok": False, "kind": "projects", "errors": errors,
                "warnings": warnings}

    projects: list[ProjectSpec] = []
    for i, row in df.iterrows():
        try:
            name = str(row[name_col]).strip()
            cost = abs(float(row[cost_col]))
            if name and cost >= 0:
                cfs = [float(row[c]) for c, _ in year_cols]
                projects.append(ProjectSpec(name, cost, np.array(cfs),
                                            "USD", ""))
        except Exception:
            errors.append(f"Row {i + 2} could not be parsed.")
    if not projects:
        errors.append("No valid project rows found.")
    if len(projects) > 30:
        warnings.append(f"{len(projects)} projects loaded — consider trimming "
                        "the catalogue for faster optimisation.")
    return {"ok": not errors, "kind": "projects", "projects": projects,
            "table": df, "errors": errors, "warnings": warnings}


def _num(df: pd.DataFrame, col: str, errors: list[str], label: str) -> np.ndarray:
    if not col:
        return np.array([])
    raw = df[col]
    if raw.dtype == object:
        raw = pd.to_numeric(raw, errors="coerce")
    arr = raw.to_numpy(dtype=float)
    if np.any(np.isnan(arr)):
        errors.append(f"'{label}' contains blank or non-numeric values.")
    return arr