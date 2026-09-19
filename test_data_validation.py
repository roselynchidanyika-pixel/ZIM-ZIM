"""Tests for CAPEXX upload validation and data transparency."""
import io

import pandas as pd

from capexx_engine import (parse_upload, make_template_csv, ProjectInput,
                           EXPECTED_COLUMNS)


def _valid_df() -> pd.DataFrame:
    row = {c: "" for c in EXPECTED_COLUMNS}
    row.update({
        "project_name": "Solar Farm", "organisation": "SunCo",
        "country": "Botswana", "location": "Gaborone",
        "project_type": "Energy / Power",
        "construction_cost": 5000000, "equipment_cost": 3000000,
        "land_building_cost": 500000, "annual_revenue": 2000000,
        "annual_opex": 400000, "annual_maintenance": 50000,
        "project_life": 20, "construction_period": 1, "tax_rate": 22,
        "discount_rate": 10,
    })
    return pd.DataFrame([row])


def test_valid_csv_passes():
    csv_bytes = _valid_df().to_csv(index=False).encode("utf-8")
    res = parse_upload(csv_bytes, "project.csv")
    assert res["ok"] is True
    assert res["input"] is not None
    assert res["input"].project_name == "Solar Farm"


def test_template_roundtrip():
    csv_bytes = make_template_csv().encode("utf-8")
    res = parse_upload(csv_bytes, "template.csv")
    assert res["ok"] is True
    assert res["input"].organisation == "Great Zimbabwe University"


def test_missing_columns_detected():
    df = pd.DataFrame([{"project_name": "Only name", "country": "Kenya"}])
    res = parse_upload(df.to_csv(index=False).encode("utf-8"), "poor.csv")
    assert res["ok"] is False
    missing_text = " ".join(res["errors"]).lower()
    assert "missing column" in missing_text


def test_invalid_file_type_rejected():
    res = parse_upload(b"garbage", "notes.txt")
    assert res["ok"] is False
    assert any("unsupported file type" in e.lower() for e in res["errors"])


def test_unparseable_csv_rejected():
    res = parse_upload(b"hello;world;this;is;garbage", "broken.csv")
    assert res["ok"] is False


def test_non_numeric_field_warns_not_crashes():
    df = _valid_df().astype(object)  # keep everything as text for the fixture
    df.loc[0, "discount_rate"] = "not-a-number"
    res = parse_upload(df.to_csv(index=False).encode("utf-8"), "weird.csv")
    # Engine tolerates the bad numeric and warns; it must not crash.
    assert "errors" in res and "warnings" in res
    assert any("not numeric" in w.lower() for w in res["warnings"])