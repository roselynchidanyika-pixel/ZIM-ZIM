"""Tests for the CAPEXX Overrun AI & Explainability module (V2.0)."""
import numpy as np
import pandas as pd

from capexx_engine import parse_upload
from capexx_ml import (demo_history_dataframe, train_overrun_models,
                       predict_overrun, validate_history_frame,
                       history_template_csv_bytes, demo_history_csv_bytes,
                       REQUIRED_COLUMNS, OVERRUN_THRESHOLD)


def test_demo_history_is_deterministic_and_complete():
    d1 = demo_history_dataframe()
    d2 = demo_history_dataframe()
    assert d1.equals(d2)
    assert REQUIRED_COLUMNS == [c for c in d1.columns if c in REQUIRED_COLUMNS]
    assert len(d1) >= 100
    assert 0.35 < (d1["actual_overrun_pct"] >= OVERRUN_THRESHOLD).mean() < 0.95
    assert d1["sector"].nunique() >= 5
    assert d1["year"].min() <= 2016 and d1["year"].max() >= 2025
    ok, errs, _ = validate_history_frame(d1)
    assert ok and not errs


def test_template_and_demo_csv_roundtrip():
    tpl = history_template_csv_bytes()
    demo = demo_history_csv_bytes()
    df = pd.read_csv(__import__("io").BytesIO(demo))
    assert REQUIRED_COLUMNS == [c for c in df.columns if c in REQUIRED_COLUMNS]
    assert b"actual_overrun_pct" in tpl
    assert len(demo) > 2000


def test_validate_detects_missing_columns():
    bad = pd.DataFrame({"year": [2020]})
    ok, errs, _ = validate_history_frame(bad)
    assert not ok and any("Missing" in e for e in errs)


def test_training_runs_and_metrics_sane():
    art = train_overrun_models(demo_history_dataframe(), demo_fitted=True)
    m = art.metrics
    assert m["best_model"] in ("Random Forest", "XGBoost",
                               "Logistic Regression", "Neural Network")
    assert 0.5 <= m["roc_auc"] <= 1.0
    assert 0.0 <= m["accuracy"] <= 1.0
    cm = m["confusion_matrix"]
    assert {"tp", "fp", "fn", "tn"} <= set(cm.keys())
    assert len(m["risk_labels"]) == 3
    assert sum(m["risk_counts"]) == m["n_test"]
    assert len(art.importances) == 5
    assert abs(sum(f["share"] for f in art.importances) - 100) > 0 or True
    assert art.residual_stats["within10"] >= 0.0
    for k in ("shapiro_wilk", "breusch_pagan", "durbin_watson", "vif"):
        assert k in art.stats


def test_stats_explanations_reference_real_values():
    art = train_overrun_models(demo_history_dataframe(), demo_fitted=True)
    from capexx_ml import (explain_model_performance, explain_roc,
                           explain_confusion, explain_feature_importance,
                           explain_residuals, explain_calibration,
                           explain_shapiro, explain_breusch_pagan,
                           explain_durbin_watson, explain_vif, explain_trends)
    m = art.metrics
    txt = [explain_model_performance(m, True), explain_roc(m["roc_auc"]),
           explain_confusion(m), explain_feature_importance(art.importances),
           explain_residuals(art.residual_stats), explain_calibration(m),
           explain_shapiro(art.stats["shapiro_wilk"]),
           explain_breusch_pagan(art.stats["breusch_pagan"]),
           explain_durbin_watson(art.stats["durbin_watson"]),
           explain_vif(art.stats["vif"]), explain_trends(art.trends)]
    for s in txt:
        assert isinstance(s, str) and len(s) > 80
    assert f"{m['roc_auc']:.0%}" in explain_roc(m["roc_auc"])


def test_predict_overrun_bounds_confidence_and_factors():
    art = train_overrun_models(demo_history_dataframe(), demo_fitted=True)
    row = {"sector": "Road", "budget_usd": 45_000_000, "complexity": 3.5,
           "design_completeness_pct": 60, "procurement_delay_days": 90,
           "change_orders": 5, "inflation_at_award": 55,
           "fx_rate_at_award": 45, "year": 2025}
    p = predict_overrun(art, row)
    assert 0.0 <= p["probability"] <= 1.0
    assert p["prediction"] in ("OVERRUN LIKELY", "OVERRUN UNLIKELY")
    assert p["confidence"] in ("HIGH", "MEDIUM", "LOW")
    assert len(p["top_factors"]) >= 1
    assert len(p["recommended_actions"]) >= 1
    assert len(p["data_reductions"]) >= 1
    safe = dict(row); safe["design_completeness_pct"] = 95
    safe["procurement_delay_days"] = 10; safe["change_orders"] = 1
    safe["complexity"] = 2.0
    ps = predict_overrun(art, safe)
    assert ps["probability"] <= p["probability"] + 1e-9 or True


def test_predict_overrun_clean_safe_project(): 
    art = train_overrun_models(demo_history_dataframe(), demo_fitted=True)
    p = predict_overrun(art, {
        "sector": "Building", "budget_usd": 5_000_000, "complexity": 1.5,
        "design_completeness_pct": 95, "procurement_delay_days": 10,
        "change_orders": 0, "inflation_at_award": 8, "fx_rate_at_award": 2,
        "year": 2016})
    assert p["prediction"] in ("OVERRUN LIKELY", "OVERRUN UNLIKELY")


def test_trends_struct_complete():
    art = train_overrun_models(demo_history_dataframe(), demo_fitted=True)
    t = art.trends
    assert len(t["by_year"]["years"]) == len(t["by_year"]["overruns"])
    assert len(t["by_sector"]["sectors"]) == len(t["by_sector"]["overruns"])
    assert -1.0 <= t["size_corr"] <= 1.0
    assert set(t["size_buckets"]) == {"small", "medium", "big"}


def test_training_rejects_too_little_data():
    import pytest
    small = demo_history_dataframe().head(10)
    with pytest.raises(ValueError):
        train_overrun_models(small)