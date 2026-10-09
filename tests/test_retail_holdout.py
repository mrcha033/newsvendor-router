import importlib.util
from copy import deepcopy
from datetime import date, timedelta
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "prepare_retail_holdout",
    Path(__file__).resolve().parents[1] / "scripts/prepare_retail_holdout.py",
)
holdout = importlib.util.module_from_spec(spec)
spec.loader.exec_module(holdout)


def test_identity_selection_excludes_stores_and_products_without_reading_outcomes():
    identities = [
        {"store_id": s, "product_id": p, "sale_amount": 900, "stock_hour6_22_cnt": 0}
        for s, p in [(1, 3), (2, 2), (2, 3), (3, 3), (4, 4), (4, 4)]
    ]
    excluded = {"retail-store:1", "retail-product:2"}
    selected = holdout.select_sources(identities, excluded, 61)
    other = deepcopy(identities[::-1])
    for row in other:
        row.update(sale_amount=-123, stock_hour6_22_cnt=99)
    assert selected == holdout.select_sources(other, excluded, 61)
    keys = [key for row in selected for key in row["keys"]]
    assert len(keys) == len(set(keys))
    assert not set(keys) & excluded
    assert len(selected) == 2


def series():
    return [
        {
            "dt": (date(2024, 3, 28) + timedelta(days=i)).isoformat(),
            "sale_amount": float(i),
            "stock_hour6_22_cnt": 0,
            "hours_sale": [0.0] * 24,
            "hours_stock_status": [0] * 24,
            "discount": 1,
            "holiday_flag": 0,
            "activity_flag": 0,
            "precpt": 0,
            "avg_temperature": 20,
            "avg_humidity": 0.5,
            "avg_wind_level": 1,
        }
        for i in range(67)
    ]


def test_holdout_future_targets_cannot_change_observed_input():
    identity = holdout.select_sources([{"store_id": 1, "product_id": 2}], set(), 61)[0]
    days = series()
    row, label = holdout.prepare_series(identity, days)
    changed = deepcopy(days)
    for day in changed[60:]:
        day.update(sale_amount=10000, stock_hour6_22_cnt=16)
    other_row, other_label = holdout.prepare_series(identity, changed)
    assert row == other_row
    assert label != other_label
    assert row["split"] == "test"
    assert len(row["input"]["observations"]) == 60
    assert row["input"]["observations"][-1]["date"] < min(label["target"]["dates"])
    assert label["target"]["latentDemandLabels"] is False


def test_missing_selected_source_period_fails_without_replacement():
    identity = holdout.select_sources([{"store_id": 1, "product_id": 2}], set(), 61)[0]
    with pytest.raises(ValueError, match="Insufficient holdout"):
        holdout.prepare_series(identity, series()[:-1])
    days = series()
    days[40]["dt"] = days[39]["dt"]
    with pytest.raises(ValueError, match="calendar gap or duplicate"):
        holdout.prepare_series(identity, days)


def test_paired_loss_keeps_censored_bounds_out_and_counts_source_groups(monkeypatch):
    directory = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(directory))
    spec = importlib.util.spec_from_file_location(
        "evaluate_retail_holdout", directory / "evaluate_retail_holdout.py"
    )
    evaluation = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluation)
    old, new = [], []
    for i in range(5):
        family = "a" if i < 2 else "b"
        row = {
            "id": str(i),
            "family": family,
            "outcomeComplete": i < 4,
            "total": 10 if i < 4 else None,
            "events": [{"input": {"task": {"sku": family, "period": "week"}}}],
        }
        old.append(row)
        new.append({**row, "total": (9 if i < 2 else 8) if i < 4 else None})
    result = evaluation.paired_difference(new, old, "total", complete=True)
    assert result["meanDifference"] == -1.5
    assert result["conditions"] == 4
    assert result["uniquePeriods"] == result["sourceFamilies"] == 2
    assert result["sourceBootstrap95"] == [-2.0, -1.0]
    unavailable = evaluation.paired_difference(new[-1:], old[-1:], "total", complete=True)
    assert unavailable["meanDifference"] is None
    assert unavailable["conditions"] == 0
