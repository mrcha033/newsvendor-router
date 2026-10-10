import copy
import math

import pytest

from newsvendor.decision_effects import (
    critical_ratio,
    pair_outcomes,
    paired_regression,
    theory_checks,
    uniform_risk,
)


def outcomes():
    rows = []
    for case, family, change in (("a", "x", -2), ("b", "x", -4), ("c", "y", 3), ("d", "z", -1)):
        for condition in ("original", "wrong_sku"):
            for method in ("parent", "scoped"):
                delta = change if condition == "wrong_sku" and method == "scoped" else 0
                rows.append(
                    {
                        "id": case,
                        "kind": "retail",
                        "family": family,
                        "period": case,
                        "complete": case != "d",
                        "condition": condition,
                        "method": method,
                        "metrics": {
                            "total": None if case == "d" else 10 + delta,
                            "questions": 10 + delta,
                        },
                    }
                )
    return rows


def test_paired_ols_retains_source_deletion_and_equal_source_weighting():
    pairs = pair_outcomes(outcomes(), "retail", "questions")
    result = paired_regression(pairs, ["original", "wrong_sku"])
    assert result["coefficients"] == pytest.approx({"intercept": 0, "wrong_sku": -1})
    changed = result["cells"]["wrong_sku"]
    assert changed["equalSourceMeanEffect"] == pytest.approx(-1 / 3)
    assert changed["leaveOneSourceOut"]["x"] == pytest.approx(1)
    assert changed["deletionRange"] == pytest.approx([-7 / 3, 1])
    assert (changed["lower"], changed["higher"], changed["unchanged"]) == (3, 1, 0)
    assert result["inference"]["pValues"] is None
    assert result["inference"]["confidenceIntervals"] is None


def test_censored_losses_do_not_remove_question_outcomes():
    rows = outcomes()
    assert len(pair_outcomes(rows, "retail", "total")) == 6
    assert len(pair_outcomes(rows, "retail", "questions")) == 8


@pytest.mark.parametrize(
    "change", ["duplicate", "unmatched", "family", "complete", "missing", "infinite"]
)
def test_invalid_pairs_fail_before_regression(change):
    rows = outcomes()
    if change == "duplicate":
        rows.append(copy.deepcopy(rows[0]))
    elif change == "unmatched":
        rows.pop(0)
    elif change in ("family", "complete"):
        rows[1][change] = "changed"
    elif change == "missing":
        rows[1]["metrics"]["total"] = None
    else:
        rows[1]["metrics"]["total"] = math.inf
    with pytest.raises(ValueError):
        pair_outcomes(rows, "retail", "total")


def test_condition_support_and_source_identity_are_checked():
    pairs = pair_outcomes(outcomes(), "retail", "questions")
    with pytest.raises(ValueError, match="Unbalanced"):
        paired_regression(pairs[1:], ["original", "wrong_sku"])
    pairs[1]["family"] = "different"
    with pytest.raises(ValueError, match="provenance"):
        paired_regression(pairs, ["original", "wrong_sku"])


def test_analytic_newsvendor_curvature_and_preference_sensitivity():
    checks = theory_checks()
    assert len(checks["cases"]) == 6
    for row in checks["cases"]:
        c, p, v, b = (row[k] for k in ("c", "p", "v", "b"))
        q, width = row["qStar"], row["width"]
        assert uniform_risk(q, 0, width, c, p, v, b) < uniform_risk(q + 0.1, 0, width, c, p, v, b)
        assert uniform_risk(q, 0, width, c, p, v, b) < uniform_risk(q - 0.1, 0, width, c, p, v, b)
        assert row["actualRegret"] == pytest.approx(row["curvatureRegret"])


@pytest.mark.parametrize("params", [(1, 1, 1, 0), (3, 1, 0, 0), (2, math.nan, 0, 1)])
def test_critical_ratio_rejects_outside_the_stated_interior_assumptions(params):
    with pytest.raises(ValueError):
        critical_ratio(*params)
