import copy

import pytest

from newsvendor.io import lines, read
from newsvendor.workload import check, comparison_contract, observe, partition, public_input


def fixtures():
    return lines("cases/pilot/inputs.jsonl"), lines("cases/pilot/annotations.jsonl")


def test_raw_fixture_numeric_and_response_contracts():
    rows, labels = fixtures()
    result = check(rows, labels, read("configs/workload-v2.json"))
    assert result["split"]["episodes"] == 14
    assert result["split"]["templates"] == 1  # Shared boilerplate cannot become a new test source.
    assert len(result["numericFixtures"]) == 6
    assert result["humanReviewed"] == result["organizationalFamilies"] == 0
    assert not result["primaryEvaluationReady"]
    assert not result["comparisonContract"]["allPrimaryAdaptersReady"]
    assert result["caseScopes"] == {"core": 12, "extension": 1, "out-of-scope": 1}


def test_private_labels_and_inferred_scope_cannot_be_provider_input():
    rows, labels = fixtures()
    payload = public_input(rows[0])
    labels[0]["numeric"]["theta"]["c"] = 1e8
    assert payload == public_input(rows[0])
    contaminated = copy.deepcopy(rows[0])
    contaminated["input"]["task"]["rho"] = {"t02": 0.98}
    with pytest.raises(ValueError, match="Generator response model"):
        public_input(contaminated)
    contaminated = copy.deepcopy(rows[0])
    contaminated["input"]["docs"][0]["sku"] = "BK-314"
    with pytest.raises(ValueError, match="Semantic document annotations"):
        public_input(contaminated)
    contaminated = copy.deepcopy(rows[0])
    contaminated["input"]["history"].append({"futureResponses": labels[0]})
    with pytest.raises(ValueError, match="Private annotations"):
        public_input(contaminated)
    contaminated = copy.deepcopy(rows[0])
    contaminated["input"]["docs"][0]["source"]["sentAt"] = "2026-10-06T10:00:00Z"
    with pytest.raises(ValueError, match="Future document"):
        public_input(contaminated)


def test_source_and_template_lineage_stay_in_one_partition():
    rows, _ = fixtures()
    first, second = copy.deepcopy(rows[:2])
    first["split"], second["split"] = "train", "test"
    second["family"] = first["family"]
    with pytest.raises(ValueError, match="family crosses splits"):
        partition([first, second])
    second["family"] = "different-source"
    second["template"] = first["template"]
    with pytest.raises(ValueError, match="template crosses splits"):
        partition([first, second])


def test_late_response_does_not_add_evidence():
    rows, labels = fixtures()
    row = next(r for r in rows if r["id"] == "4ab6")
    label = next(a for a in labels if a["id"] == row["id"])
    event = label["environment"][0]
    updated = observe(public_input(row), event["tool"], event["response"])
    assert updated["docs"] == row["input"]["docs"]
    assert updated["history"][-1]["outcome"] == "timeout"
    assert updated["history"][-1]["elapsedMinutes"] == 7
    assert updated["task"]["requestLimit"] == row["input"]["task"]["requestLimit"] - 1
    with pytest.raises(ValueError, match="after cutoff"):
        observe(updated, event["tool"], event["response"])


def test_partial_demand_response_preserves_censored_sales():
    rows, labels = fixtures()
    row = next(r for r in rows if r["id"] == "d2c8")
    label = next(a for a in labels if a["id"] == row["id"])
    event = label["environment"][0]
    updated = observe(public_input(row), event["tool"], event["response"])
    assert updated["observations"] == row["input"]["observations"]
    assert updated["history"][-1]["outcome"] == "partial"
    assert updated["docs"][: len(row["input"]["docs"])] == row["input"]["docs"]
    bad = copy.deepcopy(row)
    bad["input"]["observations"][0]["demand"] = 500
    with pytest.raises(ValueError, match="Latent demand"):
        public_input(bad)


@pytest.mark.parametrize(
    "key,value",
    [
        ("stateAccess", "reference"),
        ("responseModel", "exact-environment"),
        ("publicView", "candidate-menu"),
    ],
)
def test_privileged_baselines_cannot_enter_primary_track(key, value):
    design = read("configs/workload-v2.json")
    design["comparisons"][0][key] = value
    with pytest.raises(ValueError):
        comparison_contract(design)


def test_constraints_outside_solver_scope_have_no_scalar_numeric_score():
    rows, labels = fixtures()
    label = next(a for a in labels if a["scope"] == "extension")
    label["numeric"] = labels[0]["numeric"]
    with pytest.raises(ValueError, match="Unsupported constraints"):
        check(rows, labels, read("configs/workload-v2.json"))
