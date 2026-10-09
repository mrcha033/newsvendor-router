import copy
import runpy

import pytest

from newsvendor.io import digest, jsonl, write
from newsvendor.retail_expansion import owners, select, source_keys


def row(store, product, family, split="train"):
    return {
        "id": f"retail:{store}:{product}",
        "component": "retail",
        "family": family,
        "split": split,
    }


def test_retail_attachment_never_merges_families_or_uses_protected_identities():
    original = [row(1, 11, "a"), row(2, 22, "b"), row(8, 88, "dev", "dev")]
    pairs = [
        (1, 11),
        (2, 22),
        (3, 11),
        (3, 22),
        (3, 44),
        (1, 55),
        (1, 66),
        (8, 11),
        (1, 88),
        (9, 11),
        (1, 99),
    ]
    identities = [{"store_id": s, "product_id": p} for s, p in pairs]
    protected = source_keys(9, 99)
    selected = select(identities, original, protected, seed=53, per_family=2)
    assert selected
    combined = original + [row(r["store_id"], r["product_id"], r["family"]) for r in selected]
    assigned = owners(combined)
    assert all(assigned[k] == v for k, v in owners(original).items())
    assert all(not set(r["keys"]) & set(protected + source_keys(8, 88)) for r in selected)
    assert len({(r["store_id"], r["product_id"]) for r in selected}) == len(selected)
    assert all(sum(r["family"] == f for r in selected) <= 2 for f in ("a", "b"))
    assert not ({(3, 11), (3, 22)} <= {(r["store_id"], r["product_id"]) for r in selected})
    changed = copy.deepcopy(original)
    for r in changed:
        r["input"] = {"observations": [{"sales": 1e12, "stockoutHours": 24}]}
        r["target"] = {"hidden": "never select by these values"}
    assert select(list(reversed(identities)), changed, protected, seed=53, per_family=2) == selected


def test_retail_attachment_rejects_existing_cross_family_or_protected_train_sources():
    with pytest.raises(ValueError, match="crosses source families"):
        owners([row(1, 11, "a"), row(1, 22, "b")])
    with pytest.raises(ValueError, match="Protected identity already appears"):
        select([], [row(1, 11, "a")], source_keys(1, 22))
    with pytest.raises(ValueError, match="Positive addition limit"):
        select([], [row(1, 11, "a")], [], per_family=0)


def test_training_comparison_checks_immutable_rows_and_external_holdout_registration(tmp_path):
    audit = runpy.run_path("scripts/study_demand.py")["attachment_audit"]
    base, expanded = tmp_path / "base", tmp_path / "expanded"
    original = [row(1, 11, "a"), row(8, 88, "dev", "dev")]
    labels = [{"id": r["id"], "target": {"answer": [1]}} for r in original]
    jsonl(base / "inputs.jsonl", original)
    jsonl(base / "labels.jsonl", labels)
    jsonl(base / "collection.jsonl", [])
    write(base / "manifest.json", {"original": True})
    protected = tmp_path / "protected.json"
    write(protected, {"selected": [{"keys": source_keys(9, 99)}]})
    selected = [{"store_id": 3, "product_id": 11, "family": "a"}]
    registration = {
        "selected": selected,
        "protectedKeys": source_keys(9, 99),
        "protectedSelectionHash": digest(protected.read_bytes()),
    }
    combined = original + [row(3, 11, "a")]
    targets = labels + [{"id": "retail:3:11", "target": {"answer": [2]}}]
    jsonl(expanded / "inputs.jsonl", combined)
    jsonl(expanded / "labels.jsonl", targets)
    write(expanded / "selection.json", registration)
    write(
        expanded / "manifest.json",
        {
            "trainAttachments": {
                "baseDataset": str(base),
                "baseHashes": {p.name: digest(p.read_bytes()) for p in base.iterdir()},
                "registrationHash": digest(registration),
                "selectedHash": digest(selected),
                "protectedSelectionHash": registration["protectedSelectionHash"],
                "addedHistories": 1,
            }
        },
    )
    assert audit(expanded, base, protected)["allOriginalRowsAndLabelsUnchanged"]
    combined[-1]["split"] = "dev"
    jsonl(expanded / "inputs.jsonl", combined)
    with pytest.raises(ValueError, match="Only Train retail additions"):
        audit(expanded, base, protected)
    combined[-1]["split"] = "train"
    jsonl(expanded / "inputs.jsonl", combined)
    targets[1]["target"]["answer"] = [999]
    jsonl(expanded / "labels.jsonl", targets)
    with pytest.raises(ValueError, match="Reference rows changed"):
        audit(expanded, base, protected)
    targets[1]["target"]["answer"] = [1]
    jsonl(expanded / "labels.jsonl", targets)
    write(protected, {"selected": [{"keys": source_keys(10, 99)}]})
    with pytest.raises(ValueError, match="Changed protected source registration"):
        audit(expanded, base, protected)
