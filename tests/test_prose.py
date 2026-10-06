import pytest

from newsvendor.construction import atoms, candidates, reference
from newsvendor.corpus import document


def input_with_prose():
    input = {
        "task": {
            "sku": "A-17",
            "period": "next-day",
            "allowed": {"v": [0, 4], "b": [0, 8], "F": [[[0, 0.6], [100, 0.4]]]},
            "rho": {},
            "bounds": [0, 100],
        },
        "history": [],
        "docs": [],
        "observations": [
            {"demand": 0 if i < 6 else 100, "complete": True, "stockout": False} for i in range(10)
        ],
    }
    for id, title, text, role in (
        (
            "q",
            "Purchase quotation",
            "Each carton contains 12 units and is priced at $72 per carton.",
            "buyer",
        ),
        ("p", "Sales price list", "The retail price is $10 per unit.", "sales"),
        (
            "v",
            "Current return contract",
            "Unsold units earn a $5 refund each, less a $1 handling fee per unit.",
            "supplier",
        ),
        (
            "b",
            "Manager decision",
            "Selected policy: I choose an additional shortage cost of $8 per missed unit.",
            "manager",
        ),
    ):
        input["docs"].append(document(input, id, title, text, role))
    return input


def test_four_prose_documents_have_grounded_expressions():
    input = input_with_prose()
    assert reference(input)["values"] == {"c": 6, "p": 10, "v": 4, "b": 8}
    for slot, value in (("c", 6), ("p", 10), ("v", 4), ("b", 8)):
        assert any(c["value"] == value for c in candidates(input, slot))
    for d in input["docs"]:
        for a in atoms(d):
            lo, hi = a["span"]
            assert d["text"][lo:hi].replace(",", "") == str(int(a["value"]))


def test_word_count_and_ambiguous_currency_do_not_invent_units():
    doc = {
        "id": "s",
        "text": "A box of ten items costs USD 60. The return credit is USD 3; the basis is not specified.",
    }
    parsed = atoms(doc)
    assert any(a["value"] == 10 and doc["text"][slice(*a["span"])] == "ten" for a in parsed)
    assert any(a["value"] == 60 and a["unit"] == "currency/pack" for a in parsed)
    assert any(a["value"] == 3 and a["unit"] == "currency/unknown" for a in parsed)


@pytest.mark.parametrize("text", ["SKU-2026 expires on 2026-10-07.", "There were 100 page views."])
def test_identifiers_and_dates_are_not_currency_atoms(text):
    assert atoms({"id": "s", "text": text}) == []
