"""Attach retail Train histories without joining existing source families."""

from collections import Counter

from .io import digest, require


def source_keys(store, product):
    return [f"retail-store:{store}", f"retail-product:{product}"]


def owners(rows):
    """Every retail identity must belong to exactly one family and split."""
    result = {}
    for row in rows:
        if row["component"] != "retail":
            continue
        _, store, product = row["id"].split(":")
        for key in source_keys(store, product):
            value = (row["family"], row["split"])
            require(
                result.setdefault(key, value) == value, "Retail identity crosses source families"
            )
    return result


def select(identities, rows, protected, seed=53, per_family=3):
    """Identity-only additions, anchored to Train, with no family mergers.

    A free store or product can join one existing Train family. Edges between
    different families are always rejected, including edges created by earlier
    additions. Existing family ids and all Train/Dev/Test assignments stay fixed.
    """
    require(type(per_family) is int and per_family > 0, "Positive addition limit required")
    assigned = owners(rows)
    protected = set(protected) | {k for k, (_, split) in assigned.items() if split != "train"}
    require(
        not protected & {k for k, (_, split) in assigned.items() if split == "train"},
        "Protected identity already appears in Train",
    )
    existing = {r["id"] for r in rows if r["component"] == "retail"}
    pairs = sorted(
        {(r["store_id"], r["product_id"]) for r in identities},
        key=lambda pair: digest([seed, "anchored-retail-train-v1", *pair]),
    )
    selected, counts, seen = [], Counter(), set()
    while True:
        added = 0
        for store, product in pairs:
            if (store, product) in seen or f"retail:{store}:{product}" in existing:
                continue
            keys = source_keys(store, product)
            if protected.intersection(keys):
                continue
            groups = {assigned[k][0] for k in keys if k in assigned}
            if len(groups) != 1:
                continue
            family = next(iter(groups))
            if counts[family] >= per_family:
                continue
            for key in keys:
                require(
                    assigned.setdefault(key, (family, "train")) == (family, "train"),
                    "Source family merger",
                )
            selected.append(
                {"store_id": store, "product_id": product, "keys": keys, "family": family}
            )
            seen.add((store, product))
            counts[family] += 1
            added += 1
        if not added:
            break
    return selected
