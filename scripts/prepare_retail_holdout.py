# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["pyarrow==25.0.1"]
# ///
"""Freeze unseen retail sources before reading outcomes, preserving every existing split."""

import argparse
import hashlib
import sys
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from newsvendor.io import digest, jsonl, lines, read, require, write
from newsvendor.suite import check, payload, public_input


def source_keys(store, product):
    return [f"retail-store:{store}", f"retail-product:{product}"]


def select_sources(identities, excluded, seed):
    """Identity-only greedy matching; input ordering and outcomes cannot affect selection."""
    ordered = sorted(
        {(r["store_id"], r["product_id"]) for r in identities},
        key=lambda pair: digest([seed, "fresh-evaluation-v1", *pair]),
    )
    selected, used = [], set(excluded)
    for store, product in ordered:
        keys = source_keys(store, product)
        if used.intersection(keys):
            continue
        selected.append(
            {
                "store_id": store,
                "product_id": product,
                "keys": keys,
                "family": digest(sorted(keys)),
            }
        )
        used.update(keys)
    return selected


def prepare_series(identity, series):
    """Keep the 60 observed days separate from the seven outcome annotations."""
    fields = {
        "date": "dt",
        "sales": "sale_amount",
        "stockoutHours": "stock_hour6_22_cnt",
        "hourlySales": "hours_sale",
        "hourlyStockStatus": "hours_stock_status",
        "discount": "discount",
        "holidayFlag": "holiday_flag",
        "activityFlag": "activity_flag",
        "precipitation": "precpt",
        "temperature": "avg_temperature",
        "humidity": "avg_humidity",
        "wind": "avg_wind_level",
    }
    series = sorted(series, key=lambda row: row["dt"])
    require(len(series) >= 67, "Insufficient holdout history; do not replace selected sources")
    series = series[:67]
    dates = [row["dt"] for row in series]
    require(
        dates
        == [(date.fromisoformat(dates[0]) + timedelta(days=i)).isoformat() for i in range(67)],
        "Holdout calendar gap or duplicate; do not replace selected sources",
    )
    store, product = identity["store_id"], identity["product_id"]
    past, future = series[:60], series[60:]
    row = {
        "id": f"retail:{store}:{product}",
        "component": "retail",
        "family": identity["family"],
        "split": "test",
        "input": payload(
            f"Predict the next 7-day total demand distribution after {past[-1]['dt']} for store {store}, product {product}. "
            "Return F as [[demand, probability], ...] in globally normalized sales units, with nonnegative demand and "
            "probabilities summing to one. Use sales history and stockout information to estimate demand.",
            observations=[{name: day[source] for name, source in fields.items()} for day in past],
        ),
    }
    public_input(row)
    label = {
        "id": row["id"],
        "leakageKeys": identity["keys"],
        "officialSplit": "train",
        "target": {
            "action": "answer",
            "answer": [day["sale_amount"] for day in future],
            "dates": [day["dt"] for day in future],
            "complete": [day["stock_hour6_22_cnt"] == 0 for day in future],
            "latentDemandLabels": False,
            "cutoff": past[-1]["dt"],
        },
    }
    return row, label


def main():
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", required=True)
    args = parser.parse_args()
    plan = read(args.registration)
    require(plan["trainingAllowed"] is False, "Frozen evaluation only")
    require(
        digest(Path(__file__).read_bytes())
        == plan["scripts"][str(Path(__file__).relative_to(Path.cwd()))],
        "Changed preparation script",
    )
    base, output = Path(plan["baseDataset"]), Path(plan["dataset"])
    require(not output.exists(), "Preserve prepared holdout")
    check(base)
    for name, expected in plan["baseHashes"].items():
        require(digest((base / name).read_bytes()) == expected, "Existing dataset changed")
    for entry in plan["models"].values():
        require(digest(Path(entry["path"]).read_bytes()) == entry["hash"], "Frozen model changed")
    raw = Path(plan["source"]["path"])
    with raw.open("rb") as stream:
        require(
            hashlib.file_digest(stream, "sha256").hexdigest() == plan["source"]["sha256"],
            "Changed source capture",
        )
    old, old_labels = lines(base / "inputs.jsonl"), lines(base / "labels.jsonl")
    excluded = set()
    for row in old:
        if row["component"] == "retail":
            _, store, product = row["id"].split(":")
            excluded.update(source_keys(store, product))
    identities = (
        pq.read_table(raw, columns=["store_id", "product_id"])
        .group_by(["store_id", "product_id"])
        .aggregate([])
        .to_pylist()
    )
    selected = select_sources(identities, excluded, plan["selectionSeed"])
    require(len(selected) == plan["expectedSources"], "Identity catalog changed")
    frozen = {
        "registrationHash": digest(Path(args.registration).read_bytes()),
        "source": plan["source"],
        "selectionSeed": plan["selectionSeed"],
        "excludedKeys": sorted(excluded),
        "selected": selected,
        "selectedHash": digest(selected),
        "outcomeSelection": False,
        "pyarrow": pa.__version__,
        "identityCandidates": len(identities),
        "originalHashes": plan["baseHashes"],
    }
    output.mkdir(parents=True)
    write(output / "selection.json", frozen)
    print({"stage": "identities_frozen", "sources": len(selected)}, flush=True)
    codes = pa.array(
        [r["store_id"] * 1_000_000 + r["product_id"] for r in selected], type=pa.int64()
    )
    require(
        all(0 <= r["product_id"] < 1_000_000 and r["store_id"] >= 0 for r in identities),
        "Invalid encoded identity",
    )
    daily = defaultdict(list)
    for batch in pq.ParquetFile(raw).iter_batches(batch_size=65536):
        table = pa.Table.from_batches([batch])
        keys = pc.add(pc.multiply(table["store_id"], 1_000_000), table["product_id"])
        for row in table.filter(pc.is_in(keys, value_set=codes)).to_pylist():
            daily[row["store_id"], row["product_id"]].append(row)
    additions = [prepare_series(r, daily[r["store_id"], r["product_id"]]) for r in selected]
    rows = old + [r for r, _ in additions]
    labels = old_labels + [r for _, r in additions]
    jsonl(output / "inputs.jsonl", rows)
    jsonl(output / "labels.jsonl", labels)
    (output / "collection.jsonl").write_bytes((base / "collection.jsonl").read_bytes())
    manifest = read(base / "manifest.json")
    manifest.update(inputHash=digest(rows), labelHash=digest(labels))
    manifest["scope"] = (
        "Existing inputs unchanged plus frozen unseen-source evaluation; original public Train split reused only for previously unused sources, not model training"
    )
    manifest["evaluationExpansion"] = {
        "selectionHash": digest(frozen),
        "ids": [r["id"] for r, _ in additions],
        "families": [r["family"] for r, _ in additions],
        "sources": len(additions),
        "legacyTestChanged": False,
        "trainingAllowed": False,
        "modelHashes": {name: r["hash"] for name, r in plan["models"].items()},
    }
    write(output / "manifest.json", manifest)
    verified = check(output)
    manifest["counts"] = verified["components"]
    write(output / "manifest.json", manifest)
    for name, expected in plan["baseHashes"].items():
        require(digest((base / name).read_bytes()) == expected, "Original dataset modified")
    require(
        rows[: len(old)] == old and labels[: len(old_labels)] == old_labels, "Original rows changed"
    )
    write(
        output / "audit.json",
        {
            "checks": verified,
            "originalRowsUnchanged": True,
            "sources": len(additions),
            "selectedHash": digest(selected),
            "registrationHash": frozen["registrationHash"],
        },
    )
    print(
        {"stage": "prepared", "sources": len(additions), "overlaps": verified["sourceOverlaps"]},
        flush=True,
    )


if __name__ == "__main__":
    main()
