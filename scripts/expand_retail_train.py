# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["pyarrow==25.0.1"]
# ///
"""Add identity-selected Train histories while keeping source families and evaluations fixed."""

import argparse
import hashlib
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prepare_retail_holdout import prepare_series

from newsvendor.io import digest, jsonl, lines, read, require, write
from newsvendor.retail_expansion import owners, select
from newsvendor.suite import check


def main():
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="data/processed/core-retail-train-v1")
    parser.add_argument(
        "--protected", default="data/processed/core-retail-holdout-v1/selection.json"
    )
    parser.add_argument("--output", default="data/processed/core-retail-linked-v1")
    parser.add_argument(
        "--selection", default="results/research-checks/retail-linked-selection.json"
    )
    parser.add_argument("--identities-only", action="store_true")
    args = parser.parse_args()
    base, output, selection_path = Path(args.base), Path(args.output), Path(args.selection)
    require(not output.exists(), "Preserve prepared datasets")
    check(base)
    rows, labels = lines(base / "inputs.jsonl"), lines(base / "labels.jsonl")
    manifest = read(base / "manifest.json")
    capture = manifest["trainExpansion"]["source"]
    raw = Path(capture["path"])
    with raw.open("rb") as stream:
        require(
            hashlib.file_digest(stream, "sha256").hexdigest() == capture["sha256"],
            "Changed raw capture",
        )
    protected = read(args.protected)
    forbidden = {k for r in protected["selected"] for k in r["keys"]}
    identities = (
        pq.read_table(raw, columns=["store_id", "product_id"])
        .group_by(["store_id", "product_id"])
        .aggregate([])
        .to_pylist()
    )
    selected = select(identities, rows, forbidden, seed=53, per_family=3)
    registration = {
        "scope": "Attach up to three Train histories per existing family, never merge families, never use protected stores/products. Only identities choose additions.",
        "source": capture,
        "seed": 53,
        "perFamily": 3,
        "sourceOutcomeSelection": False,
        "selected": selected,
        "selectedHash": digest(selected),
        "identityCandidates": len(identities),
        "baseHashes": {
            n: digest((base / n).read_bytes())
            for n in ("inputs.jsonl", "labels.jsonl", "collection.jsonl", "manifest.json")
        },
        "protectedSelectionHash": digest(Path(args.protected).read_bytes()),
        "protectedKeys": sorted(forbidden),
        "scripts": {
            str(p): digest(p.read_bytes())
            for p in (
                Path(__file__).relative_to(Path.cwd()),
                Path("src/newsvendor/retail_expansion.py"),
                Path("scripts/prepare_retail_holdout.py"),
            )
        },
        "pyarrow": pa.__version__,
        "historyDays": 60,
        "horizonDays": 7,
    }
    if selection_path.exists():
        require(read(selection_path) == registration, "Identity registration changed")
    else:
        write(selection_path, registration)
    print(
        {
            "selected": len(selected),
            "familiesWithAdditions": len({r["family"] for r in selected}),
            "originalRetailTrain": sum(
                r["component"] == "retail" and r["split"] == "train" for r in rows
            ),
            "identitiesOnly": args.identities_only,
        },
        flush=True,
    )
    if args.identities_only:
        return
    require(selected, "No eligible Train additions")
    require(
        all(0 <= r["product_id"] < 1_000_000 and r["store_id"] >= 0 for r in identities),
        "Invalid identity encoding",
    )
    codes = pa.array(
        [r["store_id"] * 1_000_000 + r["product_id"] for r in selected], type=pa.int64()
    )
    daily = defaultdict(list)
    for batch in pq.ParquetFile(raw).iter_batches(batch_size=65536):
        table = pa.Table.from_batches([batch])
        keys = pc.add(pc.multiply(table["store_id"], 1_000_000), table["product_id"])
        for row in table.filter(pc.is_in(keys, value_set=codes)).to_pylist():
            daily[row["store_id"], row["product_id"]].append(row)
    additions, annotations = [], []
    for identity in selected:
        row, label = prepare_series(identity, daily[identity["store_id"], identity["product_id"]])
        row["split"] = "train"
        additions.append(row)
        annotations.append(label)
    combined, targets = rows + additions, labels + annotations
    owners(combined)
    require(
        {r["family"] for r in combined if r["component"] == "retail"}
        == {r["family"] for r in rows if r["component"] == "retail"},
        "Changed source families",
    )
    require(
        not {k for r in additions for k in owners([r])} & forbidden,
        "Protected source entered Train",
    )
    output.mkdir(parents=True)
    jsonl(output / "inputs.jsonl", combined)
    jsonl(output / "labels.jsonl", targets)
    shutil.copy2(base / "collection.jsonl", output / "collection.jsonl")
    for name in ("inputs.jsonl", "labels.jsonl"):
        require(
            (output / name).read_bytes().startswith((base / name).read_bytes()),
            "Original records changed",
        )
    manifest.update(
        inputHash=digest(combined),
        labelHash=digest(targets),
        scope="Original snapshots unchanged; extra observations within existing Train source families",
    )
    manifest["trainAttachments"] = {
        "registrationHash": digest(registration),
        "selectionFile": str(selection_path),
        "selectedHash": digest(selected),
        "addedHistories": len(additions),
        "addedByFamily": dict(Counter(r["family"] for r in additions)),
        "baseDataset": str(base),
        "baseHashes": registration["baseHashes"],
        "protectedSelectionHash": registration["protectedSelectionHash"],
        "allOriginalRowsUnchanged": True,
        "familyAssignmentsUnchanged": True,
        "sourceOutcomeSelection": False,
    }
    write(output / "manifest.json", manifest)
    checked = check(output)
    manifest["counts"] = checked["components"]
    write(output / "manifest.json", manifest)
    write(output / "selection.json", registration)
    print(
        {
            "output": str(output),
            "retail": checked["components"]["retail"],
            "sourceOverlaps": checked["sourceOverlaps"],
            "familyCountUnchanged": True,
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
