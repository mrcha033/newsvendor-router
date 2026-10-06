"""Portable licensed evaluation captures, separate from the original large raw downloads."""

import io
import lzma
import tarfile
from pathlib import Path

from .io import digest, read, require, write
from .orders import check as check_orders
from .suite import check as check_suite


def pack():
    paths = [Path("docs/sources.md"), Path("src/newsvendor/retail/LICENSE")]
    for directory in ("complementary", "orders"):
        paths += sorted(
            p
            for p in (Path("data/processed") / directory).iterdir()
            if p.suffix in {".json", ".jsonl"}
        )
    content = io.BytesIO()
    with tarfile.open(fileobj=content, mode="w") as archive:
        for path in paths:
            data = path.read_bytes()
            info = tarfile.TarInfo(str(path))
            info.size, info.mode, info.mtime = len(data), 0o644, 0
            archive.addfile(info, io.BytesIO(data))
    destination = Path("cases/evaluation.tar.xz")
    destination.write_bytes(lzma.compress(content.getvalue(), preset=6))
    receipt = {
        "sha256": digest(destination.read_bytes()),
        "bytes": destination.stat().st_size,
        "files": {str(p): digest(p.read_bytes()) for p in paths},
        "licenses": read("configs/complementary.json")["sources"],
        "orders": read("configs/orders.json"),
        "scope": "Exact processed inputs, labels and public collections for reproducible evaluation; original annotations stay separate",
    }
    write("cases/evaluation.json", receipt)
    return {k: receipt[k] for k in ("sha256", "bytes")}


def restore():
    receipt = read("cases/evaluation.json")
    path = Path("cases/evaluation.tar.xz")
    require(digest(path.read_bytes()) == receipt["sha256"], "Evaluation snapshot changed")
    with tarfile.open(path, "r:xz") as archive:
        for item in archive:
            require(item.isfile() and item.name in receipt["files"], "Unexpected snapshot member")
            data = archive.extractfile(item).read()
            require(digest(data) == receipt["files"][item.name], "Snapshot member changed")
            destination = Path(item.name)
            # Documentation and license captures are verified, not installed over local code.
            if item.name.startswith("data/processed/"):
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.exists():
                    require(
                        digest(destination.read_bytes()) == digest(data),
                        "Existing evaluation data differs; use a clean checkout",
                    )
                else:
                    destination.write_bytes(data)
    return {
        "components": check_suite("data/processed/complementary"),
        "orders": check_orders(read("configs/orders.json")),
    }
