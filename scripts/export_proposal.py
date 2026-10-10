# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["pypandoc-binary==1.15"]
# ///
"""Export the canonical research plan to Word while retaining the archived predecessor."""

import hashlib
import json
from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import ZipFile

import pypandoc


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    root = Path(__file__).resolve().parents[1]
    source, target = root / "docs/research.md", root / "docs/proposal.docx"
    metadata = root / "docs/source.json"
    old = json.loads(metadata.read_text())
    archive = json.loads((root / "docs/evidence/research-document-archive.json").read_text())
    assert sha(root / archive["path"]) == archive["sha256"], "Missing original document archive"
    assert sha(target) in {old["sha256"], archive["files"]["docs/proposal.docx"]["sha256"]}, (
        "Preserve unexpected manual edits before exporting"
    )
    pypandoc.convert_file(
        str(source),
        "docx",
        format="markdown+tex_math_single_backslash",
        outputfile=str(target),
        extra_args=["--standalone", "--toc", "--toc-depth=2", "--metadata=lang:ko-KR"],
    )
    with ZipFile(target) as document:
        tree = ET.fromstring(document.read("word/document.xml"))
        equations = len(
            tree.findall(".//{http://schemas.openxmlformats.org/officeDocument/2006/math}oMath")
        )
        assert equations == source.read_text().count("\\["), (
            "Some mathematical expressions were not converted to Word equations"
        )
        text = "".join(tree.itertext())
        assert "수리 모형" in text and "회귀" in text
    result = {
        **{k: old[k] for k in ("originalName", "originalSha256", "originalBytes")},
        "sha256": sha(target),
        "bytes": target.stat().st_size,
        "revision": old["revision"] + 1,
        "updatedAt": "2026-10-10",
        "canonicalSource": "docs/research.md",
        "canonicalSourceSha256": sha(source),
        "exportScript": "scripts/export_proposal.py",
        "exportScriptSha256": sha(__file__),
        "pandocVersion": str(pypandoc.get_pandoc_version()),
        "wordEquations": equations,
        "previousDocuments": archive,
        "measurementBaseCommit": "3dba348bd4742e82d8211a5db0adcbbe50831396",
    }
    metadata.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"document": str(target), "equations": equations, "sha256": result["sha256"]}))


if __name__ == "__main__":
    main()
