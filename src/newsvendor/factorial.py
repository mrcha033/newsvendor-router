"""Paired factorial contrasts, retaining all assigned decision outcomes."""

import math
from collections import defaultdict

import torch

from .io import require


def factorial(rows, metric):
    groups = defaultdict(dict)
    for row in rows:
        key, arm = row["id"], (row["A"], row["H"])
        require(arm not in groups[key], "Duplicate factorial cell")
        groups[key][arm] = row
    selected = []
    for key, cells in groups.items():
        require(set(cells) == {(0, 0), (0, 1), (1, 0), (1, 1)}, "Incomplete factorial block")
        require(
            len({(r["family"], r["period"], r["complete"], r["inputHash"]) for r in cells.values()})
            == 1,
            "Treatment changed observed case",
        )
        y = [cells[a, h]["metrics"][metric] for a, h in ((0, 0), (0, 1), (1, 0), (1, 1))]
        require(
            all(v is None for v in y) or all(v is not None and math.isfinite(v) for v in y),
            "Treatment-dependent outcome missingness",
        )
        if y[0] is None:
            continue
        selected.append(
            {
                "id": key,
                "family": cells[0, 0]["family"],
                "y": y,
                "contrasts": [y[2] - y[0], y[1] - y[0], y[3] - y[2] - y[1] + y[0], y[3] - y[1]],
            }
        )
    if not selected:
        return {"cases": 0, "reason": "No complete paired outcome"}
    families = sorted({r["family"] for r in selected})
    names = ["AI_without_questions", "questions_without_AI", "interaction", "AI_with_questions"]
    # Demeaning within case fits the case-fixed-effect A,H,A*H regression.
    design = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=torch.float64
    )
    design -= design.mean(0)
    x = design.repeat(len(selected), 1)
    ys = torch.tensor([r["y"] for r in selected], dtype=torch.float64)
    y = (ys - ys.mean(1, keepdim=True)).flatten()
    beta = torch.linalg.lstsq(x, y, driver="gelsd").solution.tolist()
    effects = {}
    for j, name in enumerate(names):
        values = [r["contrasts"][j] for r in selected]
        groups = {f: [r["contrasts"][j] for r in selected if r["family"] == f] for f in families}
        mean = sum(values) / len(values)
        if j < 3:
            require(math.isclose(mean, beta[j], abs_tol=1e-7), "Regression contrast disagreement")
        deleted = (
            {
                f: sum(r["contrasts"][j] for r in selected if r["family"] != f)
                / sum(r["family"] != f for r in selected)
                for f in families
            }
            if len(families) > 1
            else {}
        )
        effects[name] = {
            "effect": mean,
            "equalSourceEffect": sum(sum(v) / len(v) for v in groups.values()) / len(groups),
            "lower": sum(v < -1e-8 for v in values),
            "higher": sum(v > 1e-8 for v in values),
            "sourceEffects": {f: sum(v) / len(v) for f, v in groups.items()},
            "leaveOneSourceOut": deleted,
        }
    return {
        "cases": len(selected),
        "sourceFamilies": len(families),
        "cellMeans": dict(zip(["A0H0", "A0H1", "A1H0", "A1H1"], ys.mean(0).tolist(), strict=True)),
        "fixedEffectCoefficients": dict(zip(["A", "H", "A*H"], beta, strict=True)),
        "effects": effects,
        "pValues": None,
        "scope": "Existing Dev exploratory contrasts; source-deletion ranges are not confidence intervals",
    }
