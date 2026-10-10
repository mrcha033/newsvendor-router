"""Analyze an exported human study at the participant randomization unit."""

import argparse
import hashlib
import json
import math
import random
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import torch

from newsvendor.io import digest, require, write


def verify_export(data):
    design = data["design"]
    require(digest(data["protocol"]) == design["protocolHash"], "Exported protocol changed")
    require(
        digest(data["materials-manifest"]) == design["materialHash"], "Exported materials changed"
    )
    previous, submissions, assignments = {}, {}, {}
    request_ids = set()
    for event in data["events"]:
        person = event["participant"]
        key = (person, event["request_id"])
        require(key not in request_ids, "Duplicate event request")
        request_ids.add(key)
        require(
            event["previous_hash"] == previous.get(person, design["protocolHash"]),
            "Broken event chain",
        )
        payload = json.loads(event["payload"])
        item = {
            "participant": person,
            "requestId": event["request_id"],
            "type": event["type"],
            "at": event["at"],
            "payload": payload,
            "previousHash": event["previous_hash"],
        }
        require(digest(item) == event["hash"], "Changed event contents")
        previous[person] = event["hash"]
        if event["type"] == "submit":
            key = (person, payload["trial"])
            require(key not in submissions, "Duplicate submission event")
            submissions[key] = payload
        if event["type"] == "quiz" and payload.get("passed"):
            require(person not in assignments, "Repeated randomization")
            assignments[person] = payload["assignment"]
    exported = {(r["participant"], r["trial"]): r for r in data["answers"]}
    require(len(exported) == len(data["answers"]), "Duplicate exported answer")
    require(exported == submissions, "Answers differ from recorded submission events")
    for person in data["participants"]:
        if person["status"] == "withdrawn":
            require(
                not any(key[0] == person["id"] for key in exported), "Withdrawn responses remain"
            )
        else:
            require(
                assignments.get(person["id"]) == {"arm": person["arm"], "block": person["block"]},
                "Assignment differs from randomization event",
            )
    return {"events": len(data["events"]), "submissions": len(submissions), "verified": True}


def analyze(participants, answers, design, protocol, material_bounds):
    require(len({r["id"] for r in participants}) == len(participants), "Duplicate participant")
    by_id = {p["id"]: p for p in participants}
    require(all(p["arm"] in ("control", "ai") for p in participants), "Unassigned participant")
    groups = defaultdict(dict)
    for row in answers:
        require(row["participant"] in by_id, "Unregistered participant response")
        require(row["arm"] == by_id[row["participant"]]["arm"], "Assignment mismatch")
        if row["practice"]:
            continue
        require(row["trial"] in material_bounds, "Unknown main task")
        require(row["trial"] not in groups[row["participant"]], "Duplicate task response")
        require(
            math.isfinite(row["normalizedTotal"])
            and 0 <= row["normalizedTotal"] <= material_bounds[row["trial"]] + 1e-8,
            "Invalid bounded outcome",
        )
        groups[row["participant"]][row["trial"]] = row
    estimates = []
    for p in participants:
        rows = groups[p["id"]]
        known = sum(r["normalizedTotal"] for r in rows.values())
        missing_bound = sum(bound for key, bound in material_bounds.items() if key not in rows)
        estimates.append(
            {
                "id": p["id"],
                "arm": p["arm"],
                "block": p["block"],
                "completedTasks": len(rows),
                "complete": len(rows) == len(material_bounds),
                "lower": known / len(material_bounds),
                "upper": (known + missing_bound) / len(material_bounds),
                "mean": known / len(material_bounds) if len(rows) == len(material_bounds) else None,
            }
        )
    arms = {arm: [r for r in estimates if r["arm"] == arm] for arm in ("control", "ai")}
    report = {
        "scope": "Randomized laboratory tasks, not organizational effectiveness",
        "demo": design["demo"],
        "randomizedParticipants": len(participants),
        "completedParticipants": sum(r["complete"] for r in estimates),
        "participantOutcomes": estimates,
        "counts": {
            arm: {"assigned": len(rows), "complete": sum(r["complete"] for r in rows)}
            for arm, rows in arms.items()
        },
        "intentToTreatBounds": None,
        "completeCaseDifference": None,
        "regression": None,
        "testStatus": "No test before complete, balanced assigned outcomes",
        "protocolHash": digest(protocol),
    }
    if any(not rows for rows in arms.values()):
        report["testStatus"] = "Both randomized arms are required"
        return report
    means = {
        arm: {name: sum(r[name] for r in rows) / len(rows) for name in ("lower", "upper")}
        for arm, rows in arms.items()
    }
    report["intentToTreatBounds"] = [
        means["ai"]["lower"] - means["control"]["upper"],
        means["ai"]["upper"] - means["control"]["lower"],
    ]
    complete = {arm: [r["mean"] for r in rows if r["complete"]] for arm, rows in arms.items()}
    if all(complete.values()):
        report["completeCaseDifference"] = sum(complete["ai"]) / len(complete["ai"]) - sum(
            complete["control"]
        ) / len(complete["control"])
    if not all(r["complete"] for r in estimates):
        report["testStatus"] = (
            "Missing assigned outcomes: report bounded ITT and descriptive completers; no attrition-ignorant p-value"
        )
        return report
    blocks = defaultdict(list)
    for index, row in enumerate(estimates):
        blocks[row["block"]].append(index)
    if any(
        len(indices) != 4 or sum(estimates[i]["arm"] == "ai" for i in indices) != 2
        for indices in blocks.values()
    ):
        report["testStatus"] = "Incomplete assignment block: descriptive result only"
        return report
    order = sorted(blocks)
    a = torch.tensor([float(r["arm"] == "ai") for r in estimates], dtype=torch.float64)
    y = torch.tensor([r["mean"] for r in estimates], dtype=torch.float64)
    x = torch.tensor(
        [
            [1.0, float(r["arm"] == "ai"), *[float(r["block"] == b) for b in order[1:]]]
            for r in estimates
        ],
        dtype=torch.float64,
    )
    require(len(estimates) > x.shape[1] + 1, "Insufficient residual degrees of freedom")
    beta = torch.linalg.lstsq(x, y, driver="gelsd").solution
    residual = y - x @ beta
    inv = torch.linalg.inv(x.T @ x)
    leverage = ((x @ inv) * x).sum(1)
    meat = x.T @ ((residual.square() / (1 - leverage)).unsqueeze(1) * x)
    variance = (inv @ meat @ inv)[1, 1].clamp_min(0)
    observed = float(beta[1])
    rng = random.Random(protocol["randomizationTest"]["seed"])
    permutations = protocol["randomizationTest"]["permutations"]
    extreme = 0
    for _ in range(permutations):
        shuffled = a.clone()
        for indices in blocks.values():
            treated = set(rng.sample(indices, 2))
            for i in indices:
                shuffled[i] = float(i in treated)
        difference = float(y[shuffled == 1].mean() - y[shuffled == 0].mean())
        extreme += abs(difference) >= abs(observed) - 1e-12
    report["regression"] = {
        "formula": "participant_mean_normalized_total ~ AI + assignment_block",
        "AI": observed,
        "HC2StandardError": float(variance.sqrt()),
        "randomizationP": (1 + extreme) / (1 + permutations),
        "permutations": permutations,
        "experimentalUnit": "participant",
        "participants": len(estimates),
    }
    report["testStatus"] = (
        "All assigned outcomes observed; block-preserving randomization test of the sharp null"
    )
    if design["demo"]:
        report["testStatus"] = (
            "SOFTWARE REHEARSAL ONLY: these are not human effectiveness measurements"
        )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", required=True)
    parser.add_argument("--materials", default="results/human-materials-v2")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    require(not Path(args.output).exists(), "Preserve earlier analyses")
    with zipfile.ZipFile(args.export) as archive:
        data = {
            name: json.loads(archive.read(name + ".json"))
            for name in (
                "participants",
                "answers",
                "events",
                "protocol",
                "design",
                "materials-manifest",
            )
        }
    integrity = verify_export(data)
    root = Path(args.materials)
    from newsvendor.io import read
    from newsvendor.optimizer import risk

    manifest = read(root / "manifest.json")
    require(
        manifest == data["materials-manifest"], "Study material differs from exported experiment"
    )
    for name, item in manifest["files"].items():
        require(
            hashlib.sha256((root / name).read_bytes()).hexdigest() == item["sha256"],
            "Material file changed",
        )
    tasks = read(root / "tasks.json")
    envs = read(root / "environments.json")
    bounds = {}
    for task in tasks:
        if task["practice"]:
            continue
        value, env = task["input"], envs[task["id"]]
        worst = max(risk(q, env["theta"]) - env["optimal"]["cost"] for q in value["task"]["bounds"])
        bounds[task["id"]] = (max(worst, value["task"]["hold"]) + 8) / value["task"]["hold"]
    result = analyze(
        data["participants"], data["answers"], data["design"], data["protocol"], bounds
    )
    result["exportIntegrity"] = integrity
    result["analysisScriptHash"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result["exportHash"] = hashlib.sha256(Path(args.export).read_bytes()).hexdigest()
    participants = {p["id"]: p for p in data["participants"]}
    completed = {r["id"] for r in result["participantOutcomes"] if r["complete"]}
    result["secondaryCompleterMeans"] = {}
    for arm in ("control", "ai"):
        rows = [
            r
            for r in data["answers"]
            if not r["practice"]
            and r["participant"] in completed
            and participants[r["participant"]]["arm"] == arm
        ]
        result["secondaryCompleterMeans"][arm] = (
            {
                name: sum(float(r[name]) for r in rows) / len(rows)
                for name in (
                    "realizedOrderLoss",
                    "realizedTotal",
                    "orderRegret",
                    "questionCost",
                    "questionCount",
                    "hold",
                    "decisionSeconds",
                    "parameterAccuracy",
                    "evidenceAccuracy",
                    "confidence",
                )
            }
            if rows
            else None
        )
    deviations = [
        abs(r["q"] - r["adviceQ"])
        for r in data["answers"]
        if not r["practice"] and r["q"] is not None and r.get("adviceQ") is not None
    ]
    result["aiSuggestionDeviation"] = {
        "conditionalOn": "AI advice and a submitted order both available; descriptive, not a treatment effect",
        "tasks": len(deviations),
        "meanAbsolute": sum(deviations) / len(deviations) if deviations else None,
    }
    markdown = Path(args.output).with_suffix(".md")
    require(not markdown.exists(), "Preserve earlier narrative report")
    write(args.output, result)
    effect = result["completeCaseDifference"]
    explanation = (
        "실제 참여자 성과가 아닌 소프트웨어 리허설입니다."
        if result["demo"]
        else "생성된 실험 과제에서의 결과이며 실제 조직 업무 효과로 일반화하지 않습니다."
    )
    summary = [
        "# 인간 실험 분석",
        "",
        explanation,
        "",
        f"- 무작위 배정: {result['randomizedParticipants']}명",
        f"- 전체 과제 완료: {result['completedParticipants']}명",
        "",
        "| 조건 | 배정 | 완료 |",
        "| --- | ---: | ---: |",
    ]
    for arm, label in (("control", "AI 없음"), ("ai", "AI 있음")):
        summary.append(
            f"| {label} | {result['counts'][arm]['assigned']} | {result['counts'][arm]['complete']} |"
        )
    summary += [
        "",
        "효과는 AI 있음 − AI 없음입니다. 손실에서는 음수가 개선을 뜻합니다.",
        "",
        f"- 완료자 평균 손실 차이: {effect:.6f}"
        if effect is not None
        else "- 두 조건의 완료 자료가 아직 충분하지 않습니다.",
        f"- 중단·누락을 포함한 배정 효과 범위: {result['intentToTreatBounds']}",
        "- 위 범위는 누락 결과의 가능한 범위이며 신뢰구간이 아닙니다.",
    ]
    if result["regression"]:
        fit = result["regression"]
        summary += [
            f"- 참여자 단위 회귀 AI 계수: {fit['AI']:.6f}",
            f"- HC2 표준오차: {fit['HC2StandardError']:.6f}",
            f"- 블록 무작위화 검정 p값: {fit['randomizationP']:.6f}",
        ]
    else:
        summary += ["- 미완료 자료나 불완전 배정 블록이 있어 유의성 검정을 생략했습니다."]
    summary += [
        "",
        "질문 비용·경과 시간·모수·근거 정확도의 완료자 기술 통계와 원시 행의 hash는 같은 이름의 JSON에 있습니다. 경과 시간에는 화면을 떠난 시간도 포함됩니다.",
        "",
    ]
    markdown.write_text("\n".join(summary))
    print(
        json.dumps(
            {
                k: result[k]
                for k in ("randomizedParticipants", "completedParticipants", "testStatus")
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
