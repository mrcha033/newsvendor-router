"""Private study state and server-only answers for the participant experiment."""

import copy
import csv
import hashlib
import hmac
import io
import json
import math
import random
import re
import secrets
import socket
import sqlite3
import time
import zipfile
from pathlib import Path
from statistics import NormalDist
from urllib.parse import urlsplit

from flask import Flask, jsonify, make_response, request, send_file

from newsvendor.corpus import outcome
from newsvendor.io import digest, read, require, write
from newsvendor.optimizer import risk

HERE = Path(__file__).resolve().parent
TRANSLATIONS = [
    (r"Purchase quotation", "매입 견적서"),
    (r"Sales price list", "판매 가격표"),
    (r"Current return contract", "반품 계약서"),
    (r"Manager decision", "매니저 정책"),
    (r"Pack price", "묶음 가격"),
    (r"units per pack", "묶음당 수량"),
    (r"Purchase cost per unit", "단위 매입가"),
    (r"Selling price per unit", "단위 판매가"),
    (r"Refund per unit", "단위 환불액"),
    (r"handling fee per unit", "단위 처리 수수료"),
    (r"Unlimited returns", "반품 수량 제한 없음"),
    (r"Selected additional shortage cost per unit", "선택된 단위 추가 부족 비용"),
]


def code_hashes():
    root = HERE.parents[1]
    paths = [
        p
        for p in HERE.rglob("*")
        if p.is_file() and p.suffix in (".py", ".js", ".css", ".html", ".json")
    ]
    paths += list((root / "src/newsvendor").glob("*.py"))
    paths += [root / "pyproject.toml", root / "uv.lock"]
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)
    }


def translate(text):
    for pattern, replacement in TRANSLATIONS:
        text = re.sub(pattern, replacement, text, flags=re.I)
    return text


def local_addresses():
    try:
        return sorted(
            {
                row[4][0]
                for row in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET)
                if not row[4][0].startswith("127.")
            }
        )
    except socket.gaierror:
        return []


def connect(root):
    db = sqlite3.connect(Path(root) / "study.sqlite", timeout=30)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA foreign_keys=ON")
    return db


def initialize(
    root, materials, *, contact, compensation="보상 없음", target=128, invitations=160, demo=False
):
    root, materials = Path(root), Path(materials)
    require(not root.exists(), "Study directory already exists; preserve participants")
    require(
        contact.strip() and 0 < target <= 2000 and target % 4 == 0 and invitations >= target,
        "Invalid study setup",
    )
    manifest = read(materials / "manifest.json")
    for name, item in manifest["files"].items():
        require(
            hashlib.sha256((materials / name).read_bytes()).hexdigest() == item["sha256"],
            "Study material changed",
        )
    protocol = read(HERE / "protocol.json")
    protocol["targetRandomized"] = target
    protocol["stopping"] = (
        f"Stop enrollment at {target} randomized participants. Early close requires a recorded reason. No replacement based on performance and no effect display during enrollment."
    )
    protocol["powerPlanning"]["plannedNPerArm"] = target // 2
    normal = NormalDist()
    critical = normal.inv_cdf(1 - protocol["powerPlanning"]["alpha"] / 2)
    shift = protocol["powerPlanning"]["standardizedParticipantMeanEffect"] * math.sqrt(target / 4)
    protocol["powerPlanning"]["approximatePowerAtTarget"] = (
        1 - normal.cdf(critical - shift) + normal.cdf(-critical - shift)
    )
    if demo:
        protocol["status"] = "software-rehearsal-only"
    root.mkdir(parents=True, mode=0o700)
    seed = secrets.randbits(128)
    rng = random.Random(seed)
    schedule = []
    for block in range(target // 4):
        arms = ["control", "control", "ai", "ai"]
        rng.shuffle(arms)
        schedule.extend({"arm": arm, "block": block} for arm in arms)
    config = {
        "version": protocol["version"],
        "createdAt": time.time(),
        "contact": contact,
        "compensation": compensation,
        "target": target,
        "demo": demo,
        "materials": str(materials.resolve()),
        "materialHash": digest(manifest),
        "protocolHash": digest(protocol),
        "protocol": protocol,
        "assignmentSeed": seed,
        "schedule": schedule,
        "codeHashes": code_hashes(),
    }
    write(root / "config.json", config)
    (root / "recruitment.txt").write_text(
        f"발주 의사결정 연구 참여자를 모집합니다.\n\n"
        f"대상: 만 19세 이상, PC 웹브라우저 사용 가능자\n"
        f"내용: 판매 이력과 문서를 읽고 발주량 결정. AI 정보 지원 유무는 무작위 배정됩니다.\n"
        f"시간: 약 20~30분(예상), 연습 2문항과 본 과제 8문항\n"
        f"모집: 무작위 배정 {target}명 목표\n보상: {compensation}\n"
        f"이름·연락처는 실험 시스템에 저장하지 않으며, 참여 중단과 응답 삭제가 가능합니다.\n"
        f"참여 여부나 성과는 성적·직무 평가와 무관합니다.\n문의: {contact}\n"
        f"참여를 원하면 진행자가 제공하는 개인 초대 링크를 사용해 주세요.\n"
    )
    admin = secrets.token_urlsafe(32)
    (root / "admin-key.txt").write_text(admin + "\n")
    (root / "admin-key.txt").chmod(0o600)
    with connect(root) as db:
        db.executescript("""
        PRAGMA journal_mode=WAL;
        CREATE TABLE settings(key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE participants(id TEXT PRIMARY KEY, invite_hash TEXT UNIQUE NOT NULL,
          session_hash TEXT UNIQUE, csrf TEXT, arm TEXT, block INTEGER, status TEXT NOT NULL,
          state TEXT NOT NULL, consent_at REAL, randomized_at REAL, completed_at REAL);
        CREATE TABLE events(seq INTEGER PRIMARY KEY AUTOINCREMENT, participant TEXT NOT NULL,
          request_id TEXT NOT NULL, type TEXT NOT NULL, at REAL NOT NULL, payload TEXT NOT NULL,
          previous_hash TEXT NOT NULL, hash TEXT NOT NULL, UNIQUE(participant,request_id));
        CREATE TABLE answers(participant TEXT NOT NULL, trial TEXT NOT NULL, practice INTEGER NOT NULL,
          payload TEXT NOT NULL, PRIMARY KEY(participant,trial));
        """)
        db.execute("INSERT INTO settings VALUES('closed','false')")
        db.execute("INSERT INTO settings VALUES('config_hash',?)", (digest(config),))
        with (root / "invitations.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["participant", "token", "link_fragment"])
            for i in range(invitations):
                participant, token = f"P{i + 1:04}", secrets.token_urlsafe(24)
                db.execute(
                    "INSERT INTO participants(id,invite_hash,status,state) VALUES(?,?,?,?)",
                    (participant, digest(token), "invited", json.dumps({"phase": "consent"})),
                )
                writer.writerow([participant, token, "#join=" + token])
        (root / "invitations.csv").chmod(0o600)
    (root / "study.sqlite").chmod(0o600)
    return {"directory": str(root), "target": target, "invitations": invitations, "demo": demo}


class Study:
    def __init__(self, root):
        self.root = Path(root)
        self.config = read(self.root / "config.json")
        require(
            self.config["codeHashes"] == code_hashes(),
            "Experiment software changed after study creation",
        )
        materials = Path(self.config["materials"])
        self.manifest = read(materials / "manifest.json")
        require(
            digest(self.manifest) == self.config["materialHash"],
            "Materials changed after study creation",
        )
        for name, item in self.manifest["files"].items():
            require(
                hashlib.sha256((materials / name).read_bytes()).hexdigest() == item["sha256"],
                "Study material hash changed",
            )
        self.tasks = {r["id"]: r for r in read(materials / "tasks.json")}
        self.environment = read(materials / "environments.json")
        self.advice = read(materials / "advice.json")
        self.admin_key = (self.root / "admin-key.txt").read_text().strip()
        with connect(root) as db:
            require(
                db.execute("SELECT value FROM settings WHERE key='config_hash'").fetchone()[0]
                == digest(self.config),
                "Protocol or assignment changed after study creation",
            )

    def current(self, state):
        trial = self.tasks[state["order"][state["index"]]]
        value = copy.deepcopy(trial["input"])
        for action in state["path"]:
            value = outcome(value, action, self.environment[trial["id"]]["responses"][action])
        return trial, value

    def view(self, participant):
        state = json.loads(participant["state"])
        result = {
            "id": participant["id"],
            "phase": state["phase"],
            "csrf": participant["csrf"],
            "demo": self.config["demo"],
            "contact": self.config["contact"],
            "compensation": self.config["compensation"],
            "protocolVersion": self.config["version"],
        }
        if state["phase"] in ("task", "feedback"):
            trial, value = self.current(state)
            docs = [
                {
                    "id": d["id"],
                    "title": translate(d["title"]),
                    "text": translate(d["text"]),
                    "sku": d["sku"],
                    "period": d["period"],
                    "version": d["version"],
                }
                for d in value["docs"]
            ]
            result.update(
                trial=trial["id"],
                practice=trial["practice"],
                progress=state["index"] + 1,
                total=len(state["order"]),
                task={k: value["task"][k] for k in ("sku", "period", "bounds", "hold", "costs")},
                docs=docs,
                observations=[
                    {k: r[k] for k in ("date", "sales", "stockoutHours")}
                    for r in value["observations"]
                ],
                questions=state["path"],
                remaining=2 - len(state["path"]),
                draft=state.get("draft", {}),
                advice=None,
                managerPolicy=None,
            )
            # b was explicitly displayed in every initial manager-policy document.
            require(
                any(d["role"] == "manager" and "Selected" in d["text"] for d in value["docs"]),
                "Missing public preference",
            )
            policy = next(
                d for d in value["docs"] if d["role"] == "manager" and "Selected" in d["text"]
            )
            result["managerPolicy"] = float(
                re.search(r"=\s*([0-9.]+)", policy["text"])[1].rstrip(".")
            )
            if participant["arm"] == "ai":
                result["advice"] = self.advice[trial["id"]][",".join(state["path"])]
            if state["phase"] == "feedback":
                result["feedback"] = state["feedback"]
        return result

    def event(self, db, participant, request_id, kind, payload):
        previous = db.execute(
            "SELECT hash FROM events WHERE participant=? ORDER BY seq DESC LIMIT 1", (participant,)
        ).fetchone()
        previous_hash = previous[0] if previous else self.config["protocolHash"]
        now = time.time()
        item = {
            "participant": participant,
            "requestId": request_id,
            "type": kind,
            "at": now,
            "payload": payload,
            "previousHash": previous_hash,
        }
        db.execute(
            "INSERT INTO events(participant,request_id,type,at,payload,previous_hash,hash) VALUES(?,?,?,?,?,?,?)",
            (
                participant,
                request_id,
                kind,
                now,
                json.dumps(payload, ensure_ascii=False),
                previous_hash,
                digest(item),
            ),
        )

    def score(self, participant, state, submission):
        trial, value = self.current(state)
        env = self.environment[trial["id"]]
        action = submission.get("action")
        require(action in ("order", "hold"), "발주 또는 보류를 선택하세요.")
        q = submission.get("q")
        if action == "order":
            require(
                type(q) in (int, float) and math.isfinite(q) and 0 <= q <= 70,
                "발주량은 0~70 사이 숫자여야 합니다.",
            )
        else:
            q = None
        confidence = submission.get("confidence")
        require(type(confidence) is int and 1 <= confidence <= 7, "확신도를 선택하세요.")
        fields = submission.get("fields", {})
        require(
            isinstance(fields, dict) and set(fields) == {"c", "p", "v"},
            "세 모수의 판단을 입력하세요.",
        )
        sources = {d["id"] for d in value["docs"]}
        for field in fields.values():
            require(
                isinstance(field, dict) and set(field) == {"value", "source"},
                "잘못된 모수 입력입니다.",
            )
            require(
                field["value"] is None
                or (
                    type(field["value"]) in (float, int)
                    and math.isfinite(field["value"])
                    and 0 <= field["value"] <= 10000
                ),
                "모수 값은 양수 또는 미확정이어야 합니다.",
            )
            require(
                field["source"] is None or field["source"] in sources,
                "현재 문서에서 근거를 선택하세요.",
            )
        request_cost = sum(value["task"]["costs"][a] for a in state["path"])
        expected = risk(q, env["theta"]) if q is not None else value["task"]["hold"]
        regret = (
            max(0, expected - env["optimal"]["cost"]) if q is not None else value["task"]["hold"]
        )
        theta, d = env["theta"], env["demand"]
        realized = (
            (
                (theta["c"] - theta["v"]) * max(q - d, 0)
                + (theta["p"] - theta["c"] + theta["b"]) * max(d - q, 0)
            )
            if q is not None
            else value["task"]["hold"]
        )
        # Scoring is strictly server-side, after the participant's decision.
        from newsvendor.construction import parameter_record

        reference = parameter_record(value)
        parameter_correct, evidence_correct = [], []
        for slot, field in fields.items():
            truth = reference["values"].get(slot)
            correct = (
                (field["value"] is None)
                if truth is None
                else (
                    field["value"] is not None and math.isclose(field["value"], truth, abs_tol=1e-6)
                )
            )
            parameter_correct.append(correct)
            evidence_correct.append(correct and field["source"] == reference["links"].get(slot))
        return {
            "participant": participant["id"],
            "arm": participant["arm"],
            "block": participant["block"],
            "trial": trial["id"],
            "practice": trial["practice"],
            "position": state["index"],
            "action": action,
            "q": q,
            "fields": fields,
            "confidence": confidence,
            "questions": state["path"],
            "questionCount": len(state["path"]),
            "questionCost": request_cost,
            "expectedOrderLoss": expected,
            "orderRegret": regret,
            "total": regret + request_cost,
            "normalizedTotal": (regret + request_cost) / value["task"]["hold"],
            "realizedOrderLoss": realized,
            "realizedTotal": realized + request_cost,
            "hold": action == "hold",
            "decisionSeconds": max(0, time.time() - state["startedAt"]),
            "parameterAccuracy": sum(parameter_correct) / 3,
            "evidenceAccuracy": sum(evidence_correct) / 3,
            "adviceShown": participant["arm"] == "ai",
            "adviceHash": digest(self.advice[trial["id"]][",".join(state["path"])])
            if participant["arm"] == "ai"
            else None,
            "adviceQ": self.advice[trial["id"]][",".join(state["path"])].get("q")
            if participant["arm"] == "ai"
            else None,
            "inputHash": digest(value),
        }

    def change(self, db, participant, data):
        state = json.loads(participant["state"])
        kind, request_id = data.get("type"), data.get("requestId")
        require(
            isinstance(request_id, str) and 8 <= len(request_id) <= 100, "요청 번호가 필요합니다."
        )
        if db.execute(
            "SELECT 1 FROM events WHERE participant=? AND request_id=?",
            (participant["id"], request_id),
        ).fetchone():
            return participant
        require(
            kind == "withdraw" or state["phase"] not in ("done", "withdrawn", "declined"),
            "이미 종료된 참여입니다.",
        )
        closed = json.loads(
            db.execute("SELECT value FROM settings WHERE key='closed'").fetchone()[0]
        )
        require(not closed or kind == "withdraw", "실험이 마감되었습니다.")
        payload = {}
        if kind == "consent":
            require(state["phase"] == "consent", "동의 단계가 아닙니다.")
            if data.get("agreed") is not True or data.get("adult") is not True:
                state["phase"] = "declined"
                db.execute(
                    "UPDATE participants SET status='declined' WHERE id=?", (participant["id"],)
                )
            else:
                require(
                    data.get("experience") in ("none", "course", "work"),
                    "재고관리 경험을 선택하세요.",
                )
                state = {"phase": "instructions", "experience": data["experience"]}
                db.execute(
                    "UPDATE participants SET consent_at=?,status='consented' WHERE id=?",
                    (time.time(), participant["id"]),
                )
                payload = {
                    "adult": True,
                    "agreed": True,
                    "experience": data["experience"],
                    "protocolHash": self.config["protocolHash"],
                }
        elif kind == "quiz":
            require(state["phase"] == "instructions", "설명 단계가 아닙니다.")
            answers = data.get("answers")
            passed = answers == ["cost", "optional", "own"]
            payload = {"answers": answers, "passed": passed}
            if passed:
                assigned = db.execute(
                    "SELECT COUNT(*) FROM participants WHERE arm IS NOT NULL"
                ).fetchone()[0]
                require(assigned < self.config["target"], "모집 인원이 마감되었습니다.")
                assignment = self.config["schedule"][assigned]
                practice = [k for k, t in self.tasks.items() if t["practice"]]
                trials = [k for k, t in self.tasks.items() if not t["practice"]]
                random.Random(
                    digest([self.config["assignmentSeed"], participant["id"], "order"])
                ).shuffle(trials)
                state.update(
                    phase="task",
                    order=practice + trials,
                    index=0,
                    path=[],
                    startedAt=time.time(),
                    draft={},
                )
                db.execute(
                    "UPDATE participants SET arm=?,block=?,randomized_at=?,status='active' WHERE id=?",
                    (assignment["arm"], assignment["block"], time.time(), participant["id"]),
                )
                payload.update(assignment=assignment, order=state["order"])
            else:
                state["quizError"] = True
        elif kind in ("question", "submit", "draft"):
            require(
                state["phase"] == "task" and data.get("trial") == state["order"][state["index"]],
                "현재 과제가 변경되었습니다. 화면을 새로고침하세요.",
            )
            if kind == "question":
                action = data.get("field")
                require(
                    action in ("c", "p", "v")
                    and action not in state["path"]
                    and len(state["path"]) < 2,
                    "현재 요청할 수 없는 질문입니다.",
                )
                state["path"].append(action)
                payload = {"trial": data["trial"], "field": action}
            elif kind == "draft":
                require(isinstance(data.get("draft"), dict), "임시 입력 형식이 올바르지 않습니다.")
                require(len(json.dumps(data.get("draft", {}))) <= 6000, "입력이 너무 큽니다.")
                state["draft"] = data.get("draft", {})
                payload = {"trial": data["trial"]}
            else:
                answer = self.score(participant, state, data)
                db.execute(
                    "INSERT INTO answers VALUES(?,?,?,?)",
                    (
                        participant["id"],
                        answer["trial"],
                        int(answer["practice"]),
                        json.dumps(answer, ensure_ascii=False),
                    ),
                )
                payload = answer
                if answer["practice"]:
                    state.update(
                        phase="feedback",
                        feedback={
                            "optimalQ": self.environment[answer["trial"]]["optimal"]["q"],
                            "total": answer["total"],
                            "questionCost": answer["questionCost"],
                            "chosenQ": answer["q"],
                        },
                    )
                else:
                    self.advance(db, participant["id"], state)
        elif kind == "next":
            require(state["phase"] == "feedback", "연습 피드백 단계가 아닙니다.")
            self.advance(db, participant["id"], state)
        elif kind == "withdraw":
            db.execute("DELETE FROM answers WHERE participant=?", (participant["id"],))
            db.execute("DELETE FROM events WHERE participant=?", (participant["id"],))
            state = {"phase": "withdrawn"}
            db.execute(
                "UPDATE participants SET status='withdrawn' WHERE id=?", (participant["id"],)
            )
            payload = {"responsesDeleted": True}
        else:
            raise ValueError("지원하지 않는 작업입니다.")
        self.event(db, participant["id"], request_id, kind, payload)
        db.execute(
            "UPDATE participants SET state=? WHERE id=?",
            (json.dumps(state, ensure_ascii=False), participant["id"]),
        )
        return db.execute("SELECT * FROM participants WHERE id=?", (participant["id"],)).fetchone()

    def advance(self, db, participant_id, state):
        state["index"] += 1
        state.pop("feedback", None)
        state.update(path=[], draft={}, startedAt=time.time())
        if state["index"] == len(state["order"]):
            state["phase"] = "done"
            db.execute(
                "UPDATE participants SET status='completed',completed_at=? WHERE id=?",
                (time.time(), participant_id),
            )
        else:
            state["phase"] = "task"

    def export(self, db):
        require(
            json.loads(db.execute("SELECT value FROM settings WHERE key='closed'").fetchone()[0]),
            "실험을 마감한 뒤 결과를 내보내세요.",
        )
        participants = [
            dict(r)
            for r in db.execute(
                "SELECT id,arm,block,status,consent_at,randomized_at,completed_at FROM participants"
            )
            if r["arm"] is not None
        ]
        answers = [
            json.loads(r[0])
            for r in db.execute("SELECT payload FROM answers ORDER BY participant,trial")
        ]
        events = [
            dict(r)
            for r in db.execute(
                "SELECT seq,participant,request_id,type,at,payload,previous_hash,hash FROM events ORDER BY seq"
            )
        ]
        data = io.BytesIO()
        closure = dict(
            db.execute(
                "SELECT key,value FROM settings WHERE key IN ('closed','closed_at','close_reason')"
            )
        )
        with zipfile.ZipFile(data, "w", zipfile.ZIP_DEFLATED) as archive:
            for name, value in {
                "participants.json": participants,
                "answers.json": answers,
                "events.json": events,
                "protocol.json": self.config["protocol"],
                "materials-manifest.json": self.manifest,
                "closure.json": closure,
                "design.json": {
                    k: v for k, v in self.config.items() if k not in ("contact", "materials")
                },
            }.items():
                archive.writestr(name, json.dumps(value, ensure_ascii=False, indent=2))
            csvfile = io.StringIO()
            fields = [
                "participant",
                "arm",
                "block",
                "trial",
                "practice",
                "action",
                "q",
                "confidence",
                "questionCost",
                "questionCount",
                "orderRegret",
                "normalizedTotal",
                "realizedOrderLoss",
                "realizedTotal",
                "hold",
                "adviceQ",
                "decisionSeconds",
                "parameterAccuracy",
                "evidenceAccuracy",
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(answers)
            archive.writestr("answers.csv", csvfile.getvalue())
        data.seek(0)
        return data


def create_app(root):
    study = Study(root)
    app = Flask(__name__, static_folder=None)
    app.config.update(MAX_CONTENT_LENGTH=16000)

    @app.after_request
    def headers(response):
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; frame-ancestors 'none'; base-uri 'none'"
        )
        return response

    @app.errorhandler(ValueError)
    def invalid(error):
        return jsonify(error=str(error)), 400

    def participant(db, mutating=False):
        token = request.cookies.get("study_session", "")
        row = db.execute(
            "SELECT * FROM participants WHERE session_hash=?", (digest(token),)
        ).fetchone()
        require(row is not None, "초대 링크로 접속하세요.")
        if mutating:
            require(
                hmac.compare_digest(request.headers.get("X-CSRF-Token", ""), row["csrf"]),
                "세션 확인에 실패했습니다.",
            )
        return row

    def admin():
        require(
            hmac.compare_digest(request.headers.get("X-Admin-Key", ""), study.admin_key),
            "진행자 인증이 필요합니다.",
        )

    def body():
        data = request.get_json(silent=True)
        require(isinstance(data, dict), "JSON 객체가 필요합니다.")
        return data

    @app.get("/")
    def index():
        return send_file(HERE / "web" / "index.html")

    @app.get("/admin")
    def admin_page():
        return send_file(HERE / "web" / "admin.html")

    @app.get("/assets/<name>")
    def assets(name):
        require(name in ("app.js", "style.css", "admin.js"), "Unknown asset")
        return send_file(HERE / "web" / name)

    @app.post("/api/join")
    def join():
        data = body()
        token = data.get("token", "")
        require(
            isinstance(token, str) and 20 <= len(token) <= 100, "올바른 초대 링크를 사용하세요."
        )
        with connect(root) as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT * FROM participants WHERE invite_hash=?", (digest(token),)
            ).fetchone()
            require(row is not None, "초대 링크를 찾을 수 없습니다.")
            # Reopening a personal invitation rotates its session; one active session per invite.
            session, csrf = secrets.token_urlsafe(32), secrets.token_urlsafe(24)
            db.execute(
                "UPDATE participants SET session_hash=?,csrf=? WHERE id=?",
                (digest(session), csrf, row["id"]),
            )
            row = db.execute("SELECT * FROM participants WHERE id=?", (row["id"],)).fetchone()
            response = make_response(jsonify(study.view(row)))
            response.set_cookie(
                "study_session",
                session,
                httponly=True,
                samesite="Strict",
                secure=request.is_secure,
                max_age=86400,
            )
            return response

    @app.get("/api/state")
    def state():
        with connect(root) as db:
            return jsonify(study.view(participant(db)))

    @app.post("/api/action")
    def action():
        with connect(root) as db:
            db.execute("BEGIN IMMEDIATE")
            return jsonify(study.view(study.change(db, participant(db, True), body())))

    @app.get("/api/admin/status")
    def status():
        admin()
        with connect(root) as db:
            counts = {
                r[0]: r[1]
                for r in db.execute("SELECT status,COUNT(*) FROM participants GROUP BY status")
            }
            assigned = db.execute(
                "SELECT COUNT(*) FROM participants WHERE arm IS NOT NULL"
            ).fetchone()[0]
            parsed = urlsplit(request.host_url)
            addresses = local_addresses()
            bases = [request.host_url.rstrip("/")]
            if parsed.hostname in ("localhost", "127.0.0.1"):
                bases = [f"http://{ip}:{parsed.port or 80}" for ip in addresses] + bases
            return jsonify(
                counts=counts,
                randomized=assigned,
                target=study.config["target"],
                closed=json.loads(
                    db.execute("SELECT value FROM settings WHERE key='closed'").fetchone()[0]
                ),
                demo=study.config["demo"],
                protocolHash=study.config["protocolHash"],
                participantBases=bases,
            )

    @app.get("/api/admin/invitations")
    def invitations():
        admin()
        base = request.args.get("base", request.host_url).rstrip("/")
        parsed = urlsplit(base)
        require(
            parsed.scheme in ("http", "https")
            and parsed.hostname
            and not parsed.username
            and not parsed.password
            and parsed.path in ("", "/")
            and not parsed.query
            and not parsed.fragment,
            "참여자가 접속할 서버 주소를 입력하세요.",
        )
        stream = io.StringIO()
        writer = csv.writer(stream)
        writer.writerow(["participant", "url"])
        with (Path(root) / "invitations.csv").open() as file:
            for row in csv.DictReader(file):
                writer.writerow([row["participant"], base + "/" + row["link_fragment"]])
        data = io.BytesIO(stream.getvalue().encode("utf-8-sig"))
        return send_file(
            data, as_attachment=True, download_name="invitations.csv", mimetype="text/csv"
        )

    @app.post("/api/admin/close")
    def close():
        admin()
        with connect(root) as db:
            db.execute("BEGIN IMMEDIATE")
            assigned = db.execute(
                "SELECT COUNT(*) FROM participants WHERE arm IS NOT NULL"
            ).fetchone()[0]
            reason = body().get("reason", "")
            require(
                assigned >= study.config["target"]
                or (isinstance(reason, str) and 5 <= len(reason.strip()) <= 500),
                "목표 인원 전에 마감하려면 사유를 기록하세요.",
            )
            db.execute("UPDATE settings SET value='true' WHERE key='closed'")
            db.execute("INSERT OR REPLACE INTO settings VALUES('close_reason',?)", (reason,))
            db.execute("INSERT OR REPLACE INTO settings VALUES('closed_at',?)", (str(time.time()),))
        return jsonify(closed=True)

    @app.get("/api/admin/export")
    def export():
        admin()
        with connect(root) as db:
            return send_file(
                study.export(db),
                as_attachment=True,
                download_name="study-results.zip",
                mimetype="application/zip",
            )

    return app
