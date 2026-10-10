import copy
import csv
import hashlib
import io
import json
import sys
import uuid
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.participant.analyze import analyze, verify_export
from experiments.participant.app import Study, connect, create_app, initialize
from experiments.participant.prepare import build_tasks
from newsvendor.io import read, write


@pytest.fixture()
def study(tmp_path):
    materials = tmp_path / "materials"
    tasks, env = build_tasks()
    write(materials / "tasks.json", tasks)
    write(materials / "environments.json", env)
    advice = {r["id"]: {"": {"q": 21, "valid": True}} for r in tasks}
    write(materials / "advice.json", advice)
    write(
        materials / "manifest.json",
        {
            "files": {
                p.name: {"sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                for p in materials.iterdir()
            },
            "scope": "UNIT TEST FIXTURE, not model advice",
        },
    )
    root = tmp_path / "study"
    initialize(root, materials, contact="Test researcher", target=4, invitations=6, demo=True)
    app = create_app(root)
    app.config["TESTING"] = True
    with (root / "invitations.csv").open() as stream:
        invites = list(csv.DictReader(stream))
    return root, app, invites


def join(app, invite):
    client = app.test_client()
    response = client.post("/api/join", json={"token": invite["token"]})
    assert response.status_code == 200
    return client, response.json


def act(client, state, kind, **kwargs):
    response = client.post(
        "/api/action",
        json={"type": kind, "requestId": str(uuid.uuid4()), **kwargs},
        headers={"X-CSRF-Token": state["csrf"]},
    )
    assert response.status_code == 200, response.json
    return response.json


def enroll(app, invite):
    client, state = join(app, invite)
    state = act(client, state, "consent", adult=True, agreed=True, experience="none")
    state = act(client, state, "quiz", answers=["cost", "optional", "own"])
    return client, state


def submit(client, state):
    return act(
        client,
        state,
        "submit",
        trial=state["trial"],
        action="order",
        q=25,
        confidence=4,
        fields={s: {"value": None, "source": None} for s in ("c", "p", "v")},
    )


def test_blinded_assignment_and_no_environment_in_participant_payload(study):
    root, app, invites = study
    for invite in invites[:4]:
        _, state = enroll(app, invite)
        assert "theta" not in state and "demand" not in state and "optimal" not in state
        assert "arm" not in state and "block" not in state
    with connect(root) as db:
        assigned = list(db.execute("SELECT arm,block FROM participants WHERE arm IS NOT NULL"))
    assert sorted(r["arm"] for r in assigned) == ["ai", "ai", "control", "control"]
    assert {r["block"] for r in assigned} == {0}
    assert app.test_client().get("/api/admin/status").status_code == 400
    assert app.test_client().get("/assets/environments.json").status_code == 400


def test_csrf_and_request_id_prevent_forged_or_duplicate_submission(study):
    _, app, invites = study
    client, state = enroll(app, invites[0])
    body = {
        "type": "submit",
        "trial": state["trial"],
        "requestId": str(uuid.uuid4()),
        "action": "hold",
        "confidence": 4,
        "fields": {s: {"value": None, "source": None} for s in ("c", "p", "v")},
    }
    assert client.post("/api/action", json=body).status_code == 400
    headers = {"X-CSRF-Token": state["csrf"]}
    first = client.post("/api/action", json=body, headers=headers)
    second = client.post("/api/action", json=body, headers=headers)
    assert first.status_code == second.status_code == 200
    assert first.json == second.json and first.json["phase"] == "feedback"
    assert client.post("/api/action", json=[], headers=headers).status_code == 400


def test_invalid_field_or_draft_data_does_not_break_the_study(study):
    _, app, invites = study
    client, state = enroll(app, invites[0])
    headers = {"X-CSRF-Token": state["csrf"]}
    for fields in (None, [], {"c": None, "p": {}, "v": {}}):
        response = client.post(
            "/api/action",
            json={
                "type": "submit",
                "requestId": str(uuid.uuid4()),
                "trial": state["trial"],
                "action": "hold",
                "confidence": 4,
                "fields": fields,
            },
            headers=headers,
        )
        assert response.status_code == 400
    assert (
        client.post(
            "/api/action",
            json={
                "type": "draft",
                "requestId": str(uuid.uuid4()),
                "trial": state["trial"],
                "draft": [],
            },
            headers=headers,
        ).status_code
        == 400
    )
    assert submit(client, state)["phase"] == "feedback"


def test_personal_invitation_urls_and_target_are_ready_for_recruitment(study):
    root, app, invites = study
    admin = {"X-Admin-Key": (root / "admin-key.txt").read_text().strip()}
    client = app.test_client()
    response = client.get("/api/admin/invitations?base=http://192.0.2.5:8795", headers=admin)
    assert response.status_code == 200
    links = list(csv.DictReader(io.StringIO(response.data.decode("utf-8-sig"))))
    assert links[0]["url"] == "http://192.0.2.5:8795/" + invites[0]["link_fragment"]
    assert len(links) == len(invites)
    for base in ("javascript:alert(1)", "http://host/path", "http://user:pass@host"):
        assert (
            client.get(
                "/api/admin/invitations", query_string={"base": base}, headers=admin
            ).status_code
            == 400
        )
    config = read(root / "config.json")
    assert config["protocol"]["targetRandomized"] == config["target"] == 4
    assert config["protocol"]["powerPlanning"]["plannedNPerArm"] == 2
    assert "4명" in (root / "recruitment.txt").read_text()
    assert "Test researcher" in (root / "recruitment.txt").read_text()


def test_resume_rotates_session_and_keeps_draft(study):
    _, app, invites = study
    old, state = enroll(app, invites[0])
    state = act(old, state, "draft", trial=state["trial"], draft={"q": 27})
    current, restored = join(app, invites[0])
    assert restored["draft"] == {"q": 27}
    assert old.get("/api/state").status_code == 400
    assert current.get("/api/state").json["trial"] == restored["trial"]


def test_withdrawal_deletes_answers_and_retains_assignment_attrition(study):
    root, app, invites = study
    client, state = enroll(app, invites[0])
    state = submit(client, state)
    state = act(client, state, "withdraw")
    assert state["phase"] == "withdrawn"
    with connect(root) as db:
        assert db.execute("SELECT COUNT(*) FROM answers").fetchone()[0] == 0
        assert db.execute(
            "SELECT status,arm FROM participants WHERE id=?", (invites[0]["participant"],)
        ).fetchone()["arm"] in ("control", "ai")
        assert db.execute("SELECT type FROM events").fetchone()[0] == "withdraw"


def test_ending_export_is_separate_from_practice_and_contains_no_secrets(study):
    root, app, invites = study
    client, state = enroll(app, invites[0])
    while state["phase"] != "done":
        state = (
            act(client, state, "next") if state["phase"] == "feedback" else submit(client, state)
        )
    admin = {"X-Admin-Key": (root / "admin-key.txt").read_text().strip()}
    assert client.get("/api/admin/export", headers=admin).status_code == 400
    assert (
        client.post(
            "/api/admin/close", json={"reason": "unit test rehearsal complete"}, headers=admin
        ).status_code
        == 200
    )
    response = client.get("/api/admin/export", headers=admin)
    assert response.status_code == 200
    with zipfile.ZipFile(io.BytesIO(response.data)) as archive:
        participants = json.loads(archive.read("participants.json"))
        answers = json.loads(archive.read("answers.json"))
        assert len(answers) == 10 and sum(not r["practice"] for r in answers) == 8
        assert all("invite_hash" not in r and "session_hash" not in r for r in participants)
        assert "closure.json" in archive.namelist()
        exported = {
            name[:-5]: json.loads(archive.read(name))
            for name in archive.namelist()
            if name.endswith(".json")
        }
    assert verify_export(exported)["submissions"] == 10
    changed = copy.deepcopy(exported)
    changed["answers"][0]["q"] += 1
    with pytest.raises(ValueError, match="Answers differ"):
        verify_export(changed)
    changed = copy.deepcopy(exported)
    changed["events"][0]["at"] += 1
    with pytest.raises(ValueError, match="Changed event"):
        verify_export(changed)


def test_material_or_protocol_edits_cannot_silently_change_a_live_study(study):
    root, _, _ = study
    config = read(root / "config.json")
    for path in ("src/newsvendor/optimizer.py", "src/newsvendor/construction.py", "uv.lock"):
        assert config["codeHashes"][path] == hashlib.sha256(Path(path).read_bytes()).hexdigest()
    config["target"] = 8
    write(root / "config.json", config)
    with pytest.raises(ValueError, match="changed"):
        Study(root)


def test_lab_tasks_are_deterministic_and_do_not_expose_future_labels():
    tasks, env = build_tasks()
    assert (tasks, env) == build_tasks()
    assert len(tasks) == 10 and sum(t["practice"] for t in tasks) == 2
    assert not {"theta", "responses", "optimal", "target", "gold"} & set(tasks[0]["input"])
    assert len({t["family"] for t in tasks}) == 10


def test_participant_level_regression_and_missing_outcome_bounds():
    participants = [
        {"id": str(i), "arm": "ai" if i % 2 else "control", "block": i // 4} for i in range(8)
    ]
    answers = [
        {
            "participant": p["id"],
            "arm": p["arm"],
            "practice": False,
            "trial": "task",
            "normalizedTotal": 0.3 if p["arm"] == "ai" else 0.8,
        }
        for p in participants
    ]
    protocol = copy.deepcopy(read("experiments/participant/protocol.json"))
    protocol["randomizationTest"]["permutations"] = 100
    result = analyze(participants, answers, {"demo": False}, protocol, {"task": 1.5})
    assert result["regression"]["AI"] == pytest.approx(-0.5)
    assert result["regression"]["participants"] == 8
    incomplete = analyze(participants, answers[:-1], {"demo": False}, protocol, {"task": 1.5})
    assert incomplete["regression"] is None
    assert incomplete["intentToTreatBounds"][0] <= -0.5 <= incomplete["intentToTreatBounds"][1]
    assert incomplete["completedParticipants"] == 7
