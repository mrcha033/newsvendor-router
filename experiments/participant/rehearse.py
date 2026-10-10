"""Drive four SOFTWARE participants in a demo-only study; never enroll real participants."""

import argparse
import csv
import json
from pathlib import Path

from playwright.sync_api import sync_playwright


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True)
    parser.add_argument("--url", default="http://127.0.0.1:8795")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    directory, output = Path(args.directory), Path(args.output)
    config = json.loads((directory / "config.json").read_text())
    assert config["demo"] is True and config["target"] == 4, "Demo-only, four software participants"
    assert not output.exists(), "Preserve previous browser checks"
    output.mkdir(parents=True)
    with (directory / "invitations.csv").open() as stream:
        invites = list(csv.DictReader(stream))[:4]
    errors, arms, count = [], [], 0
    with sync_playwright() as runner:
        browser = runner.chromium.launch(headless=True)
        for index, invite in enumerate(invites):
            context = browser.new_context(viewport={"width": 1440, "height": 1000})
            page = context.new_page()
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.goto(args.url + "/" + invite["link_fragment"])
            page.wait_for_selector("#consent")
            page.select_option("#experience", "course")
            page.check("#adult")
            page.check("#agree")
            page.click("#consent")
            page.wait_for_selector("#quiz")
            for name, val in (("q1", "cost"), ("q2", "optional"), ("q3", "own")):
                page.check(f'input[name="{name}"][value="{val}"]')
            page.locator("#quiz button").click()
            page.wait_for_selector("#decision")
            ai = page.locator(".advice").count() > 0
            arms.append("ai" if ai else "control")
            page.screenshot(path=str(output / f"{arms[-1]}-desktop.png"), full_page=True)
            if index == 0:
                page.set_viewport_size({"width": 390, "height": 844})
                page.screenshot(path=str(output / "mobile.png"), full_page=True)
                assert page.evaluate(
                    "document.documentElement.scrollWidth <= window.innerWidth + 1"
                )
                page.set_viewport_size({"width": 1440, "height": 1000})
                page.fill("#quantity", "23")
                page.select_option("#confidence", "4")
                page.wait_for_timeout(850)
                page.reload()
                page.wait_for_selector("#quantity")
                assert page.input_value("#quantity") == "23", "Refresh lost a saved draft"
            for trial_index in range(10):
                page.wait_for_selector("#decision")
                state = page.request.get(args.url + "/api/state").json()
                assert "theta" not in state and "optimal" not in state and "demand" not in state
                assert (state["advice"] is not None) == ai
                for field in ("c", "v") if index % 2 else ("v", "c"):
                    with page.expect_response(
                        lambda r: (
                            r.url.endswith("/api/action")
                            and r.request.post_data_json.get("type") == "question"
                        )
                    ) as reply:
                        page.click(f'[data-question="{field}"]')
                    assert reply.value.status == 200, reply.value.text()
                    page.wait_for_function(
                        "document.querySelectorAll('[data-question]:disabled').length >= "
                        + str(1 if field == (("c", "v") if index % 2 else ("v", "c"))[0] else 2)
                    )
                if ai:
                    page.click("#use-fields")
                page.fill("#quantity", "28")
                page.select_option("#confidence", "4")
                with page.expect_response(
                    lambda r: (
                        r.url.endswith("/api/action")
                        and r.request.post_data_json.get("type") == "submit"
                    )
                ) as submitted:
                    page.locator("#decision button.primary").click()
                response = submitted.value
                assert response.status == 200, response.text()
                current = response.json()
                count += 1
                if current["phase"] == "feedback":
                    page.wait_for_selector("#next")
                    page.click("#next")
                elif current["phase"] == "done":
                    page.wait_for_selector(".complete")
                else:
                    page.wait_for_function(
                        "document.querySelector('#status').textContent.includes('"
                        + str(trial_index + 2)
                        + " / 10')"
                    )
            assert page.request.get(args.url + "/api/state").json()["phase"] == "done"
            context.close()
            print(
                json.dumps(
                    {"softwareParticipant": index + 1, "arm": arms[-1], "completedTasks": 10}
                ),
                flush=True,
            )
        assert sorted(arms) == ["ai", "ai", "control", "control"]
        admin = browser.new_page(viewport={"width": 1440, "height": 1000})
        admin.goto(args.url + "/admin")
        admin.fill("#key", (directory / "admin-key.txt").read_text().strip())
        admin.click("#connect")
        admin.wait_for_selector("#close")
        admin.screenshot(path=str(output / "admin.png"), full_page=True)
        with admin.expect_download() as invitation_download:
            admin.click("#invitations")
        rows = list(
            csv.DictReader(
                Path(invitation_download.value.path()).read_text(encoding="utf-8-sig").splitlines()
            )
        )
        assert len(rows) == 8
        assert all(row["url"].startswith("http") and "/#join=" in row["url"] for row in rows)
        admin.on("dialog", lambda dialog: dialog.accept())
        admin.click("#close")
        admin.wait_for_selector("#export")
        with admin.expect_download() as download:
            admin.click("#export")
        download.value.save_as(output / "software-rehearsal-results.zip")
        browser.close()
    report = {
        "humanParticipants": 0,
        "softwareParticipants": 4,
        "practiceSubmissions": 8,
        "mainSubmissions": 32,
        "totalSubmissions": count,
        "arms": arms,
        "pageErrors": errors,
        "refreshResume": True,
        "managerRequests": 80,
        "sourceAnswersAbsentFromPayload": True,
        "exportVerified": True,
        "personalInvitationURLs": True,
        "scope": "Browser and data-pipeline rehearsal; not human treatment-effect evidence",
    }
    assert not errors, errors
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
