"""Create, serve, back up and export a participant study without changing its design."""

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from app import connect, create_app, initialize, local_addresses
from waitress import serve


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("init", "serve", "backup", "export", "start"))
    parser.add_argument("--directory", default="results/human-study-live")
    parser.add_argument("--materials", default="results/human-materials-v2")
    parser.add_argument("--contact")
    parser.add_argument("--compensation", default="보상 없음")
    parser.add_argument("--target", type=int, default=128)
    parser.add_argument("--invitations", type=int, default=160)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8795)
    parser.add_argument("--output")
    args = parser.parse_args()
    root = Path(args.directory)
    if args.command == "init" or (args.command == "start" and not root.exists()):
        contact = args.contact or input("참여자에게 표시할 연구 문의 연락처: ").strip()
        print(
            json.dumps(
                initialize(
                    root,
                    args.materials,
                    contact=contact,
                    compensation=args.compensation,
                    target=args.target,
                    invitations=args.invitations,
                    demo=args.demo,
                ),
                ensure_ascii=False,
            )
        )
    if args.command in ("serve", "start"):
        app = create_app(root)
        admin = (root / "admin-key.txt").read_text().strip()
        base = f"http://127.0.0.1:{args.port}"
        print(f"진행자 화면: {base}/admin#key={admin}", flush=True)
        print(f"개인 초대 목록: {root.resolve() / 'invitations.csv'}", flush=True)
        if args.host == "0.0.0.0":
            addresses = local_addresses()
            print(
                "같은 네트워크의 참여자 접속 주소: "
                + ", ".join(f"http://{ip}:{args.port}" for ip in addresses),
                flush=True,
            )
        if args.demo:
            with (root / "invitations.csv").open() as stream:
                first = next(csv.DictReader(stream))
            print(f"리허설 참여 링크: {base}/{first['link_fragment']}", flush=True)
        serve(
            app,
            host=args.host,
            port=args.port,
            threads=8,
            max_request_body_size=16000,
            ident="Study",
        )
    elif args.command == "backup":
        import sqlite3

        target = Path(args.output or (str(root) + "-backup"))
        if target.exists():
            parser.error("Backup destination exists; choose a new path")
        shutil.copytree(root, target, ignore=shutil.ignore_patterns("study.sqlite*"))
        with connect(root) as source, sqlite3.connect(target / "study.sqlite") as destination:
            source.backup(destination)
        target.chmod(0o700)
        (target / "study.sqlite").chmod(0o600)
        print(json.dumps({"backup": str(target)}))
    elif args.command == "export":
        from app import Study

        target = Path(args.output or (str(root) + "-results.zip"))
        if target.exists():
            parser.error("Export exists; preserve previous outputs")
        with connect(root) as db:
            target.write_bytes(Study(root).export(db).getvalue())
        target.chmod(0o600)
        print(json.dumps({"export": str(target)}))


if __name__ == "__main__":
    main()
