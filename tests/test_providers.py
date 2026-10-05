import json
import threading
import urllib.error
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from newsvendor.io import read
from newsvendor.providers import Provider, no_labels, normalize, request, settings


def test_probability_contract_and_label_guard():
    criteria = {"yes": "permit", "no": "reject"}
    good = {"answers": {"choice": {"choice": "yes", "probabilities": {"yes": 0.7, "no": 0.3}}}}
    assert normalize(good, "choice", criteria)["choice"] == "yes"
    for p in ({"yes": 0.9, "no": 0.3}, {"yes": 0.7}, {"yes": True, "no": 0}):
        with pytest.raises(ValueError, match="probabilit"):
            normalize(
                {"answers": {"choice": {"choice": "yes", "probabilities": p}}}, "choice", criteria
            )
    with pytest.raises(ValueError, match="Oracle"):
        no_labels({"input": {"gold": {"answer": 5}}})
    with pytest.raises(ValueError, match="HTTPS"):
        settings("sglang", {"SGLANG_URL": "http://remote.test/api", "SGLANG_MODEL": "model"})
    with pytest.raises(ValueError, match="SGLANG_URL and SGLANG_MODEL"):
        settings("sglang", {})


def test_systemone_wire_protocol_and_usage_capture():
    received = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            received.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("x-request-id", "contract-test")
            self.end_headers()
            self.wfile.write(
                json.dumps(
                    {
                        "answers": {"choice": {"choice": "hold", "probabilities": {"hold": 1.0}}},
                        "usage": {"input_tokens": 10, "output_tokens": 1},
                    }
                ).encode()
            )

        def log_message(self, *_):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        answer, trace = request(
            Provider("sglang", f"http://127.0.0.1:{server.server_port}/v1/systemone", "test-model"),
            {"docs": ["observable text"]},
            "Next action?",
            {"hold": "Hold order"},
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    assert answer["choice"] == "hold"
    assert received[0]["questions"]["choice"]["type"] == "choice"
    assert json.loads(received[0]["state"]) == {"docs": ["observable text"]}
    assert trace["usage"]["input_tokens"] == 10 and trace["requestId"] == "contract-test"


def test_failed_calls_are_recorded_without_keys(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    def failure(*_, **__):
        raise urllib.error.HTTPError("http://localhost", 503, "unavailable", {}, None)

    monkeypatch.setattr("urllib.request.urlopen", failure)
    with pytest.raises(RuntimeError, match="recorded"):
        request(
            Provider("sglang", "http://localhost/v1/systemone", "fixture", "private-test-key"),
            {"docs": ["visible"]},
            "Next?",
            {"hold": "Hold"},
        )
    file = next((tmp_path / "results/provider-errors").glob("*.json"))
    assert read(file)["status"] == 503
    assert "private-test-key" not in file.read_text()
