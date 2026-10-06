import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from newsvendor.providers import Provider, request


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
