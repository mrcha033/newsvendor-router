from newsvendor import benchmark
from newsvendor.construction import reference
from newsvendor.corpus import SLOTS, generate
from newsvendor.encoder import ACTIONS, CONSTRAINT, DEMAND, Embeddings, texts
from newsvendor.io import read
from newsvendor.providers import Provider


def test_shared_training_and_policy_cross_with_fixed_provider(monkeypatch, tmp_path):
    config = {
        "seed": 42,
        "groups": 10,
        "variants": 1,
        "budget": 2,
        "epochs": 1,
        "bootstrap": 30,
        "encoder": {"dim": 4, "revision": "test-vectors"},
    }
    vocabulary = {
        *SLOTS.values(),
        *ACTIONS.values(),
        DEMAND,
        CONSTRAINT,
        *(t for e in generate(config) for t in texts(e["input"])),
    }
    cache = Embeddings(config["encoder"], {t: [1.0, 0.0, 0.5, -0.5] for t in vocabulary})

    class Fixed:
        def __init__(self, *_):
            self.calls, self.snapshots = 0, {}
            self.identity = {"model": "protocol-fixture-only"}

        def __call__(self, input):
            return reference(input)

        def calibrate(self, _):
            return {"source": "test fixture"}

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(benchmark, "TypedBuilder", Fixed)
    monkeypatch.setattr(benchmark, "embed", lambda *_: cache)
    monkeypatch.setattr(
        benchmark, "settings", lambda *_: Provider("sglang", "http://localhost", "fixture")
    )
    monkeypatch.setattr(benchmark, "provenance", lambda *_: {"testFixture": True})
    benchmark.run(config, "sglang", 0)
    metrics = read("results/benchmarks/sglang/42/metrics.json")
    assert not metrics["primaryEvaluationReady"]
    assert not any(
        key.startswith(("planner/", "reference/", "one_step/", "oracle/"))
        for key in metrics["summary"]
    )
    assert metrics["comparisons"] and all(
        not c["b"].endswith("/rules") for c in metrics["comparisons"]
    )
    assert metrics["summary"]["learned/typed"]["n"] == 12
    training = read("results/benchmarks/sglang/42/training.json")
    assert training["newCalls"] == 0
    assert set(training["fitFamilies"]).isdisjoint(training["testFamilies"])
    assert training["losses"]["value"]["rows"] > 0
