from newsvendor.corpus import audit, generate
from newsvendor.io import digest, read
from newsvendor.optimizer import optimal


def test_revised_workload_has_independent_economics_and_no_theta_overlap():
    config = read("configs/full.json")
    episodes = generate(config)
    assert episodes == generate({**config, "seed": config["seed"] + 1})
    assert audit(episodes)["counts"] == {"train": 360, "dev": 60, "cal": 60, "test": 120}
    assert len({digest(e["gold"]["theta"]) for e in episodes}) == 120
    train = {digest(e["gold"]["theta"]) for e in episodes if e["split"] == "train"}
    assert all(digest(e["gold"]["theta"]) not in train for e in episodes if e["split"] == "test")
    assert len({e["gold"]["theta"]["c"] / e["gold"]["theta"]["p"] for e in episodes}) == 120
    assert all(len(e["gold"]["theta"]["F"]) == 5 for e in episodes)
    assert all(
        optimal(e["gold"]["theta"], e["input"]["task"]["bounds"])["q"]
        not in e["input"]["task"]["bounds"]
        for e in episodes
    )
    assert all(e["scenario"] not in e["input"]["task"]["sku"] for e in episodes)
    for family in {e["family"] for e in episodes}:
        variants = [e for e in episodes if e["family"] == family]
        assert all(
            e["gold"] == variants[0]["gold"] and e["input"]["task"] == variants[0]["input"]["task"]
            for e in variants
        )
    assert len({digest(e["input"]["task"]["costs"]) for e in episodes}) == 120
