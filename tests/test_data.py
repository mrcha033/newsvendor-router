from newsvendor.data import normalize_sharc, normalize_tat, select, split_retail


def test_label_separation_in_official_schemas():
    tat = normalize_tat(
        [
            {
                "table": {"table": [["Cost", "5"]]},
                "paragraphs": [],
                "questions": [
                    {"uid": "q", "question": "Cost?", "answer": 5, "derivation": "private-answer"}
                ],
            }
        ],
        "train",
    )[0]
    assert "answer" not in tat["input"] and "derivation" not in tat["input"]
    assert tat["label"]["answer"] == 5
    sharc = normalize_sharc(
        [
            {
                "utterance_id": "s",
                "question": "Allowed?",
                "snippet": "Rule",
                "scenario": "",
                "answer": "What age?",
                "evidence": ["private"],
            }
        ],
        "dev",
    )[0]
    assert "evidence" not in sharc["input"] and sharc["label"]["action"] == "ask"


def test_source_duplicate_replenishment():
    train = [{"id": f"t{i}", "group": str(i)} for i in range(6)]
    dev = [{"id": f"d{i}", "group": str(i)} for i in range(10)]
    selected = select(train, dev, 10)
    assert len(selected) == 10
    assert {r["group"] for r in selected[:6]}.isdisjoint(r["group"] for r in selected[6:])


def test_retail_time_split_preserves_stockout_observation_type():
    rows = [
        {
            "store_id": 1,
            "product_id": 2,
            "dt": f"day-{i:03}",
            "sale_amount": i / 100,
            "stock_hour6_22_cnt": i % 2,
            "hours_stock_status": [],
        }
        for i in range(90)
    ]
    result = split_retail(rows[::-1], 1)[0]
    assert [len(result[k]) for k in ("train", "cal", "test")] == [54, 18, 18]
    assert result["train"][-1]["date"] < result["cal"][0]["date"] < result["test"][0]["date"]
    assert not result["latentDemandLabels"]
    assert not result["train"][1]["complete"] and result["train"][1]["stockout"]
