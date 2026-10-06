import copy

import numpy as np
import pytest
import torch

from newsvendor.construction import training_rows
from newsvendor.corpus import SLOTS, generate
from newsvendor.encoder import ACTIONS, CONSTRAINT, DEMAND, Embeddings, texts
from newsvendor.heads import Head
from newsvendor.train import HEADS, checkpoint, head_dim, load


def test_checkpoint_encoder_guard_and_label_free_features(tmp_path):
    config = {"seed": 42, "groups": 10, "variants": 1, "budget": 2}
    episode = generate(config)[0]
    vocabulary = {*texts(episode["input"]), *ACTIONS.values(), *SLOTS.values(), DEMAND, CONSTRAINT}
    cache = Embeddings(
        {"dim": 4, "revision": "fixture"}, {t: [1.0, 0.0, 0.5, -0.5] for t in vocabulary}
    )
    rows = training_rows([episode], cache)
    poison = copy.deepcopy(episode)
    poison["gold"]["theta"]["v"] = 1e8
    other = training_rows([poison], cache)
    for name in ("evidence", "type", "state"):
        assert [r["y"] for r in rows[name]] == [r["y"] for r in other[name]]
        assert np.array_equal(
            np.array([r["x"] for r in rows[name]]), np.array([r["x"] for r in other[name]])
        )
    model = {k: Head(head_dim(k, cache), *shape) for k, shape in HEADS.items()}
    model.update(temperatures={k: {"temperature": 1.0} for k in HEADS}, threshold=0.5)
    path = str(tmp_path / "model.pt")
    checkpoint(path, model, config, cache, [episode])
    restored, saved = load(path, cache)
    assert saved == config
    for name in HEADS:
        for original, recovered in zip(
            model[name].parameters(), restored[name].parameters(), strict=True
        ):
            assert torch.equal(original, recovered)
    with pytest.raises(ValueError, match="encoder differs"):
        load(path, Embeddings({"dim": 4, "revision": "different"}, cache.values))
    payload = torch.load(path, weights_only=True)
    payload["schema"] = 2
    torch.save(payload, path)
    with pytest.raises(ValueError, match="retrain"):
        load(path, cache)
