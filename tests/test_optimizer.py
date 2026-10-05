import pytest

from newsvendor.optimizer import empirical, finite, minimax, optimal, risk


def theta(v, b):
    return {"c": 6, "p": 10, "v": v, "b": b, "F": [[0, 0.6], [100, 0.4]]}


@pytest.mark.parametrize(
    "v,b,q,cost", [(0, 0, 0, 160), (4, 0, 100, 120), (0, 8, 100, 360), (4, 8, 100, 120)]
)
def test_proposal_optima(v, b, q, cost):
    assert optimal(theta(v, b)) == {"q": q, "cost": pytest.approx(cost)}


@pytest.mark.parametrize(
    "pairs,q,gamma",
    [
        ([(4, 0), (4, 8)], 100, 0),
        ([(0, 0), (4, 0)], 50 / 3, 100 / 3),
        ([(0, 0), (0, 8)], 37.5, 75),
        ([(0, 0), (0, 8), (4, 0), (4, 8)], 450 / 7, 900 / 7),
    ],
)
def test_proposal_minimax(pairs, q, gamma):
    result = minimax([theta(v, b) for v, b in pairs])
    assert result["q"] == pytest.approx(q)
    assert result["gamma"] == pytest.approx(gamma)


def test_sequential_budget_and_response_probability():
    omega = [theta(v, b) for v in (0, 4) for b in (0, 8)]
    assert minimax(omega, integer=True)["gamma"] == pytest.approx(129.6)
    two = finite(omega, 2)
    assert two["values"]["v"] == two["values"]["b"] == pytest.approx(30)
    assert two["action"] == "v"
    one = finite(omega, 1)
    assert one["values"]["v"] == pytest.approx(85)
    assert one["values"]["b"] == pytest.approx(160 / 3)
    assert one["action"] == "b"
    assert finite([theta(0, 0), theta(0, 8)], 1, {"b": 20}, rho={"b": 0.2})["action"] == "handoff"
    assert finite([theta(0, 0), theta(0, 8)], 1, {"b": 20}, rho={"b": 0.5})["values"][
        "b"
    ] == pytest.approx(57.5)


def test_general_envelope_and_invalid_inputs():
    omega = [
        {"c": 5, "p": 11, "v": 1, "b": 0, "F": [[0, 0.2], [40, 0.5], [100, 0.3]]},
        {"c": 6, "p": 12, "v": 4, "b": 4, "F": [[0, 0.5], [60, 0.2], [100, 0.3]]},
    ]
    fit = minimax(omega)
    costs = [optimal(t)["cost"] for t in omega]
    for i in range(401):
        assert (
            fit["gamma"]
            <= max(risk(i / 4, t) - cost for t, cost in zip(omega, costs, strict=True)) + 1e-8
        )
    with pytest.raises(ValueError, match="Censored"):
        empirical([{"demand": 4, "complete": False, "stockout": True}])
    with pytest.raises(ValueError, match="empty"):
        minimax([])
    with pytest.raises(ValueError, match="Negative"):
        optimal(theta(7, 0))
