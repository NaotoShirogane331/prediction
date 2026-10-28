"""Equation and contract tests for the independent PyTorch model."""

import math

import pytest
import torch

from cognitive_trajectory.losses import pinball_loss
from cognitive_trajectory.model import (
    CognitiveTrajectoryModel,
    RelationalGraphConv,
    TimeAwareLSTMCell,
)


def batch():
    torch.manual_seed(2026)
    return {
        "x": torch.randn(2, 3, 14),
        "missing": torch.zeros(2, 3, 14, dtype=torch.bool),
        "dt_days": torch.tensor([[0.0, 30.0, 90.0], [0.0, 60.0, 0.0]]),
        "lengths": torch.tensor([3, 2]),
        "baseline": torch.tensor([22.0, 25.0]),
        "horizon_days": torch.tensor([90.0, 365.0]),
    }


def test_pinball_matches_hand_computed_asymmetric_error():
    # target 2, errors [2, 1, 0, -1, -2]: [.1, .25, 0, .25, .1].
    prediction = torch.tensor([[0.0, 1.0, 2.0, 3.0, 4.0]], requires_grad=True)
    loss = pinball_loss(prediction, torch.tensor([2.0]))
    assert loss.item() == pytest.approx(0.14)
    loss.backward()
    assert prediction.grad[0, 0].item() == pytest.approx(-0.01)
    assert prediction.grad[0, 4].item() == pytest.approx(0.01)


def test_relations_share_both_directions_and_use_independent_self_transform():
    layer = RelationalGraphConv(1, 1)
    with torch.no_grad():
        layer.relation_weight.copy_(torch.tensor([[[2.0]], [[3.0]], [[5.0]], [[7.0]]]))
        layer.self_linear.weight.fill_(11.0)
    nodes = torch.tensor([[[1.0], [10.0], [20.0], [30.0], [40.0]]])
    edges = torch.tensor([[0, 1, 0, 2, 0, 3, 0, 4], [1, 0, 2, 0, 3, 0, 4, 0]])
    kinds = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    result = layer(nodes, edges, kinds)
    # Center = 11 + 20 + 60 + 150 + 280. Each leaf sees center once.
    torch.testing.assert_close(result.squeeze(-1), torch.tensor([[521.0, 112.0, 223.0, 335.0, 447.0]]))


def test_relational_mean_is_per_destination_and_per_relation():
    layer = RelationalGraphConv(1, 1, num_relations=2)
    with torch.no_grad():
        layer.relation_weight.copy_(torch.tensor([[[2.0]], [[3.0]]]))
        layer.self_linear.weight.zero_()
    nodes = torch.tensor([[[1.0], [3.0], [8.0]]])
    edges = torch.tensor([[0, 1, 2, 0], [2, 2, 2, 1]])
    kinds = torch.tensor([0, 0, 1, 0])
    # Destination2 gets mean(2,6) + 24 = 28, not global mean or degree sum.
    torch.testing.assert_close(layer(nodes, edges, kinds).squeeze(-1), torch.tensor([[0.0, 2.0, 28.0]]))


def test_tlstm_discounts_only_short_memory_before_ordinary_gates():
    cell = TimeAwareLSTMCell(1, 1)
    with torch.no_grad():
        for parameter in cell.parameters():
            parameter.zero_()
        cell.short_memory.weight.fill_(1.0)
    x = torch.zeros(1, 1)
    h = torch.zeros(1, 1)
    c = torch.ones(1, 1)
    h0, c0 = cell(x, (h, c), torch.tensor([0.0]))
    hd, cd = cell(x, (h, c), torch.tensor([30.0]))
    expected_c = 0.5 * (1 - math.tanh(1) + math.tanh(1) / math.log(math.e + 1))
    assert c0.item() == pytest.approx(0.5)
    assert h0.item() == pytest.approx(0.5 * math.tanh(0.5))
    assert cd.item() == pytest.approx(expected_c)
    assert hd.item() == pytest.approx(0.5 * math.tanh(expected_c))
    assert cd.item() < c0.item()


def test_model_orders_outputs_uses_horizon_and_adds_baseline_residually():
    inputs = batch()
    model = CognitiveTrajectoryModel().eval()
    original = model(**inputs)
    assert original.shape == (2, 5)
    assert torch.isfinite(original).all()
    assert (original[:, 1:] > original[:, :-1]).all()
    inputs["baseline"] += 2
    torch.testing.assert_close(model(**inputs), original + 2)
    inputs["horizon_days"] += 1000
    assert not torch.allclose(model(**inputs), original + 2)


def test_padding_preserves_state_and_ignores_invalid_padded_values():
    inputs = batch()
    model = CognitiveTrajectoryModel().eval()
    result = model(**inputs)
    inputs["x"][1, 2] = float("nan")
    inputs["dt_days"][1, 2] = float("nan")
    torch.testing.assert_close(model(**inputs), result)
    single = {key: value[1:2] for key, value in inputs.items()}
    for key in ("x", "missing", "dt_days"):
        single[key] = single[key][:, :2]
    torch.testing.assert_close(model(**single), result[1:2])


def test_missing_garbage_is_sanitized_but_missing_flags_remain_informative():
    inputs = batch()
    inputs["missing"][:, :, 0] = True
    model = CognitiveTrajectoryModel().eval()
    result = model(**inputs)
    inputs["x"][:, :, 0] = float("nan")
    torch.testing.assert_close(model(**inputs), result)
    inputs["x"][:, :, 0] = float("inf")
    torch.testing.assert_close(model(**inputs), result)
    inputs["x"][:, :, 0] = 0
    inputs["missing"][:, :, 0] = False
    assert not torch.allclose(model(**inputs), result)


def test_real_loss_updates_graph_temporal_and_quantile_parameters():
    inputs = batch()
    model = CognitiveTrajectoryModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    loss = pinball_loss(model(**inputs), torch.tensor([20.0, 23.0]))
    loss.backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
    optimizer.step()
    for prefix in ("group_encoders", "graph_layers", "temporal", "readout"):
        assert any(
            not torch.equal(before[name], parameter)
            for name, parameter in model.named_parameters()
            if name.startswith(prefix)
        )


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("x", torch.zeros(2, 3, 13)),
        ("missing", torch.zeros(2, 3, 14)),
        ("lengths", torch.tensor([0, 2])),
        ("lengths", torch.tensor([4, 2])),
        ("lengths", torch.tensor([3.0, 2.0])),
        ("dt_days", torch.tensor([[0.0, -1.0, 10.0], [0.0, 1.0, 0.0]])),
        ("dt_days", torch.tensor([[1.0, 1.0, 10.0], [0.0, 1.0, 0.0]])),
        ("dt_days", torch.zeros(2, 2)),
        ("horizon_days", torch.tensor([-1.0, 90.0])),
        ("horizon_days", torch.tensor([float("nan"), 90.0])),
        ("baseline", torch.tensor([22.0, float("inf")])),
        ("baseline", torch.zeros(2, 1)),
    ],
)
def test_model_rejects_malformed_or_invalid_inputs(key, value):
    inputs = batch()
    inputs[key] = value
    with pytest.raises((ValueError, TypeError)):
        CognitiveTrajectoryModel()(**inputs)


def test_model_rejects_nonfinite_observed_features_and_intervals():
    inputs = batch()
    inputs["x"][0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        CognitiveTrajectoryModel()(**inputs)
    inputs = batch()
    inputs["dt_days"][0, 1] = float("inf")
    with pytest.raises(ValueError, match="finite"):
        CognitiveTrajectoryModel()(**inputs)


@pytest.mark.parametrize("key", ["dt_days", "baseline", "horizon_days"])
@pytest.mark.parametrize("dtype", [torch.bool, torch.complex64])
def test_model_rejects_boolean_or_complex_numeric_inputs(key, dtype):
    inputs = batch()
    inputs[key] = inputs[key].to(dtype)
    with pytest.raises((ValueError, TypeError)):
        CognitiveTrajectoryModel()(**inputs)


@pytest.mark.parametrize(
    ("predictions", "targets"),
    [
        (torch.zeros(2, 4), torch.zeros(2)),
        (torch.zeros(2, 5), torch.zeros(2, 1)),
        (torch.zeros(0, 5), torch.zeros(0)),
        (torch.full((2, 5), float("nan")), torch.zeros(2)),
        (torch.zeros(2, 5), torch.tensor([float("inf"), 0.0])),
    ],
)
def test_pinball_rejects_malformed_or_nonfinite_inputs(predictions, targets):
    with pytest.raises((ValueError, TypeError)):
        pinball_loss(predictions, targets)
