"""Lock down the real head path, masking, and frozen-Planner gradient routing."""
import pytest
import torch

from radar_vla.model import ModelConfig, RadarVLAGrounder
from radar_vla.planner import PlannerConfig, RiskConditionedPlanner
from radar_vla.risk_distillation import (configure_risk_head_training, first_mode_logits,
    is_risk_parameter, oracle_mode_kl, refinement_loss, risk_from_frozen_tokens)
from test_radar_vla_hf import tiny_local_qwen, make_planner


def systems():
    torch.manual_seed(42)
    grounder = RadarVLAGrounder(ModelConfig(hidden_dim=8, num_heads=2, num_queries=2,
        use_detection=False, planner_ego_context=True))
    planner = RiskConditionedPlanner(PlannerConfig(hidden_dim=8, radar_dim=8,
        num_heads=2, num_layers=1, max_length=128))
    return grounder, planner


def test_masked_kl_detaches_teacher_and_keeps_student_gradients():
    student = torch.tensor([[1., -1.], [-2., 2.]], requires_grad=True)
    teacher = torch.tensor([[-1., 1.], [float('nan'), 0.]], requires_grad=True)
    loss = oracle_mode_kl(student, teacher, torch.tensor([True, False]))
    expected = torch.nn.functional.kl_div(student[:1].log_softmax(-1),
        teacher[:1].detach().softmax(-1), reduction='batchmean')
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert teacher.grad is None
    assert student.grad[0].abs().sum() > 0
    assert student.grad[1].abs().sum() == 0
    empty = oracle_mode_kl(student, teacher, torch.tensor([False, False]))
    assert empty.item() == 0 and empty.requires_grad


def test_real_optimizer_step_only_changes_risk_head_path():
    grounder, planner = systems()
    parameters = configure_risk_head_training(grounder, planner)
    assert not grounder.training and not planner.training
    assert all(not p.requires_grad for p in planner.parameters())
    assert all(p.requires_grad == is_risk_parameter(n) for n, p in grounder.named_parameters())
    before = {n: p.detach().clone() for n, p in grounder.named_parameters()}
    planner_before = {n: p.detach().clone() for n, p in planner.named_parameters()}
    tokens = torch.randn(2, 3, 8, requires_grad=True)
    batch = dict(radar_tokens=tokens, instruction=['drive', 'drive'],
        risk_target=torch.tensor([[0., 0., 0., 10., 0.], [1., 1., 1., 0., 2.]]),
        risk_mask=torch.ones(2, 5, dtype=torch.bool),
        oracle_mode_logits=torch.tensor([[-3., 3.], [3., -3.]], requires_grad=True))
    optimizer = torch.optim.AdamW(parameters, lr=.001)
    losses = refinement_loss(grounder, planner, batch)
    losses['total'].backward()
    assert tokens.grad is None and batch['oracle_mode_logits'].grad is None
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in parameters)
    assert all(p.grad is None for p in planner.parameters())
    optimizer.step()
    assert any(not torch.equal(p, before[n]) for n, p in grounder.named_parameters() if is_risk_parameter(n))
    assert all(torch.equal(p, before[n]) for n, p in grounder.named_parameters() if not is_risk_parameter(n))
    assert all(torch.equal(p, planner_before[n]) for n, p in planner.named_parameters())


def test_cached_head_matches_full_sensor_forward_exactly():
    grounder, planner = systems();grounder.eval()
    batch = dict(radar=torch.rand(2, 2, 2, 16, 20),
        range_m=torch.linspace(0., 50., 16).repeat(2, 1),
        azimuth_rad=torch.linspace(-1., 1., 20).repeat(2, 1),
        time_offsets_s=torch.tensor([[-.1, 0.], [-.1, 0.]]),
        ego=torch.tensor([[5., 0., .1], [0., 0., 0.]]),
        future_times_s=torch.arange(1, 7).repeat(2, 1).float() * .5)
    with torch.no_grad():
        full = grounder(batch)
        cached = risk_from_frozen_tokens(grounder, full['radar_tokens'])
    for key in ['risk', 'risk_logits']:
        torch.testing.assert_close(cached[key], full[key], rtol=0, atol=0)


def test_kl_reaches_risk_query_attention_and_mlp_through_frozen_planner():
    grounder, planner = systems();configure_risk_head_training(grounder, planner)
    tokens = torch.randn(2, 3, 8)
    output = risk_from_frozen_tokens(grounder, tokens)
    student = first_mode_logits(planner, tokens, output['risk'], ['go', 'go'])
    teacher = torch.tensor([[6., -6.], [-6., 6.]])
    oracle_mode_kl(student, teacher, torch.tensor([True, True])).backward()
    for part in ['risk_query', 'risk_attention', 'risk_head']:
        grads = [p.grad for n, p in grounder.named_parameters() if n == part or n.startswith(part + '.')]
        assert any(g is not None and g.abs().sum() > 0 for g in grads), part
    assert all(p.grad is None for p in planner.parameters())


@pytest.mark.parametrize('temperature', [0., -1., float('nan')])
def test_invalid_temperature_rejected(temperature):
    with pytest.raises(ValueError):
        oracle_mode_kl(torch.zeros(1, 2), torch.zeros(1, 2), torch.tensor([True]), temperature)


def test_invalid_projection_or_missing_shared_ego_is_rejected():
    grounder, planner = systems();planner.config.risk_projection = 'isotonic'
    with pytest.raises(ValueError):configure_risk_head_training(grounder, planner)
    planner.config.risk_projection = 'none';grounder.config.planner_ego_context = False
    with pytest.raises(ValueError):configure_risk_head_training(grounder, planner)


def test_actual_local_qwen_propagates_kl_to_risk_without_updating_planner(tiny_local_qwen):
    grounder, _ = systems()
    planner = make_planner(tiny_local_qwen, lora=True, gradient_checkpointing=True)
    parameters = configure_risk_head_training(grounder, planner)
    frozen = {name: value.detach().clone() for name, value in planner.state_dict().items()}
    tokens = torch.randn(2, 3, 8)
    prediction = risk_from_frozen_tokens(grounder, tokens)
    logits = first_mode_logits(planner, tokens, prediction['risk'], ['Drive safely', 'turn left'])
    kl = oracle_mode_kl(logits, torch.tensor([[4., -4.], [-4., 4.]]), torch.tensor([True, True]))
    optimizer = torch.optim.AdamW(parameters, lr=.001)
    kl.backward();optimizer.step()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in parameters)
    assert all(p.grad is None for p in planner.parameters())
    assert all(torch.equal(value, frozen[name]) for name, value in planner.state_dict().items())
