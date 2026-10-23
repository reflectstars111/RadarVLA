"""Real local Qwen action-query tests, including LoRA and checkpoint contracts."""
import pytest
import torch

from test_radar_vla_hf import tiny_local_qwen
from test_parallel_action import config, inputs


@pytest.mark.parametrize('lora,dtype', [(False, 'float32'), (True, 'bfloat16')])
def test_real_qwen_action_query_prefill_gradient_and_description_independence(tiny_local_qwen, lora, dtype):
    from radar_vla.hf_planner import HFRiskConditionedPlanner
    from radar_vla.action_head import parallel_action_loss
    model = HFRiskConditionedPlanner(config(), tiny_local_qwen, lora=lora, dtype=dtype,
                                    gradient_checkpointing=True)
    radar, risk, text, ego, times = inputs()
    calls = []
    handle = model._native_model().base_model.register_forward_hook(lambda *args: calls.append(1))
    output = model(radar, risk, text, observed_ego=ego, future_times_s=times)
    handle.remove()
    assert calls == [1]
    batch = dict(ego_future=torch.zeros(2, 6, 2), ego_future_mask=torch.ones(2, 6, dtype=torch.bool),
                 action_mode_target=torch.tensor([0, 1]), action_mode_mask=torch.ones(2, dtype=torch.bool))
    loss, _ = parallel_action_loss(output, batch, model.config)
    loss.backward()
    for value in (model.action_query, model.radar_projector.weight, model.action_ego_projector[0].weight):
        assert value.grad is not None and torch.isfinite(value.grad).all() and value.grad.abs().sum() > 0
    if lora:
        gradients = [p.grad for n, p in model.named_parameters() if 'lora_B' in n]
        assert any(g is not None and g.abs().sum() > 0 for g in gradients)
    model.eval()
    base = model(radar, risk, text, observed_ego=ego, future_times_s=times)
    teacher = model(radar, risk, text, torch.ones(2, 8, dtype=torch.long), observed_ego=ego, future_times_s=times)
    torch.testing.assert_close(base['positions'], teacher['positions'])
    teacher['description_logits'].square().mean().backward()
    assert model.physical_head.weight.grad is not None
    assert model.plan_action(radar, risk, text, ego, times)[0]['valid']


def test_hf_checkpoint_protocol_roundtrip_and_reject_legacy(tiny_local_qwen):
    from radar_vla.hf_planner import HFRiskConditionedPlanner
    model = HFRiskConditionedPlanner(config(), tiny_local_qwen, lora=False, gradient_checkpointing=False).eval()
    saved = model.checkpoint_state()
    with torch.no_grad():
        model.action_query.add_(1.)
    model.load_checkpoint_state(saved)
    torch.testing.assert_close(model.action_query.cpu(), saved['parameters']['action_query'])
    old = HFRiskConditionedPlanner(config(action_representation='token_spline_v1'), tiny_local_qwen,
                                   lora=False, gradient_checkpointing=False)
    with pytest.raises(ValueError, match='representation'):
        old.load_checkpoint_state(saved)
    with pytest.raises(ValueError, match='representation'):
        model.load_checkpoint_state(old.checkpoint_state())
