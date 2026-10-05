"""Structured physical generation, supervision integrity, and causal gradients."""
import math

import pytest
import torch

from radar_vla.planner import (PlannerConfig, RiskConditionedPlanner, create_targets,
                              create_supervision, planner_physics_loss)
from radar_vla.tokenizer import PhysicalTokenizer, interpolate_trajectory, token_cross_entropy


def tiny_config(**kwargs):
    values = dict(hidden_dim=16, radar_dim=16, num_heads=4, num_layers=1,
                  bins=65, max_agents=2, horizon_steps=3)
    values.update(kwargs)
    return PlannerConfig(**values)


def training_batch():
    return {
        'agent_state': torch.tensor([[[8., 1., -2., 3., 0.], [0., 0., 0., 0., 0.]]]),
        'agent_mask': torch.tensor([[True, False]]),
        'agent_state_mask': torch.tensor([[[True] * 5, [False] * 5]]),
        'agent_future': torch.tensor([[[[9., 1.], [10., 1.], [11., 1.]], [[0., 0.], [0., 0.], [0., 0.]]]]),
        'agent_future_mask': torch.tensor([[[True, True, True], [False, False, False]]]),
        'ego_future': torch.tensor([[[1., 0.], [2., 0.], [3., 0.]]]),
        'ego_future_mask': torch.tensor([[True, False, True]]),
        'future_times_s': torch.tensor([[.5, 1., 1.5]]),
        'ego_velocity': torch.tensor([[5., 0.]]),
        'agent_radial_observed': torch.tensor([[-2., 0.]]),
        'agent_radial_mask': torch.tensor([[True, False]]),
        'agent_los': torch.tensor([[[1., 0.], [0., 0.]]]),
        'risk_target': torch.tensor([[0., 0., 0., 15., 0.]]),
        'risk_mask': torch.ones((1, 5), dtype=torch.bool),
        'agent_risk': torch.tensor([[[0., 0., 0., 15., 0.], [0., 0., 0., 0., 0.]]]),
        'agent_risk_mask': torch.tensor([[[True] * 5, [False] * 5]]),
        'critical_agent_order': torch.tensor([[0, 1]]),
        'road_centerline': torch.tensor([[[0., 0.], [5., 0.], [10., .5], [15., 1.5]]]),
        'road_centerline_mask': torch.ones(1, 4, dtype=torch.bool),
        'road_width': torch.tensor([[3.6]]),
        'road_mask': torch.ones(1, 1, dtype=torch.bool),
    }


def test_signed_log_round_trip_gaussian_proximity_and_overflow_rejection():
    tokenizer = PhysicalTokenizer(bins=65, coordinate_limit_m=80.)
    for value in [-80., -4., 0., 4., 80.]:
        restored = tokenizer.decode_scalar(tokenizer.encode_scalar(value))
        assert abs(restored - value) <= max(.15, abs(value) * .09)
    assert tokenizer.decode_scalar(tokenizer.encode_scalar(0.)) == pytest.approx(0.)
    q = tokenizer.soft_targets(0., sigma=.7)
    assert q.sum() == pytest.approx(1.) and q[32] > q[31] > q[25]
    with pytest.raises(ValueError, match='finite'):
        tokenizer.encode_scalar(float('nan'))
    with pytest.raises(ValueError, match='exceeds'):
        tokenizer.encode_scalar(80.01)


def test_actual_road_present_agents_and_spline_controls_replace_dummy_fields():
    model = RiskConditionedPlanner(tiny_config())
    batch = training_batch()
    supervision = create_supervision(batch, model.tokenizer, model.config)
    ids = supervision['target_ids'][0].tolist()
    decoded = model.decode(ids, batch['future_times_s'][0])
    assert decoded['mode'] == 'long' and decoded['physically_complete']
    assert len(decoded['agent_states']) == 1
    assert len(decoded['agent_trajectories']) == 1
    assert decoded['road']['width_m'] == pytest.approx(3.6, abs=.5)
    assert len(decoded['road']['centerline_controls']) == 4
    assert len(decoded['ego_control_points']) == 4
    assert len(decoded['ego_trajectory']) == 3  # Three dense samples are not four controls.
    assert '<ROAD_UNKNOWN>' not in model.tokenizer.special_ids
    assert '<UNKNOWN>' not in model.tokenizer.special_ids
    assert ids.count(model.tokenizer.token('<AGENT>')) == 1


def test_missing_endpoint_masks_entire_control_teacher_without_fabricated_future():
    model = RiskConditionedPlanner(tiny_config())
    batch = training_batch()
    batch['ego_future_mask'][0, -1] = False
    supervision = create_supervision(batch, model.tokenizer, model.config)
    ego = supervision['rows'][0]['ego']
    assert not ego['fitted']
    indices = torch.tensor(ego['indices'])
    assert (supervision['target_ids'][0, indices] == model.tokenizer.pad_id).all()
    decoded = model.decode(supervision['target_ids'][0].tolist())
    assert decoded['valid'] and not decoded['ego_complete']
    assert all(p is None for p in decoded['ego_trajectory'])


def test_missing_road_interface_is_rejected_and_unobserved_width_is_masked():
    model = RiskConditionedPlanner(tiny_config())
    batch = training_batch()
    del batch['road_centerline']
    with pytest.raises(ValueError, match='map-derived'):
        create_targets(batch, model.tokenizer, model.config)
    batch = training_batch()
    batch['road_mask'][:] = False
    decoded = model.decode(create_targets(batch, model.tokenizer, model.config)[0].tolist())
    assert decoded['road']['width_m'] is None and not decoded['physically_complete']


def test_short_teacher_selects_actual_counterfactual_hazard_not_nearest_agent():
    model = RiskConditionedPlanner(tiny_config(short_max_agents=1))
    batch = training_batch()
    batch['agent_mask'][:] = True
    batch['agent_state_mask'][:] = True
    batch['agent_state'][0, 0, :2] = torch.tensor([1., 5.])
    batch['agent_state'][0, 1, :2] = torch.tensor([30., 0.])
    batch['risk_target'][0, 0] = 1.
    batch['agent_risk'][0, 1] = torch.tensor([1., 1., 1., 0., 6.])
    batch['agent_risk_mask'][:] = True
    batch['critical_agent_order'][0] = torch.tensor([1, 0])
    supervision = create_supervision(batch, model.tokenizer, model.config)
    assert supervision['rows'][0]['agents'][0]['source_index'] == 1
    decoded = model.decode(supervision['target_ids'][0].tolist())
    assert decoded['mode'] == 'short' and decoded['agent_states'][0]['x'] > 20
    assert decoded['ego_times_s'] == [.5, 1.]
    del batch['agent_risk']
    with pytest.raises(ValueError, match='counterfactual'):
        create_targets(batch, model.tokenizer, model.config)


def test_causal_predictions_cannot_see_future_teacher_tokens():
    torch.manual_seed(9)
    model = RiskConditionedPlanner(tiny_config()).eval()
    targets = create_targets(training_batch(), model.tokenizer, model.config)
    changed = targets.clone()
    changed[:, 6:] = model.tokenizer.encode_scalar(27.)
    radar = torch.randn(1, 4, 16)
    risk = torch.tensor([[.1, .2, .3, 5., 1.]])
    with torch.no_grad():
        a = model(radar, risk, ['继续'], targets)
        b = model(radar, risk, ['继续'], changed)
    torch.testing.assert_close(a[:, :7], b[:, :7])
    assert not torch.allclose(a[:, 7:], b[:, 7:])


def test_loss_backpropagates_to_continuous_radar_and_risk_conditions():
    torch.manual_seed(2)
    model = RiskConditionedPlanner(tiny_config())
    supervision = create_supervision(training_batch(), model.tokenizer, model.config)
    targets = supervision['target_ids']
    radar = torch.randn(1, 4, 16, requires_grad=True)
    risk = torch.tensor([[.1, .2, .3, 5., 1.]], requires_grad=True)
    loss = token_cross_entropy(model(radar, risk, ['drive'], targets), targets, model.tokenizer,
                               continuous_targets=supervision['continuous_targets'])
    loss.backward()
    assert math.isfinite(loss.item()) and radar.grad.abs().sum() > 0 and risk.grad.abs().sum() > 0


def test_gaussian_loss_is_centered_on_original_continuous_label_not_rounded_bin():
    tokenizer = PhysicalTokenizer(bins=9)
    value = 1.2
    ids = torch.tensor([[tokenizer.encode_scalar(value)]])
    logits = torch.linspace(-2., 2., tokenizer.vocab_size)[None, None]
    continuous = torch.tensor([[value]])
    actual = token_cross_entropy(logits, ids, tokenizer, continuous_targets=continuous)
    expected = -(tokenizer.soft_targets(value) * logits.log_softmax(-1)[0, 0,
                 tokenizer.position_offset:tokenizer.position_offset + tokenizer.bins]).sum()
    torch.testing.assert_close(actual, expected)
    assert not torch.isclose(actual, token_cross_entropy(logits, ids, tokenizer))


def test_padding_has_no_token_loss_and_near_physical_prediction_is_better_than_far():
    tokenizer = PhysicalTokenizer(bins=65)
    target = torch.tensor([[tokenizer.encode_scalar(0.), tokenizer.pad_id]])
    near = torch.zeros(1, 2, tokenizer.vocab_size)
    far = near.clone()
    near[0, 0, tokenizer.encode_scalar(.5)] = 4.
    far[0, 0, tokenizer.encode_scalar(80.)] = 4.
    near[0, 1, :] = 1.e4
    assert token_cross_entropy(near, target, tokenizer) < token_cross_entropy(far, target, tokenizer)
    assert token_cross_entropy(near[:, :1], target[:, :1], tokenizer) == token_cross_entropy(near, target, tokenizer)


@pytest.mark.parametrize('policy,mode', [('adaptive', None), ('always_long', 'long'), ('always_short', 'short')])
def test_grammar_generation_is_parseable_and_q5_policy_only_forces_mode(policy, mode):
    torch.manual_seed(5)
    model = RiskConditionedPlanner(tiny_config(reasoning_policy=policy)).eval()
    ids = model.generate(torch.randn(1, 3, 16), torch.zeros(1, 5), ['go'], max_new_tokens=160)[0]
    decoded = model.decode(ids)
    assert decoded['valid'] and decoded['physically_complete']
    assert decoded['mode'] in {'short', 'long'}
    assert mode is None or decoded['mode'] == mode
    assert len(decoded['ego_trajectory']) in {2, 3}
    assert not model.decode(ids[:-1])['valid']
    assert model.tokenizer.pad_id not in ids


def test_no_risk_ablation_really_removes_prefix_and_is_invariant_to_risk():
    model = RiskConditionedPlanner(tiny_config(risk_source='none')).eval()
    radar = torch.randn(1, 3, 16)
    risk = torch.randn(1, 5, requires_grad=True)
    ids = create_targets(training_batch(), model.tokenizer, model.config)
    prefix, _ = model.encode_prefix(radar, risk, ['go'])
    assert prefix.shape[1] == 3 + 2
    torch.testing.assert_close(model(radar, risk, ['go'], ids), model(radar, risk * 100, ['go'], ids))


def test_decode_reports_partial_observed_agent_velocity_without_padding_agents():
    model = RiskConditionedPlanner(tiny_config())
    batch = training_batch()
    batch['agent_state_mask'][0, 0, 3:] = False
    ids = create_targets(batch, model.tokenizer, model.config)[0].tolist()
    result = model.decode(ids, future_times_s=[.5, 1., 1.5])
    assert result['ego_times_s'] == [.5, 1., 1.5]
    assert result['agent_states'][0]['vx'] is None
    assert result['agent_states'][0]['x'] is not None
    assert not result['physically_complete']


def test_unknown_bootstrap_modes_are_excluded_and_do_not_block_other_gradients():
    model = RiskConditionedPlanner(tiny_config())
    batch = {key: value.repeat(2, *([1] * (value.ndim - 1))) for key, value in training_batch().items()}
    batch['risk_mask'][0, :2] = False
    targets = create_targets(batch, model.tokenizer, model.config)
    assert (targets[0] == model.tokenizer.pad_id).all()
    assert targets[1, 0] == model.tokenizer.token('<LONG>')
    logits = torch.zeros(2, targets.shape[1], model.tokenizer.vocab_size, requires_grad=True)
    loss = token_cross_entropy(logits, targets, model.tokenizer)
    loss.backward()
    assert logits.grad[0].abs().sum() == 0 and logits.grad[1].abs().sum() > 0
    assert token_cross_entropy(logits[:1], targets[:1], model.tokenizer) == 0
    batch['risk_mask'][0, 0] = True
    assert (create_targets(batch, model.tokenizer, model.config)[0] == model.tokenizer.pad_id).all()
    batch['risk_target'][0, 0] = 1.
    assert create_targets(batch, model.tokenizer, model.config)[0, 0] == model.tokenizer.token('<SHORT>')


def test_physics_losses_train_planner_tokens_and_exclude_unmeasured_doppler():
    model = RiskConditionedPlanner(tiny_config())
    batch = training_batch()
    supervision = create_supervision(batch, model.tokenizer, model.config)
    logits = torch.zeros((*supervision['target_ids'].shape, model.tokenizer.vocab_size), requires_grad=True)
    losses = planner_physics_loss(logits, supervision, batch, model.tokenizer, model.config)
    assert losses['ego_trajectory'] > 0 and losses['agent_trajectory'] > 0 and losses['road'] > 0
    assert losses['doppler'] > 0
    losses['total'].backward()
    ego_indices = torch.tensor(supervision['rows'][0]['ego']['indices'])
    assert logits.grad[0, ego_indices].abs().sum() > 0
    batch['agent_radial_mask'][:] = False
    batch['agent_radial_observed'][:] = float('nan')
    assert planner_physics_loss(logits, supervision, batch, model.tokenizer, model.config)['doppler'] == 0


def test_trajectory_wrapper_uses_shared_decoder_and_rejects_missing_controls():
    assert interpolate_trajectory([[0., 0.], [4., 2.]], [0., 2.], [0., 1., 2.]) == [[0., 0.], [2., 1.], [4., 2.]]
    with pytest.raises(ValueError, match='missing'):
        interpolate_trajectory([[0., 0.], None], [0., 2.], [1.])


def test_weighted_doppler_projection_is_not_renormalized_to_center_los():
    model = RiskConditionedPlanner(tiny_config())
    batch = training_batch()
    batch['agent_doppler_projection'] = torch.tensor([[[.5, 0.], [0., 0.]]])
    batch['agent_radial_observed'][0, 0] = -2.5
    supervision = create_supervision(batch, model.tokenizer, model.config)
    logits = torch.zeros((*supervision['target_ids'].shape, model.tokenizer.vocab_size))
    # Symmetric bins predict all velocity expectations zero. Mean-cell Doppler
    # is (0-5)*0.5=-2.5, exactly observed; vr-head error contributes Huber(2.5)/2=1.
    loss = planner_physics_loss(logits, supervision, batch, model.tokenizer, model.config)['doppler']
    assert float(loss) == pytest.approx(1., abs=1e-5)


def test_ego_origin_is_a_coordinate_constraint_even_with_even_number_of_bins():
    model = RiskConditionedPlanner(tiny_config(bins=8, reasoning_policy='always_long')).eval()
    ids = model.generate(torch.randn(1, 3, 16), torch.zeros(1, 5), ['go'], max_new_tokens=160)[0]
    plan = model.decode(ids)
    assert plan['valid'] and plan['ego_control_points'][0] == [0., 0.]


def test_instruction_budget_rejects_overflow_instead_of_truncating_route():
    tokenizer = PhysicalTokenizer()
    assert len(tokenizer.encode_instruction('左转', max_bytes=6)) == 6
    with pytest.raises(ValueError, match='exceeding'):
        tokenizer.encode_instruction('左转后右转', max_bytes=6)
