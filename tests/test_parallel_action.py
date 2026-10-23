import json
import numpy as np
import pytest
import torch

from radar_vla.planner import PlannerConfig, RiskConditionedPlanner


def config(**changes):
    values = dict(hidden_dim=16, radar_dim=8, num_heads=2, num_layers=1,
                  max_length=128, bins=8, action_representation='parallel_acceleration_v1')
    values.update(changes)
    return PlannerConfig(**values)


def inputs(batch=2):
    return (torch.randn(batch, 3, 8), torch.zeros(batch, 5), ['Drive'] * batch,
            dict(velocity=torch.tensor([[4., 0.]]).repeat(batch, 1),
                 acceleration=torch.tensor([[0., .4]]).repeat(batch, 1),
                 yaw_rate=torch.full((batch,), .1)), torch.arange(1, 7) * .5)


def test_parallel_action_one_backbone_call_gradient_and_exact_initial_state():
    from radar_vla.action_head import parallel_action_loss
    model = RiskConditionedPlanner(config())
    radar, risk, text, ego, times = inputs()
    calls = []
    handle = model.decoder.register_forward_hook(lambda *args: calls.append(1))
    out = model(radar, risk, text, observed_ego=ego, future_times_s=times)
    handle.remove()
    assert len(calls) == 1
    assert out['positions'].shape == (2, 6, 2)
    torch.testing.assert_close(out['accelerations'][:, 0], ego['acceleration'], check_dtype=False)
    assert out['accelerations'].norm(dim=-1).max() <= model.config.action_max_acceleration_mps2 + 1e-6
    assert out['jerks'].norm(dim=-1).max() <= model.config.action_max_jerk_mps3 + 1e-6
    batch = dict(ego_future=torch.zeros(2, 6, 2), ego_future_mask=torch.ones(2, 6, dtype=torch.bool),
                 mode_target=torch.tensor([0, 1]), mode_supervision_mask=torch.ones(2, dtype=torch.bool))
    loss, details = parallel_action_loss(out, batch, model.config)
    loss.backward()
    for param in (model.action_query, model.radar_projector.weight, model.decoder.layers[0].self_attn.in_proj_weight):
        assert param.grad is not None and torch.isfinite(param.grad).all() and param.grad.abs().sum() > 0
    assert details['mode_count'] == 2


def test_plan_native_sampler_matches_differentiable_integration_and_json():
    from radar_vla.evaluation import spline_plan_sampler
    model = RiskConditionedPlanner(config(reasoning_policy='always_long')).eval()
    radar, risk, text, ego, times = inputs()
    out = model(radar, risk, text, observed_ego=ego, future_times_s=times)
    plans = model.plan_action(radar, risk, text, ego, times)
    plan = json.loads(json.dumps(plans[0]))
    assert plan['schema'] == 'radar_vla_parallel_acceleration_v1'
    sampler = spline_plan_sampler(plan)
    np.testing.assert_allclose(sampler(times), out['positions'][0].detach().numpy(), atol=1e-5)
    np.testing.assert_allclose(sampler([0.], 1)[0], ego['velocity'][0], atol=1e-6)
    np.testing.assert_allclose(sampler([0.], 2)[0], ego['acceleration'][0], atol=1e-6)
    with pytest.raises(ValueError, match='native horizon'):
        sampler([3.01])


def test_short_nonzero_terminal_acceleration_has_bounded_explicit_continuation():
    from radar_vla.execution import execution_plan_sampler
    model = RiskConditionedPlanner(config(reasoning_policy='always_short')).eval()
    radar, risk, text, ego, times = inputs(1)
    plan = model.plan_action(radar, risk, text, ego, times)[0]
    sample, meta = execution_plan_sampler(plan)
    assert plan['ego_times_s'] == [.5, 1.]
    assert meta['native_horizon_s'] == 1.
    assert meta['fallback'] == 'acceleration_release_then_jerk_limited_stop'
    assert meta['execution_horizon_s'] > 1.
    t = np.linspace(0, meta['execution_horizon_s'], 1001)
    assert np.linalg.norm(sample(t, 2), axis=-1).max() <= 8. + 1e-5
    assert np.linalg.norm(sample(t, 3), axis=-1).max() <= 20. + 1e-5
    np.testing.assert_allclose(sample([meta['execution_horizon_s']], 1), 0., atol=1e-6)
    for d in (0, 1, 2):
        np.testing.assert_allclose(sample([1.-1e-8], d), sample([1.+1e-8], d), atol=1e-5)


def test_unknown_mask_no_nan_loss_no_future_prefix_and_fixed_grid():
    from radar_vla.action_head import parallel_action_loss
    model = RiskConditionedPlanner(config())
    radar, risk, text, ego, times = inputs()
    out = model(radar, risk, text, observed_ego=ego, future_times_s=times)
    batch = dict(ego_future=torch.full((2, 6, 2), float('nan')),
                 ego_future_mask=torch.zeros(2, 6, dtype=torch.bool),
                 mode_target=torch.tensor([-1, -1]), mode_supervision_mask=torch.zeros(2, dtype=torch.bool))
    loss, details = parallel_action_loss(out, batch, model.config)
    assert loss.item() == 0 and details['mode_count'] == 0
    loss.backward()
    with_teacher = model(radar, risk, text, torch.ones(2, 5, dtype=torch.long), observed_ego=ego, future_times_s=times)
    torch.testing.assert_close(with_teacher['positions'], out['positions'])
    assert with_teacher['description_logits'].shape[:2] == (2, 5)
    with pytest.raises(ValueError, match='0.5'):
        model(radar, risk, text, observed_ego=ego, future_times_s=times + .1)


def test_checkpoint_representation_rejects_cross_load_and_old_path_unchanged():
    old = RiskConditionedPlanner(config(action_representation='token_spline_v1'))
    new = RiskConditionedPlanner(config())
    with pytest.raises(ValueError, match='representation'):
        new.load_state_dict(old.state_dict(), strict=False)
    with pytest.raises(ValueError, match='representation'):
        old.load_state_dict(new.state_dict(), strict=False)
    new.load_state_dict(new.state_dict())
    radar, risk, text, _, _ = inputs()
    assert old(radar, risk, text, torch.ones(2, 4, dtype=torch.long)).shape[:2] == (2, 4)


def test_new_action_refuses_legacy_token_rl_and_timing_uses_no_generation_loop():
    from radar_vla.execution import measure_plan
    from radar_vla.rl_policy import sample_plan
    model = RiskConditionedPlanner(config()).eval()
    radar, risk, text, ego, times = inputs(1)
    with pytest.raises(ValueError, match='continuous'):
        sample_plan(model, radar, risk, text)
    model.generate = lambda *a, **k: pytest.fail('action waited for description token generation')
    report = measure_plan(model, radar, risk, text, observed_ego=ego, future_times_s=times)
    assert report['ids'] == [] and report['plan']['valid']


def test_invalid_initial_acceleration_is_rejected_not_clipped():
    model = RiskConditionedPlanner(config()).eval()
    radar, risk, text, ego, times = inputs(1)
    ego['acceleration'][:] = torch.tensor([10., 0.])
    out = model(radar, risk, text, observed_ego=ego, future_times_s=times)
    torch.testing.assert_close(out['accelerations'][0, 0], ego['acceleration'][0], check_dtype=False)
    assert not out['initial_state_valid'][0]
    assert not model.plan_action(radar, risk, text, ego, times)[0]['valid']


def test_native_action_execution_adoption_and_replay_share_sampler():
    from radar_vla.execution import prepare_adoption, execution_plan_sampler, execution_payload, sample_baseline
    from radar_vla.response_replay import compose_execution
    from radar_vla.response_reward import ResponseRewardConfig
    model = RiskConditionedPlanner(config(reasoning_policy='always_long')).eval()
    radar, risk, text, ego, times = inputs(1)
    raw = model.plan_action(radar, risk, text, ego, times)[0]
    record = dict(ego=dict(velocity=[4., 0.], yaw_rate=.1, box_size=[1., .5]))
    prepared = prepare_adoption(record, raw, .2, adoption_transition_s=.5)
    adapted, meta = execution_plan_sampler(prepared)
    original, _ = execution_plan_sampler(raw)
    for d in range(3):
        np.testing.assert_allclose(adapted([.5, 1., 3.], d), original([.5, 1., 3.], d), atol=1e-6)
    payload = execution_payload(record, raw, .2, adoption_transition_s=.5)
    assert payload['valid']
    json.dumps(payload, allow_nan=False)
    # A LONG action later used as a baseline retains its 3 s native grid and
    # gets the same bounded acceleration-release/stop, not spline conversion.
    assert np.isfinite(sample_baseline(record, [3., 4., 8.], raw)).all()
    composite, _ = compose_execution(record, [dict(plan=raw, ready_s=.2)], ResponseRewardConfig(adoption_transition_s=.5))
    assert composite['valid']


def test_projection_bounds_vector_norm_even_for_large_network_outputs():
    model = RiskConditionedPlanner(config(action_max_acceleration_mps2=2., action_max_jerk_mps3=1.))
    with torch.no_grad():
        model.action_head.residual[-1].weight.zero_()
        model.action_head.residual[-1].bias.copy_(torch.tensor([100., -200., -100., 300.] * 3))
    radar, risk, text, ego, times = inputs()
    out = model(radar, risk, text, observed_ego=ego, future_times_s=times)
    assert out['accelerations'].norm(dim=-1).max() <= 2. + 1e-6
    assert out['jerks'].norm(dim=-1).max() <= 1. + 1e-6
    # Exact segment integral, not position clipping or a post-hoc spline fit.
    a0, a1 = out['accelerations'][:, 0], out['accelerations'][:, 1]
    expected = ego['velocity']*.5 + a0*.5**2/2 + (a1-a0)/.5*.5**3/6
    torch.testing.assert_close(out['positions'][:, 0], expected)


def test_missing_acceleration_explicit_ctrv_fallback_invalid_velocity_refused():
    model = RiskConditionedPlanner(config()).eval()
    radar, risk, text, ego, times = inputs(1)
    ego['acceleration'][:] = float('nan')
    ego['acceleration_valid'] = torch.tensor([False])
    plan = model.plan_action(radar, risk, text, ego, times)[0]
    assert plan['acceleration_source'] == 'ctrv_fallback'
    np.testing.assert_allclose(plan['acceleration_knots_mps2'][0], [0., .4], atol=1e-6)
    ego['velocity_valid'] = torch.tensor([False])
    with pytest.raises(ValueError, match='valid observed velocity'):
        model.plan_action(radar, risk, text, ego, times)


def test_parallel_configuration_and_checkpoint_limits_are_strict():
    with pytest.raises(ValueError, match='six 0.5'):
        config(horizon_steps=3)
    with pytest.raises(ValueError, match='description_loss'):
        config(action_description_loss_weight=float('nan'))
    model = RiskConditionedPlanner(config())
    checkpoint = model.state_dict()
    checkpoint.pop('_action_representation_contract')
    with pytest.raises(ValueError, match='representation'):
        model.load_state_dict(checkpoint, strict=False)
    with pytest.raises(ValueError, match='representation'):
        model.load_state_dict(RiskConditionedPlanner(config(action_max_jerk_mps3=5.)).state_dict())


def test_short_mode_and_unknown_targets_preserve_mask_semantics():
    from radar_vla.action_head import parallel_action_loss
    model = RiskConditionedPlanner(config())
    radar, risk, text, ego, times = inputs()
    out = model(radar, risk, text, observed_ego=ego, future_times_s=times)
    batch = dict(ego_future=torch.zeros(2, 6, 2), ego_future_mask=torch.ones(2, 6, dtype=torch.bool),
                 action_mode_target=torch.tensor([0, -1]), action_mode_mask=torch.tensor([True, False]))
    _, metrics = parallel_action_loss(out, batch, model.config)
    assert metrics['trajectory_count'] == 8 and metrics['mode_count'] == 1
    batch['ego_future'][0, 3:] = 1e8
    loss1, _ = parallel_action_loss(out, batch, model.config)
    batch['ego_future'][0, 3:] = -1e8
    loss2, _ = parallel_action_loss(out, batch, model.config)
    torch.testing.assert_close(loss1, loss2)


def test_native_execution_limits_are_part_of_action_protocol_not_sampler_defaults():
    from radar_vla.execution import execution_plan_sampler
    model = RiskConditionedPlanner(config(reasoning_policy='always_short',
        action_max_acceleration_mps2=2., action_max_jerk_mps3=1.)).eval()
    radar, risk, text, ego, times = inputs(1)
    plan = model.plan_action(radar, risk, text, ego, times)[0]
    sample, meta = execution_plan_sampler(plan)
    query = np.linspace(0, meta['execution_horizon_s'], 1001)
    assert np.linalg.norm(sample(query, 2), axis=-1).max() <= 2.+1e-6
    assert np.linalg.norm(sample(query, 3), axis=-1).max() <= 1.+1e-6


def test_action_query_positions_do_not_depend_on_other_samples_text_padding():
    model = RiskConditionedPlanner(config()).eval()
    radar, risk, text, ego, times = inputs()
    text = ['Go', 'Turn left carefully']
    batched = model(radar, risk, text, observed_ego=ego, future_times_s=times)
    single = model(radar[:1], risk[:1], text[:1],
        observed_ego={key:value[:1] for key,value in ego.items()}, future_times_s=times)
    torch.testing.assert_close(batched['positions'][:1], single['positions'], atol=1e-5, rtol=1e-5)


def test_saturated_projection_respects_execution_tolerance_after_json_precision():
    from radar_vla.action_head import ParallelAccelerationHead
    torch.manual_seed(132)
    head = ParallelAccelerationHead(16, config())
    with torch.no_grad():
        head.residual[-1].weight.mul_(1000.)
    b = 100
    state = (torch.zeros(b,2), torch.zeros(b,2), torch.zeros(b),
             torch.ones(b,dtype=torch.bool), torch.ones(b,dtype=torch.bool))
    out = head(torch.randn(b,16), state, torch.arange(1,7).float().expand(b,-1)*.5)
    acceleration = out['accelerations'].detach().double()
    assert acceleration.norm(dim=-1).max() <= 8.+1e-8
    assert (acceleration.diff(dim=1)/.5).norm(dim=-1).max() <= 20.+1e-8


def test_evaluation_reports_native_acceleration_representation():
    from radar_vla.evaluation import evaluate_plan
    model = RiskConditionedPlanner(config(reasoning_policy='always_long')).eval()
    radar, risk, text, ego, times = inputs(1)
    plan = model.plan_action(radar, risk, text, ego, times)[0]
    record = dict(future_times_s=times.tolist(), agents=[],
        ego=dict(velocity=[4., 0.], yaw_rate=.1, box_size=[1., .5],
                 future_xy=plan['ego_trajectory'], future_valid=[True]*6))
    metrics = evaluate_plan(record, plan)
    assert metrics['valid_plan']
    assert metrics['trajectory_sampling'] == (
        'native piecewise-linear acceleration; exact integration and analytic derivatives; sampled collision checks')
