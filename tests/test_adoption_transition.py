"""Opt-in adoption must be continuous, bounded and shared by all consumers."""
import copy
import json

import numpy as np
import pytest


def plan(speed=4., horizon=1., mode='short'):
    return dict(valid=True, ego_complete=True, mode=mode,
                control_times_s=[0., horizon],
                ego_control_points=[[0., 0.], [speed*horizon, 0.]])


def record(yaw=.1):
    return dict(ego=dict(velocity=[4., 0.], yaw_rate=yaw, box_size=[1., .5]))


def scene(yaw=.1):
    row = record(yaw)
    times = np.arange(.5, 6.01, .5).tolist()
    row.update(future_times_s=times, agent_supervision_available=True,
        tracking_coverage_valid=[True]*len(times),
        agents=[dict(id='target', position=[50., 0.], velocity=[0., 0.],
                     heading=0., size=[1., .5], future_xy=[[50., 0.]]*len(times),
                     future_yaw=[0.]*len(times), future_valid=[True]*len(times))],
        map=dict(lane_centerlines=[dict(id='lane', points=[[-100., 0.], [100., 0.]],
                     left_boundary_id='left', right_boundary_id='right')],
                 lane_boundaries=[dict(id='left', points=[[-100., 50.], [100., 50.]]),
                                  dict(id='right', points=[[-100., -50.], [100., -50.]])]))
    return row


def test_turning_adoption_is_executable_and_serializes_raw_and_adapted_diagnostics():
    from radar_vla.execution import execution_payload, execution_plan_sampler
    raw = plan()
    old = execution_payload(record(), raw, .2)
    assert not old['valid'] and old['adoption_acceleration_jump_mps2'] == pytest.approx(.4)
    adopted = execution_payload(record(), raw, .2, adoption_transition_s=.5)
    assert adopted['valid']
    assert adopted['adoption_acceleration_jump_mps2'] < 1e-6
    assert adopted['initial_velocity_error_mps'] < 1e-6
    assert adopted['adoption_transition']['raw_acceleration_jump_mps2'] == pytest.approx(.4)
    assert adopted['adoption_transition']['duration_s'] == .5
    _, original = execution_plan_sampler(raw)
    assert adopted['native_horizon_s'] == original['native_horizon_s']
    assert adopted['execution_horizon_s'] == original['execution_horizon_s']
    assert adopted['fallback'] == 'jerk_limited_stop'
    json.dumps(adopted, allow_nan=False)


def test_prepared_plan_matches_pva_at_adoption_and_join_then_preserves_suffix():
    from radar_vla.execution import prepare_adoption, sample_executed_plan, sample_baseline
    raw = plan()
    untouched = copy.deepcopy(raw)
    prepared = prepare_adoption(record(), raw, .2, adoption_transition_s=.5)
    for derivative in range(3):
        np.testing.assert_allclose(sample_executed_plan(record(), prepared, .2, [.2], derivative=derivative),
                                   sample_baseline(record(), [.2], derivative=derivative), atol=1e-8)
        np.testing.assert_allclose(sample_executed_plan(record(), prepared, .2, [.7, .9, 1.2], derivative=derivative),
                                   sample_executed_plan(record(), raw, .2, [.7, .9, 1.2], derivative=derivative), atol=1e-8)
    assert raw == untouched


def test_nonzero_acceleration_replan_uses_previous_execution_state():
    from radar_vla.execution import execution_payload, sample_baseline
    previous = dict(valid=True, mode='long', control_times_s=[0., 1., 2., 3.],
                    ego_control_points=[[0., 0.], [3.9, 0.], [7.6, 0.], [11.1, 0.]])
    speed = sample_baseline(record(0.), [1.], previous, derivative=1)[0, 0]
    raw = plan(speed)
    adopted = execution_payload(record(0.), raw, 1., baseline=previous, adoption_transition_s=.5)
    assert adopted['valid']
    assert adopted['adoption_transition']['raw_acceleration_jump_mps2'] > .1
    assert adopted['adoption_acceleration_jump_mps2'] < 1e-7


def test_zero_transition_keeps_legacy_behavior_identical():
    from radar_vla.execution import execution_payload
    assert execution_payload(record(), plan(), .1) == execution_payload(
        record(), plan(), .1, adoption_transition_s=0.)


@pytest.mark.parametrize('yaw,duration', [(.9, 1.), (.1, 1.01)])
def test_infeasible_or_overlong_bridge_is_rejected_without_extending_horizon(yaw, duration):
    from radar_vla.execution import execution_payload
    raw = plan()
    result = execution_payload(record(yaw), raw, .1, adoption_transition_s=duration)
    assert not result['valid'] and result['reason'] == 'invalid_adoption_transition'
    assert raw['control_times_s'][-1] == 1.
    assert result['adoption_transition']['status'] == 'rejected'


def test_bridge_extrema_catch_peak_between_regular_grid_samples():
    from radar_vla.execution import execution_payload
    row = dict(ego=dict(velocity=[0., 0.], yaw_rate=0.))
    # The bridge has acceleration maximum 3.94023 at t~.24274; .05s grid
    # maximum is only 3.9375. Reject using polynomial extrema, not that grid.
    result = execution_payload(row, plan(1.), .1, adoption_transition_s=1.,
                               max_acceleration_mps2=3.939, max_jerk_mps3=40.)
    assert not result['valid'] and result['reason'] == 'invalid_adoption_transition'
    assert result['adoption_transition']['max_acceleration_mps2'] > 3.939


def test_clock_uses_transition_but_never_future_annotations():
    from radar_vla.response_clock import response_clock
    from radar_vla.response_reward import ResponseRewardConfig
    cfg = ResponseRewardConfig(adoption_transition_s=.5)
    actual = response_clock(record(), plan(), .2, config=cfg)
    annotated = response_clock(scene(), plan(), .2, config=cfg)
    assert actual == annotated
    assert actual['response_clock_status'] != 'invalid_plan'
    assert actual['adoption_transition']['duration_s'] == .5


def test_reward_uses_same_prepared_execution_as_export_and_clock():
    from radar_vla.execution import execution_payload
    from radar_vla.response_reward import ResponseRewardConfig, response_reward
    from radar_vla.response_clock import response_clock
    cfg = ResponseRewardConfig(adoption_transition_s=.5)
    row = scene()
    output = execution_payload(row, plan(), .2, adoption_transition_s=.5)
    reward = response_reward(row, plan(), 'target', .2, config=cfg)
    clock = response_clock(row, plan(), .2, config=cfg)
    assert reward['reason'] != 'invalid_dynamics'
    assert reward['adoption_acceleration_jump_mps2'] == pytest.approx(output['adoption_acceleration_jump_mps2'])
    assert reward['adoption_transition'] == output['adoption_transition']
    assert reward['reaction_s'] == clock['reaction_s']


def test_composed_adoptions_are_continuous_and_not_bridged_twice():
    from radar_vla.execution import prepare_adoption, sample_executed_plan
    from radar_vla.response_replay import compose_execution
    from radar_vla.response_reward import ResponseRewardConfig
    cfg = ResponseRewardConfig(adoption_transition_s=.5)
    row = record()
    composite, first = compose_execution(row, [dict(plan=plan(), ready_s=.1),
        dict(plan=plan(), ready_s=.4)], cfg)
    assert composite['execution_failure'] is None
    assert max(composite['adoption_acceleration_jumps_mps2']) < 1e-6
    assert prepare_adoption(row, composite, first, adoption_transition_s=.5) is composite
    for when in [.1, .4]:
        for derivative in range(3):
            before, after = sample_executed_plan(row, composite, first, [when-1e-7, when+1e-7], derivative=derivative)
            np.testing.assert_allclose(before, after, atol=2e-5)


def test_rejected_later_transition_continues_previous_adopted_action():
    from radar_vla.execution import prepare_adoption, sample_executed_plan
    from radar_vla.response_replay import compose_execution
    from radar_vla.response_reward import ResponseRewardConfig
    cfg = ResponseRewardConfig(adoption_transition_s=.5)
    row = record()
    first_plan = prepare_adoption(row, plan(), .1, adoption_transition_s=.5)
    composite, first = compose_execution(row, [dict(plan=plan(), ready_s=.1),
        dict(plan=plan(20.), ready_s=.4)], cfg)
    assert composite['execution_failure'] == 'invalid_adoption_transition'
    times = [0., .1, .4, .8, 2., 5.]
    np.testing.assert_allclose(sample_executed_plan(row, composite, first, times),
                               sample_executed_plan(row, first_plan, .1, times), atol=1e-8)


def test_already_continuous_plan_is_not_rewritten_by_enabled_adapter():
    from radar_vla.execution import prepare_adoption, execution_plan_sampler
    raw = dict(valid=True, ego_complete=True, mode='short',
               control_times_s=[0., .1, .3, 1.],
               ego_control_points=[[0., 0.], [.4, .0001], [1.2, .0002], [4., 0.]])
    native, _ = execution_plan_sampler(raw)
    row = dict(ego=dict(velocity=native([0.], derivative=1)[0].tolist(), yaw_rate=0.))
    prepared = prepare_adoption(row, raw, .1, adoption_transition_s=.5)
    adapted, _ = execution_plan_sampler(prepared)
    for derivative in range(3):
        np.testing.assert_allclose(adapted([.05, .15, .25, .4], derivative=derivative),
                                   native([.05, .15, .25, .4], derivative=derivative), atol=1e-12)
    assert prepared['adoption_transition']['status'] == 'not_needed'


def test_prepared_long_plan_still_gets_terminal_stop_when_used_as_prior():
    from radar_vla.execution import prepare_adoption, sample_baseline, execution_plan_sampler
    raw = plan(horizon=3., mode='long')
    prepared = prepare_adoption(record(), raw, .1, adoption_transition_s=.5)
    native, metadata = execution_plan_sampler(prepared)
    assert metadata['fallback'] is None and metadata['execution_horizon_s'] == 3.
    with pytest.raises(ValueError, match='outside'):
        native([3.1])
    np.testing.assert_allclose(sample_baseline(record(), [6., 7.], prepared, derivative=1), 0., atol=1e-10)
    np.testing.assert_allclose(sample_baseline(record(), [6.], prepared), sample_baseline(record(), [7.], prepared))


def test_failed_transition_clock_has_no_fictitious_response_and_serializes():
    from radar_vla.response_clock import response_clock
    from radar_vla.response_reward import ResponseRewardConfig
    result = response_clock(record(.9), plan(), .2,
                            config=ResponseRewardConfig(adoption_transition_s=1.))
    assert result['response_clock_status'] == 'invalid_adoption_transition'
    assert not result['response_detected'] and result['reaction_s'] is None
    assert result['adoption_transition']['status'] == 'rejected'
    json.dumps(result, allow_nan=False)


def test_composed_export_and_reward_keep_the_same_adoption_history():
    from radar_vla.execution import execution_payload
    from radar_vla.response_replay import compose_execution
    from radar_vla.response_reward import ResponseRewardConfig, response_reward
    cfg = ResponseRewardConfig(adoption_transition_s=.5)
    row = scene()
    composite, first = compose_execution(row, [dict(plan=plan(), ready_s=.1),
        dict(plan=plan(), ready_s=.4)], cfg)
    exported = execution_payload(row, composite, first, adoption_transition_s=.5)
    score = response_reward(row, composite, 'target', first, config=cfg)
    assert exported['valid'] and score['reason'] != 'invalid_dynamics'
    assert exported['adoption_transition'] == score['adoption_transition']
    assert len(exported['adoption_transition']['decisions']) == 2
    json.dumps(exported, allow_nan=False)
    json.dumps(score, allow_nan=False)


@pytest.mark.parametrize('duration', [-1., float('nan'), float('inf')])
def test_invalid_transition_configuration_is_rejected(duration):
    from radar_vla.response_reward import ResponseRewardConfig
    with pytest.raises(ValueError, match='adoption_transition_s'):
        ResponseRewardConfig(adoption_transition_s=duration)
