"""Behavioral regressions for delayed, causal response reward semantics."""
from functools import lru_cache

import numpy as np
import pytest

from radar_vla.response_reward import event_baseline, response_reward


def _scene(speed, target_x, target_speed=0., horizon=5.):
    times = np.arange(.5, horizon + .01, .5).tolist()
    return dict(future_times_s=times, agent_supervision_available=True,
                tracking_coverage_valid=[True] * len(times),
                ego=dict(velocity=[speed, 0.], acceleration=[0., 0.], yaw_rate=0., box_size=[1., .5]),
                agents=[dict(id='obstacle', category='vehicle', position=[target_x, 0.],
                    velocity=[target_speed, 0.], heading=0., size=[1., .5],
                    future_xy=[[target_x + target_speed * t, 0.] for t in times],
                    future_valid=[True] * len(times), future_yaw=[0.] * len(times))],
                map=dict(lane_centerlines=[dict(id='lane', points=[[-50., 0.], [100., 0.]],
                    left_boundary_id='left', right_boundary_id='right')],
                    lane_boundaries=[dict(id='left', points=[[-50., 10.], [100., 10.]]),
                                     dict(id='right', points=[[-50., -10.], [100., -10.]])]))


@lru_cache(maxsize=1)
def _planner():
    from radar_vla.planner import PlannerConfig, RiskConditionedPlanner
    return RiskConditionedPlanner(PlannerConfig(hidden_dim=8, radar_dim=8,
        num_heads=2, num_layers=1)).eval()


def _legal_plan(controls, mode='long'):
    from radar_vla.planner import _layout
    planner = _planner()
    fields, *_ = _layout(mode, planner.config, 0)
    start = fields.index('<TRAJECTORY>') + 1
    values = np.asarray(controls)[1:].ravel().tolist()
    ids = []
    for index, field in enumerate(fields):
        if start <= index < start + len(values):
            token = planner.tokenizer.encode_scalar(values[index - start])
        elif field in ('position', 'width'):
            token = planner.tokenizer.encode_scalar(2. if field == 'width' else 0.)
        else:
            token = planner.tokenizer.token(field)
        assert token in planner._allowed_next(ids)
        ids.append(token)
    plan = planner.decode(ids)
    assert plan['valid'] and plan['physically_complete']
    return plan


def _braking_plan():
    return dict(valid=True, ego_complete=True, control_times_s=[0., 1., 2., 3.],
                ego_control_points=[[0., 0.], [3.6, 0.], [5.6, 0.], [6., 0.]],
                ego_trajectory=[[3.6, 0.], [5.6, 0.], [6., 0.]], ego_times_s=[1., 2., 3.])


def test_default_vocabulary_delayed_tail_collision_cannot_earn_success():
    record = _scene(4.4, 11.25)
    plan = _legal_plan([[0., 0.], [4., .5], [6.8, 2.], [10.5, 0.]])
    for delay in (.05, .45):
        score = response_reward(record, plan, 'obstacle', delay)
        assert score['collision'] == 1
        assert not score['effective'] and score['reward'] < 0


def test_delayed_tail_requires_observed_gt_through_actual_execution_end():
    score = response_reward(_scene(4., 10., horizon=3.), _braking_plan(), 'obstacle', .05)
    assert score['reason'] == 'unknown_coverage'
    assert not score['effective']
    assert score['safety_coverage'] < 1.


def test_default_vocabulary_harmful_early_brake_cannot_claim_lateral_onset():
    record = _scene(3.8, -11., 8.)
    plan = _legal_plan(list(zip([0., 3.3, 4.8, 5.6], [0., .1, 2., 4.])))
    score = response_reward(record, plan, 'obstacle', .05)
    assert not score['effective']
    assert score['reason'] == 'timeout'
    # The early brake is a real motion response, but not effective avoidance.
    assert score['reaction_s'] == pytest.approx(.25)
    assert score['effective_reaction_s'] >= 1.2
    assert score['reward'] < 0.


def test_invalid_output_reward_never_improves_by_waiting_for_collision():
    record = _scene(4., 10.)
    scores = [response_reward(record, {'valid': False}, 'obstacle', delay)
              for delay in (.05, .45, 2.4)]
    assert all(later['reward'] < earlier['reward'] for earlier, later in zip(scores, scores[1:]))
    assert scores[-1]['collision'] == 1


def test_known_prefix_collision_survives_missing_later_coverage():
    record = _scene(4., 10.)
    record['tracking_coverage_valid'][5:] = [False] * 5
    score = response_reward(record, {'valid': False}, 'obstacle', 2.4)
    assert score['reason'] == 'collision'
    assert score['collision'] == 1


def test_known_executed_collision_survives_missing_later_coverage():
    record = _scene(4.4, 11.25)
    record['tracking_coverage_valid'][6:] = [False] * 4
    plan = _legal_plan([[0., 0.], [4., .5], [6.8, 2.], [10.5, 0.]])
    score = response_reward(record, plan, 'obstacle', .45)
    assert score['reason'] == 'collision'
    assert score['collision'] == 1
    assert score['safety_coverage'] < 1.


def test_currently_executing_brake_is_baseline_not_new_response():
    record, plan = _scene(4., 10.), _braking_plan()
    baseline = event_baseline(record, 'obstacle', baseline_plan=plan)
    assert not baseline['hazard']
    score = response_reward(record, plan, 'obstacle', 0., baseline_plan=plan)
    assert score['reason'] == 'safe_control'
    assert score['action_onset_s'] is None
    assert not score['effective']


def test_safe_braking_still_rewards_faster_effective_response():
    record, plan = _scene(4., 10.), _braking_plan()
    fast = response_reward(record, plan, 'obstacle', .05)
    slow = response_reward(record, plan, 'obstacle', .25)
    assert fast['effective'] and slow['effective']
    assert slow['reaction_s'] - fast['reaction_s'] == pytest.approx(.2)
    assert fast['reward'] > slow['reward']


def test_late_effective_action_failures_get_worse_with_more_generation_delay():
    record = _scene(3.8, -11., 8.)
    plan = _legal_plan(list(zip([0., 4., 8., 12.], [0., .1, 2., 4.])))
    scores = [response_reward(record, plan, 'obstacle', delay) for delay in (.05, .15, .25)]
    assert all(score['reason'] == 'timeout' for score in scores)
    assert all(b['reward'] < a['reward'] for a, b in zip(scores, scores[1:]))


def test_no_response_failures_get_worse_with_more_generation_delay():
    # Constant-velocity baseline passes the obstacle with a .8m clearance:
    # hazardous under 1m, but no sampled contact and no clearance violation.
    record = _scene(4., 10.)
    record['agents'][0]['position'][1] = 1.3
    record['agents'][0]['future_xy'] = [[10., 1.3]] * len(record['future_times_s'])
    plan = _braking_plan()
    plan['ego_control_points'] = [[4. * t, 0.] for t in plan['control_times_s']]
    scores = [response_reward(record, plan, 'obstacle', delay) for delay in (.05, .15, .25)]
    assert all(score['reason'] == 'no_response' for score in scores)
    assert all(b['reward'] < a['reward'] for a, b in zip(scores, scores[1:]))


def test_short_braking_has_full_safe_stopping_continuation_without_lowering_horizon():
    record = _scene(2., 5.)
    # This has initial speed 2m/s, peak deceleration 1.6m/s2, jerk 4.8m/s3.
    plan = dict(valid=True, ego_complete=True, mode='short',
                control_times_s=[0., 1/3, 2/3, 1.],
                ego_control_points=[[0., 0.], [.633333333333, 0.], [1.133333333333, 0.], [1.5, 0.]])
    score = response_reward(record, plan, 'obstacle', .05)
    assert score['effective']
    assert score['evaluated_horizon_s'] == pytest.approx(3.05)
    assert score['execution']['fallback'] == 'jerk_limited_stop'
    assert score['safety_coverage'] == 1.


@pytest.mark.parametrize('replay', [False, True])
def test_default_vocabulary_short_success_preserves_three_second_safety_horizon(replay):
    from radar_vla.execution import execution_plan_sampler
    from radar_vla.response_reward import ResponseRewardConfig
    cfg = ResponseRewardConfig()
    plan = _legal_plan([[0., 0.], [.633333333333, 0.], [1.133333333333, 0.], [1.5, 0.]], mode='short')
    assert _planner().config.bins == 64
    assert plan['control_times_s'][-1] == 1.
    record = _scene(2., 5.)
    candidate, ready = plan, .05
    if replay:
        from radar_vla.response_replay import compose_execution
        candidate, ready = compose_execution(record, [dict(plan=plan, ready_s=.05)], cfg)
    score = response_reward(record, candidate, 'obstacle', ready, config=cfg)
    assert score['effective'] and score['reward'] > 0
    assert score['evaluated_horizon_s'] == pytest.approx(3.05)
    assert score['safety_coverage'] == 1.
    assert score['collision'] == 0
    assert score['max_acceleration_mps2'] <= cfg.max_acceleration_mps2
    assert score['max_jerk_mps3'] <= cfg.max_jerk_mps3
    sampler, execution = execution_plan_sampler(plan)
    assert execution['fallback'] == 'jerk_limited_stop'
    assert execution['execution_horizon_s'] > 1.
    np.testing.assert_allclose(sampler([3.], derivative=1), [[0., 0.]])


def test_all_rejected_replans_preserve_nominal_motion_with_lateral_velocity():
    from radar_vla.execution import sample_executed_plan
    from radar_vla.response_replay import compose_execution
    from radar_vla.response_reward import ResponseRewardConfig
    record = _scene(4., 50.)
    record['ego']['velocity'] = [4., 1.]
    composite, first = compose_execution(record, [
        dict(plan={'valid': False}, ready_s=.2),
        dict(plan={'valid': False}, ready_s=.4)], ResponseRewardConfig())
    actual = sample_executed_plan(record, composite, first, [0., .2, 1.])
    np.testing.assert_allclose(actual, [[0., 0.], [.8, .2], [4., 1.]], atol=1e-10)


@pytest.mark.parametrize('velocity_jump,acceleration_jump', [(0.8, 0.), (0., .5)])
def test_later_plan_adoption_cannot_hide_dynamics_discontinuity(velocity_jump, acceleration_jump):
    from radar_vla.evaluation import spline_plan_sampler
    native = spline_plan_sampler(_braking_plan())

    def sample(times, derivative=0):
        times = np.asarray(times)
        after = np.maximum(times - 1., 0.)
        result = native(times, derivative=derivative)
        if derivative == 0:
            result[:, 0] += velocity_jump * after + .5 * acceleration_jump * after**2
        elif derivative == 1:
            result[:, 0] += velocity_jump * (times >= 1.) + acceleration_jump * after
        elif derivative == 2:
            result[:, 0] += acceleration_jump * (times >= 1.)
        return result

    plan = dict(valid=True, _sampler=sample, execution_horizon_s=3.,
                adoption_times_s=[1.], adoption_velocity_errors_mps=[velocity_jump],
                adoption_acceleration_jumps_mps2=[acceleration_jump])
    score = response_reward(_scene(4., 10.), plan, 'obstacle', .05)
    assert score['reason'] == 'invalid_dynamics'
    assert not score['effective']


def test_first_adoption_checks_acceleration_continuity_of_executing_plan():
    def previous(times, derivative=0):
        times = np.asarray(times)
        values = {0: 4 * times - .5 * times**2,
                  1: 4 - times, 2: np.full_like(times, -1.), 3: np.zeros_like(times)}
        return np.column_stack((values[derivative], np.zeros_like(times)))

    score = response_reward(_scene(4., 10.), _braking_plan(), 'obstacle', .1,
                            baseline_plan={'_sampler': previous})
    assert score['reason'] == 'invalid_dynamics'


def test_invalid_replan_cannot_claim_success_from_previous_safe_motion():
    from radar_vla.evaluation import spline_plan_sampler
    plan = dict(valid=True, _sampler=spline_plan_sampler(_braking_plan()),
                execution_horizon_s=3., execution_failure='invalid_plan')
    score = response_reward(_scene(4., 10.), plan, 'obstacle', .05)
    assert score['reason'] == 'invalid_plan'
    assert not score['effective']


def test_invalid_replan_still_reports_collision_on_continuing_motion():
    from radar_vla.evaluation import spline_plan_sampler
    previous = _braking_plan()
    previous['ego_control_points'] = [[4. * t, 0.] for t in previous['control_times_s']]
    plan = dict(valid=True, _sampler=spline_plan_sampler(previous),
                execution_horizon_s=3., execution_failure='invalid_plan')
    score = response_reward(_scene(4., 10.), plan, 'obstacle', .05)
    assert score['reason'] == 'collision'
    assert score['collision'] == 1


@pytest.mark.parametrize('composed', [False, True])
def test_dynamics_checks_native_knots_between_regular_samples(composed):
    from radar_vla.response_reward import ResponseRewardConfig
    # A 10m/s2 triangular acceleration peak at .01s lies between .05s samples.
    # The initial velocity is 4m/s and the jerk peaks at 2000m/s3, below the
    # deliberately separate jerk limit: acceleration alone must reject it.
    plan = dict(valid=True, ego_complete=True,
                control_times_s=[0., .005, .01, .015, .02, 3.],
                ego_control_points=[[0., 0.], [.02, 0.], [.04004166666666666, 0.],
                                    [.06025, 0.], [.0805, 0.], [12.1495, 0.]])
    if composed:
        from radar_vla.evaluation import spline_plan_sampler
        plan = dict(valid=True, _sampler=spline_plan_sampler(plan),
                    execution_horizon_s=3., dynamics_times_s=plan['control_times_s'])
    score = response_reward(_scene(4., 50.), plan, 'obstacle', .05,
                            config=ResponseRewardConfig(max_jerk_mps3=3000.))
    assert score['reason'] == 'invalid_dynamics'
    assert score['max_acceleration_mps2'] == pytest.approx(10.)
