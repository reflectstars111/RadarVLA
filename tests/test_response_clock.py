"""The motion response clock needs ego motion, never future annotations."""
import numpy as np
import pytest


def _record():
    return {'ego': {'velocity': [4., 0.], 'yaw_rate': 0.}}


def _cubic_plan(direction=-1., horizon=3.):
    # x(t) = 4t + direction*t^3/6, so the longitudinal acceleration
    # threshold of .5m/s2 is first reached at t=.5s in either direction.
    def sample(times, derivative=0):
        t = np.asarray(times, dtype=float)
        if (t > horizon + 1e-8).any():
            raise ValueError('Native plan must not be extrapolated')
        x = {0: 4*t + direction*t**3/6, 1: 4+direction*t*t/2,
             2: direction*t, 3: np.full_like(t, direction)}[derivative]
        return np.column_stack((x, np.zeros_like(t)))
    return dict(valid=True, ego_complete=True, _sampler=sample,
                native_horizon_s=horizon, execution_horizon_s=horizon)


def test_response_clock_works_with_ego_only_and_counts_queued_generation():
    from radar_vla.response_clock import response_clock
    result = response_clock(_record(), _cubic_plan(), .2, waiting_s=.1, perception_s=.4)
    assert result['response_detected']
    assert result['response_clock_status'] == 'changed'
    assert result['action_onset_s'] == pytest.approx(.5)
    assert result['reaction_s'] == pytest.approx(.8)
    assert result['response_time_reward'] == pytest.approx(1.2)


def test_annotation_content_never_changes_motion_clock():
    from radar_vla.response_clock import response_clock
    empty = _record()
    annotated = {**empty, 'future_times_s': [.25], 'tracking_coverage_valid': [False],
                 'agents': [], 'risk_agents': [], 'map': None}
    assert response_clock(empty, _cubic_plan(), .1) == response_clock(annotated, _cubic_plan(), .1)


def test_same_executing_plan_is_not_a_new_response():
    from radar_vla.response_clock import response_clock
    plan = _cubic_plan()
    result = response_clock(_record(), plan, 0., baseline_plan=plan)
    assert not result['response_detected']
    assert result['response_clock_status'] == 'no_change'
    assert result['reaction_s'] is None and result['action_onset_s'] is None


def test_acceleration_avoidance_counts_as_motion_change():
    from radar_vla.response_clock import response_clock
    result = response_clock(_record(), _cubic_plan(direction=1.), .1)
    assert result['response_detected']
    assert result['reaction_s'] == pytest.approx(.6)


def test_native_spline_controls_measure_braking_without_annotations():
    from radar_vla.response_clock import response_clock
    plan = dict(valid=True, ego_complete=True, control_times_s=[0., 1., 2., 3.],
                ego_control_points=[[0., 0.], [3.6, 0.], [5.6, 0.], [6., 0.]])
    result = response_clock(_record(), plan, .2)
    # Natural-spline acceleration over the first second is -1.92*t;
    # the first .05s grid point above .5m/s2 is therefore .3s.
    assert result['response_detected']
    assert result['reaction_s'] == pytest.approx(.5)


def test_lateral_change_counts_without_longitudinal_acceleration_change():
    from radar_vla.response_clock import response_clock
    def sample(times, derivative=0):
        t = np.asarray(times)
        x = {0: 4*t, 1: np.full_like(t, 4.), 2: np.zeros_like(t), 3: np.zeros_like(t)}[derivative]
        y = {0: .25*t*t, 1: .5*t, 2: np.full_like(t, .5), 3: np.zeros_like(t)}[derivative]
        return np.column_stack((x, y))
    result = response_clock(_record(), dict(valid=True, _sampler=sample, execution_horizon_s=2.), .1)
    assert result['response_detected']
    assert result['action_onset_s'] == pytest.approx(1.1)


@pytest.mark.parametrize('knots', ['control_times_s', 'adoption_times_s', 'dynamics_times_s'])
def test_execution_knots_preserve_response_time_between_regular_samples(knots):
    from radar_vla.response_clock import response_clock
    def sample(times, derivative=0):
        t = np.asarray(times)
        elapsed = np.maximum(t-.025, 0.)
        x = {0: 4*t+.5*elapsed*elapsed, 1: 4+elapsed,
             2: (t >= .025-1e-10).astype(float), 3: np.zeros_like(t)}[derivative]
        return np.column_stack((x, np.zeros_like(t)))
    plan = dict(valid=True, _sampler=sample, execution_horizon_s=1., **{knots: [.025]})
    result = response_clock(_record(), plan, .1)
    assert result['action_onset_s'] == pytest.approx(.025)


def test_finite_plan_can_record_response_without_longer_execution_or_gt():
    from radar_vla.response_clock import response_clock
    result = response_clock(_record(), _cubic_plan(horizon=.7), .2)
    assert result['response_detected']
    assert result['action_onset_s'] == pytest.approx(.5)


def test_small_acceleration_and_lateral_jitter_do_not_count():
    from radar_vla.response_clock import response_clock
    def sample(times, derivative=0):
        t = np.asarray(times)
        x = {0: 4*t+.2*t*t, 1: 4+.4*t, 2: np.full_like(t, .4), 3: np.zeros_like(t)}[derivative]
        y = {0: .05*t, 1: np.full_like(t, .05), 2: np.zeros_like(t), 3: np.zeros_like(t)}[derivative]
        return np.column_stack((x, y))
    plan = dict(valid=True, _sampler=sample, execution_horizon_s=3.)
    result = response_clock(_record(), plan, .2)
    assert not result['response_detected']
    assert result['response_time_reward'] == pytest.approx(-6.2)


def test_short_transient_above_threshold_does_not_count():
    from radar_vla.response_clock import response_clock
    # A .05s acceleration pulse must not pass the .1s sustained-change filter.
    def sample(times, derivative=0):
        t = np.asarray(times)
        u, v = np.maximum(t-.5, 0.), np.maximum(t-.55, 0.)
        x = {0: 4*t + .5*(u*u-v*v), 1: 4+u-v,
             2: ((t >= .5) & (t < .55)).astype(float), 3: np.zeros_like(t)}[derivative]
        return np.column_stack((x, np.zeros_like(t)))
    plan = dict(valid=True, _sampler=sample, execution_horizon_s=1., dynamics_times_s=[.5, .55])
    assert not response_clock(_record(), plan, .1)['response_detected']


def test_short_plan_counts_its_explicit_stop_continuation():
    from radar_vla.response_clock import response_clock
    plan = dict(valid=True, ego_complete=True, mode='short',
                control_times_s=[0., 1.], ego_control_points=[[0., 0.], [4., 0.]])
    result = response_clock(_record(), plan, .1)
    assert result['response_detected']
    assert result['action_onset_s'] > 1.


@pytest.mark.parametrize('latency', [-.1, float('nan'), float('inf')])
def test_invalid_latency_cannot_produce_response_time(latency):
    from radar_vla.response_clock import response_clock
    with pytest.raises(ValueError):
        response_clock(_record(), _cubic_plan(), latency)


def test_invalid_output_has_no_motion_response():
    from radar_vla.response_clock import response_clock
    result = response_clock(_record(), {'valid': False}, .2)
    assert not result['response_detected']
    assert result['response_clock_status'] == 'invalid_plan'
    assert result['response_time_reward'] == pytest.approx(-6.2)
