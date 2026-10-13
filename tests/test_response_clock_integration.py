"""Response latency exists independently of evidence about eventual safety."""
import pytest


def scene(horizon=3.):
    times = [.5*i for i in range(1, int(2*horizon)+1)]
    return dict(future_times_s=times, agent_supervision_available=True,
        tracking_coverage_valid=[True]*len(times),
        ego=dict(velocity=[4., 0.], yaw_rate=0., box_size=[1., .5]),
        agents=[dict(id='obstacle', position=[10., 0.], velocity=[0., 0.], heading=0.,
            size=[1., .5], future_xy=[[10., 0.]]*len(times),
            future_valid=[True]*len(times), future_yaw=[0.]*len(times))],
        map=dict(lane_centerlines=[dict(id='lane', left_boundary_id='left', right_boundary_id='right')],
            lane_boundaries=[dict(id='left', points=[[-10., 10.], [100., 10.]]),
                             dict(id='right', points=[[-10., -10.], [100., -10.]])]))


def brake():
    return dict(valid=True, ego_complete=True, control_times_s=[0., 1., 2., 3.],
        ego_control_points=[[0., 0.], [3.6, 0.], [5.6, 0.], [6., 0.]],
        ego_trajectory=[[3.6, 0.], [5.6, 0.], [6., 0.]], ego_times_s=[1., 2., 3.])


def test_unknown_tail_keeps_response_clock_and_latency_learning_signal():
    from radar_vla.response_reward import response_reward
    fast = response_reward(scene(), brake(), 'obstacle', .05)
    slow = response_reward(scene(), brake(), 'obstacle', .25)
    complete = response_reward(scene(6.), brake(), 'obstacle', .05)
    assert fast['reason'] == slow['reason'] == 'unknown_coverage'
    assert fast['response_detected'] and not fast['effective']
    assert fast['reaction_s'] == complete['reaction_s']
    assert slow['reaction_s'] - fast['reaction_s'] == pytest.approx(.2)
    assert fast['reward'] == fast['response_time_reward']
    assert fast['reward'] > slow['reward'] > 0.


def test_observed_later_collision_does_not_erase_earlier_response_time():
    from radar_vla.response_reward import response_reward
    record = scene(6.)
    record['agents'][0].update(position=[5., 0.], future_xy=[[5., 0.]]*12)
    result = response_reward(record, brake(), 'obstacle', .05)
    assert result['collision'] == 1 and result['reward'] < 0.
    assert result['response_detected'] and result['reaction_s'] is not None
    assert result['safety_status'] == 'collision'


def test_unknown_tail_does_not_make_impossible_or_unnecessary_changes_good():
    from radar_vla.response_reward import response_reward
    invalid = brake()
    invalid['ego_control_points'] = [[0., 0.]]*4
    bad = response_reward(scene(), invalid, 'obstacle', .05)
    assert bad['reason'] == 'invalid_dynamics' and bad['reward'] < 0.
    control = scene()
    control['agents'][0].update(position=[10., 8.], future_xy=[[10., 8.]]*6)
    result = response_reward(control, brake(), 'obstacle', .05)
    assert result['response_detected'] and result['reward'] < 0.


def test_response_statistics_include_responses_with_unknown_safety():
    from radar_vla.response_rl import _aggregate, _validation_key
    from radar_vla.response_reward import response_reward
    report = dict(event_id='e', detected=True, eligible=True, hazard=True, reason='eligible')
    fast = dict(event_id='e', ids=[1], score=response_reward(scene(), brake(), 'obstacle', .05))
    slow = dict(event_id='e', ids=[1], score=response_reward(scene(), brake(), 'obstacle', .25))
    result = _aggregate([fast], [report])
    assert result['response_p50_s'] == fast['score']['reaction_s']
    assert result['response_p50_population'] == 'detected_trajectory_changes'
    assert result['effective_response_p50_s'] is None
    assert result['effective_response_rate'] == 0.
    assert _validation_key([fast], [report]) < _validation_key([slow], [report])


def test_acceleration_avoiding_a_rear_threat_is_a_response_and_can_be_effective():
    from radar_vla.response_reward import response_reward
    record = scene(6.)
    record['agents'][0].update(position=[-5., 0.], velocity=[6., 0.],
        future_xy=[[-5.+6*t, 0.] for t in record['future_times_s']])
    plan = brake()
    plan['ego_control_points'] = [[0., 0.], [4.4, 0.], [10.4, 0.], [18., 0.]]
    score = response_reward(record, plan, 'obstacle', .05)
    assert score['response_detected'] and score['collision'] == 0
    assert score['effective'] and score['reward'] > 0.


def test_control_change_with_unknown_tail_still_counts_as_unnecessary_response():
    from radar_vla.response_reward import response_reward
    from radar_vla.response_rl import _aggregate
    record = scene()
    record['agents'][0].update(position=[10., 8.], future_xy=[[10., 8.]]*6)
    score = response_reward(record, brake(), 'obstacle', .05)
    report = dict(event_id='e', detected=True, eligible=True, hazard=False, reason='eligible')
    result = _aggregate([dict(event_id='e', ids=[1], score=score)], [report])
    assert result['unnecessary_response_rate'] == 1.
