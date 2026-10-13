"""Chronological replay must retain elapsed time and already adopted actions."""
import numpy as np
import pytest


def _plan(speed, horizon=3.):
    return dict(valid=True, ego_complete=True, mode='long',
                control_times_s=[0., 1., 2., horizon],
                ego_control_points=[[0., 0.], [speed, 0.], [2*speed, 0.], [horizon*speed, 0.]])


def test_replanning_preserves_position_and_checks_each_velocity_jump():
    from radar_vla.response_replay import compose_execution
    from radar_vla.execution import sample_executed_plan
    from radar_vla.response_reward import ResponseRewardConfig
    record = dict(ego=dict(velocity=[4., 0.], yaw_rate=0.))
    plan, first_ready = compose_execution(record, [
        dict(plan=_plan(4.), ready_s=.2), dict(plan=_plan(2.), ready_s=.7)], ResponseRewardConfig())
    assert first_ready == .2
    assert plan['adoption_times_s'] == pytest.approx([0., .5])
    assert plan['adoption_velocity_errors_mps'] == pytest.approx([0., 2.])
    points = sample_executed_plan(record, plan, first_ready, [0., .2, .7, 1.7])
    np.testing.assert_allclose(points[:, 0], [0., .8, 2.8, 4.8])
    assert plan['execution_horizon_s'] == pytest.approx(3.5)


def test_missing_detector_event_stays_in_hazard_response_denominator():
    from radar_vla.response_rl import _aggregate
    score = dict(effective=True, hazard=True, reaction_s=.2, reward=1.8,
                 collision=0, reason='effective_response', generation_s=.05)
    rows = [dict(event_id='seen', ids=[1], score=score)]
    reports = [dict(event_id='seen', detected=True, eligible=True, hazard=True, reason='eligible'),
               dict(event_id='missed', detected=False, eligible=False, hazard=True, reason='missed_detection')]
    result = _aggregate(rows, reports)
    assert result['effective_response_rate'] == .5
    assert result['hazard_events'] == 2
    assert result['response_p50_population'] == 'detected_trajectory_changes'
    assert result['effective_response_p50_s'] == .2
    assert result['outcomes']['missed_detection'] == 1


def test_fifo_planner_queue_does_not_reset_at_later_frame(monkeypatch):
    from radar_vla.response_rl import _rollout, ResponseRLConfig
    import torch
    measured = []
    def generate(*args, **kwargs):
        measured.append(kwargs.get('future_times_s'))
        return dict(ids=[1], plan=_plan(4.), generation_s=.4)
    monkeypatch.setattr('radar_vla.response_rl.measure_plan', generate)
    monkeypatch.setattr('radar_vla.response_rl.response_reward',
        lambda record, plan, target, generation_s, **kw: dict(
            generation_s=generation_s, waiting_s=kw['waiting_s'], perception_s=kw['perception_s']))
    record = dict(timestamp_s=10., future_times_s=[.5, 1., 2., 3.],
                  ego=dict(velocity=[4., 0.], yaw_rate=0.))
    frames = [dict(record={**record, 'timestamp_s':10.+t}, ready=10.+t+.1,
                   radar=torch.zeros(1, 2, 8), risk=torch.zeros(1, 5), instructions=['go'],
                   future_times_s=[.5, 1., 2., 3.]) for t in [0., .1]]
    context = dict(record=record, event=dict(event_id='e', target_agent_id='x'), frames=frames,
                   report=dict(detected_timestamp_s=10.1), perception_s=.1)
    result = _rollout(None, context, ResponseRLConfig(), 'cpu', greedy=True)
    assert [d['ready_s'] for d in result['decisions']] == pytest.approx([.5, .9])
    assert result['decisions'][1]['queue_s'] == pytest.approx(.3)
    assert result['score']['perception_s'] == .1
    assert measured == [[.5, 1., 2., 3.]] * 2


def test_failed_episode_penalizes_slower_later_generation(monkeypatch):
    from radar_vla.response_rl import _rollout, ResponseRLConfig
    import torch
    record = dict(timestamp_s=0., future_times_s=[.5 * i for i in range(1, 13)],
                  ego=dict(velocity=[4., 0.], yaw_rate=0., box_size=[1., .5]),
                  tracking_coverage_valid=[True] * 12, agent_supervision_available=True,
                  agents=[dict(id='x', position=[10., 0.], velocity=[0., 0.], heading=0.,
                               size=[1., .5], future_xy=[[10., 0.]]*12,
                               future_valid=[True]*12, future_yaw=[0.]*12)])
    frames = [dict(record=record, ready=t, radar=torch.zeros(1, 2, 8),
                   risk=torch.zeros(1, 5), instructions=['go'], future_times_s=[.5, 1., 2., 3.])
              for t in [.05, .15]]
    context = dict(record=record, event=dict(event_id='e', target_agent_id='x'), frames=frames,
                   report=dict(detected_timestamp_s=.05), perception_s=.05)
    def score(later):
        durations = iter([.05, later])
        monkeypatch.setattr('radar_vla.response_rl.measure_plan', lambda *a, **kw:
            dict(ids=[1], plan=_plan(4.), generation_s=next(durations)))
        return _rollout(None, context, ResponseRLConfig(), 'cpu', greedy=True)['score']
    fast, slow = score(.05), score(.45)
    assert fast['reason'] == slow['reason'] == 'collision'
    assert slow['reward'] < fast['reward']
    assert slow['reward'] - fast['reward'] == pytest.approx(-.4)


def test_rejected_replan_checks_entire_added_stopping_continuation():
    from radar_vla.response_replay import compose_execution
    from radar_vla.response_reward import response_reward, ResponseRewardConfig
    record = dict(future_times_s=[.5*i for i in range(1, 13)],
        ego=dict(velocity=[4., 0.], yaw_rate=0., box_size=[1., .5]),
        agent_supervision_available=True, tracking_coverage_valid=[True]*12,
        agents=[dict(id='x', position=[15., 0.], velocity=[0., 0.], heading=0., size=[1., .5],
                     future_xy=[[15., 0.]]*12, future_yaw=[0.]*12, future_valid=[True]*12)])
    composite, first = compose_execution(record, [dict(plan=_plan(4.), ready_s=.05),
        dict(plan={'valid': False}, ready_s=.4)], ResponseRewardConfig())
    score = response_reward(record, composite, 'x', first)
    assert score['collision'] == 1
    assert score['reason'] == 'collision'
    assert score['evaluated_horizon_s'] > 4.1


def _annotated_record():
    times = [.5, 1., 1.5, 2., 2.5, 3., 3.5, 4., 4.5, 5.]
    return dict(future_times_s=times,
        ego=dict(velocity=[4., 0.], yaw_rate=0., box_size=[1., .5]),
        agent_supervision_available=True, tracking_coverage_valid=[True] * len(times),
        agents=[dict(id='target', position=[10., 0.], velocity=[0., 0.],
            heading=0., size=[1., .5], future_xy=[[10., 0.]] * len(times),
            future_valid=[True] * len(times), future_yaw=[0.] * len(times))])


def test_rejected_later_plan_keeps_previously_adopted_braking_motion():
    from radar_vla.response_replay import compose_execution
    from radar_vla.execution import sample_executed_plan
    from radar_vla.response_reward import ResponseRewardConfig
    record = _annotated_record()
    brake = _plan(4.)
    brake['ego_control_points'] = [[0., 0.], [3.6, 0.], [5.6, 0.], [6., 0.]]
    composed, first = compose_execution(record, [dict(plan=brake, ready_s=.05),
        dict(plan={'valid': False}, ready_s=.4)], ResponseRewardConfig())
    times = [.05, .4, 1.05, 2.05]
    expected = sample_executed_plan(record, brake, .05, times)
    actual = sample_executed_plan(record, composed, first, times)
    np.testing.assert_allclose(actual, expected)
    assert composed['execution_failure'] == 'invalid_plan'
    assert composed['adoption_times_s'] == [0.]


def test_rejected_later_plan_cannot_erase_collision_of_adopted_motion():
    from radar_vla.response_replay import compose_execution
    from radar_vla.response_reward import ResponseRewardConfig, response_reward
    record = _annotated_record()
    composed, first = compose_execution(record, [dict(plan=_plan(4.), ready_s=.05),
        dict(plan={'valid': False}, ready_s=.4)], ResponseRewardConfig())
    score = response_reward(record, composed, 'target', first)
    assert score['collision'] == 1
    assert score['reason'] == 'collision'
    assert not score['effective']


@pytest.mark.parametrize('decisions', [[], [dict(plan={'valid': False}, ready_s=float('nan'))],
    [dict(plan={'valid': False}, ready_s=-.1)],
    [dict(plan={'valid': False}, ready_s=.4), dict(plan={'valid': False}, ready_s=.3)]])
def test_replay_rejects_invalid_timeline_even_when_outputs_are_rejected(decisions):
    from radar_vla.response_replay import compose_execution
    from radar_vla.response_reward import ResponseRewardConfig
    with pytest.raises(ValueError, match='ready|chronological|decision'):
        compose_execution(_annotated_record(), decisions, ResponseRewardConfig())


def test_composed_prior_state_retains_diagonal_heading_after_terminal_stop():
    from radar_vla.response_replay import compose_execution
    from radar_vla.execution import sample_executed_plan
    from radar_vla.response_reward import ResponseRewardConfig
    record = dict(ego=dict(velocity=[1., 0.], yaw_rate=0.))
    diagonal = dict(valid=True, ego_complete=True, mode='short', control_times_s=[0., 1.],
                    ego_control_points=[[0., 0.], [1., 1.]])
    composed, first = compose_execution(record, [dict(plan=diagonal, ready_s=.1),
        dict(plan=_plan(1.), ready_s=5.)], ResponseRewardConfig())
    positions = sample_executed_plan(record, composed, first, [5., 5.1])
    np.testing.assert_allclose(positions[1]-positions[0], [.1/np.sqrt(2)]*2, atol=1e-8)


@pytest.mark.parametrize('detect_last', [True, False])
def test_detection_after_passing_obstacle_retains_trigger_collision_and_misses(tmp_path, monkeypatch, detect_last):
    """Detector outcomes are controlled; geometry, contexts and reward are real."""
    import json
    from radar_vla.synthetic import write_synthetic_dataset
    from radar_vla.pipeline import resolve_config
    from radar_vla.model import ModelConfig, RadarVLAGrounder
    from radar_vla.response_rl import _contexts
    from radar_vla.response_reward import ResponseRewardConfig, response_reward
    manifest = write_synthetic_dataset(tmp_path / 'data', scenes_per_split=1,
        frames_per_scene=2, range_bins=8, azimuth_bins=8)
    records = [json.loads(line) for line in manifest.read_text().splitlines()]
    for row in records:
        now = 0. if row['timestamp_s'] == 0. else 3.
        row['timestamp_s'] = now
        row['ego'].update(velocity=[4., 0.], box_size=[1., .5],
                         future_xy=[[4.*t, 0.] for t in row['future_times_s']])
        row['agents'][0].update(position=[10.-4.*now, 0.], velocity=[0., 0.],
            heading=0., size=[1., .5], future_xy=[[10.-4.*now, 0.]] * 6,
            future_yaw=[0.] * 6)
    manifest.write_text(''.join(json.dumps(row) + '\n' for row in records))
    cfg = resolve_config(dict(model=dict(hidden_dim=8, num_heads=2, num_queries=2),
                              language_model=dict(backend='tiny')))
    grounder = RadarVLAGrounder(ModelConfig(**cfg['model'])).eval()
    reports = iter([dict(detections=[], detection_truncated=False),
                    dict(detections=[dict(position_m=[-2., 0.])] if detect_last else [],
                         detection_truncated=False)])
    monkeypatch.setattr('radar_vla.response_rl.decode_detections', lambda *args: [next(reports)])
    clock = iter([0., .1, 1., 1.1])
    monkeypatch.setattr('radar_vla.response_rl.time.perf_counter', lambda: next(clock))
    ids = [row['sample_id'] for row in records if row['split'] == 'train']
    event = dict(event_id='late', target_agent_id='target_0', split='train',
        scene_id='train_scene_0000', sample_ids=ids, trigger_sample_id=ids[0],
        event_kind='hazard_start', hazard=True, response_window_complete=False)
    contexts, reports = _contexts(grounder, manifest, [event], 'train', cfg, 'cpu', ResponseRewardConfig())
    assert len(reports) == 1 and reports[0]['hazard'] is True
    if not detect_last:
        assert contexts == []
        assert not reports[0]['detected'] and reports[0]['reason'] == 'missed_detection'
        return
    context, report = contexts[0], reports[0]
    assert context['record']['sample_id'] == ids[0]
    assert report['sample_id'] == ids[1]
    assert report['matched_observation_timestamp_s'] == 3.
    assert report['matched_frame_perception_s'] == pytest.approx(.1)
    assert report['perception_s'] == pytest.approx(3.1)
    assert report['response_window_censored'] is True
    # At the detected frame the stationary obstacle is already behind ego.
    # The actual collision occurred during the previously undetected prefix.
    score = response_reward(context['record'], {'valid': False}, 'target_0', .05,
                            perception_s=context['perception_s'])
    assert score['collision'] == 1 and score['reason'] == 'collision'
