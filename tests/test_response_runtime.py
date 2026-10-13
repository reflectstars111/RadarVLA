"""Deployment timing, execution stitching and explicit SHORT continuation."""
import numpy as np
import pytest
import torch


def _planner():
    from radar_vla.planner import PlannerConfig, RiskConditionedPlanner
    return RiskConditionedPlanner(PlannerConfig(hidden_dim=8, radar_dim=8, num_heads=2,
        num_layers=1, bins=8, max_agents=1, max_length=128)).eval()


def _straight_plan(speed=4., horizon=1., mode='short'):
    return dict(valid=True, ego_complete=True, mode=mode,
                control_times_s=[0., horizon], ego_control_points=[[0., 0.], [speed*horizon, 0.]])


def test_measured_replay_matches_shared_generation_without_probability_work(monkeypatch):
    import radar_vla.execution as execution
    from radar_vla.rl_policy import sample_plan
    planner = _planner()
    radar, risk = torch.zeros(1, 2, 8), torch.zeros(1, 5)
    sampled = sample_plan(planner, radar, risk, ['go'], max_new_tokens=100)
    def forbidden(*args, **kwargs):
        raise AssertionError('probability diagnostics entered deployment clock')
    monkeypatch.setattr(torch.distributions, 'Categorical', forbidden)
    rng = torch.get_rng_state().clone()
    replay = execution.measure_plan(planner, radar, risk, ['go'], ids=sampled['ids'],
                                    max_new_tokens=100, clock=iter([3., 3.25]).__next__)
    assert replay['ids'] == sampled['ids']
    assert replay['plan'] == planner.decode(sampled['ids'])
    assert replay['generation_s'] == .25
    assert torch.equal(rng, torch.get_rng_state())
    greedy = execution.measure_plan(planner, radar, risk, ['go'], max_new_tokens=100)
    assert greedy['ids'] == planner.generate(radar, risk, ['go'], max_new_tokens=100)[0]


def test_short_stopping_continuation_has_continuous_bounded_dynamics():
    from radar_vla.execution import execution_plan_sampler
    plan = _straight_plan()
    sample, metadata = execution_plan_sampler(plan, max_acceleration_mps2=3., max_jerk_mps3=6.)
    assert metadata['fallback'] == 'jerk_limited_stop'
    assert metadata['native_horizon_s'] == 1.
    end = metadata['execution_horizon_s']
    t = np.linspace(1., end, 1001)
    assert np.linalg.norm(sample(t, derivative=2), axis=1).max() <= 3. + 1e-8
    assert np.linalg.norm(sample(t, derivative=3), axis=1).max() <= 6. + 1e-8
    np.testing.assert_allclose(sample([1.]), [[4., 0.]])
    np.testing.assert_allclose(sample([1.], derivative=1), [[4., 0.]])
    np.testing.assert_allclose(sample([1., end], derivative=2), [[0., 0.], [0., 0.]], atol=1e-7)
    np.testing.assert_allclose(sample([end, end+2.], derivative=1), 0., atol=1e-7)
    np.testing.assert_allclose(sample([end]), sample([end+2.]))
    assert plan['control_times_s'] == [0., 1.]  # native teacher horizon is unchanged


def test_execution_continues_existing_plan_then_rebases_at_its_actual_pose():
    from radar_vla.execution import sample_executed_plan
    record = {'ego': {'velocity': [8., 0.], 'yaw_rate': 0.}}
    baseline = _straight_plan(2., horizon=3., mode='long')
    candidate = _straight_plan(2.)
    xy = sample_executed_plan(record, candidate, 1., [0., .5, 1., 1.5], baseline_plan=baseline)
    np.testing.assert_allclose(xy, [[0., 0.], [1., 0.], [2., 0.], [3., 0.]])


def test_execution_rebases_heading_without_teleporting_at_adoption():
    from radar_vla.execution import sample_executed_plan
    record = {'ego': {'velocity': [1., 0.], 'yaw_rate': np.pi/2}}
    candidate = _straight_plan(1.)
    xy = sample_executed_plan(record, candidate, 1., [1., 1.5])
    np.testing.assert_allclose(xy[0], [2/np.pi, 2/np.pi], atol=1e-7)
    np.testing.assert_allclose(xy[1]-xy[0], [0., .5], atol=1e-7)


def test_zero_delay_preserves_current_frame_plan_with_lateral_ego_velocity():
    from radar_vla.execution import sample_executed_plan
    record = {'ego': {'velocity': [4., 1.], 'yaw_rate': 0.}}
    plan = _straight_plan()
    plan['ego_control_points'][-1] = [4., 1.]
    np.testing.assert_allclose(sample_executed_plan(record, plan, 0., [0., .5, 1.]),
                               [[0., 0.], [2., .5], [4., 1.]])


def test_replanning_after_a_diagonal_stop_preserves_last_body_heading():
    from radar_vla.execution import sample_executed_plan, sample_baseline
    record = {'ego': {'velocity': [1., 0.], 'yaw_rate': 0.}}
    baseline = _straight_plan(speed=1.)
    baseline['ego_control_points'][-1] = [1., 1.]
    origin = sample_baseline(record, [5.], baseline)[0]
    xy = sample_executed_plan(record, _straight_plan(speed=1.), 5., [5.1], baseline)
    np.testing.assert_allclose(xy[0] - origin, [.1/np.sqrt(2), .1/np.sqrt(2)], atol=1e-8)


def test_replanning_accepts_legacy_waypoint_baseline():
    from radar_vla.execution import sample_executed_plan
    record = {'ego': {'velocity': [2., 0.], 'yaw_rate': 0.}}
    baseline = dict(valid=True, ego_trajectory=[[2., 0.], [4., 0.], [6., 0.]],
                    ego_times_s=[1., 2., 3.])
    xy = sample_executed_plan(record, _straight_plan(speed=2.), 1., [1., 1.5], baseline)
    np.testing.assert_allclose(xy, [[2., 0.], [3., 0.]])


def test_restarting_forward_after_reverse_stop_preserves_body_heading():
    from radar_vla.execution import sample_executed_plan, sample_baseline
    record = {'ego': {'velocity': [-1., 0.], 'yaw_rate': 0.}}
    baseline = _straight_plan(speed=-1.)
    origin = sample_baseline(record, [5.], baseline)[0]
    xy = sample_executed_plan(record, _straight_plan(speed=1.), 5., [5.1], baseline)
    np.testing.assert_allclose(xy[0] - origin, [.1, 0.], atol=1e-8)


def test_long_plan_does_not_silently_fabricate_missing_horizon():
    from radar_vla.execution import execution_plan_sampler
    sample, metadata = execution_plan_sampler(_straight_plan(horizon=3., mode='long'))
    assert metadata['fallback'] is None
    with pytest.raises(ValueError, match='outside'):
        sample([3.1])


def test_composed_episode_sampler_keeps_explicit_horizon_and_derivatives():
    from radar_vla.execution import execution_plan_sampler
    def path(times, derivative=0):
        t = np.asarray(times)
        return np.column_stack((2*t, np.zeros_like(t))) if derivative == 0 else np.column_stack((np.full_like(t, 2.), np.zeros_like(t)))
    sampler, metadata = execution_plan_sampler(dict(valid=True, _sampler=path, execution_horizon_s=4.))
    np.testing.assert_allclose(sampler([2.]), [[4., 0.]])
    np.testing.assert_allclose(sampler([2.], derivative=1), [[2., 0.]])
    assert metadata['execution_horizon_s'] == 4.
    assert metadata['fallback'] is None


def test_prediction_execution_payload_includes_waiting_and_explicit_stop():
    from radar_vla.execution import execution_payload, sample_executed_plan
    record = dict(ego=dict(velocity=[4., 0.]), future_times_s=[.5, 1., 2., 3.])
    plan = _straight_plan()
    payload = execution_payload(record, plan, .25)
    assert payload['valid']
    assert payload['adoption_s'] == .25
    assert payload['fallback'] == 'jerk_limited_stop'
    assert payload['ego_times_s'][-1] == pytest.approx(.25 + payload['execution_horizon_s'])
    np.testing.assert_allclose(payload['ego_trajectory'], sample_executed_plan(
        record, plan, .25, payload['ego_times_s']))
    assert len(payload['ego_trajectory']) > 3


def test_prediction_execution_payload_marks_infeasible_or_incomplete_outputs_invalid():
    from radar_vla.execution import execution_payload
    record = dict(ego=dict(velocity=[4., 0.]), future_times_s=[.5, 1., 2., 3.])
    assert not execution_payload(record, {'valid': False}, .25)['valid']
    result = execution_payload(record, _straight_plan(speed=20.), .25)
    assert not result['valid']
    assert result['reason'] == 'invalid_dynamics'


def test_execution_payload_rejects_acceleration_jump_at_replanning():
    from radar_vla.execution import execution_payload, sample_baseline
    record = dict(ego=dict(velocity=[4., 0.]))
    baseline = dict(valid=True, mode='long', control_times_s=[0., 1., 2., 3.],
                    ego_control_points=[[0., 0.], [3.6, 0.], [5.6, 0.], [6., 0.]])
    speed = float(sample_baseline(record, [1.], baseline, derivative=1)[0, 0])
    candidate = _straight_plan(speed)
    result = execution_payload(record, candidate, 1., baseline=baseline)
    assert not result['valid']
    assert result['adoption_acceleration_jump_mps2'] > 1.


def test_prediction_reports_detection_before_shared_per_record_plan_runtime(tmp_path, monkeypatch):
    import radar_vla.detection as detection
    from radar_vla.pipeline import prediction_rows
    from radar_vla.model import ModelConfig, RadarVLAGrounder
    from radar_vla.synthetic import write_synthetic_dataset
    from radar_vla.data import RadarDataset
    manifest = write_synthetic_dataset(tmp_path / 'data', scenes_per_split=1,
        frames_per_scene=2, range_bins=8, azimuth_bins=8)
    dataset = RadarDataset(manifest, 'test', supervision=False)
    grounder = RadarVLAGrounder(ModelConfig(hidden_dim=8, num_heads=2, num_queries=2)).eval()
    planner = _planner()
    planner.config.reasoning_policy = 'always_short'
    seen = []
    decode = detection.decode_detections
    generate = planner._generate_ids
    def record_detection(*args, **kwargs):
        output = decode(*args, **kwargs)
        seen.append('detection')
        return output
    def record_generation(*args, **kwargs):
        seen.append('generation')
        assert args[0].shape[0] == 1
        return generate(*args, **kwargs)
    monkeypatch.setattr(detection, 'decode_detections', record_detection)
    monkeypatch.setattr(planner, '_generate_ids', record_generation)
    state = {'protocol': {'precision': 'float32', 'rl_config': {'reward': {
        'max_acceleration_mps2': 3., 'max_jerk_mps3': 6., 'initial_velocity_tolerance_mps': .2}}}}
    rows = prediction_rows(grounder, planner, dataset, state,
                           batch_size=2, device='cpu', max_new_tokens=100)
    assert seen == ['detection', 'generation', 'generation']
    assert rows[1]['waiting_seconds'] >= rows[0]['generation_seconds']
    for row in rows:
        assert row['detection_seconds'] > 0.
        assert row['generation_seconds'] > 0.
        assert 'executed_plan' in row
        assert row['executed_plan']['adoption_s'] == pytest.approx(
            row['detection_seconds'] + row['waiting_seconds'] + row['generation_seconds'])
        from radar_vla.evaluation import spline_plan_sampler
        native = spline_plan_sampler(row['plan'])
        speed = float(np.linalg.norm(native([row['plan']['control_times_s'][-1]], derivative=1)))
        assert row['executed_plan']['stop_duration_s'] == pytest.approx(max(.5*speed, np.sqrt(speed), .05))
