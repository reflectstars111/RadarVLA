"""Response RL: latency must buy an effective, physically plausible action."""
import copy
import json

import pytest
import torch


def reaction_scene():
    return dict(future_times_s=[.5 * i for i in range(1, 13)],
                ego=dict(velocity=[4., 0.], yaw_rate=0., box_size=[1., .5]),
                map=dict(lane_centerlines=[dict(id='lane', left_boundary_id='left', right_boundary_id='right')],
                         lane_boundaries=[dict(id='left', points=[[-5., 2.], [30., 2.]]),
                                          dict(id='right', points=[[-5., -2.], [30., -2.]])]),
                tracking_coverage_valid=[True] * 12, agent_supervision_available=True,
                agents=[dict(id='obstacle', position=[10., 0.], velocity=[0., 0.],
                             heading=0., size=[1., .5], future_xy=[[10., 0.]] * 12,
                             future_valid=[True] * 12, future_yaw=[0.] * 12)])


def braking_plan():
    # Natural cubic: initial velocity 4 m/s, then smooth deceleration to rest.
    return dict(valid=True, ego_complete=True, control_times_s=[0., 1., 2., 3.],
                ego_control_points=[[0., 0.], [3.6, 0.], [5.6, 0.], [6., 0.]],
                ego_trajectory=[[3.6, 0.], [5.6, 0.], [6., 0.]], ego_times_s=[1., 2., 3.])


def test_effective_braking_rewards_less_measured_latency():
    from radar_vla.response_reward import ResponseRewardConfig, response_reward
    cfg = ResponseRewardConfig()
    fast = response_reward(reaction_scene(), braking_plan(), 'obstacle', .05, config=cfg)
    slow = response_reward(reaction_scene(), braking_plan(), 'obstacle', .25, config=cfg)
    assert fast['effective'] and slow['effective']
    assert slow['reaction_s'] - fast['reaction_s'] == pytest.approx(.2)
    assert fast['reward'] > slow['reward']


def test_generation_delay_cannot_hide_a_collision_before_action():
    from radar_vla.response_reward import response_reward
    result = response_reward(reaction_scene(), braking_plan(), 'obstacle', 2.4)
    assert not result['effective']
    assert result['collision'] == 1
    assert result['reason'] == 'collision'


def test_short_or_invalid_output_cannot_hide_collision_during_generation():
    from radar_vla.response_reward import response_reward
    short = braking_plan()
    short['control_times_s'] = [0., .3, .6, 1.]
    for plan in (short, {'valid': False}):
        result = response_reward(reaction_scene(), plan, 'obstacle', 2.4)
        assert result['collision'] == 1
        assert result['reason'] == 'collision'
        assert result['reward'] <= -6.


def test_avoidance_cannot_earn_reward_by_leaving_known_road():
    from radar_vla.response_reward import response_reward
    plan = braking_plan()
    plan['ego_control_points'] = [[0., 0.], [4., 1.], [8., 4.], [12., 5.]]
    result = response_reward(reaction_scene(), plan, 'obstacle', .05)
    assert not result['effective'] and result['reason'] == 'road_departure'
    record = reaction_scene()
    record.pop('map')
    assert response_reward(record, braking_plan(), 'obstacle', .05)['reason'] == 'unknown_road'


def test_missing_coverage_short_plan_and_teleport_never_succeed():
    from radar_vla.response_reward import response_reward
    record = reaction_scene()
    record['tracking_coverage_valid'][3] = False
    assert response_reward(record, braking_plan(), 'obstacle', .05)['reason'] == 'unknown_coverage'
    plan = braking_plan()
    plan['control_times_s'] = [0., .3, .6, 1.]
    assert response_reward(reaction_scene(), plan, 'obstacle', .05)['reason'] == 'short_horizon'
    plan = braking_plan()
    plan['ego_control_points'] = [[0., 0.]] * 4
    assert response_reward(reaction_scene(), plan, 'obstacle', .05)['reason'] == 'invalid_dynamics'


def test_no_response_is_penalized_and_control_event_does_not_reward_braking():
    from radar_vla.response_reward import response_reward
    plan = braking_plan()
    plan['ego_control_points'] = [[4. * t, 0.] for t in plan['control_times_s']]
    failed = response_reward(reaction_scene(), plan, 'obstacle', .001)
    assert not failed['effective'] and failed['reward'] < 0
    record = reaction_scene()
    record['agents'][0]['position'] = [10., 8.]
    record['agents'][0]['future_xy'] = [[10., 8.]] * 12
    unnecessary = response_reward(record, braking_plan(), 'obstacle', .05)
    steady = response_reward(record, plan, 'obstacle', .05)
    assert not unnecessary['hazard']
    assert unnecessary['reason'] == 'unnecessary_response'
    assert unnecessary['reward'] < steady['reward']


def test_waiting_is_in_reward_clock_and_perception_is_in_physical_clock():
    from radar_vla.response_reward import response_reward
    base = response_reward(reaction_scene(), braking_plan(), 'obstacle', .05)
    wait = response_reward(reaction_scene(), braking_plan(), 'obstacle', .05, waiting_s=.3)
    assert wait['reaction_s'] == pytest.approx(base['reaction_s'] + .3)
    delayed = response_reward(reaction_scene(), braking_plan(), 'obstacle', .05, perception_s=2.4)
    assert delayed['collision'] == 1
    with pytest.raises(ValueError):
        response_reward(reaction_scene(), braking_plan(), 'obstacle', float('nan'))


def tiny_policy():
    from radar_vla.planner import PlannerConfig, RiskConditionedPlanner
    return RiskConditionedPlanner(PlannerConfig(hidden_dim=8, radar_dim=8, num_heads=2,
        num_layers=1, bins=8, max_agents=1, max_length=128)).eval()


def test_sampled_grammar_logprob_matches_rescore_and_has_policy_gradient():
    from radar_vla.rl_policy import sample_plan, policy_statistics
    torch.manual_seed(11)
    planner = tiny_policy()
    radar, risk = torch.randn(1, 2, 8), torch.zeros(1, 5)
    sampled = sample_plan(planner, radar, risk, ['go'], max_new_tokens=100, temperature=.8)
    assert planner.decode(sampled['ids'])['valid']
    stats = policy_statistics(planner, radar, risk, ['go'], sampled['ids'], temperature=.8)
    assert stats['log_prob_sum'].item() == pytest.approx(sampled['log_prob_sum'], abs=1e-5)
    assert 0 < stats['decisions'] < len(sampled['ids'])
    (-stats['log_prob_sum']).backward()
    assert planner.lm_head.weight.grad.abs().sum() > 0
    reference = copy.deepcopy(planner).requires_grad_(False)
    assert policy_statistics(planner, radar, risk, ['go'], sampled['ids'],
                             temperature=.8, reference=reference)['kl'].item() == pytest.approx(0., abs=1e-7)


def test_sampling_rejects_dropout_training_and_illegal_teacher_tokens():
    from radar_vla.rl_policy import sample_plan, policy_statistics
    planner = tiny_policy()
    radar, risk = torch.zeros(1, 2, 8), torch.zeros(1, 5)
    with pytest.raises(ValueError, match='grammar'):
        policy_statistics(planner, radar, risk, ['go'], [planner.tokenizer.eos_id])
    planner.train()
    with pytest.raises(ValueError, match='eval'):
        sample_plan(planner, radar, risk, ['go'])


def test_event_matching_is_one_to_one_and_does_not_invent_track_ids():
    from radar_vla.response_rl import match_event_target
    agents = [dict(id='near', position=[1., 0.]), dict(id='far', position=[2., 0.])]
    detections = [dict(position_m=[1., 0.], local_id=19)]
    assert match_event_target(detections, agents, 'near', 2.) == 0
    assert match_event_target(detections, agents, 'far', 2.) is None


def test_event_builder_requires_observed_absence_and_preserves_split(tmp_path):
    from radar_vla.response_rl import prepare_response_events, load_events
    records = []
    for split in ('train', 'val'):
        for index in range(4):
            records.append(dict(sample_id=f'{split}{index}', scene_id=split, split=split,
                                timestamp_s=index * .1, agent_supervision_available=True,
                                agents=[] if index == 0 else [dict(id='new')]))
    manifest, events = tmp_path / 'frames.jsonl', tmp_path / 'events.jsonl'
    manifest.write_text(''.join(json.dumps(r) + '\n' for r in records))
    prepare_response_events(manifest, events)
    rows = load_events(events, records)
    assert len(rows) == 2
    assert rows[0]['sample_ids'][0] == 'train0'
    broken = json.loads(events.read_text().splitlines()[0])
    broken['sample_ids'][-1] = 'val2'
    events.write_text(json.dumps(broken) + '\n')
    with pytest.raises(ValueError, match='scene|split'):
        load_events(events, records)


def test_leave_one_out_baseline_retains_reward_scale_and_equal_rewards_have_no_gradient():
    from radar_vla.rl_policy import leave_one_out_advantages
    assert leave_one_out_advantages([1., 3.]) == [-2., 2.]
    assert leave_one_out_advantages([-2., -2.]) == [0., 0.]


def test_detector_replay_includes_backlog_before_event_window(tmp_path, monkeypatch):
    from radar_vla.synthetic import write_synthetic_dataset
    from radar_vla.pipeline import resolve_config
    from radar_vla.model import ModelConfig, RadarVLAGrounder
    from radar_vla.response_rl import _contexts
    from radar_vla.response_reward import ResponseRewardConfig
    manifest = write_synthetic_dataset(tmp_path / 'data', scenes_per_split=1, frames_per_scene=5,
                                       range_bins=8, azimuth_bins=8)
    records = [json.loads(line) for line in manifest.read_text().splitlines()]
    for record in records:
        if record['timestamp_s'] < .4:
            record['agents'] = []
    manifest.write_text(''.join(json.dumps(r) + '\n' for r in records))
    cfg = resolve_config(dict(model=dict(hidden_dim=8, num_heads=2, num_queries=2,
        detection_confidence=.001, detection_match_distance_m=100.), language_model=dict(backend='tiny')))
    model = RadarVLAGrounder(ModelConfig(**cfg['model'])).eval()
    clock_values = iter([v for i in range(5) for v in (float(i), i + .2)])
    monkeypatch.setattr('radar_vla.response_rl.time.perf_counter', lambda: next(clock_values))
    event = dict(event_id='one', target_agent_id='target_0', split='train', scene_id='train_scene_0000',
                 sample_ids=['train_scene_0000_frame_0003', 'train_scene_0000_frame_0004'])
    contexts, reports = _contexts(model, manifest, [event], 'train', cfg, 'cpu', ResponseRewardConfig())
    assert reports[0]['detected_timestamp_s'] == pytest.approx(1.)
    assert reports[0]['perception_s'] == pytest.approx(.6)


def test_tiny_rl_updates_only_planner_and_checkpoint_remains_predictable(tmp_path):
    from radar_vla.synthetic import write_synthetic_dataset
    from radar_vla.pipeline import resolve_config, make_planner, load_models, predict_checkpoint
    from radar_vla.model import ModelConfig, RadarVLAGrounder
    from radar_vla.response_rl import run_response_rl, prepare_response_events, evaluate_response_rl
    # Real detector, real grammar sampler, real optimizer. A tiny random model
    # verifies plumbing only; its invalid trajectories are not evidence of skill.
    torch.manual_seed(5)
    manifest = write_synthetic_dataset(tmp_path / 'data', scenes_per_split=1, frames_per_scene=3,
                                       range_bins=8, azimuth_bins=8)
    records = [json.loads(line) for line in manifest.read_text().splitlines()]
    for row in records:
        if row['timestamp_s'] == 0:
            row['agents'] = []
    manifest.write_text(''.join(json.dumps(r) + '\n' for r in records))
    cfg = resolve_config(dict(model=dict(hidden_dim=8, num_heads=2, num_queries=2,
        detection_confidence=.001, detection_match_distance_m=100.),
        planner=dict(hidden_dim=8, num_heads=2, num_layers=1, bins=8, max_agents=1,
                     max_instruction_bytes=64, max_length=192), language_model=dict(backend='tiny')))
    grounder = RadarVLAGrounder(ModelConfig(**cfg['model']))
    planner = make_planner(cfg)
    checkpoint = tmp_path / 'sft.pt'
    torch.save(dict(protocol=dict(schema='radar_vla_pipeline_v2', stage='sft', config=cfg,
                   language_model_files=None, precision='float32'),
                    grounder=grounder.state_dict(), planner=planner.state_dict()), checkpoint)
    events = tmp_path / 'events.jsonl'
    prepare_response_events(manifest, events)
    options = dict(checkpoint=checkpoint, manifest=manifest, events=events, output=tmp_path / 'rl',
                   epochs=1, config=dict(group_size=2, require_controls=False,
                                        reward=dict(safety_horizon_s=1.)))
    result = run_response_rl(**options)
    assert result['stage'] == 'rl' and result['state'] == 'complete'
    restored, trained, state = load_models(tmp_path / 'rl/last.pt')
    for key, value in grounder.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], value, rtol=0, atol=0)
    assert state['optimizer']['state']  # Gradients reached optimizer parameters.
    assert state['history'][1]['nonzero_advantages'] > 0
    assert any(not torch.equal(value, trained.state_dict()[name]) for name, value in planner.state_dict().items())
    assert state['protocol']['stage'] == 'rl'
    assert state['history'][0]['source'] == 'initial_sft'
    assert (tmp_path / 'rl/validation_epoch_0.json').is_file()
    assert state['history'][1]['train']['sampled_plans'] == 2
    assert state['history'][1]['train']['generated_plans'] == 4
    predictions = predict_checkpoint(tmp_path / 'rl/last.pt', manifest, tmp_path / 'pred.jsonl')
    assert len(predictions) == 3
    report = evaluate_response_rl(tmp_path / 'rl/last.pt', manifest, events, tmp_path / 'eval.json')
    assert report['split'] == 'test' and report['events'] == 1
    # Resume at completed epoch is safe; changed reward/provenance is rejected.
    run_response_rl(**options, resume=True)
    changed = {**options, 'config': {**options['config'], 'temperature': .8}}
    with pytest.raises(ValueError, match='provenance'):
        run_response_rl(**changed, resume=True)
