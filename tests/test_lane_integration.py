"""LONG lane perception stays observation-only and shares deployment timing."""
import copy
from dataclasses import asdict
import json

import pytest
import torch

from radar_vla.planner import PlannerConfig, RiskConditionedPlanner, _layout


def _planner(mode='long'):
    return RiskConditionedPlanner(PlannerConfig(hidden_dim=8, radar_dim=8,
        num_heads=2, num_layers=1, bins=65, max_agents=1,
        reasoning_policy='always_' + mode)).eval()


def _ids(planner, mode='long'):
    fields, _, _ = _layout(mode, planner.config, 0)
    return [planner.tokenizer.encode_scalar(3.6 if field == 'width' else 0.)
            if field in ('position', 'width') else planner.tokenizer.token(field)
            for field in fields]


def _perception():
    lanes = []
    for index in range(2):
        y = 3.6 * index
        lanes.append(dict(lane_id=index, confidence=.9,
            centerline=[[0., y], [20., y]],
            left_boundary=[[0., y + 1.8], [20., y + 1.8]],
            right_boundary=[[0., y - 1.8], [20., y - 1.8]],
            left_neighbor_id=1 if index == 0 else None,
            right_neighbor_id=0 if index == 1 else None))
    return dict(available=True, lanes=lanes, num_lanes=2,
                schema='radar_vla_lanes_v1', coordinate_frame='current_ego', distance_unit='m')


def test_long_preserves_each_lane_geometry_adjacency_and_original_road():
    planner = _planner()
    payload = _perception()
    original = copy.deepcopy(payload)
    decoded = planner.decode(_ids(planner), lane_perception=payload)
    assert decoded['valid'] and decoded['physically_complete']
    assert decoded['schema'] == 'radar_vla_structured_v3'
    assert decoded['road']['complete']
    assert decoded['lanes'] == payload['lanes']
    assert decoded['lane_perception']['available']
    assert decoded['lane_perception']['source'] == 'radar_encoder'
    assert decoded['lane_perception']['schema'] == 'radar_vla_lanes_v1'
    assert decoded['lane_perception']['coordinate_frame'] == 'current_ego'
    incomplete = planner.decode(_ids(planner)[:-1], lane_perception=payload)
    assert not incomplete['valid'] and incomplete['lanes'] == payload['lanes']
    decoded['lanes'][0]['centerline'][0][0] = 9.
    assert payload == original  # Callers may reuse cached perception across RL samples.


def test_long_distinguishes_missing_lane_head_from_observed_empty_result():
    planner = _planner()
    missing = planner.decode(_ids(planner))
    empty = planner.decode(_ids(planner), lane_perception=dict(
        available=True, lanes=[], num_lanes=0, schema='radar_vla_lanes_v1'))
    assert missing['lanes'] == empty['lanes'] == []
    assert not missing['lane_perception']['available']
    assert empty['lane_perception']['available']


def test_short_output_is_identical_with_or_without_lane_perception():
    planner = _planner('short')
    ids = _ids(planner, 'short')
    plain = planner.decode(ids)
    attached = planner.decode(ids, lane_perception=_perception())
    assert plain == attached
    assert plain['schema'] == 'radar_vla_structured_v2'
    assert 'lanes' not in plain and 'lane_perception' not in plain


def test_long_lane_relationships_cannot_reference_an_absent_instance():
    planner = _planner()
    payload = _perception()
    payload['lanes'][0]['left_neighbor_id'] = 17
    with pytest.raises(ValueError, match='neighbor'):
        planner.decode(_ids(planner), lane_perception=payload)


def test_timed_deployment_replay_uses_the_same_lane_payload():
    from radar_vla.execution import measure_plan
    planner = _planner()
    ids = _ids(planner)
    payload = _perception()
    result = measure_plan(planner, torch.zeros(1, 2, 8), torch.zeros(1, 5), ['go'],
        ids=ids, lane_perception=payload, clock=iter([2., 2.25]).__next__)
    assert result['generation_s'] == .25
    assert result['plan'] == planner.decode(ids, lane_perception=payload)


def test_legacy_checkpoint_loads_without_a_random_lane_head(tmp_path):
    from radar_vla.model import ModelConfig, RadarVLAGrounder
    from radar_vla.pipeline import load_models
    model = RadarVLAGrounder(ModelConfig(hidden_dim=8, num_heads=2, use_lanes=False))
    config = asdict(model.config)
    # Actual checkpoints written before lanes had no lane configuration keys.
    config = {key: value for key, value in config.items()
              if key != 'use_lanes' and not key.startswith('lane_')}
    checkpoint = tmp_path / 'legacy.pt'
    torch.save(dict(protocol=dict(schema='radar_vla_pipeline_v2', config=dict(model=config)),
                    grounder=model.state_dict(), planner=None), checkpoint)
    restored, planner, _ = load_models(checkpoint)
    assert not restored.config.use_lanes
    assert planner is None
    assert set(restored.state_dict()) == set(model.state_dict())


def test_prediction_decodes_lanes_before_planner_and_without_map_labels(tmp_path, monkeypatch):
    import radar_vla.lanes as lanes
    from radar_vla.data import RadarDataset
    from radar_vla.model import ModelConfig, RadarVLAGrounder
    from radar_vla.pipeline import prediction_rows
    from radar_vla.synthetic import write_synthetic_dataset
    manifest = write_synthetic_dataset(tmp_path / 'data', scenes_per_split=1,
        frames_per_scene=1, range_bins=8, azimuth_bins=8)
    dataset = RadarDataset(manifest, 'test', supervision=False)
    grounder = RadarVLAGrounder(ModelConfig(hidden_dim=8, num_heads=2, num_queries=2)).eval()
    planner = _planner()
    seen = []
    payload = _perception()
    def perception(prediction, config):
        seen.append('lanes')
        return [copy.deepcopy(payload)] * len(prediction['radar_tokens'])
    def generate(*args, **kwargs):
        seen.append('generation')
        return [_ids(planner)]
    monkeypatch.setattr(lanes, 'decode_lanes', perception)
    monkeypatch.setattr(planner, 'generate', generate)
    rows = prediction_rows(grounder, planner, dataset, {'protocol': {}}, 1, 'cpu')
    assert seen == ['lanes', 'generation']
    assert rows[0]['plan']['lanes'] == payload['lanes']
    assert rows[0]['detection_seconds'] > 0


def test_rl_rollout_carries_frame_lane_perception_to_shared_deployment(monkeypatch):
    import radar_vla.response_rl as response_rl
    planner = _planner()
    payload = _perception()
    decoded = planner.decode(_ids(planner), lane_perception=payload)
    seen = []
    def measured(*args, **kwargs):
        seen.append(kwargs.get('lane_perception'))
        return dict(ids=_ids(planner), plan=decoded, generation_s=.1)
    monkeypatch.setattr(response_rl, 'measure_plan', measured)
    monkeypatch.setattr(response_rl, 'compose_execution', lambda *args: (decoded, .2))
    monkeypatch.setattr(response_rl, 'response_reward', lambda *args, **kwargs: {})
    frame = dict(radar=torch.zeros(1, 2, 8), risk=torch.zeros(1, 5),
        instructions=['go'], future_times_s=[.5, 1., 1.5], ready=.1,
        lane_perception=payload)
    context = dict(record={'timestamp_s': 0.}, frames=[frame],
        report={'detected_timestamp_s': .1}, perception_s=.1,
        event={'event_id': 'event', 'target_agent_id': 'obstacle'})
    result = response_rl._rollout(planner, context, response_rl.ResponseRLConfig(), 'cpu', greedy=True)
    assert seen == [payload]
    assert result['decisions'][0]['plan']['lanes'] == payload['lanes']


def test_rl_context_decodes_lanes_inside_perception_clock_and_keeps_frame_instances(tmp_path, monkeypatch):
    import radar_vla.lanes as lanes
    import radar_vla.response_rl as response_rl
    from radar_vla.model import ModelConfig, RadarVLAGrounder
    from radar_vla.pipeline import resolve_config
    from radar_vla.response_reward import ResponseRewardConfig
    from radar_vla.synthetic import write_synthetic_dataset
    manifest = write_synthetic_dataset(tmp_path / 'data', scenes_per_split=1,
        frames_per_scene=2, range_bins=8, azimuth_bins=8)
    records = [r for r in map(json.loads, manifest.read_text().splitlines()) if r['split'] == 'train']
    config = resolve_config(dict(model=dict(hidden_dim=8, num_heads=2, num_queries=2),
                                 language_model=dict(backend='tiny')))
    grounder = RadarVLAGrounder(ModelConfig(**config['model'])).eval()
    detection_rows = iter([dict(detections=[dict(position_m=r['agents'][0]['position'])],
                                detection_truncated=False) for r in records])
    monkeypatch.setattr(response_rl, 'decode_detections', lambda *args: [next(detection_rows)])
    payloads = [_perception(), dict(available=True, lanes=[], schema='radar_vla_lanes_v1')]
    iterator = iter(payloads)
    seen = []
    def decode(*args):
        seen.append('lanes')
        return [next(iterator)]
    ticks = iter([0., .1, 1., 1.1])
    def clock():
        seen.append('clock')
        return next(ticks)
    monkeypatch.setattr(lanes, 'decode_lanes', decode)
    monkeypatch.setattr(response_rl.time, 'perf_counter', clock)
    ids = [r['sample_id'] for r in records]
    event = dict(event_id='lane-event', target_agent_id='target_0', split='train',
        scene_id=records[0]['scene_id'], sample_ids=ids, trigger_sample_id=ids[0],
        event_kind='hazard_start', hazard=True)
    contexts, _ = response_rl._contexts(grounder, manifest, [event], 'train', config,
                                       'cpu', ResponseRewardConfig())
    assert seen == ['clock', 'lanes', 'clock'] * 2
    assert [frame['lane_perception'] for frame in contexts[0]['frames']] == payloads
