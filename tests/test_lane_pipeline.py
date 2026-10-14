"""Real multilaned data trains both stages and reloads through public inference."""
import json

from radar_vla.pipeline import run_training, load_models, predict_checkpoint
from radar_vla.synthetic import write_synthetic_dataset


def test_multilane_training_checkpoint_and_long_prediction_with_custom_samples(tmp_path):
    manifest = write_synthetic_dataset(tmp_path / 'data', scenes_per_split=1,
        frames_per_scene=1, range_bins=16, azimuth_bins=16)
    records = [json.loads(line) for line in manifest.read_text().splitlines()]
    for record in records:
        road = record['map']
        lane = road['lane_centerlines'][0]
        lane.update(left_neighbor_id='next_lane', right_neighbor_id=None)
        road['lane_centerlines'].append(dict(id='next_lane',
            points=[[-10., 4., 0.], [70., 4., 0.]], left_boundary_id='outer_left',
            right_boundary_id='left', left_neighbor_id=None, right_neighbor_id='route_lane'))
        road['lane_boundaries'].append(dict(id='outer_left',
            points=[[-10., 6., 0.], [70., 6., 0.]]))
    manifest.write_text(''.join(json.dumps(row) + '\n' for row in records))
    config = dict(model=dict(hidden_dim=8, num_heads=2, num_queries=2, lane_samples=6),
                  planner=dict(num_layers=1, reasoning_policy='always_long'),
                  language_model=dict(backend='tiny'))
    run_training(manifest, tmp_path / 'grounding', stage='grounding', epochs=1,
                 batch_size=1, config=config)
    run_training(manifest, tmp_path / 'sft', stage='sft', epochs=1, batch_size=1,
                 config=config, init_grounding=tmp_path / 'grounding' / 'best.pt')
    checkpoint = tmp_path / 'sft' / 'best.pt'
    grounder, _, _ = load_models(checkpoint)
    assert grounder.config.lane_samples == 6
    assert grounder.lane_supervised
    assert grounder.lane_boundary_supervised.tolist() == [True, True]
    assert grounder.lane_topology_supervised.tolist() == [True, True]
    destination = tmp_path / 'predictions.jsonl'
    predict_checkpoint(checkpoint, manifest, destination, batch_size=1)
    rows = [json.loads(line) for line in destination.read_text().splitlines()]
    assert len(rows) == 1
    plan = rows[0]['plan']
    assert plan['mode'] == 'long' and plan['schema'] == 'radar_vla_structured_v3'
    assert plan['lane_perception']['available']
    assert plan['lane_perception']['num_lanes'] == len(plan['lanes'])
    for lane in plan['lanes']:
        assert len(lane['centerline']) == 6
