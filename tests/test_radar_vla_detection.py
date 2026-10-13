"""Dense radar detection, semantic labels and observable inference contracts."""
import json
import math

import pytest
import torch


def batch():
    from test_radar_vla_model import example_batch
    value = example_batch(1)
    value.update(agent_class=torch.tensor([[0, 1]]),
                 agent_class_mask=torch.ones(1, 2, dtype=torch.bool),
                 agent_size=torch.tensor([[[4.5, 1.8], [.6, .6]]]),
                 agent_heading=torch.tensor([[0., math.pi / 2]]),
                 agent_box_mask=torch.ones(1, 2, dtype=torch.bool))
    return value


def predicted():
    # Two duplicate cars, an overlapping pedestrian, an uncertain class,
    # and a low-confidence background candidate.
    state = torch.tensor([[[10., 0., -2., 3., 0.], [10.1, 0., -2., 3., 0.],
                           [10., 0., 0., 0., 1.], [20., 0., 0., 0., 0.],
                           [30., 0., 0., 0., 0.]]])
    classes = torch.full((1, 5, 4), -8.)
    classes[0, :2, 0] = 8.; classes[0, 2, 1] = 8.
    classes[0, 3] = 0.; classes[0, 4, 0] = 8.
    future = state[:, :, None, :2].repeat(1, 1, 2, 1)
    future[0, 0, :, 0] += 3.
    return dict(detection_logits=torch.tensor([[8., 7., 8., 8., -8.]]),
                detection_class_logits=classes, detection_state=state,
                detection_size=torch.tensor([[[4., 2.]] * 5]),
                detection_heading=torch.zeros(1, 5),
                detection_future=future,
                detection_future_velocity=torch.zeros(1, 5, 2, 2))


def test_categories_are_explicit_and_missing_is_unknown():
    from radar_vla.categories import category_name
    assert category_name({'category': 'car'}) == 'vehicle'
    assert category_name({'class_name': 'pedestrian'}) == 'pedestrian'
    assert category_name({'type': 'bicycle'}) == 'cyclist'
    assert category_name({}) == 'unknown'
    with pytest.raises(ValueError, match='category'):
        category_name({'category': 'car', 'type': 'pedestrian'})
    with pytest.raises(ValueError, match='category'):
        category_name({'category': 'typo'})


def test_raw_import_preserves_classes_and_dataset_masks_unknown(tmp_path):
    from radar_vla.records import prepare_frames
    from radar_vla.data import RadarDataset
    from test_radar_vla_records import frame_fixture
    frames = [frame_fixture(tmp_path, i) for i in range(2)]
    frames[0]['agents'][0]['category'] = 'truck'
    raw = tmp_path / 'raw.jsonl'
    raw.write_text(''.join(json.dumps(f)+'\n' for f in frames))
    manifest = prepare_frames(raw, tmp_path / 'prepared', history_frames=1)
    ds = RadarDataset(manifest, 'train')
    assert ds.records[0]['agents'][0]['category'] == 'vehicle'
    assert ds[0]['agent_class'][0] == 0
    assert ds[0]['agent_box_mask'][0]
    assert ds[1]['agent_class'][0] == -1
    assert not ds[1]['agent_class_mask'][0]
    observation = RadarDataset(manifest, 'train', supervision=False)[0]
    assert not observation['agent_class_mask'].any()
    assert not observation['agent_box_mask'].any()
    assert observation['agent_size'].sum() == 0


def test_dense_head_keeps_spatial_resolution_and_learns_all_outputs():
    from radar_vla.model import ModelConfig, RadarVLAGrounder
    from radar_vla.losses import loss_grounding
    cfg = ModelConfig(hidden_dim=16, max_agents=3, max_grid_size=2)
    model = RadarVLAGrounder(cfg)
    data = batch()
    output = model(data)
    # Encoder stride four: 16x20 -> 4x5, two hypotheses per cell.
    assert output['detection_logits'].shape == (1, 40)
    assert output['detection_class_logits'].shape == (1, 40, 4)
    assert output['detection_future'].shape == (1, 40, 6, 2)
    assert (output['detection_size'] > 0).all()
    losses = loss_grounding(output, data)
    for key in ('classification', 'box_size', 'heading', 'agent_future'):
        assert losses['detection_'+key] > 0
    losses['total'].backward()
    for name in ('object_head', 'class_head', 'box_head', 'motion_head'):
        parameters = getattr(model.detector, name).parameters()
        assert sum(p.grad.abs().sum().item() for p in parameters if p.grad is not None) > 0
    assert model.encoder[0].weight.grad.abs().sum() > 0


def test_unknown_labels_do_not_train_vehicle_classifier():
    from radar_vla.model import ModelConfig, RadarVLAGrounder
    from radar_vla.losses import loss_grounding
    data = batch(); data['agent_class_mask'][:] = False; data['agent_class'][:] = -1
    model = RadarVLAGrounder(ModelConfig(hidden_dim=16))
    losses = loss_grounding(model(data), data)
    assert losses['detection_classification'] == 0
    assert losses['detection_position'] > 0
    assert losses['detection_objectness'] > 0


def test_same_cell_targets_get_distinct_dense_assignments():
    from radar_vla.model import ModelConfig, RadarVLAGrounder
    from radar_vla.losses import loss_grounding
    data = batch(); data['agent_state'][0, 1, :2] = data['agent_state'][0, 0, :2]
    model = RadarVLAGrounder(ModelConfig(hidden_dim=16))
    losses = loss_grounding(model(data), data)
    assert torch.isfinite(losses['total'])
    # Reordering annotation slots must leave the complete detector loss unchanged.
    reverse = dict(data)
    for key, value in data.items():
        if key.startswith('agent_') and value.ndim >= 2 and value.shape[1] == 2:
            reverse[key] = value[:, [1, 0]]
    other = loss_grounding(model(reverse), reverse)
    for key in losses:
        if key.startswith('detection_'):
            torch.testing.assert_close(losses[key], other[key])


def test_rotated_iou_respects_heading():
    from radar_vla.detection import rotated_iou
    assert rotated_iou([0, 0, 8, 2, 0], [0, 0, 8, 2, math.pi]) == pytest.approx(1.)
    assert rotated_iou([0, 0, 8, 2, 0], [0, 0, 8, 2, math.pi/2]) == pytest.approx(1/7)
    assert rotated_iou([0, 0, 8, 2, 0], [20, 0, 8, 2, 0]) == 0


def test_decode_filters_background_and_duplicates_preserving_classes_and_trajectories():
    from radar_vla.detection import decode_detections
    from radar_vla.model import ModelConfig
    row = decode_detections(predicted(), ModelConfig(), [[.5, 1.]])[0]
    assert row['num_agents'] == 3
    assert row['num_vehicles'] == 1
    assert row['class_counts'] == {'vehicle': 1, 'pedestrian': 1, 'cyclist': 0, 'other': 0, 'unknown': 1}
    car = next(d for d in row['detections'] if d['category'] == 'vehicle')
    assert car['position_m'] == [10., 0.]
    assert car['future_xy'] == [[13., 0.], [13., 0.]]
    assert car['future_times_s'] == [.5, 1.]
    assert not row['detection_truncated']
    json.dumps(row, allow_nan=False)


def test_zero_detections_and_capacity_are_not_fixed_to_eight():
    from radar_vla.detection import decode_detections
    from radar_vla.model import ModelConfig
    output = predicted(); output['detection_logits'].fill_(-10)
    row = decode_detections(output, ModelConfig())[0]
    assert row['num_agents'] == row['num_vehicles'] == 0
    assert row['detections'] == []
    output = predicted()
    row = decode_detections(output, ModelConfig(detection_max_objects=1))[0]
    assert row['num_agents'] == 1 and row['detection_truncated']
    output = {key: value[:, :1].repeat(1, 80, *([1] * (value.ndim - 2)))
              for key, value in predicted().items()}
    output['detection_state'][0, :, 0] = torch.arange(80) * 10.
    row = decode_detections(output, ModelConfig())[0]
    assert row['num_agents'] == row['num_vehicles'] == 80
    assert not row['detection_truncated']


def test_detector_anchors_respect_sensor_rotation_and_translation():
    from radar_vla.detection import cell_centers
    data = batch()
    original = cell_centers(data, 4, 5)
    pose = torch.eye(4).repeat(1, 3, 1, 1)
    pose[:, :, :2, :2] = torch.tensor([[0., -1.], [1., 0.]])
    pose[:, :, :2, 3] = torch.tensor([3., -4.])
    data['radar_pose'] = pose
    transformed = cell_centers(data, 4, 5)
    expected = torch.stack((-original[..., 1] + 3., original[..., 0] - 4.), -1)
    torch.testing.assert_close(transformed, expected)
    radius, angle = data['range_m'][:, :, None], data['azimuth_rad'][:, None, :]
    xy = torch.stack((-radius * angle.sin() + 3., radius * angle.cos() - 4.), -1)
    data['radar_cartesian'] = xy[:, None].repeat(1, 3, 1, 1, 1)
    torch.testing.assert_close(cell_centers(data, 4, 5), transformed)


def test_legacy_checkpoint_does_not_report_random_detector_counts(tmp_path):
    from dataclasses import asdict
    from radar_vla.model import ModelConfig, RadarVLAGrounder
    from radar_vla.pipeline import load_models, prediction_rows
    from radar_vla.synthetic import write_synthetic_dataset
    from radar_vla.data import RadarDataset
    cfg = ModelConfig(hidden_dim=16, use_detection=False)
    old_config = {key: value for key, value in asdict(cfg).items()
                  if key != 'use_detection' and not key.startswith('detection_')}
    state = dict(protocol=dict(schema='radar_vla_pipeline_v2', config={'model': old_config}),
                 grounder=RadarVLAGrounder(cfg).state_dict(), planner=None)
    checkpoint = tmp_path / 'legacy.pt'
    torch.save(state, checkpoint)
    model, planner, state = load_models(checkpoint)
    assert not model.config.use_detection
    manifest = write_synthetic_dataset(tmp_path / 'data', scenes_per_split=1, frames_per_scene=1)
    dataset = RadarDataset(manifest, 'test', supervision=False)
    row = prediction_rows(model, planner, dataset, state, 1, 'cpu')[0]
    assert row['detection_status'] == 'unavailable_legacy'
    assert row['num_agents'] is None and row['num_vehicles'] is None
    assert row['detections'] is None


def test_detection_metrics_count_false_positives_misses_and_unknown_class_coverage():
    from radar_vla.detection_metrics import DetectionMetrics
    from radar_vla.detection import decode_detections
    from radar_vla.model import ModelConfig
    row = decode_detections(predicted(), ModelConfig(), [[.5, 1.]])[0]
    data = batch()
    data['agent_state'][0, :, :2] = torch.tensor([[10., 0.], [40., 0.]])
    data['agent_future'] = data['agent_future'][:, :, :2]
    data['agent_future_mask'] = data['agent_future_mask'][:, :, :2]
    meter = DetectionMetrics(match_distance_m=2.)
    meter.update([row], data)
    result = meter.compute()
    assert result['true_positives'] == 1
    assert result['false_positives'] == 2
    assert result['false_negatives'] == 1
    assert result['count_mae'] == 1
    empty = dict(data, agent_mask=torch.zeros_like(data['agent_mask']))
    meter.update([row], empty)
    assert meter.compute()['false_positives'] == 5
    unknown = dict(data, agent_class_mask=torch.zeros_like(data['agent_class_mask']))
    meter = DetectionMetrics(); meter.update([row], unknown)
    assert meter.compute()['vehicle_count_samples'] == 0
    assert meter.compute()['vehicle_count_mae'] is None


def test_formal_training_rejects_unlabeled_categories(tmp_path):
    from radar_vla.synthetic import write_synthetic_dataset
    from radar_vla.pipeline import run_training
    manifest = write_synthetic_dataset(tmp_path/'data', scenes_per_split=1, frames_per_scene=1)
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    rows[0]['agents'][0].pop('category', None)
    manifest.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    with pytest.raises(ValueError, match='category'):
        run_training(manifest, tmp_path/'run', epochs=1,
                     config={'data': {'require_detection_classes': True}})


def crowded_manifest(tmp_path):
    import copy
    from radar_vla.synthetic import write_synthetic_dataset
    manifest = write_synthetic_dataset(tmp_path, scenes_per_split=1, frames_per_scene=2)
    records = [json.loads(line) for line in manifest.read_text().splitlines()]
    for i, record in enumerate(records):
        source = record['agents'][0]
        record['agents'] = [dict(copy.deepcopy(source), id=f'actor-{j}',
                                category='vehicle' if j % 2 else 'pedestrian')
                            for j in range(12 if i % 2 == 0 else 0)]
    manifest.write_text(''.join(json.dumps(r)+'\n' for r in records))
    return manifest


def test_variable_agent_batches_pad_without_discarding_objects(tmp_path):
    from radar_vla.data import RadarDataset, collate_batch
    dataset = RadarDataset(crowded_manifest(tmp_path), 'train')
    crowded, empty = dataset[0], dataset[1]
    assert crowded['agent_state'].shape == (12, 5)
    assert empty['agent_state'].shape == (0, 5)
    merged = collate_batch([crowded, empty])
    assert merged['agent_state'].shape == (2, 12, 5)
    assert merged['agent_mask'].sum(1).tolist() == [12, 0]
    assert not merged['agent_class_mask'][1].any()
    assert merged['agent_class'][1].tolist() == [-1] * 12
    assert merged['critical_agent_order'][1].tolist() == list(range(12))
    all_empty = collate_batch([empty, empty])
    assert all_empty['agent_state'].shape == (2, 0, 5)


def test_dense_loss_supervises_every_object_without_legacy_queries():
    from radar_vla.model import ModelConfig, RadarVLAGrounder
    from radar_vla.losses import loss_grounding
    data = batch()
    for key, value in list(data.items()):
        if key.startswith('agent_'):
            data[key] = value.repeat(1, 6, *([1] * (value.ndim - 2)))
    model = RadarVLAGrounder(ModelConfig(hidden_dim=16, max_agents=1))
    assert not hasattr(model, 'object_queries')
    output = model(data)
    output['detection_logits'].retain_grad()
    losses = loss_grounding(output, data)
    losses['total'].backward()
    # Exactly one positive candidate per GT; all 12 objects get supervision.
    assert (output['detection_logits'].grad < 0).sum() == 12


def test_crowded_grounding_train_evaluate_predict_never_constructs_llm(tmp_path, monkeypatch):
    from radar_vla import pipeline
    def forbidden(*args, **kwargs):
        raise AssertionError('Stage 1 must not construct an LLM')
    monkeypatch.setattr(pipeline, 'make_planner', forbidden)
    manifest = crowded_manifest(tmp_path / 'data')
    config = {'model': {'hidden_dim': 16, 'num_heads': 2, 'max_agents': 1},
              'data': {'require_detection_classes': True}}
    pipeline.run_training(manifest, tmp_path / 'run', epochs=1, config=config)
    checkpoint = tmp_path / 'run/best.pt'
    report = pipeline.evaluate_checkpoint(checkpoint, manifest, tmp_path / 'metrics.json')
    assert report['detection']['category_labeled_objects'] == 12
    assert report['detection']['samples'] == 2
    predictions = pipeline.predict_checkpoint(checkpoint, manifest, tmp_path / 'predictions.jsonl')
    assert len(predictions) == 2
    assert all(row['plan'] is None for row in predictions)
    for row in predictions:
        assert len(row['agent_state']) == len(row['agent_future']) == row['num_agents']


def test_dense_cli_and_cohort_do_not_reapply_legacy_capacity(tmp_path):
    import subprocess
    import sys
    from radar_vla.cohort import build_cohort
    manifest = crowded_manifest(tmp_path / 'data')
    result = subprocess.run([sys.executable, '-m', 'radar_vla', 'validate', '--manifest', str(manifest)],
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    report = build_cohort(manifest, tmp_path / 'cohort.jsonl',
                          config={'model': {'max_agents': 1}, 'planner': {'max_agents': 16}})
    assert report['retained_samples'] == 6


def test_oracle_preflight_uses_dynamic_dense_capacity(tmp_path, monkeypatch):
    from radar_vla.experiments import build_experiment_plan, run_experiment_plan
    manifest = crowded_manifest(tmp_path / 'data')
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    for row in rows:
        row['risk_label'] = dict(pcol=[0., 0., 0.], pcol_mask=[True]*3,
                                 dmin=20., dmin_mask=True, areq=0., areq_mask=True)
    manifest.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    plan = build_experiment_plan({'model': {'max_agents': 1}, 'planner': {'max_agents': 16}},
                                 manifest, tmp_path / 'runs', seeds=[42])
    path = tmp_path / 'plan.json'; path.write_text(json.dumps(plan))
    class LaunchReached(Exception):
        pass
    def do_not_launch(*args, **kwargs):
        raise LaunchReached
    monkeypatch.setattr('radar_vla.experiments.subprocess.run', do_not_launch)
    with pytest.raises(LaunchReached):
        run_experiment_plan(path, device='cpu')
