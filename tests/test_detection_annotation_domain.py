"""Limited annotation coverage must never become full-range background truth."""
import json
import math

import pytest
import torch


def domain():
    return {'type': 'radar_bev_sector', 'range_m': 50.,
            'azimuth_min_rad': -math.radians(53), 'azimuth_max_rad': math.radians(53),
            'source': 'recorded annotation coverage'}


def batch():
    return dict(agent_mask=torch.tensor([[True]]),
                agent_state=torch.tensor([[[10., 0., 0., 0., 0.]]]),
                agent_class=torch.tensor([[0]]), agent_class_mask=torch.tensor([[True]]),
                agent_supervision_mask=torch.tensor([True]),
                agent_future=torch.tensor([[[[10., 0.]]]]),
                agent_future_mask=torch.tensor([[[True]]]),
                radar_pose=torch.eye(4).reshape(1, 1, 4, 4),
                detection_annotation_domain=torch.tensor([[50., -math.radians(53), math.radians(53)]]),
                detection_annotation_domain_mask=torch.tensor([True]))


def detection(x, y=0., size=(4., 2.)):
    return dict(position_m=[x, y], size_m=list(size), heading_rad=0.,
                category='vehicle', future_xy=[[x, y]])


def test_metrics_ignore_only_unmatched_boxes_outside_explicit_domain_and_keep_raw_counts():
    from radar_vla.detection_metrics import DetectionMetrics
    predictions = [detection(10.), detection(51.), detection(75.)]
    metrics = DetectionMetrics()
    metrics.update([dict(detections=predictions, num_vehicles=3, detection_truncated=False)], batch())
    result = metrics.compute()
    assert result['false_positives'] == 1  # The box at 51 m intersects the 50 m sector.
    assert result['ignored_outside_annotation_domain'] == 1
    assert result['raw_predictions'] == 3 and result['evaluated_predictions'] == 2
    assert result['raw_count_mae'] == 2. and result['count_mae'] == 1.
    assert result['raw_vehicle_count_mae'] == 2. and result['vehicle_count_mae'] == 1.
    assert result['annotation_domain_samples'] == 1


def test_metric_matching_retains_boundary_ground_truth_even_if_predicted_box_outside():
    from radar_vla.detection_metrics import DetectionMetrics
    data = batch()
    data['agent_state'][0, 0, 0] = 51.
    metrics = DetectionMetrics()
    metrics.update([dict(detections=[detection(51., size=(.2, .2))], num_vehicles=1,
                         detection_truncated=False)], data)
    result = metrics.compute()
    assert result['true_positives'] == 1 and result['ignored_outside_annotation_domain'] == 0


def test_legacy_metrics_do_not_infer_a_fifty_metre_domain():
    from radar_vla.detection_metrics import DetectionMetrics
    data = batch()
    data.pop('detection_annotation_domain')
    data.pop('detection_annotation_domain_mask')
    metrics = DetectionMetrics()
    metrics.update([dict(detections=[detection(10.), detection(75.)], num_vehicles=2,
                         detection_truncated=False)], data)
    assert metrics.compute()['false_positives'] == 1


def test_domain_background_mask_uses_current_sensor_pose_and_legacy_default():
    from radar_vla.annotation_domain import annotation_domain_mask
    data = batch()
    data['radar_pose'][0, 0, :2, :2] = torch.tensor([[0., -1.], [1., 0.]])
    data['radar_pose'][0, 0, :2, 3] = torch.tensor([3., 4.])
    points = torch.tensor([[[3., 14.], [3., 65.], [20., 4.]]])
    assert annotation_domain_mask(points, data).tolist() == [[True, False, False]]
    data['detection_annotation_domain_mask'][:] = False
    assert annotation_domain_mask(points, data).all()
    assert annotation_domain_mask(points, {}).all()


def test_box_sector_overlap_handles_edges_and_rotation():
    from radar_vla.annotation_domain import box_intersects_annotation_domain
    limits = [50., -math.radians(53), math.radians(53)]
    pose = torch.eye(4).numpy()
    assert box_intersects_annotation_domain([51., 0.], [4., 2.], 0., limits, pose)
    assert not box_intersects_annotation_domain([53., 0.], [4., 2.], 0., limits, pose)
    # An edge crosses the circle while no corner is in the sector.
    assert box_intersects_annotation_domain([50., 0.], [2., 30.], 0., limits, pose)
    assert not box_intersects_annotation_domain([10., 40.], [2., 2.], 0., limits, pose)
    pose[:2, :2] = [[0., -1.], [1., 0.]]
    pose[:2, 3] = [3., 4.]
    assert box_intersects_annotation_domain([3., 55.], [4., 2.], math.pi/2, limits, pose)
    assert not box_intersects_annotation_domain([3., 58.], [4., 2.], math.pi/2, limits, pose)


def test_manifest_domain_is_explicit_validated_and_collated_without_cropping(tmp_path):
    from radar_vla.synthetic import write_synthetic_dataset
    from radar_vla.data import RadarDataset, collate_batch, load_manifest
    manifest = write_synthetic_dataset(tmp_path/'synthetic', scenes_per_split=1, frames_per_scene=1)
    records = [json.loads(line) for line in manifest.read_text().splitlines()]
    original = RadarDataset(manifest, 'train')[0]
    assert not original['detection_annotation_domain_mask'].item()
    for record in records:
        record['detection_annotation_domain'] = domain()
    manifest.write_text(''.join(json.dumps(record)+'\n' for record in records))
    item = RadarDataset(manifest, 'train')[0]
    assert item['detection_annotation_domain_mask'].item()
    assert item['radar'].shape == original['radar'].shape
    assert torch.equal(item['radar'], original['radar'])
    assert collate_batch([item])['detection_annotation_domain'].shape == (1, 3)
    mixed = collate_batch([item, original])
    assert mixed['detection_annotation_domain_mask'].tolist() == [True, False]
    records[0]['detection_annotation_domain']['range_m'] = -1.
    manifest.write_text(''.join(json.dumps(record)+'\n' for record in records))
    with pytest.raises(ValueError, match='annotation domain'):
        load_manifest(manifest)


def test_sensor_velocity_accounts_for_rotating_offset_in_current_ego_coordinates():
    from radar_vla.collection_adapter import _radar_sensor_velocity
    frame = {'ego': {'pose': {'rotation_deg': [0., 0., 90.]},
                     'velocity_world_mps': [0., 10., 0.],
                     'angular_velocity_world_degps': [0., 0., 90.]}}
    # CARLA rotation +90deg/s: forward 2m mounting acquires world vx=-pi.
    assert _radar_sensor_velocity(frame, 90., 2.) == pytest.approx([10., -math.pi])
    frame['ego']['angular_velocity_world_degps'][2] = 0.
    assert _radar_sensor_velocity(frame, 90., 2.) == pytest.approx([10., 0.])


def test_prepared_reuse_rejects_old_contract_before_copying_assets(tmp_path):
    from radar_vla.collection_adapter import _reuse_record
    with pytest.raises(ValueError, match='preparation contract'):
        _reuse_record({}, {}, tmp_path/'old', tmp_path/'new')


def test_empty_annotated_scene_keeps_raw_outside_predictions_visible():
    from radar_vla.detection_metrics import DetectionMetrics
    data = batch()
    data['agent_mask'][:] = False
    metrics = DetectionMetrics()
    metrics.update([dict(detections=[detection(75.)], num_vehicles=1,
                         detection_truncated=False)], data)
    result = metrics.compute()
    assert result['false_positives'] == 0 and result['raw_predictions'] == 1
    assert result['ignored_outside_annotation_domain'] == 1
    assert result['count_mae'] == 0. and result['raw_count_mae'] == 1.
