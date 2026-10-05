import copy
import json

import numpy as np
import pytest
from PIL import Image

from radar_vla.coordinates import transform_points, transform_vectors, validate_transform
from radar_vla.records import load_frame, prepare_frames
from radar_vla.radar_processing import unfold_doppler, radar_geometry, associate_doppler
from radar_vla.data import RadarDataset


def frame_fixture(tmp_path, index=0, split='train'):
    shape = (8, 7)
    np.save(tmp_path / f'p{index}.npy', np.ones(shape, np.float32))
    np.save(tmp_path / f'd{index}.npy', np.full(shape, -4., np.float32))
    np.save(tmp_path / 'pc.npy', np.array([[1., 2., 3., .4]], np.float32))
    Image.fromarray(np.zeros((6, 9, 3), np.uint8)).save(tmp_path / 'front.png')
    pose = np.eye(4); pose[0, 3] = index
    calibration = np.eye(4).tolist()
    ts = index * .1
    return dict(schema_version='radar_frame_v2', sample_id=f'f{index}', scene_id='s', split=split,
                timestamp_s=ts, coordinate_frame='world',
                sensors=dict(radar=dict(power=f'p{index}.npy', unfolded_doppler=f'd{index}.npy',
                    range_m=np.arange(1, 9).tolist(), azimuth_rad=np.linspace(-.3, .3, 7).tolist(),
                    timestamp_s=ts, T_ego_sensor=calibration),
                    camera=dict(front_rgb=dict(path='front.png', timestamp_s=ts,
                        intrinsics=[[10., 0, 4], [0, 10., 3], [0, 0, 1]], T_ego_sensor=calibration)),
                    lidar=dict(pointcloud='pc.npy', timestamp_s=ts, T_ego_sensor=calibration)),
                ego=dict(pose=pose.tolist(), velocity=[10., 0., 0.], acceleration=[0., 0., 0.],
                    yaw_rate=0., box_size=[4., 2.], future_trajectory=[dict(timestamp_s=ts+t,
                        position=[index+10*t, 0., 0.], yaw=0., valid=True) for t in [.5, 1., 2., 3.]]),
                agents=[dict(id='a', bbox3d=dict(center=[8., 0., 0.], size=[4., 2., 1.5], yaw=0.),
                    velocity=[0., 0., 0.], future_trajectory=[dict(timestamp_s=ts+t,
                        position=[8., 0., 0.], yaw=0., valid=True) for t in [.5, 1., 2., 3.]])],
                map=dict(lane_centerlines=[dict(id='lane', points=[[0., 0., 0.], [50., 0., 0.]],
                    left_boundary_id='left', right_boundary_id='right')], lane_boundaries=[
                    dict(id='left', points=[[0., 2., 0.], [50., 2., 0.]]),
                    dict(id='right', points=[[0., -2., 0.], [50., -2., 0.]])],
                    traffic_elements=[dict(id='sign', type='stop_sign', position=[20., 2., 0.])]),
                language=dict(instruction='Go forward.'))


def test_pose_rotation_and_velocity_not_translation():
    pose = np.array([[0., -1, 0, 10], [1, 0, 0, 20], [0, 0, 1, 0], [0, 0, 0, 1]])
    np.testing.assert_allclose(transform_points([[1, 0, 0]], pose), [[10, 21, 0]])
    np.testing.assert_allclose(transform_vectors([[1, 0, 0]], pose), [[0, 1, 0]])
    invalid = pose.copy(); invalid[0, 0] = 2
    with pytest.raises(ValueError, match='rotation'):
        validate_transform(invalid)


def test_full_frame_loads_optional_sensors_and_normalizes_world(tmp_path):
    frame = frame_fixture(tmp_path, 1)
    loaded = load_frame(frame, tmp_path)
    assert loaded['sensor_payload']['camera']['front_rgb'].shape == (6, 9, 3)
    assert loaded['sensor_payload']['lidar'].shape == (1, 4)
    assert loaded['agents'][0]['position'] == [7., 0.]
    assert loaded['agents'][0]['bbox3d']['center'] == [7., 0., 0.]
    assert loaded['map']['traffic_elements'][0]['position'] == [19., 2., 0.]
    assert loaded['ego']['future_xy'][0] == [5., 0.]


def test_prepare_history_has_motion_pose_and_no_future_observation_leak(tmp_path):
    frames = [frame_fixture(tmp_path, i) for i in range(3)]
    source = tmp_path / 'frames.jsonl'
    source.write_text('\n'.join(json.dumps(f) for f in frames))
    manifest = prepare_frames(source, tmp_path / 'prepared', history_frames=2)
    rows = [json.loads(s) for s in manifest.read_text().splitlines()]
    assert [r['sample_id'] for r in rows] == ['f1', 'f2']
    item = RadarDataset(manifest, 'train')[0]
    assert item['radar'].shape == (2, 2, 8, 7)
    assert item['radar_pose'][0, 0, 3] == -1
    assert item['radar_pose'][1, 0, 3] == 0
    assert item['road_mask'].item() and item['road_width'].item() == pytest.approx(4.)
    assert item['agent_radial_observed'][0] == -4.
    assert item['agent_state'][0, 2] == pytest.approx(-10 * item['agent_doppler_projection'][0, 0].item())
    assert item['agent_state'][0, 2] < -9.  # observation independent of GT radial
    assert item['agent_radial_mask'][0]
    changed = copy.deepcopy(frames)
    changed[2]['sensors']['radar']['power'] = 'missing-future-power.npy'
    # First output must only depend on f0,f1. Future annotations never supply input pixels.
    assert np.all(np.load(manifest.parent / rows[0]['sensors']['radar']['power']) == 1.)


def test_unfold_requires_independent_velocity_prior_and_handles_sign():
    folded = np.array([-4., 4., 0.])
    result, valid = unfold_doppler(folded, np.array([16., -16., 0.]), 5., max_prior_error=2.)
    np.testing.assert_allclose(result, [16., -16., 0.])
    assert valid.all()
    _, valid = unfold_doppler(np.array([0.]), np.array([5.]), 5., max_prior_error=2.)
    assert not valid[0]
    with pytest.raises(ValueError):
        unfold_doppler(folded, None, 5.)


def test_sensor_poses_rotate_los_without_rotating_doppler_scalar():
    pose = np.array([[0., -1, 0, 2], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    xy, los = radar_geometry([2.], [0.], pose[None])
    np.testing.assert_allclose(xy[0, 0, 0], [2., 2.])
    np.testing.assert_allclose(los[0, 0, 0], [0., 1.])


def test_raw_sensor_validation_and_scene_split_leak_rejected(tmp_path):
    frame = frame_fixture(tmp_path)
    frame['sensors']['camera']['front_rgb']['intrinsics'][0][0] = -1
    with pytest.raises(ValueError, match='intrinsics'):
        load_frame(frame, tmp_path)
    frames = [frame_fixture(tmp_path, 0), frame_fixture(tmp_path, 1, split='val')]
    source = tmp_path / 'frames.jsonl'; source.write_text('\n'.join(map(json.dumps, frames)))
    with pytest.raises(ValueError, match='split'):
        prepare_frames(source, tmp_path / 'prepared', history_frames=1)


def test_observation_only_inference_does_not_require_agents_or_futures(tmp_path):
    frame = frame_fixture(tmp_path)
    frame.pop('agents'); frame['ego'].pop('future_trajectory')
    source = tmp_path / 'frames.jsonl'; source.write_text(json.dumps(frame))
    manifest = prepare_frames(source, tmp_path / 'prepared', history_frames=1, supervision=False)
    item = RadarDataset(manifest, 'train', supervision=False)[0]
    assert not item['agent_supervision_mask']
    assert not item['risk_mask'].any() and not item['ego_future_mask'].any()


def test_future_radar_values_cannot_change_earlier_history_window(tmp_path):
    frames = [frame_fixture(tmp_path, i) for i in range(3)]
    source = tmp_path / 'frames.jsonl'; source.write_text('\n'.join(map(json.dumps, frames)))
    first = prepare_frames(source, tmp_path / 'one', history_frames=2)
    before = RadarDataset(first, 'train', supervision=False)[0]['radar'].clone()
    np.save(tmp_path / 'p2.npy', np.full((8, 7), 999., np.float32))
    second = prepare_frames(source, tmp_path / 'two', history_frames=2)
    np.testing.assert_array_equal(before.numpy(), RadarDataset(second, 'train', supervision=False)[0]['radar'].numpy())
    assert RadarDataset(second, 'train', supervision=False)[1]['radar'][-1, 0, 0, 0] == 999.


def test_folded_import_uses_independent_prior_and_binary_validity(tmp_path):
    frame = frame_fixture(tmp_path)
    radar = frame['sensors']['radar']
    radar['folded_doppler'] = radar.pop('unfolded_doppler')
    np.save(tmp_path / 'prior.npy', np.full((8, 7), 16., np.float32))
    radar.update(doppler_prior='prior.npy', doppler_prior_source='causal_tracker', max_unambiguous_velocity=5.)
    loaded = load_frame(frame, tmp_path)
    np.testing.assert_allclose(loaded['sensor_payload']['unfolded_doppler'], 16.)
    radar['doppler_prior_source'] = 'future_ground_truth'
    with pytest.raises(ValueError, match='causal'):
        load_frame(frame, tmp_path)


def test_missing_map_width_is_unknown_and_turning_sensor_has_lever_arm(tmp_path):
    from radar_vla.coordinates import sensor_velocity
    from radar_vla.records import road_targets
    frame = frame_fixture(tmp_path)
    frame['map']['lane_centerlines'][0].pop('right_boundary_id')
    target = road_targets(load_frame(frame, tmp_path)['map'])
    assert target['road_centerline_mask'].all()
    assert not target['road_mask'].any() and target['road_width'][0] == 0.
    mounting = np.eye(4); mounting[0, 3] = 2
    np.testing.assert_allclose(sensor_velocity([10., 0.], .5, mounting), [10., 1., 0.])


def test_future_stable_id_tracks_fill_labels_and_entering_agents_affect_risk(tmp_path):
    frames = [frame_fixture(tmp_path, i) for i in range(3)]
    for frame in frames:
        frame['ego'].pop('future_trajectory')
        for agent in frame['agents']:
            agent.pop('future_trajectory')
    frames[0]['agents'] = []
    # At t=.1, ego is at worldx=1 and the entering actor overlaps it.
    for frame in frames[1:]:
        frame['agents'][0]['bbox3d']['center'] = [1., 0., 0.]
    source = tmp_path / 'frames.jsonl'; source.write_text('\n'.join(map(json.dumps, frames)))
    manifest = prepare_frames(source, tmp_path / 'prepared', history_frames=1, future_times_s=(.1, .2))
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    assert rows[0]['agents'] == [] and len(rows[0]['risk_agents']) == 1
    assert rows[0]['risk_agents'][0]['present_valid'] is False
    item = RadarDataset(manifest, 'train')[0]
    assert item['risk_target'][0] == 1. and item['risk_mask'][0]
    np.testing.assert_allclose(rows[0]['ego']['future_xy'], [[1., 0.], [2., 0.]])


def test_tracking_coverage_must_be_known_to_label_empty_scene_safe(tmp_path):
    from radar_vla.geometry import build_risk_labels
    frame = frame_fixture(tmp_path); frame['agents'] = []
    unknown = load_frame(frame, tmp_path)
    assert build_risk_labels(unknown)['pcol_mask'] == [False] * 3
    frame['tracking'] = {'coverage': 'complete_relevant_agents'}
    known = load_frame(frame, tmp_path)
    assert build_risk_labels(known)['pcol_mask'] == [True] * 3


def test_dense_risk_interpolation_finds_between_annotation_collision():
    from radar_vla.geometry import build_risk_labels
    source = dict(future_times_s=[1., 2., 3.], ego=dict(velocity=[0., 0.], yaw_rate=0., box_size=[1., 1.]),
                  agents=[dict(id='crossing', position=[-5., 0.], size=[1., 1.], heading=0.,
                    future_xy=[[5., 0.], [15., 0.], [25., 0.]], future_yaw=[0., 0., 0.],
                    future_valid=[True] * 3)])
    assert build_risk_labels(source, sample_dt_s=1.)['pcol'][0] == 0.
    assert build_risk_labels(source, sample_dt_s=.05)['pcol'][0] == 1.


def test_asynchronous_radar_requires_actual_pose_and_state(tmp_path):
    frame = frame_fixture(tmp_path, 1)
    radar = frame['sensors']['radar']; radar['timestamp_s'] = .08
    with pytest.raises(ValueError, match='ego_pose_at_timestamp'):
        load_frame(frame, tmp_path)
    sensor_pose = np.eye(4); sensor_pose[0, 3] = .8
    radar['ego_pose_at_timestamp'] = sensor_pose.tolist()
    with pytest.raises(ValueError):
        load_frame(frame, tmp_path)
    radar['ego_state_at_timestamp'] = dict(velocity_world=[8., 0., 0.], yaw_rate=0.)
    source = tmp_path/'frame.jsonl'; source.write_text(json.dumps(frame))
    manifest = prepare_frames(source, tmp_path/'prepared', history_frames=1)
    sample = RadarDataset(manifest, 'train')[0]
    assert sample['time_offsets_s'][0].item() == pytest.approx(-.02)
    assert sample['radar_pose'][0, 0, 3].item() == pytest.approx(-.2)
    assert sample['radar_sensor_velocity'][0, 0].item() == pytest.approx(8.)
    radar['timestamp_s'] = .11
    with pytest.raises(ValueError, match='future'):
        load_frame(frame, tmp_path)


def test_future_entrant_is_unknown_before_first_observation(tmp_path):
    frames = [frame_fixture(tmp_path, i) for i in range(2)]
    frames[0]['agents'] = []
    frames[1]['timestamp_s'] = .5
    for sensor in (frames[1]['sensors']['radar'], frames[1]['sensors']['lidar'], frames[1]['sensors']['camera']['front_rgb']):
        sensor['timestamp_s'] = .5
    frames[1]['agents'][0].pop('future_trajectory')
    frames[1]['ego'].pop('future_trajectory')
    source = tmp_path/'frames.jsonl'; source.write_text('\n'.join(map(json.dumps, frames)))
    manifest = prepare_frames(source, tmp_path/'prepared', history_frames=1, future_times_s=(.25, .5))
    row = json.loads(manifest.read_text().splitlines()[0])
    assert row['risk_agents'][0]['future_valid'] == [False, True]
    assert not row['risk_agents'][0]['present_valid']


def test_full_raw_fixture_import_and_cached_per_agent_risk(tmp_path, monkeypatch):
    from radar_vla.synthetic import write_synthetic_frames
    from radar_vla.data import prepare_labels
    frames = write_synthetic_frames(tmp_path/'raw', range_bins=16, azimuth_bins=11, frames_per_scene=3)
    manifest = prepare_frames(frames, tmp_path/'prepared', history_frames=2)
    labelled = prepare_labels(manifest, tmp_path/'labels'/'manifest.jsonl')
    rows = [json.loads(line) for line in labelled.read_text().splitlines()]
    assert {row['split'] for row in rows} == {'train', 'val', 'test'}
    assert all(row['agent_risk_labels'] for row in rows)
    def unexpected_recompute(*args, **kwargs):
        raise AssertionError('cached risk labels should eliminate per-epoch geometry recomputation')
    monkeypatch.setattr('radar_vla.data.build_risk_labels', unexpected_recompute)
    for split in ('train', 'val', 'test'):
        sample = RadarDataset(labelled, split)[0]
        assert sample['road_mask'].item()
        assert sample['agent_risk_mask'].any()
        assert sample['metadata']['map']['traffic_elements'][0]['type'] == 'speed_limit'


def test_missing_map_and_inference_preparation_do_not_consume_future_annotations(tmp_path):
    frame = frame_fixture(tmp_path); frame['map'] = None
    # These deliberately invalid annotations must be irrelevant to observation-only import.
    frame['ego']['future_trajectory'] = 'not future GT'
    frame['agents'][0]['future_trajectory'] = 'not future GT'
    source = tmp_path/'frame.jsonl'; source.write_text(json.dumps(frame))
    manifest = prepare_frames(source, tmp_path/'prepared', history_frames=1, supervision=False)
    sample = RadarDataset(manifest, 'train', supervision=False)[0]
    assert not sample['road_mask'].item() and not sample['risk_mask'].any()


def test_doppler_projection_is_power_weighted_without_unit_renormalization():
    angles = [-.4, 0., .4]
    power = np.array([[1., 1., 10.]])
    los = np.stack((np.cos(angles), np.sin(angles)), -1)
    relative_velocity = np.array([3., -4.])
    doppler = (los@relative_velocity)[None]
    result = associate_doppler(power, doppler, [10.], angles, np.eye(4),
                              [dict(position=[10., 0.], size=[10., 12.], heading=0.)])
    expected = np.average(los, axis=0, weights=power[0])
    np.testing.assert_allclose(result['projection'][0], expected, atol=1e-7)
    assert np.linalg.norm(result['projection'][0]) < 1.
    assert result['observed'][0] == pytest.approx(expected@relative_velocity)


def test_nonplanar_ego_motion_rejected_but_sensor_tilt_kept(tmp_path):
    frame = frame_fixture(tmp_path)
    frame['ego']['velocity'][2] = 1.
    with pytest.raises(ValueError, match='planar'):
        load_frame(frame, tmp_path)
    frame['ego']['velocity'][2] = 0.
    angle = .2
    tilted = np.array([[np.cos(angle), 0., np.sin(angle), 0.], [0., 1., 0., 0.],
                      [-np.sin(angle), 0., np.cos(angle), 0.], [0., 0., 0., 1.]])
    frame['ego']['pose'] = tilted.tolist()
    with pytest.raises(ValueError, match='vertical'):
        load_frame(frame, tmp_path)
    frame['ego']['pose'] = np.eye(4).tolist()
    frame['sensors']['radar']['T_ego_sensor'] = tilted.tolist()
    load_frame(frame, tmp_path)
    _, los = radar_geometry([1.], [0.], [tilted])
    assert np.linalg.norm(los[0, 0, 0]) == pytest.approx(np.cos(angle))


def test_asynchronous_radar_has_no_unsynchronized_object_doppler_supervision(tmp_path):
    frame = frame_fixture(tmp_path, 1)
    radar = frame['sensors']['radar']; radar['timestamp_s'] = .08
    radar['ego_pose_at_timestamp'] = np.eye(4).tolist()
    radar['ego_state_at_timestamp'] = dict(velocity_world=[8., 0., 0.], yaw_rate=0.)
    source = tmp_path/'frame.jsonl'; source.write_text(json.dumps(frame))
    manifest = prepare_frames(source, tmp_path/'prepared', history_frames=1)
    item = RadarDataset(manifest, 'train')[0]
    assert not item['agent_radial_mask'].any()
    assert not item['agent_state_mask'][:, 2].any()
    assert item['agent_state_mask'][0, 0]  # current position still has valid GT


def test_two_same_side_boundaries_cannot_invent_road_width(tmp_path):
    from radar_vla.records import road_targets
    frame = frame_fixture(tmp_path)
    frame['map']['lane_boundaries'][1]['points'] = [[0., 3., 0.], [50., 3., 0.]]
    result = road_targets(load_frame(frame, tmp_path)['map'])
    assert result['road_centerline_mask'].all()
    assert not result['road_mask'].item()
