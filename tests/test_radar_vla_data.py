import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from radar_vla.data import RadarDataset, collate_batch, load_manifest, prepare_labels
from radar_vla.geometry import box_distance, build_risk_labels, nominal_rollout
from radar_vla.synthetic import write_synthetic_dataset


def record():
    times = np.arange(.5, 3.01, .5).tolist()
    return {
        'future_times_s': times,
        'ego': {'velocity': [10., 0.], 'acceleration': [-4., 0.], 'yaw_rate': 0.,
                'box_size': [4., 2.], 'future_xy': [[0., 0.]] * 6},
        'agents': [{'id': 'leader', 'position': [16., 0.], 'velocity': [0., 0.],
                    'heading': 0., 'size': [4., 2.], 'future_xy': [[16., 0.]] * 6,
                    'future_yaw': [0.] * 6, 'future_valid': [True] * 6}],
    }


def test_exact_oriented_box_distance():
    assert box_distance([0, 0], 0, [4, 2], [7, 0], 0, [4, 2]) == pytest.approx(3.)
    assert box_distance([0, 0], 0, [4, 2], [0, 0], np.pi / 4, [4, 2]) == 0
    assert box_distance([0, 0], 0, [2, 2], [3, 3], 0, [2, 2]) == pytest.approx(np.sqrt(2))


def test_counterfactual_ignores_actual_braked_ego_future():
    source = record()
    before = build_risk_labels(source)
    source['ego']['future_xy'] = [[999., 999.]] * 6
    assert build_risk_labels(source) == before
    assert before['pcol'] == [0., 1., 1.]
    assert before['pcol_mask'] == [True] * 3
    assert before['dmin'] == 0.
    assert before['areq_mask'] and 0 < before['areq'] <= 8.


def test_missing_future_is_unknown_not_safe():
    source = record()
    source['agents'][0]['future_valid'] = [False] * 6
    labels = build_risk_labels(source)
    assert labels['pcol_mask'] == [False] * 3
    assert not labels['dmin_mask'] and not labels['areq_mask']


def test_positive_conflict_remains_known_with_later_missing_observations():
    source = record()
    source['agents'][0]['future_valid'] = [True, True, True, False, False, False]
    labels = build_risk_labels(source)
    assert labels['pcol'] == [0., 1., 1.]
    assert labels['pcol_mask'] == [True, True, True]
    assert not labels['areq_mask']


def test_partial_coverage_and_empty_scene_distance_cap():
    source = record()
    source['agents'] = []
    labels = build_risk_labels(source)
    assert labels['pcol'] == [0.] * 3
    assert labels['dmin'] == 80. and labels['dmin_mask']
    assert labels['areq'] == 0. and labels['areq_mask']
    source['future_times_s'] = [.5, 1.]
    labels = build_risk_labels(source)
    assert labels['pcol_mask'] == [True, False, False]
    assert not labels['dmin_mask']


def test_unavoidable_braking_has_no_fabricated_cap_target():
    source = record()
    source['agents'][0]['position'] = [1., 0.]
    source['agents'][0]['future_xy'] = [[1., 0.]] * 6
    labels = build_risk_labels(source)
    assert labels['braking_feasible'] is False
    assert labels['areq_mask'] is False
    assert labels['areq'] == 0.


def test_rollout_stops_without_reversing_and_turns():
    xy, yaw = nominal_rollout([10., 0.], 0., [1., 2., 3.], deceleration=10.)
    np.testing.assert_allclose(xy[:, 0], [5., 5., 5.])
    xy, yaw = nominal_rollout([2., 0.], np.pi / 2, [1.])
    np.testing.assert_allclose(xy[0], [4 / np.pi, 4 / np.pi], atol=1e-6)


def test_almost_straight_braking_avoids_ctrv_cancellation():
    xy, _ = nominal_rollout([10., 0.], 1e-8, [1., 2., 3.], deceleration=2.)
    np.testing.assert_allclose(xy[:, 0], [9., 16., 21.], atol=1e-8)
    assert np.all(xy[:, 1] > 0)


def test_stopped_rollout_freezes_heading_and_stationary_collision_is_known():
    xy, yaw = nominal_rollout([2., 0.], .2, [1., 2., 3.], deceleration=2.)
    np.testing.assert_allclose(xy, np.broadcast_to(xy[0], xy.shape))
    np.testing.assert_allclose(yaw, [.2, .2, .2])
    source = record()
    source['ego']['velocity'] = [0., 0.]
    source['agents'][0]['position'] = [1., 0.]
    source['agents'][0]['future_valid'] = [False] * 6
    labels = build_risk_labels(source)
    assert labels['pcol'] == [1., 1., 1.]
    assert labels['pcol_mask'] == [True] * 3
    assert labels['dmin'] == 0. and labels['dmin_mask']
    assert not labels['areq_mask']
    assert labels['braking_feasible'] is False


def test_horizon_endpoint_collision_and_invalid_gap():
    source = record()
    source['ego']['velocity'] = [0., 0.]
    agent = source['agents'][0]
    agent['future_xy'] = [[10., 0.], [4., 0.], [10., 0.], [10., 0.], [10., 0.], [10., 0.]]
    labels = build_risk_labels(source)
    assert labels['pcol'] == [1., 1., 1.]
    agent['future_valid'][1] = False
    labels = build_risk_labels(source)
    assert labels['pcol'] == [0., 0., 0.]
    assert labels['pcol_mask'] == [False] * 3
    assert not labels['dmin_mask'] and not labels['areq_mask']


def test_synthetic_roundtrip_and_relocated_label_manifest(tmp_path):
    manifest = write_synthetic_dataset(tmp_path / 'source', scenes_per_split=1, frames_per_scene=2)
    dataset = RadarDataset(manifest, 'train', max_agents=4)
    assert len(dataset) == 2
    item = dataset[0]
    assert item['radar'].shape == (3, 2, 32, 24)
    assert item['agent_state'].shape == (4, 5)
    assert item['risk_target'].shape == (5,)
    assert torch.isfinite(item['radar']).all()
    assert not torch.equal(dataset[0]['radar'], dataset[1]['radar'])
    batch = collate_batch([dataset[0], dataset[1]])
    assert batch['radar'].shape[0] == 2
    assert isinstance(batch['instruction'], list)
    relocated = tmp_path / 'labels' / 'manifest.jsonl'
    prepare_labels(manifest, relocated)
    assert torch.equal(RadarDataset(relocated, 'train', max_agents=4)[0]['radar'], item['radar'])


def test_reject_scene_leakage_duplicate_ids_and_folded_doppler(tmp_path):
    manifest = write_synthetic_dataset(tmp_path, scenes_per_split=1, frames_per_scene=1)
    records = [json.loads(line) for line in manifest.read_text().splitlines()]
    bad = copy.deepcopy(records)
    bad[1]['scene_id'] = bad[0]['scene_id']
    path = tmp_path / 'bad.jsonl'
    path.write_text('\n'.join(json.dumps(r) for r in bad))
    with pytest.raises(ValueError, match='scene'):
        load_manifest(path)
    bad = copy.deepcopy(records)
    bad[1]['sample_id'] = bad[0]['sample_id']
    path.write_text('\n'.join(json.dumps(r) for r in bad))
    with pytest.raises(ValueError, match='sample_id'):
        load_manifest(path)
    bad = copy.deepcopy(records)
    bad[0]['sensors']['radar']['folded_doppler'] = bad[0]['sensors']['radar'].pop('unfolded_doppler')
    path.write_text('\n'.join(json.dumps(r) for r in bad))
    with pytest.raises(ValueError, match='unfolded_doppler'):
        load_manifest(path)


def test_nonfinite_radar_and_excess_agents_rejected(tmp_path):
    manifest = write_synthetic_dataset(tmp_path, scenes_per_split=1, frames_per_scene=1)
    with pytest.raises(ValueError, match='max_agents'):
        RadarDataset(manifest, 'train', max_agents=0)
    rec = json.loads(manifest.read_text().splitlines()[0])
    radar_path = tmp_path / rec['sensors']['radar']['power']
    values = np.load(radar_path)
    values[0, 0, 0] = np.nan
    np.save(radar_path, values)
    with pytest.raises(ValueError, match='finite'):
        RadarDataset(manifest, 'train')[0]


def test_relative_doppler_and_missing_ego_action_mask(tmp_path):
    manifest = write_synthetic_dataset(tmp_path, scenes_per_split=1, frames_per_scene=1)
    records = [json.loads(line) for line in manifest.read_text().splitlines()]
    records[0]['ego'].pop('future_xy')
    records[0]['ego'].pop('future_valid')
    manifest.write_text('\n'.join(json.dumps(r) for r in records))
    sample = RadarDataset(manifest, 'train')[0]
    assert sample['agent_state'][0, 2] < 0
    assert sample['agent_state'][0, 3] == 0
    assert not sample['ego_future_mask'].any()
    assert sample['agent_supervision_mask']
    extra = copy.deepcopy(records[0]['agents'][0])
    extra['id'] = 'second'
    records[0]['agents'].append(extra)
    manifest.write_text('\n'.join(json.dumps(r) for r in records))
    with pytest.raises(ValueError, match='max_agents'):
        RadarDataset(manifest, 'train', max_agents=1)


def test_synthetic_refuses_to_overwrite_existing_data(tmp_path):
    marker = tmp_path / 'keep.txt'
    marker.write_text('existing data')
    with pytest.raises(FileExistsError, match='nonempty'):
        write_synthetic_dataset(tmp_path)
    assert marker.read_text() == 'existing data'
    assert not (tmp_path / 'radar').exists()


@pytest.mark.parametrize('pcol, mask, message', [
    ([.3, 1., 1.], [True] * 3, 'binary'),
    ([1., 0., 1.], [True] * 3, 'cumulative'),
    ([1., 0., 0.], [True, False, True], 'cumulative'),
])
def test_cached_collision_labels_are_binary_and_known_horizons_monotonic(tmp_path, pcol, mask, message):
    manifest = write_synthetic_dataset(tmp_path, scenes_per_split=1, frames_per_scene=1)
    records = [json.loads(line) for line in manifest.read_text().splitlines()]
    records[0]['risk_label']['pcol'] = pcol
    records[0]['risk_label']['pcol_mask'] = mask
    manifest.write_text('\n'.join(json.dumps(r) for r in records))
    with pytest.raises(ValueError, match=message):
        load_manifest(manifest)
