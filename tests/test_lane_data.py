import copy
import json

import numpy as np
import pytest
import torch

from radar_vla import records
from radar_vla.data import RadarDataset, collate_batch, load_manifest
from radar_vla.synthetic import write_synthetic_dataset


def lane_map():
    return {
        'route_lane_id': 'a',
        'lane_centerlines': [
            dict(id='b', points=[[0., 4.], [30., 4.]],
                 left_boundary_id=None, right_boundary_id='shared', right_neighbor_id='a'),
            dict(id='a', points=[[0., 0.], [30., 0.]],
                 left_boundary_id='shared', right_boundary_id='right',
                 left_neighbor_id='b', right_neighbor_id=None),
            dict(id='opposite', points=[[30., -4.], [0., -4.]]),
        ],
        'lane_boundaries': [dict(id='shared', points=[[0., 2.], [30., 2.]]),
                            dict(id='right', points=[[0., -2.], [30., -2.]])],
    }


def test_lane_targets_retain_every_direction_and_actual_geometry():
    target = records.lane_targets(lane_map())
    assert target['lane_ids'] == ['a', 'b', 'opposite']
    assert target['lane_points'].shape == (3, 3, 16, 2)
    np.testing.assert_allclose(target['lane_points'][0, :, -1], [[30., 0.], [30., 2.], [30., -2.]])
    np.testing.assert_allclose(target['lane_points'][2, 0, [0, -1]], [[30., -4.], [0., -4.]])
    assert target['lane_mask'].all() and target['lane_supervision_mask']
    assert target['lane_point_mask'][0].all()
    assert not target['lane_point_mask'][2, 1:].any()
    np.testing.assert_array_equal(target['lane_boundary_known'], [[True, True], [True, True], [False, False]])


def test_adjacency_distinguishes_explicit_absent_from_unknown():
    target = records.lane_targets(lane_map())
    assert target['lane_adjacency'][0, 1, 0] == 1
    assert target['lane_adjacency'][1, 0, 1] == 1
    assert target['lane_adjacency_mask'][0].all()
    assert not target['lane_adjacency'][0, :, 1].any()
    assert not target['lane_adjacency_mask'][1, :, 0].any()
    assert not target['lane_adjacency_mask'][2].any()


def test_shared_boundary_supplies_positive_adjacency_without_inventing_negatives():
    source = lane_map()
    source['lane_centerlines'][0].pop('right_neighbor_id')
    source['lane_centerlines'][1].pop('left_neighbor_id')
    target = records.lane_targets(source)
    assert target['lane_adjacency_mask'][0, 1, 0]
    assert target['lane_adjacency'][0, 1, 0] == 1
    assert target['lane_adjacency_mask'][0, :, 0].sum() == 1
    assert target['lane_adjacency_mask'][1, :, 1].sum() == 1
    # A missing referenced lane/boundary cannot label all retained lanes absent.
    source['lane_centerlines'][1].update(left_neighbor_id='outside', right_boundary_id='unavailable')
    target = records.lane_targets(source)
    assert not target['lane_adjacency_mask'][0, :, 0].any()
    assert not target['lane_boundary_known'][0, 1]
    assert not target['lane_point_mask'][0, 2].any()


def test_shared_boundary_does_not_override_explicit_neighbor_absence():
    source = lane_map()
    source['lane_centerlines'][0].pop('right_neighbor_id')
    source['lane_centerlines'][1]['left_neighbor_id'] = None
    target = records.lane_targets(source)
    assert not target['lane_adjacency_mask'][1, :, 1].any()


def test_boundary_sampling_follows_its_lane_direction():
    source = lane_map()
    source['lane_boundaries'][0]['points'].reverse()
    target = records.lane_targets(source)
    np.testing.assert_allclose(target['lane_points'][0, 1, [0, -1]], [[0., 2.], [30., 2.]])
    np.testing.assert_allclose(target['lane_points'][1, 2, [0, -1]], [[0., 2.], [30., 2.]])


def test_shared_boundary_does_not_connect_disjoint_lane_sections():
    source = lane_map()
    source['lane_centerlines'][0].pop('right_neighbor_id')
    source['lane_centerlines'][1].pop('left_neighbor_id')
    source['lane_centerlines'][0]['points'] = [[40., 4.], [60., 4.]]
    target = records.lane_targets(source)
    assert not target['lane_adjacency_mask'][0, :, 0].any()
    assert not target['lane_adjacency_mask'][1, :, 1].any()


@pytest.mark.parametrize('source, known', [(None, False), ({}, False), ({'lane_centerlines': []}, True)])
def test_missing_lane_labels_are_not_observed_empty(source, known):
    target = records.lane_targets(source)
    assert target['lane_supervision_mask'].item() is known
    assert target['lane_points'].shape == (0, 3, 16, 2)
    assert target['lane_adjacency'].shape == (0, 0, 2)


def test_partial_map_coverage_cannot_supply_objectness_negatives():
    source = lane_map()
    source['lane_coverage_valid'] = False
    target = records.lane_targets(source)
    assert target['lane_mask'].all()
    assert not target['lane_supervision_mask']
    source['lane_coverage_valid'] = 'false'
    with pytest.raises(ValueError, match='lane_coverage_valid'):
        records.lane_targets(source)


@pytest.mark.parametrize('change, error', [
    (lambda m: m['lane_centerlines'].append(copy.deepcopy(m['lane_centerlines'][0])), 'unique'),
    (lambda m: m['lane_centerlines'][0].update(points=[[0., 0.], [0., 0.]]), 'distinct'),
    (lambda m: m['lane_centerlines'][0].update(points=[[0., 0.], [float('nan'), 0.]]), 'finite'),
    (lambda m: m['lane_centerlines'][0].update(left_neighbor_id=[]), 'left_neighbor_id'),
    (lambda m: m['lane_centerlines'][0].update(right_neighbor_id='b'), 'itself'),
])
def test_invalid_map_geometry_and_topology_raise_informative_errors(change, error):
    source = lane_map()
    change(source)
    with pytest.raises(ValueError, match=error):
        records.lane_targets(source)


def test_contradictory_explicit_neighbor_labels_are_rejected():
    source = lane_map()
    source['lane_centerlines'][0]['right_neighbor_id'] = None
    with pytest.raises(ValueError, match='contradict'):
        records.lane_targets(source)
    source = lane_map()
    source['lane_centerlines'][1]['right_neighbor_id'] = 'b'
    with pytest.raises(ValueError, match='both sides'):
        records.lane_targets(source)


def dataset_with_maps(tmp_path, maps, *, supervision=True):
    manifest = write_synthetic_dataset(tmp_path, scenes_per_split=1, frames_per_scene=len(maps))
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    for row, map_record in zip(rows, maps):
        row['map'] = map_record
    manifest.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    return RadarDataset(manifest, 'train', supervision=supervision)


def test_variable_lane_batches_pad_both_adjacency_axes_without_truncation(tmp_path):
    source = lane_map()
    single = dict(lane_centerlines=[dict(id='only', points=[[0., 0.], [10., 0.]])])
    dataset = dataset_with_maps(tmp_path, [source, single, {'lane_centerlines': []}])
    batch = collate_batch([dataset[0], dataset[1], dataset[2]])
    assert batch['lane_points'].shape == (3, 3, 3, 16, 2)
    assert batch['lane_adjacency'].shape == (3, 3, 3, 2)
    assert batch['lane_mask'].tolist() == [[True, True, True], [True, False, False], [False, False, False]]
    assert batch['lane_ids'][1] == ['only', '', '']
    assert not batch['lane_adjacency_mask'][1:].any()
    assert batch['lane_adjacency'][0, 0, 1, 0] == 1
    assert not batch['lane_point_mask'][1, 1:].any()


def test_inference_has_no_map_label_tensors_even_when_manifest_retains_annotations(tmp_path):
    dataset = dataset_with_maps(tmp_path, [lane_map()], supervision=False)
    row = dataset[0]
    assert not row['lane_supervision_mask']
    assert row['lane_points'].shape[0] == 0
    assert row['lane_ids'] == []
    assert not row['road_centerline_mask'].any()
    assert not row['road_mask'].any()
    batch = collate_batch([row])
    assert batch['lane_points'].shape == (1, 0, 3, 16, 2)
    assert not torch.any(batch['lane_mask'])


def test_map_normalization_preserves_unknown_lane_coverage_and_topology():
    transform = np.eye(4)
    transform[0, 3] = -5.
    assert not records.lane_targets(records.normalize_map({}, transform))['lane_supervision_mask']
    source = records.normalize_map(lane_map(), transform)
    target = records.lane_targets(source)
    np.testing.assert_allclose(target['lane_points'][0, 0, 0], [-5., 0.])
    assert target['lane_adjacency'][0, 1, 0] == 1


def test_dataset_lane_sampling_matches_configurable_head_resolution(tmp_path):
    manifest = write_synthetic_dataset(tmp_path, scenes_per_split=1, frames_per_scene=1)
    dataset = RadarDataset(manifest, 'train', lane_samples=7)
    assert dataset[0]['lane_points'].shape == (1, 3, 7, 2)
    with pytest.raises(ValueError, match='lane_samples'):
        RadarDataset(manifest, 'train', lane_samples=1)


def test_map_topology_validated_before_starting_training(tmp_path):
    manifest = write_synthetic_dataset(tmp_path, scenes_per_split=1, frames_per_scene=1)
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    rows[0]['map']['lane_centerlines'][0]['left_neighbor_id'] = []
    manifest.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    with pytest.raises(ValueError, match='left_neighbor_id'):
        load_manifest(manifest)
    with pytest.raises(ValueError, match='left_neighbor_id'):
        records.normalize_map(rows[0]['map'], np.eye(4))
