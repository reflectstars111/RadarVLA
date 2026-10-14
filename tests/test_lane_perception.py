"""Radar lane learning uses variable instances, masked geometry and topology."""
import json
from types import SimpleNamespace

import pytest
import torch


def lane_config(**kwargs):
    return SimpleNamespace(**dict(dict(lane_confidence=.5, lane_boundary_confidence=.5,
        lane_adjacency_confidence=.5, lane_nms_distance_m=1.), **kwargs))


def lane_predictions(offsets=(0., 3.6), samples=16):
    n = len(offsets)
    points = torch.zeros(1, n, 3, samples, 2)
    points[..., 0] = torch.linspace(0., 30., samples)
    for i, y in enumerate(offsets):
        points[0, i, :, :, 1] = torch.tensor([y, y + 1.8, y - 1.8])[:, None]
    return dict(lane_points=points, lane_logits=torch.full((1, n), 8.),
        lane_boundary_logits=torch.full((1, n, 2), 8.),
        lane_adjacency_source=torch.zeros(1, n, 2, 4),
        lane_adjacency_target=torch.zeros(1, n, 4),
        lane_adjacency_none_logits=torch.full((1, n, 2), -8.))


def lane_batch(points):
    b, n, _, s, _ = points.shape
    mask = torch.ones(b, n, n, 2, dtype=torch.bool)
    adjacency = torch.zeros(b, n, n, 2)
    if n >= 2:
        adjacency[:, 0, 1, 0] = 1
        adjacency[:, 1, 0, 1] = 1
    return dict(lane_points=points.clone(), lane_mask=torch.ones(b, n, dtype=torch.bool),
        lane_point_mask=torch.ones(b, n, 3, s, dtype=torch.bool),
        lane_supervision_mask=torch.ones(b, dtype=torch.bool), lane_adjacency=adjacency,
        lane_adjacency_mask=mask, lane_boundary_known=torch.ones(b, n, 2, dtype=torch.bool))


def test_lane_head_uses_radar_features_calibrated_geometry_and_backpropagates():
    from radar_vla.lanes import DenseLaneDetector, lane_losses
    torch.manual_seed(9)
    head = DenseLaneDetector(16, samples=16)
    spatial = torch.randn(1, 16, 3, 4, requires_grad=True)
    tokens = torch.randn(1, 6, 16)
    ego = torch.randn(1, 16)
    batch = lane_batch(lane_predictions()['lane_points'])
    batch.update(range_m=torch.linspace(0, 30, 12)[None],
                 azimuth_rad=torch.linspace(-1., 1., 16)[None])
    output = head(spatial, tokens, ego, batch)
    assert output['lane_logits'].shape == (1, 12)
    assert output['lane_points'].shape == (1, 12, 3, 16, 2)
    assert output['lane_features'].shape == (1, 12, 16)
    # The pairwise topology matrix must not scale with every spatial pair.
    assert output['lane_adjacency_source'].shape[:3] == (1, 12, 2)
    losses = lane_losses(output, batch)
    assert all(torch.isfinite(value) for value in losses.values())
    losses['lane_total'].backward()
    assert spatial.grad.abs().sum() > 0
    for name in ('object_head', 'geometry_head', 'boundary_head', 'adjacency_source',
                 'adjacency_target', 'adjacency_none'):
        assert sum(p.grad.abs().sum() for p in getattr(head, name).parameters()) > 0
    other = head(spatial.detach() + 2., tokens, ego, batch)
    assert not torch.allclose(other['lane_points'], output['lane_points'])


def test_lane_loss_matches_every_lane_and_is_permutation_invariant():
    from radar_vla.lanes import lane_losses
    output = lane_predictions(tuple(float(i) * 3.6 for i in range(11)))
    batch = lane_batch(output['lane_points'])
    output['lane_points'] = output['lane_points'] + .4
    first = lane_losses(output, batch)
    order = torch.arange(10, -1, -1)
    reverse = {key: (value[:, order][:, :, order] if key.startswith('lane_adjacency')
                    else value[:, order] if value.ndim >= 2 else value)
               for key, value in batch.items()}
    second = lane_losses(output, reverse)
    assert first['lane_geometry'] > 0
    for key in first:
        torch.testing.assert_close(first[key], second[key])


def test_lane_loss_ignores_unknown_boundaries_topology_and_unlabelled_frames():
    from radar_vla.lanes import lane_losses
    output = {k: v.requires_grad_() for k, v in lane_predictions().items()}
    batch = lane_batch(output['lane_points'].detach())
    batch['lane_point_mask'][:, :, 1:] = False
    batch['lane_boundary_known'][:] = False
    batch['lane_adjacency_mask'][:] = False
    batch['lane_points'][:, :, 1:] = float('nan')
    losses = lane_losses(output, batch)
    assert losses['lane_boundary'] == losses['lane_adjacency'] == losses['lane_geometry'] == 0
    batch['lane_supervision_mask'][:] = False
    losses = lane_losses(output, batch)
    assert losses['lane_total'] == 0
    losses['lane_total'].backward()
    assert all(v.grad is not None and v.grad.abs().sum() == 0 for v in output.values())


def test_reversed_boundary_points_do_not_penalize_same_geometry():
    from radar_vla.lanes import lane_losses
    output = lane_predictions()
    batch = lane_batch(output['lane_points'])
    batch['lane_points'][:, :, 1:] = batch['lane_points'][:, :, 1:].flip(-2)
    assert lane_losses(output, batch)['lane_geometry'] == 0


def test_lane_capacity_overflow_is_explicit_instead_of_dropping_truth():
    from radar_vla.lanes import lane_losses
    with pytest.raises(ValueError, match='candidates'):
        lane_losses(lane_predictions((0.,)), lane_batch(lane_predictions()['lane_points']))


def test_decode_lanes_keeps_more_than_eight_with_no_label_dependency():
    from radar_vla.lanes import decode_lanes
    row = decode_lanes(lane_predictions(tuple(i * 3.6 for i in range(12))), lane_config())[0]
    assert row['available'] and len(row['lanes']) == 12
    assert row['lanes'][0]['centerline'][0] == [0., 0.]
    assert row['lanes'][0]['left_boundary'][0] == pytest.approx([0., 1.8])
    json.dumps(row, allow_nan=False)


def test_lane_dedup_preserves_adjacent_and_opposite_direction_instances():
    from radar_vla.lanes import decode_lanes
    output = lane_predictions((0., .1, .7, 0.))
    output['lane_logits'][0] = torch.tensor([9., 8., 7., 6.])
    # A distinct narrow neighboring lane shares a boundary with lane 0.
    output['lane_points'][0, 0, 1, :, 1] = .35
    output['lane_points'][0, 2, 2, :, 1] = .35
    output['lane_points'][0, 3] = output['lane_points'][0, 3].flip(-2)
    row = decode_lanes(output, lane_config())[0]
    assert len(row['lanes']) == 3


def test_decode_topology_uses_selected_ids_with_reciprocal_edges():
    from radar_vla.lanes import decode_lanes
    output = lane_predictions((0., 3.6, 7.2))
    output['lane_logits'][0, 2] = -8.
    output['lane_adjacency_source'][0, 0, 0, 0] = 10.
    output['lane_adjacency_target'][0, 1, 0] = 10.
    output['lane_adjacency_source'][0, 1, 1, 1] = 10.
    output['lane_adjacency_target'][0, 0, 1] = 10.
    row = decode_lanes(output, lane_config())[0]
    a, b = row['lanes']
    assert a['left_neighbor_id'] == b['lane_id']
    assert b['right_neighbor_id'] == a['lane_id']
    assert a['left_neighbor_status'] == b['right_neighbor_status'] == 'linked'
    assert a['right_neighbor_id'] is None
    assert a['right_neighbor_status'] == 'unknown'


def test_decode_unknown_none_missing_boundary_and_untrained_status_are_distinct():
    from radar_vla.lanes import decode_lanes
    output = lane_predictions((0.,))
    output['lane_boundary_logits'][0, 0, 1] = -8.
    output['lane_adjacency_none_logits'][0, 0, 0] = 8.
    row = decode_lanes(output, lane_config())[0]
    assert row['lanes'][0]['right_boundary'] is None
    assert row['lanes'][0]['left_neighbor_status'] == 'none'
    assert row['lanes'][0]['right_neighbor_status'] == 'unknown'
    output['lane_available'] = torch.tensor([False])
    unavailable = decode_lanes(output, lane_config())[0]
    assert not unavailable['available'] and unavailable['lanes'] == []
    assert not decode_lanes({'risk': torch.zeros(2, 5)}, lane_config())[1]['available']


def test_decode_rejects_nonfinite_lane_geometry():
    from radar_vla.lanes import decode_lanes
    output = lane_predictions()
    output['lane_points'][0, 0, 0, 0, 0] = float('nan')
    with pytest.raises(ValueError, match='finite'):
        decode_lanes(output, lane_config())


def test_lanes_without_longitudinal_overlap_are_not_lateral_neighbors():
    from radar_vla.lanes import decode_lanes
    output = lane_predictions((0., 3.6))
    output['lane_points'][0, 1, ..., 0] += 100.
    output['lane_adjacency_source'].fill_(10.)
    output['lane_adjacency_target'].fill_(10.)
    row = decode_lanes(output, lane_config())[0]
    assert all(lane['left_neighbor_id'] is None and lane['right_neighbor_id'] is None
               for lane in row['lanes'])


def test_narrow_lanes_with_learned_adjacency_survive_dedup_without_boundaries():
    from radar_vla.lanes import decode_lanes
    output = lane_predictions((0., .7))
    output['lane_boundary_logits'].fill_(-8.)
    output['lane_adjacency_source'][0, 0, 0, 0] = 10.
    output['lane_adjacency_target'][0, 1, 0] = 10.
    output['lane_adjacency_source'][0, 1, 1, 1] = 10.
    output['lane_adjacency_target'][0, 0, 1] = 10.
    row = decode_lanes(output, lane_config())[0]
    assert len(row['lanes']) == 2
    assert row['lanes'][0]['left_neighbor_id'] == row['lanes'][1]['lane_id']


def test_untrained_topology_and_boundary_sides_remain_unknown():
    from radar_vla.lanes import decode_lanes
    output = lane_predictions()
    output['lane_adjacency_source'].fill_(10.)
    output['lane_adjacency_target'].fill_(10.)
    output['lane_adjacency_none_logits'].fill_(8.)
    output['lane_topology_available'] = torch.tensor([[False, False]])
    output['lane_boundary_available'] = torch.tensor([[True, False]])
    row = decode_lanes(output, lane_config())[0]
    for lane in row['lanes']:
        assert lane['left_neighbor_status'] == lane['right_neighbor_status'] == 'unknown'
        assert lane['left_neighbor_id'] is None and lane['right_neighbor_id'] is None
        assert lane['left_boundary'] is not None and lane['right_boundary'] is None
