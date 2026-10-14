"""The learned lane branch is trained, causal, and connected to planning tokens."""
import pytest
import torch

from radar_vla.model import ModelConfig, RadarVLAGrounder
from radar_vla.losses import loss_grounding
from test_radar_vla_model import example_batch


def lane_batch():
    batch = example_batch(1)
    x = torch.linspace(0., 40., 16)
    points = torch.stack([torch.stack((x, torch.full_like(x, y)), -1)
                          for y in (0., 2., -2.)])[None, None]
    batch.update(lane_points=points, lane_point_mask=torch.ones(1, 1, 3, 16, dtype=torch.bool),
                 lane_mask=torch.ones(1, 1, dtype=torch.bool),
                 lane_supervision_mask=torch.tensor([True]),
                 lane_boundary_known=torch.ones(1, 1, 2, dtype=torch.bool),
                 lane_adjacency=torch.zeros(1, 1, 1, 2),
                 lane_adjacency_mask=torch.ones(1, 1, 1, 2, dtype=torch.bool))
    return batch


def test_lane_branch_trains_geometry_and_informs_planning_tokens():
    model = RadarVLAGrounder(ModelConfig(hidden_dim=16, use_lanes=True))
    batch = lane_batch()
    model.mark_lane_supervision(batch)
    result = model(batch)
    assert result['lane_points'].shape == (1, 20, 3, 16, 2)
    losses = loss_grounding(result, batch)
    assert losses['lane_geometry'] > 0
    losses['total'].backward()
    assert sum(p.grad.abs().sum() for p in model.lane_detector.parameters() if p.grad is not None) > 0
    # A planner-only gradient must reach the lane geometry branch as well.
    model.zero_grad(set_to_none=True)
    model(batch)['radar_tokens'][..., 0].sum().backward()
    assert sum(p.grad.abs().sum() for p in model.lane_detector.parameters() if p.grad is not None) > 0


def test_lane_forward_never_reads_annotation_geometry_or_topology():
    model = RadarVLAGrounder(ModelConfig(hidden_dim=16, use_lanes=True)).eval()
    batch = lane_batch()
    with torch.no_grad():
        expected = model(batch)
        poisoned = dict(batch, lane_points=torch.full_like(batch['lane_points'], float('nan')),
                        lane_adjacency=torch.full_like(batch['lane_adjacency'], float('nan')),
                        metadata=[{'map': 'must never be read'}])
        actual = model(poisoned)
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)


def test_untrained_lane_branch_stays_unavailable_and_checkpoint_tracks_training():
    from radar_vla.lanes import decode_lanes
    model = RadarVLAGrounder(ModelConfig(hidden_dim=16, use_lanes=True)).eval()
    batch = lane_batch()
    assert not decode_lanes(model(batch), model.config)[0]['available']
    model.train()
    model.mark_lane_supervision(batch)
    loss_grounding(model(batch), batch)['total'].backward()
    restored = RadarVLAGrounder(model.config).eval()
    restored.load_state_dict(model.state_dict())
    assert decode_lanes(restored(batch), restored.config)[0]['available']


def test_boundary_and_topology_availability_requires_its_own_supervision():
    model = RadarVLAGrounder(ModelConfig(hidden_dim=16, use_lanes=True))
    batch = lane_batch()
    batch['lane_boundary_known'][..., 1] = False
    batch['lane_adjacency_mask'][..., 0] = False
    model.mark_lane_supervision(batch)
    prediction = model(batch)
    assert prediction['lane_boundary_available'].tolist() == [[True, False]]
    assert prediction['lane_topology_available'].tolist() == [[False, True]]
    model.eval()
    model.mark_lane_supervision(lane_batch())
    assert model(batch)['lane_boundary_available'].tolist() == [[True, False]]


@pytest.mark.parametrize('kwargs', [dict(lane_samples=1), dict(lane_samples=True),
                                    dict(lane_confidence=0.), dict(lane_adjacency_confidence=float('nan')),
                                    dict(lane_nms_distance_m=-1.), dict(use_lanes=1)])
def test_lane_configuration_rejects_invalid_geometry_and_thresholds(kwargs):
    with pytest.raises(ValueError):
        ModelConfig(**kwargs)
