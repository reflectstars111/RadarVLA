"""Response replay keeps inference contracts while excluding teacher labels."""

import pytest
import torch

from radar_vla.data import RadarDataset
from radar_vla.model import ModelConfig, RadarVLAGrounder
from radar_vla.response_rl import _observation
from radar_vla.risk_contract import SHORT_HORIZONS
from radar_vla.synthetic import write_synthetic_dataset


@pytest.fixture
def sample(tmp_path):
    manifest = write_synthetic_dataset(
        tmp_path / 'data', scenes_per_split=1, frames_per_scene=1,
        range_bins=8, azimuth_bins=8,
    )
    # Include real teacher tensors to exercise the observation whitelist itself.
    return RadarDataset(manifest, 'train')[0]


def short_model():
    return RadarVLAGrounder(ModelConfig(
        hidden_dim=8, num_heads=2, num_queries=2,
        risk_horizon_schema=SHORT_HORIZONS, detection_class_schema='agent_v1',
    )).eval()


@torch.no_grad()
def test_krsv2_grounder_accepts_filtered_response_observation(sample):
    sample['risk_horizon_schema_id'] = torch.tensor(1)
    sample['detection_class_schema_id'] = torch.tensor(1)

    observation = _observation([sample], 0, 'cpu')
    output = short_model()(observation)

    assert output['risk'].shape == (1, 5)
    assert torch.isfinite(output['risk']).all()
    assert output['detection_class_logits'].shape[-1] == 1


@pytest.mark.parametrize('field, error', [
    ('risk_horizon_schema_id', 'Batch and model Risk horizons differ'),
    ('detection_class_schema_id', 'Detector input class schema differs'),
])
@torch.no_grad()
def test_response_observation_preserves_mismatches_for_model_rejection(sample, field, error):
    sample['risk_horizon_schema_id'] = torch.tensor(1)
    sample['detection_class_schema_id'] = torch.tensor(1)
    sample[field] = torch.tensor(0)

    with pytest.raises(ValueError, match=error):
        short_model()(_observation([sample], 0, 'cpu'))


@pytest.mark.parametrize('include_schema_metadata', [False, True])
@torch.no_grad()
def test_legacy_response_observation_remains_compatible(sample, include_schema_metadata):
    if not include_schema_metadata:
        sample.pop('risk_horizon_schema_id')
        sample.pop('detection_class_schema_id')
    grounder = RadarVLAGrounder(ModelConfig(
        hidden_dim=8, num_heads=2, num_queries=2,
    )).eval()

    output = grounder(_observation([sample], 0, 'cpu'))

    assert output['risk'].shape == (1, 5)
    assert torch.isfinite(output['risk']).all()


def test_response_observation_excludes_ground_truth_and_future_labels(sample):
    assert sample['agent_future'].numel() > 0
    assert sample['ego_future'].numel() > 0
    assert sample['risk_target'].numel() == 5
    teacher_fields = (
        'agent_state', 'agent_future', 'agent_future_mask',
        'ego_future', 'ego_future_mask', 'risk_target', 'risk_mask',
        'agent_risk', 'agent_risk_mask', 'road_centerline', 'road_width',
        'metadata',
    )

    observation = _observation([sample], 0, 'cpu')

    assert not set(teacher_fields).intersection(observation)
    # Future knot times specify the output grid; they contain no future states.
    assert observation['future_times_s'].tolist() == [[.5, 1., 1.5, 2., 2.5, 3.]]

    changed = dict(sample)
    for field in teacher_fields:
        if torch.is_tensor(changed[field]):
            changed[field] = torch.zeros_like(changed[field])
    changed['metadata'] = dict(map={'teacher_only': True})
    filtered = _observation([changed], 0, 'cpu')
    assert observation.keys() == filtered.keys()
    for field in observation:
        torch.testing.assert_close(observation[field], filtered[field])
