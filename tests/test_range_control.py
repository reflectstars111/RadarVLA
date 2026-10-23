import importlib.util
from pathlib import Path

import pytest
import torch

spec = importlib.util.spec_from_file_location('range_control', Path(__file__).parents[1] / 'scripts' / 'train_range_control.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def example():
    r = 6
    xy = torch.arange(2 * 4 * r * 3 * 2).reshape(2, 4, r, 3, 2).float()
    return dict(range_m=torch.tensor([[0., 10., 30., 50., 70., 110.]]).expand(2, -1),
                radar=torch.arange(2 * 4 * 2 * r * 3).reshape(2, 4, 2, r, 3).float(),
                radar_doppler_valid=torch.ones(2, 4, r, 3, dtype=torch.bool),
                radar_cartesian=xy, radar_los=xy.clone(),
                radar_pose=torch.eye(4)[None, None].expand(2, 4, 4, 4),
                risk_target=torch.tensor([[0., 0., 1., 0., 2.]]).expand(2, -1),
                agent_state=torch.tensor([[[51., 0., -3., 0., 0.]]]).expand(2, -1, -1))


def test_crop_preserves_alignment_and_supervision():
    original = example()
    cropped = module.crop_batch(original)
    assert cropped['radar'].shape == (2, 4, 2, 4, 3)
    torch.testing.assert_close(cropped['radar'], original['radar'][:, :, :, :4])
    for key in ['radar_doppler_valid', 'radar_cartesian', 'radar_los']:
        torch.testing.assert_close(cropped[key], original[key][:, :, :4])
    for key in ['agent_state', 'risk_target', 'radar_pose']:
        assert cropped[key] is original[key]
    # A boundary-intersecting GT with center beyond 50 m must not be deleted.
    assert cropped['agent_state'][0, 0, 0] == 51.
    assert original['radar'].shape[3] == 6
    assert cropped['range_m'][0, -1] == 50.


def test_full_range_is_identity_in_value():
    original = example()
    result = module.crop_batch(original, 200.)
    for key in original:
        torch.testing.assert_close(result[key], original[key])


def test_incompatible_axes_are_rejected():
    original = example()
    original['range_m'] = original['range_m'].clone()
    original['range_m'][1, 3] = 60.
    with pytest.raises(ValueError, match='equal'):
        module.crop_batch(original)


def test_misaligned_geometry_is_rejected():
    original = example()
    original['radar_los'] = original['radar_los'][:, :, :-1]
    with pytest.raises(ValueError, match='aligned'):
        module.crop_batch(original)
