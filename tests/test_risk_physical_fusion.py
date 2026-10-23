"""The opt-in Risk readout consumes predictions, never target annotations."""
from dataclasses import asdict
import importlib.util
from pathlib import Path
import sys

import pytest
import torch

from radar_vla.model import ModelConfig, RadarVLAGrounder
from test_radar_vla_model import example_batch


def make_model(**kwargs):
    torch.manual_seed(731)
    return RadarVLAGrounder(ModelConfig(hidden_dim=16, num_heads=4, num_queries=3,
                                      risk_input='predicted_geometry_v1', **kwargs)).eval()


def test_fusion_uses_real_predictions_and_keeps_radar_gradients():
    model = make_model()
    batch = example_batch(1)
    output = model(batch)
    output['risk'].sum().backward()
    for module in (model.encoder, model.risk_attention, model.risk_head, model.risk_fusion):
        gradients = [p.grad for p in module.parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients)
        assert sum(g.abs().sum().item() for g in gradients) > 0
    assert all(p.grad is None for p in model.detector.parameters())
    assert output['detection_logits'].shape == (1, 40)
    assert output['agent_state'] is output['detection_state']


def test_grid_metadata_is_opt_in_and_preserves_legacy_output_contract():
    batch = example_batch(1)
    old = make_model()(batch)
    new = make_model(detection_assignment_policy='local_quality_v1')(batch)
    assert 'detection_grid_shape' not in old
    assert new['detection_grid_shape'].tolist() == [4, 5]
    for key in old:
        torch.testing.assert_close(old[key], new[key], rtol=0, atol=0)


def test_fusion_ignores_all_gt_and_annotation_domain_metadata():
    model, batch = make_model(), example_batch(1)
    before = model(batch)
    changed = {k: (torch.full_like(v, float('nan')) if k in
                   ('agent_state', 'agent_future', 'risk_target') else v)
               for k, v in batch.items()}
    changed.update(annotation_domain=torch.full((1, 6), float('nan')),
                   agent_mask=torch.zeros_like(batch['agent_mask']))
    after = model(changed)
    assert before.keys() == after.keys()
    for key in before:
        torch.testing.assert_close(before[key], after[key], atol=0, rtol=0)


def test_all_low_confidence_candidates_are_available_despite_description_caps():
    model = make_model(max_agents=1, detection_max_candidates=1, detection_max_objects=1)
    with torch.no_grad():
        model.detector.object_head.weight.zero_()
        model.detector.object_head.bias.fill_(-12.)
    batch = example_batch(1)
    output = model(batch)
    features, mask = model.risk_fusion.physical_features(output, batch)
    assert features.shape[:2] == mask.shape == (1, 40)
    assert mask.all() and (output['detection_logits'].sigmoid() < .5).all()
    changed = dict(output, detection_state=output['detection_state'].clone(),
                   detection_future=output['detection_future'].clone())
    changed['detection_state'][:, -1, :2] += 70.
    changed['detection_future'][:, -1] += 70.
    # Hold the radar and Ego contexts fixed: the final low-score candidate alone
    # must reach the Risk head, even though it would be removed by public decode.
    tokens = output['radar_tokens']
    ego = torch.zeros(1, model.config.hidden_dim)
    first = model.risk_from_context(tokens, ego, predictions=output, batch=batch)['risk']
    second = model.risk_from_context(tokens, ego, predictions=changed, batch=batch)['risk']
    assert not torch.allclose(first, second, atol=1e-8, rtol=1e-7)
    other = make_model(max_agents=99)
    other.load_state_dict(model.state_dict())
    torch.testing.assert_close(output['risk'], other(batch)['risk'], atol=0, rtol=0)


def test_candidate_mask_excludes_invalid_predictions_without_nan():
    model, batch = make_model(), example_batch(1)
    output = model(batch)
    mask = torch.ones_like(output['detection_logits'], dtype=torch.bool)
    mask[:, -1] = False
    output = dict(output, detection_candidate_mask=mask)
    clean, valid = model.risk_fusion.physical_features(output, batch)
    damaged = dict(output)
    for key in ('detection_state', 'detection_size', 'detection_heading',
                'detection_future', 'detection_future_velocity', 'detection_logits'):
        damaged[key] = output[key].clone()
        damaged[key][:, -1] = float('nan')
    dirty, dirty_valid = model.risk_fusion.physical_features(damaged, batch)
    torch.testing.assert_close(clean, dirty, atol=0, rtol=0)
    assert torch.equal(valid, dirty_valid)
    a = model.risk_from_context(output['radar_tokens'], torch.zeros(1, 16),
                                predictions=output, batch=batch)['risk']
    b = model.risk_from_context(output['radar_tokens'], torch.zeros(1, 16),
                                predictions=damaged, batch=batch)['risk']
    torch.testing.assert_close(a, b, atol=0, rtol=0)


def test_all_empty_candidate_mask_is_stable_in_real_forward():
    model = make_model(detection_candidate_policy='supervised_sector_v1',
                       detection_candidate_range_m=.001)
    batch = example_batch(2)
    batch['radar_pose'] = torch.eye(4).expand(2, 3, 4, 4).clone()
    output = model(batch)
    assert not output['detection_candidate_mask'].any()
    assert torch.isfinite(output['risk']).all()
    output['risk'].sum().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    assert any(p.grad is not None for p in model.risk_fusion.parameters())


def test_physical_features_use_ground_velocity_relative_to_ego_and_ctrv():
    model, batch = make_model(), example_batch(1)
    output = model(batch)
    features, _ = model.risk_fusion.physical_features(output, batch)
    moving = dict(batch, ego_velocity=torch.tensor([[0., 10.]]))
    altered, _ = model.risk_fusion.physical_features(output, moving)
    assert not torch.allclose(features, altered)
    assert not features.requires_grad
    # Feature construction is independent of the model's learned Ego embedding.
    risk = model.risk_from_context(output['radar_tokens'], torch.zeros(1, 16),
                                   predictions=output, batch=batch)['risk']
    changed = model.risk_from_context(output['radar_tokens'], torch.zeros(1, 16),
                                      predictions=output, batch=moving)['risk']
    assert not torch.allclose(risk, changed)


def test_geometry_contract_matches_observed_stationary_and_turning_motion():
    from radar_vla.risk_fusion import nominal_ego_motion, point_to_oriented_box_distance
    times = torch.tensor([[.5, 1., 2.]])
    position, velocity = nominal_ego_motion(torch.tensor([[3., 4.]]), torch.zeros(1), times)
    torch.testing.assert_close(position, times[..., None] * torch.tensor([3., 4.]))
    torch.testing.assert_close(velocity, torch.tensor([3., 4.]).expand(1, 3, 2))
    position, velocity = nominal_ego_motion(torch.zeros(1, 2), torch.ones(1), times)
    assert torch.equal(position, torch.zeros_like(position))
    assert torch.equal(velocity, torch.zeros_like(velocity))
    from radar_vla.geometry import nominal_rollout
    position, _ = nominal_ego_motion(torch.tensor([[3., 4.]]), torch.tensor([.7]), times)
    expected, _ = nominal_rollout([3., 4.], .7, times[0].numpy())
    torch.testing.assert_close(position[0], torch.tensor(expected).float(), atol=1e-6, rtol=1e-6)
    distance = point_to_oriented_box_distance(torch.tensor([[[3., 0.], [0., 3.], [0., 0.]]]),
                                            torch.zeros(1, 3), torch.tensor([[[4., 2.]]]).expand(1, 3, 2))
    torch.testing.assert_close(distance, torch.tensor([[1., 2., 0.]]))


def test_stationary_target_doppler_is_not_confused_with_ground_velocity():
    model, batch = make_model(), example_batch(1)
    predictions = dict(detection_logits=torch.tensor([[-12.]]),
                       detection_state=torch.tensor([[[10., 0., -5., 0., 0.]]]),
                       detection_size=torch.tensor([[[4., 2.]]]),
                       detection_heading=torch.zeros(1, 1),
                       detection_future=torch.tensor([10., 0.]).expand(1, 1, 6, 2),
                       detection_future_velocity=torch.zeros(1, 1, 6, 2))
    features, _ = model.risk_fusion.physical_features(predictions, batch)
    # A stationary target has zero ground velocity; ego at +5 m/s yields
    # relative vx=-5 and positive closing speed, not target vx=-5.
    torch.testing.assert_close(features[0, 0, 2:7], torch.tensor([0., 0., -5/15, 0., -5/15]))
    torch.testing.assert_close(features[0, 0, 17:20], torch.tensor([.08, .1, 5/15]))
    future = features[0, 0, 20:].reshape(6, 10)
    torch.testing.assert_close(future[:, 2], torch.tensor([.075, .05, .025, 0., -.025, -.05]))
    torch.testing.assert_close(future[:, 8], torch.tensor([.055, .03, .005, 0., .005, .03]), atol=1e-7, rtol=1e-6)


def test_geometry_attention_handles_mixed_empty_rows_and_zero_candidates():
    model, batch = make_model(), example_batch(2)
    output = model(batch)
    mask = torch.ones_like(output['detection_logits'], dtype=torch.bool)
    mask[0] = False
    predictions = dict(output, detection_candidate_mask=mask)
    risk = model.risk_from_context(output['radar_tokens'], torch.zeros(2, 16),
                                   predictions=predictions, batch=batch)['risk']
    assert torch.isfinite(risk).all()
    zero_candidates = {key: value[:, :0] for key, value in predictions.items()
                       if key in ('detection_logits', 'detection_state', 'detection_size',
                                  'detection_heading', 'detection_future',
                                  'detection_future_velocity', 'detection_candidate_mask')}
    empty_risk = model.risk_from_context(output['radar_tokens'], torch.zeros(2, 16),
                                         predictions=zero_candidates, batch=batch)['risk']
    torch.testing.assert_close(risk[0], empty_risk[0])
    empty_risk.sum().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_bfloat16_autocast_keeps_physical_fusion_and_backward_finite():
    model = make_model()
    with torch.autocast('cpu', dtype=torch.bfloat16):
        output = model(example_batch(1))
        loss = output['risk'].sum()
    loss.backward()
    assert torch.isfinite(output['risk']).all()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_no_ego_ablation_does_not_leak_ego_through_physical_features():
    model, batch = make_model(use_ego=False), example_batch(1)
    changed = dict(batch, ego=torch.full_like(batch['ego'], 10.),
                   ego_velocity=torch.full_like(batch['ego_velocity'], 100.),
                   ego_acceleration=torch.full_like(batch['ego_velocity'], -20.))
    torch.testing.assert_close(model(batch)['risk'], model(changed)['risk'], atol=0, rtol=0)


def test_readout_mode_freezes_perception_and_trains_only_risk():
    model = make_model()
    trainable = model.configure_risk_readout_training()
    assert not model.training and not model.encoder.training
    selected = {name for name, p in model.named_parameters() if p.requires_grad}
    expected = {name for name, _ in model.named_parameters()
                if name == 'risk_query' or name.startswith(('risk_attention.', 'risk_head.', 'risk_fusion.'))}
    assert selected == expected
    assert {id(p) for p in trainable} == {id(p) for p in model.parameters() if p.requires_grad}
    model(example_batch(1))['risk'].sum().backward()
    assert all(p.grad is None for name, p in model.named_parameters() if name not in expected)


def test_fusion_requires_predictions_in_cached_context_path():
    model = make_model()
    with pytest.raises(ValueError, match='predictions|geometry'):
        model.risk_from_context(torch.zeros(1, 3, 16), torch.zeros(1, 16))


def test_default_is_exact_legacy_parameter_and_output_contract():
    # The fixture is an immutable copy of the pre-fusion CNN implementation.
    name = 'radar_vla._test_legacy_fusion_reference'
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parent / 'fixtures/legacy_cnn_model.py')
    legacy_module = importlib.util.module_from_spec(spec)
    sys.modules[name] = legacy_module
    try:
        spec.loader.exec_module(legacy_module)
        for detection in (False, True):
            torch.manual_seed(101)
            legacy = legacy_module.RadarVLAGrounder(legacy_module.ModelConfig(
                hidden_dim=16, num_queries=3, use_detection=detection)).eval()
            torch.manual_seed(101)
            current = RadarVLAGrounder(ModelConfig(hidden_dim=16, num_queries=3,
                                                   use_detection=detection)).eval()
            assert current.config.risk_input == 'radar_v1'
            assert legacy.state_dict().keys() == current.state_dict().keys()
            for key, tensor in legacy.state_dict().items():
                torch.testing.assert_close(tensor, current.state_dict()[key], atol=0, rtol=0)
            old, new = legacy(example_batch(1)), current(example_batch(1))
            for key in old:
                torch.testing.assert_close(old[key], new[key], atol=0, rtol=0)
    finally:
        sys.modules.pop(name, None)


@pytest.mark.parametrize('strict', [False, True])
def test_schema_rejects_implicit_conversion_in_both_directions(strict):
    old = RadarVLAGrounder(ModelConfig(hidden_dim=16, num_heads=4, num_queries=3))
    new = make_model()
    with pytest.raises(ValueError, match='Risk input|risk_input'):
        new.load_state_dict(old.state_dict(), strict=strict)
    with pytest.raises(ValueError, match='Risk input|risk_input'):
        old.load_state_dict(new.state_dict(), strict=strict)
    corrupted = dict(new.state_dict(), _risk_input_contract=torch.tensor(42))
    with pytest.raises(ValueError, match='Risk input|risk_input'):
        new.load_state_dict(corrupted, strict=strict)


def test_real_tiny_checkpoint_round_trip_preserves_fusion_output(tmp_path):
    model, batch = make_model(risk_horizon_schema='0p5_1_2_v2'), example_batch(1)
    batch['risk_horizon_schema_id'] = torch.ones(1, dtype=torch.long)
    optimizer = torch.optim.Adam(model.configure_risk_readout_training(), lr=.001)
    model(batch)['risk'].sum().backward()
    optimizer.step()
    path = tmp_path / 'tiny_fusion.pt'
    torch.save(dict(model_config=asdict(model.config), model=model.state_dict()), path)
    saved = torch.load(path, weights_only=True)
    restored = RadarVLAGrounder(ModelConfig(**saved['model_config'])).eval()
    restored.load_state_dict(saved['model'])
    torch.testing.assert_close(model(batch)['risk'], restored(batch)['risk'], atol=0, rtol=0)


@pytest.mark.parametrize('options', [dict(risk_input='typo'),
    dict(risk_input='predicted_geometry_v1', use_detection=False),
    dict(detection_assignment_policy='typo'),
    dict(detection_assignment_policy='local_quality_v1', use_detection=False),
    dict(detection_assignment_warmup_epochs=-1), dict(detection_assignment_warmup_epochs=True),
    dict(detection_assignment_warmup_epochs=2.5)])
def test_invalid_new_model_contracts_are_rejected(options):
    with pytest.raises(ValueError):
        ModelConfig(**options)
