"""Behavior tests for the independent RadarVLA grounding stage."""

import pytest
import torch


def example_batch(batch_size=2):
    generator = torch.Generator().manual_seed(17)
    radar = torch.randn(batch_size, 3, 2, 16, 20, generator=generator)
    radar[:, :, 0] = radar[:, :, 0].abs()
    state = torch.tensor([[10., 1., -3., 2., 0.], [20., -2., 0., 5., 0.]])
    times = torch.arange(1, 7).float() * .5
    agents = state.unsqueeze(0).repeat(batch_size, 1, 1)
    return dict(
        radar=radar,
        range_m=torch.linspace(0., 50., 16).repeat(batch_size, 1),
        azimuth_rad=torch.linspace(-1., 1., 20).repeat(batch_size, 1),
        time_offsets_s=torch.tensor([-.2, -.1, 0.]).repeat(batch_size, 1),
        ego=torch.tensor([5., 0., 0.]).repeat(batch_size, 1),
        ego_velocity=torch.tensor([5., 0.]).repeat(batch_size, 1),
        future_times_s=times.repeat(batch_size, 1),
        agent_state=agents,
        agent_mask=torch.ones(batch_size, 2, dtype=torch.bool),
        agent_state_mask=torch.ones(batch_size, 2, 5, dtype=torch.bool),
        agent_future=agents[:, :, None, :2] + agents[:, :, None, 3:5] * times[None, None, :, None],
        agent_future_mask=torch.ones(batch_size, 2, 6, dtype=torch.bool),
        risk_target=torch.tensor([0., 1., 1., 2., 3.]).repeat(batch_size, 1),
        risk_mask=torch.ones(batch_size, 5, dtype=torch.bool),
    )


def small_model():
    from radar_vla.model import ModelConfig, RadarVLAGrounder
    return RadarVLAGrounder(ModelConfig(hidden_dim=16, num_queries=4, max_agents=3))


def test_grounder_outputs_physical_heads_and_backpropagates():
    from radar_vla.losses import loss_grounding
    model, batch = small_model(), example_batch()
    output = model(batch)
    assert output['radar_tokens'].shape == (2, 4, 16)
    assert output['risk_logits'].shape == (2, 3)
    assert output['risk'].shape == (2, 5)
    assert output['agent_state'].shape == (2, 3, 5)
    assert output['agent_future'].shape == (2, 3, 6, 2)
    assert ((output['risk'][:, :3] >= 0) & (output['risk'][:, :3] <= 1)).all()
    assert (output['risk'][:, 3:] > 0).all()
    losses = loss_grounding(output, batch)
    assert all(value.ndim == 0 and torch.isfinite(value) for value in losses.values())
    losses['total'].backward()
    for module in (model.encoder, model.risk_head):
        gradients = [p.grad for p in module.parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients)
        assert sum(g.abs().sum().item() for g in gradients) > 0


def test_grounder_never_reads_ground_truth_future_or_risk_labels():
    model, batch = small_model().eval(), example_batch()
    with torch.no_grad():
        before = model(batch)
        changed = dict(batch, agent_future=torch.full_like(batch['agent_future'], float('nan')),
                       risk_target=torch.full_like(batch['risk_target'], 1e9),
                       agent_state=torch.full_like(batch['agent_state'], -1e9))
        after = model(changed)
    for key in before:
        torch.testing.assert_close(before[key], after[key], rtol=0, atol=0)


def test_normalization_preserves_signed_doppler_and_geometry_matters():
    model, batch = small_model().eval(), example_batch()
    radar = torch.zeros(1, 1, 2, 1, 2)
    radar[0, 0, 1, 0] = torch.tensor([-15., 15.])
    normalized = model.normalize_radar(radar)
    torch.testing.assert_close(normalized[0, 0, 1, 0], torch.tensor([-1., 1.]))
    with torch.no_grad():
        original = model(batch)['radar_tokens']
        altered = model(dict(batch, range_m=batch['range_m'] + 100.))['radar_tokens']
    assert not torch.allclose(original, altered)


def test_matching_is_invariant_to_ground_truth_object_order():
    from radar_vla.losses import loss_grounding
    model, batch = small_model(), example_batch()
    output = model(batch)
    original = loss_grounding(output, batch)
    permuted = dict(batch)
    for key in ('agent_state', 'agent_state_mask', 'agent_mask', 'agent_future', 'agent_future_mask'):
        permuted[key] = batch[key][:, [1, 0]]
    reordered = loss_grounding(output, permuted)
    for key in original:
        torch.testing.assert_close(original[key], reordered[key])


def test_missing_labels_are_ignored_even_when_filled_with_nan():
    from radar_vla.losses import loss_grounding
    model, batch = small_model(), example_batch()
    output = model(batch)
    batch['risk_target'][:] = float('nan')
    batch['risk_mask'][:] = False
    batch['agent_state'][:] = float('nan')
    batch['agent_future'][:] = float('nan')
    batch['agent_mask'][:] = False
    batch['agent_future_mask'][:] = False
    losses = loss_grounding(output, batch)
    assert losses['objectness'] > 0  # observed empty scenes still supervise absence
    for key, value in losses.items():
        if key not in ('objectness', 'total'):
            assert value == 0
    batch['agent_supervision_mask'] = torch.zeros(2, dtype=torch.bool)
    all_missing = loss_grounding(output, batch)
    assert all(torch.isfinite(value) and value == 0 for value in all_missing.values())
    all_missing['total'].backward()


def test_doppler_consistency_subtracts_ego_velocity():
    from radar_vla.losses import loss_grounding
    batch = example_batch(1)
    state = torch.tensor([[[10., 0., -5., 0., 0.]]], requires_grad=True)
    batch.update(agent_state=state.detach().clone(), agent_mask=torch.ones(1, 1, dtype=torch.bool),
                 agent_state_mask=torch.ones(1, 1, 5, dtype=torch.bool),
                 agent_future=torch.tensor([[[[10., 0.]] * 6]]),
                 agent_future_mask=torch.ones(1, 1, 6, dtype=torch.bool))
    output = dict(agent_state=state, object_logits=torch.zeros(1, 1, requires_grad=True),
                  agent_future=batch['agent_future'].clone().requires_grad_(),
                  risk_logits=torch.zeros(1, 3, requires_grad=True),
                  risk=torch.tensor([[.5, .5, .5, 2., 3.]], requires_grad=True))
    batch.update(agent_radial_observed=torch.tensor([[-5.]]), agent_radial_mask=torch.tensor([[True]]),
                 agent_los=torch.tensor([[[1., 0.]]]))
    assert loss_grounding(output, batch)['doppler'] == 0
    wrong_frame = dict(batch, ego_velocity=torch.zeros(1, 2))
    assert loss_grounding(output, wrong_frame)['doppler'] > 0


def test_physical_time_and_config_dimensions_are_checked():
    from radar_vla.model import ModelConfig, RadarVLAGrounder
    with pytest.raises(ValueError, match='heads'):
        RadarVLAGrounder(ModelConfig(hidden_dim=15, num_heads=4))
    model, batch = small_model(), example_batch()
    batch['future_times_s'][:, 2] = .1
    with pytest.raises(ValueError, match='future_times_s'):
        model(batch)


def test_ablation_switches_remove_only_the_selected_observation():
    from radar_vla.model import ModelConfig, RadarVLAGrounder
    model = RadarVLAGrounder(ModelConfig(hidden_dim=16, use_doppler=False, use_ego=False)).eval()
    batch = example_batch()
    altered = dict(batch, radar=batch['radar'].clone(), ego=-100. * batch['ego'])
    altered['radar'][:, :, 1] *= -50.
    with torch.no_grad():
        before, after = model(batch), model(altered)
    for key in before:
        torch.testing.assert_close(before[key], after[key], rtol=0, atol=0)


def test_large_grid_is_compressed_before_temporal_attention():
    model, batch = small_model().eval(), example_batch(1)
    batch.update(radar=torch.ones(1, 3, 2, 65, 97),
                 range_m=torch.linspace(0, 50, 65)[None],
                 azimuth_rad=torch.linspace(-1, 1, 97)[None])
    lengths = []
    hook = model.temporal_attention.register_forward_pre_hook(
        lambda module, args: lengths.append(args[1].shape[1]))
    with torch.no_grad():
        output = model(batch)
    hook.remove()
    assert lengths == [3 * model.config.max_grid_size ** 2]
    assert torch.isfinite(output['risk']).all()


def test_missing_ego_velocity_does_not_create_nan_gradients():
    from radar_vla.losses import loss_grounding
    model, batch = small_model(), example_batch()
    output = model(batch)
    batch['ego_velocity'][0] = float('nan')
    losses = loss_grounding(output, batch)
    losses['total'].backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_full_lateral_ego_state_and_motion_geometry_change_predictions():
    model, batch = small_model().eval(), example_batch(1)
    batch['ego_acceleration'] = torch.zeros(1, 2)
    with torch.no_grad():
        original = model(batch)
        lateral = model(dict(batch, ego_velocity=torch.tensor([[0., 5.]])))
    assert not torch.allclose(original['risk'], lateral['risk'])
    r = batch['range_m'][:, None, :, None]
    a = batch['azimuth_rad'][:, None, None, :]
    xy = torch.stack((r * a.cos(), r * a.sin()), -1).expand(-1, 3, -1, -1, -1).clone()
    los = torch.stack((a.cos().expand_as(xy[..., 0]), a.sin().expand_as(xy[..., 0])), -1)
    pose = torch.eye(4)[None, None].repeat(1, 3, 1, 1)
    original_geometry = dict(batch, radar_cartesian=xy, radar_los=los, radar_pose=pose)
    moved = xy.clone()
    moved[:, 0, ..., 0] += 10
    with torch.no_grad():
        a = model(original_geometry)['radar_tokens']
        b = model(dict(original_geometry, radar_cartesian=moved))['radar_tokens']
    assert not torch.allclose(a, b)


def test_doppler_consistency_is_grounded_in_measured_radar():
    from radar_vla.losses import loss_grounding
    model, batch = small_model(), example_batch(1)
    output = model(batch)
    batch.update(agent_radial_observed=torch.tensor([[0., 0.]]),
                 agent_radial_mask=torch.tensor([[True, True]]),
                 agent_los=torch.tensor([[[1., 0.], [1., 0.]]]),
                 agent_sensor_velocity=torch.zeros(1, 2, 2))
    before = loss_grounding(output, batch)['doppler']
    batch['agent_radial_observed'] += 30
    after = loss_grounding(output, batch)['doppler']
    assert after > before
    batch['agent_radial_mask'][:] = False
    assert loss_grounding(output, batch)['doppler'] == 0


def test_kinematic_consistency_checks_every_nonuniform_interval():
    from radar_vla.losses import loss_grounding
    batch = example_batch(1)
    times = torch.tensor([[.1, .3, .6, 1., 2., 3.]])
    batch['future_times_s'] = times
    model = small_model()
    output = model(batch)
    velocity = output['agent_state'][:, :, 3:5]
    output['agent_future'] = output['agent_state'][:, :, None, :2] + velocity[:, :, None] * times[:, None, :, None]
    output['agent_future_velocity'] = velocity[:, :, None].expand(-1, -1, 6, -1).clone()
    baseline = loss_grounding(output, batch)['kinematic'].clone()
    assert baseline.abs() < 1e-6
    output['agent_future_velocity'][:, :, 3] += 10
    assert loss_grounding(output, batch)['kinematic'] > baseline + 1e-3


def test_single_frame_ablation_ignores_all_history_observations():
    from radar_vla.model import ModelConfig, RadarVLAGrounder
    model = RadarVLAGrounder(ModelConfig(hidden_dim=16, use_temporal=False)).eval()
    batch = example_batch(1)
    altered = dict(batch, radar=batch['radar'].clone())
    altered['radar'][:, :-1] *= 100
    with torch.no_grad():
        before, after = model(batch), model(altered)
    for key in before:
        torch.testing.assert_close(before[key], after[key], rtol=0, atol=0)


def test_sensor_lever_arm_velocity_is_used_for_measured_doppler():
    from radar_vla.losses import loss_grounding
    model, batch = small_model(), example_batch(1)
    output = model(batch)
    batch.update(agent_radial_observed=torch.tensor([[0., 0.]]),
                 agent_radial_mask=torch.tensor([[True, True]]),
                 agent_los=torch.tensor([[[0., 1.], [0., 1.]]]),
                 agent_sensor_velocity=torch.zeros(1, 2, 2))
    before = loss_grounding(output, batch)['doppler']
    batch['agent_sensor_velocity'][:, :, 1] = 30
    assert loss_grounding(output, batch)['doppler'] > before


def test_nonuniform_accelerating_trajectory_is_kinematically_consistent():
    from radar_vla.losses import loss_grounding
    model, batch = small_model(), example_batch(1)
    times = torch.tensor([[.1, .3, .6, 1., 2., 3.]])
    batch['future_times_s'] = times
    output = model(batch)
    position, velocity = output['agent_state'][:, :, :2], output['agent_state'][:, :, 3:5]
    acceleration = torch.tensor([1., -2.])[None, None, None]
    t = times[:, None, :, None]
    output['agent_future'] = position[:, :, None] + velocity[:, :, None] * t + .5 * acceleration * t.square()
    output['agent_future_velocity'] = velocity[:, :, None] + acceleration * t
    assert loss_grounding(output, batch)['kinematic'].abs() < 1e-6


def test_invalid_observation_vectors_are_rejected_before_forward():
    model, batch = small_model(), example_batch(1)
    batch['ego_velocity'][0, 1] = float('nan')
    with pytest.raises(ValueError, match='ego_velocity'):
        model(batch)


def test_kinematics_does_not_bridge_missing_future_annotations():
    from radar_vla.losses import loss_grounding
    model, batch = small_model(), example_batch(1)
    output = model(batch)
    batch['agent_future_mask'][:, :, 2:] = False
    baseline = loss_grounding(output, batch)['kinematic']
    output['agent_future_velocity'][:, :, 3:] += 100
    output['agent_future'][:, :, 3:] += 100
    torch.testing.assert_close(loss_grounding(output, batch)['kinematic'], baseline)


def test_doppler_projection_uses_weighted_cell_los_without_normalizing():
    from radar_vla.losses import loss_grounding
    batch = example_batch(1)
    state = torch.tensor([[[10., 0., -2., 0., 0.]]], requires_grad=True)
    batch.update(agent_state=state.detach().clone(), agent_mask=torch.ones(1, 1, dtype=torch.bool),
                 agent_state_mask=torch.ones(1, 1, 5, dtype=torch.bool),
                 agent_future=torch.tensor([[[[10., 0.]] * 6]]), agent_future_mask=torch.ones(1, 1, 6, dtype=torch.bool),
                 agent_radial_observed=torch.tensor([[-2.]]), agent_radial_mask=torch.tensor([[True]]),
                 agent_los=torch.tensor([[[1., 0.]]]), agent_doppler_projection=torch.tensor([[[.4, 0.]]]))
    output = dict(agent_state=state, object_logits=torch.zeros(1, 1, requires_grad=True),
                  agent_future=batch['agent_future'].clone().requires_grad_(), risk_logits=torch.zeros(1, 3, requires_grad=True),
                  risk=torch.tensor([[.5, .5, .5, 2., 3.]], requires_grad=True))
    assert loss_grounding(output, batch)['doppler'] == 0


def test_missing_doppler_mask_differs_from_a_measured_zero_and_is_removed_by_ablation():
    from radar_vla.model import ModelConfig, RadarVLAGrounder
    batch = example_batch(1)
    batch['radar'][:, :, 1] = 0
    shape = (1, 3, 16, 20)
    absent = dict(batch, radar_doppler_valid=torch.zeros(shape, dtype=torch.bool))
    present = dict(batch, radar_doppler_valid=torch.ones(shape, dtype=torch.bool))
    model = small_model().eval()
    with torch.no_grad():
        assert not torch.allclose(model(absent)['radar_tokens'], model(present)['radar_tokens'])
    ablated = RadarVLAGrounder(ModelConfig(hidden_dim=16, use_doppler=False)).eval()
    with torch.no_grad():
        for name, value in ablated(absent).items():
            torch.testing.assert_close(value, ablated(present)[name], rtol=0, atol=0)


def test_delayed_causal_radar_times_remain_supported():
    model, batch = small_model(), example_batch(1)
    batch['time_offsets_s'] -= .03
    assert torch.isfinite(model(batch)['risk']).all()
    batch['time_offsets_s'][:, -1] = .03
    with pytest.raises(ValueError, match='historical observation'):
        model(batch)
