"""Radar measurement errors remain distinct from state-derived radial velocity."""
import pytest
import torch


def sample():
    prediction = dict(risk=torch.zeros(1, 5), agent_state=torch.tensor([[[10., 0., -2., 3., 0.]]]),
                      agent_future=torch.zeros(1, 1, 2, 2))
    batch = dict(risk_target=torch.zeros(1, 5), risk_mask=torch.zeros(1, 5, dtype=torch.bool),
                 agent_state=torch.tensor([[[10., 0., -9., 3., 0.]]]),
                 agent_state_mask=torch.ones(1, 1, 5, dtype=torch.bool), agent_mask=torch.ones(1, 1, dtype=torch.bool),
                 agent_future=torch.zeros(1, 1, 2, 2), agent_future_mask=torch.ones(1, 1, 2, dtype=torch.bool),
                 agent_radial_observed=torch.tensor([[-2.]]), agent_radial_mask=torch.tensor([[True]]),
                 agent_los=torch.tensor([[[1., 0.]]]), agent_sensor_velocity=torch.tensor([[[5., 0.]]]))
    return prediction, batch


def test_radial_metric_scores_observed_doppler_not_derived_gt_velocity():
    from radar_vla.metrics import MetricAccumulator
    prediction, batch = sample()
    accumulator = MetricAccumulator()
    accumulator.update(prediction, batch)
    result = accumulator.compute()
    assert result['radial_mae'] == 0
    assert result['state_derived_radial_mae'] == 7
    assert result['doppler_projection_mae'] == 0
    assert result['radial_target_sources'] == {'measured_radar': 1, 'legacy_state': 0}


def test_missing_observation_has_no_fake_radial_success():
    from radar_vla.metrics import MetricAccumulator
    prediction, batch = sample()
    batch['agent_radial_mask'][:] = False
    accumulator = MetricAccumulator()
    accumulator.update(prediction, batch)
    result = accumulator.compute()
    assert result['radial_mae'] is None
    assert result['radial_mae_count'] == 0
    assert result['doppler_projection_mae'] is None


def test_measured_projection_metric_preserves_effective_los_magnitude():
    from radar_vla.metrics import MetricAccumulator
    prediction, batch = sample()
    batch['agent_doppler_projection'] = torch.tensor([[[.5, 0.]]])
    batch['agent_radial_observed'][:] = -1.
    accumulator = MetricAccumulator()
    accumulator.update(prediction, batch)
    assert accumulator.compute()['doppler_projection_mae'] == 0
