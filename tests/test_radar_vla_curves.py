"""Independent numeric contracts for the shared spline representation."""
import pytest
import torch

from radar_vla.curves import decode_spline, fit_control_points, spline_basis


def test_natural_cubic_matches_known_three_knot_solution_and_derivatives():
    times = torch.tensor([0., 1., 2.], dtype=torch.float64)
    points = torch.tensor([[0.], [1.], [0.]], dtype=torch.float64)
    result = decode_spline(points, times, [0., .5, 1., 1.5, 2.])
    torch.testing.assert_close(result[:, 0], torch.tensor([0., .6875, 1., .6875, 0.], dtype=torch.float64))
    torch.testing.assert_close((spline_basis(times, [0., 1., 2.], derivative=1) @ points)[:, 0],
                              torch.tensor([1.5, 0., -1.5], dtype=torch.float64))
    torch.testing.assert_close((spline_basis(times, [0., 1., 2.], derivative=2) @ points)[:, 0],
                              torch.tensor([0., -3., 0.], dtype=torch.float64))


def test_nonuniform_metric_times_preserve_constant_velocity_and_gradient():
    times = torch.tensor([0., .2, 1.4, 3.], dtype=torch.float64)
    query = torch.tensor([0., .1, .7, 2.5, 3.], dtype=torch.float64)
    velocity = torch.tensor([4., -2.], dtype=torch.float64)
    controls = (times[:, None] * velocity).requires_grad_()
    decoded = decode_spline(controls, times, query)
    torch.testing.assert_close(decoded, query[:, None] * velocity)
    decoded.sum().backward()
    assert torch.isfinite(controls.grad).all() and controls.grad.abs().sum() > 0
    torch.testing.assert_close(spline_basis(times, query, derivative=2) @ controls,
                              torch.zeros_like(decoded), atol=1e-12, rtol=0)


def test_fitting_uses_actual_times_and_never_extrapolates_missing_endpoint():
    times = torch.tensor([0., .3, 1., 2., 3.])
    points = torch.stack((2 * times, -times), -1)
    valid = torch.tensor([True, False, True, True, True])
    controls, knots, fitted = fit_control_points(points, times, valid, num_controls=4)
    assert fitted
    torch.testing.assert_close(decode_spline(controls, knots, times), points, atol=1e-5, rtol=1e-5)
    valid[-1] = False
    controls, _, fitted = fit_control_points(points, times, valid, num_controls=4)
    assert not fitted and torch.equal(controls, torch.zeros_like(controls))


def test_sparse_two_observation_fit_has_explicit_linear_interpolation():
    points = torch.tensor([[0., 0.], [2., -1.]])
    controls, knots, fitted = fit_control_points(points, torch.tensor([0., .5]), num_controls=4)
    assert fitted
    torch.testing.assert_close(decode_spline(controls, knots, [.25]), torch.tensor([[1., -.5]]))


def test_spline_rejects_nonmonotonic_times_and_extrapolation():
    with pytest.raises(ValueError, match='strictly increasing'):
        spline_basis([0., 0., 1.], [.2])
    with pytest.raises(ValueError, match='outside'):
        spline_basis([0., 1.], [1.01])
    with pytest.raises(ValueError, match='finite'):
        decode_spline(torch.tensor([[0., 0.], [float('nan'), 1.]]), [0., 1.], [.5])


def test_compact_least_squares_fit_preserves_observed_interval_endpoints():
    times = torch.linspace(0., 3., 20)
    points = torch.stack((times, times.square()), -1)
    controls, _, fitted = fit_control_points(points, times, num_controls=4)
    assert fitted
    torch.testing.assert_close(controls[0], points[0], atol=0, rtol=0)
    torch.testing.assert_close(controls[-1], points[-1], atol=0, rtol=0)
