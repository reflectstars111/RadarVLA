"""Differentiable, metric-time natural cubic spline representation.

Controls are positions at explicit knot times (not un-timed waypoints). Natural
boundary conditions set endpoint second derivatives to zero. The same linear
basis is used for fitting, physical losses, and inference. Two observations
identify a straight line; no extrapolated or unobserved endpoints are teachers.
"""
from __future__ import annotations

import torch
from torch import Tensor


def spline_basis(control_times, query_times, derivative: int = 0) -> Tensor:
    """Return ``[number_of_queries, number_of_controls]`` interpolation weights.

    Float64 is preserved for numerical checking; other input types calculate in
    float32. Coordinates may be either seconds or a road's arc length in metres.
    Query extrapolation is deliberately rejected.
    """
    if derivative not in (0, 1, 2, 3):
        raise ValueError('spline derivative must be 0, 1, 2, or 3')
    times = torch.as_tensor(control_times)
    if not times.is_floating_point():
        times = times.float()
    if times.dtype not in (torch.float32, torch.float64):
        times = times.float()
    query = torch.as_tensor(query_times, device=times.device, dtype=times.dtype)
    if times.ndim != 1 or times.numel() < 2:
        raise ValueError('at least two one-dimensional control times are required')
    if query.ndim != 1:
        raise ValueError('query times must be one-dimensional')
    if not bool(torch.isfinite(times).all() and torch.isfinite(query).all()):
        raise ValueError('spline times must be finite')
    h = times[1:] - times[:-1]
    if bool((h <= 0).any()):
        raise ValueError('control times must be strictly increasing')
    tolerance = torch.finfo(times.dtype).eps * times.abs().max().clamp_min(1) * 8
    if bool(((query < times[0] - tolerance) | (query > times[-1] + tolerance)).any()):
        raise ValueError('query time lies outside the control-point interval')
    query = query.clamp(times[0], times[-1])
    n = times.numel()
    eye = torch.eye(n, device=times.device, dtype=times.dtype)
    second = torch.zeros_like(eye)
    if n > 2:
        system = torch.diag(2 * (h[:-1] + h[1:]))
        if n > 3:
            system = system + torch.diag(h[1:-1], 1) + torch.diag(h[1:-1], -1)
        rhs = 6 * ((eye[2:] - eye[1:-1]) / h[1:, None]
                   - (eye[1:-1] - eye[:-2]) / h[:-1, None])
        second[1:-1] = torch.linalg.solve(system, rhs)
    index = torch.searchsorted(times, query, right=True).sub(1).clamp(0, n - 2)
    delta = h[index]
    left = (times[index + 1] - query) / delta
    right = (query - times[index]) / delta
    if derivative == 1:
        return ((eye[index + 1] - eye[index]) / delta[:, None]
                + ((1 - 3 * left.square())[:, None] * second[index]
                   + (3 * right.square() - 1)[:, None] * second[index + 1]) * delta[:, None] / 6)
    if derivative == 2:
        return left[:, None] * second[index] + right[:, None] * second[index + 1]
    if derivative == 3:
        return (second[index + 1] - second[index]) / delta[:, None]
    return (left[:, None] * eye[index] + right[:, None] * eye[index + 1]
            + ((left.pow(3) - left)[:, None] * second[index]
               + (right.pow(3) - right)[:, None] * second[index + 1])
            * delta[:, None].square() / 6)


def decode_spline(control_points: Tensor, control_times, query_times) -> Tensor:
    """Decode ``[..., K, dimensions]`` controls without breaking gradients."""
    points = torch.as_tensor(control_points)
    if not points.is_floating_point():
        points = points.float()
    if points.ndim < 2 or not bool(torch.isfinite(points).all()):
        raise ValueError('control points must be finite [...,K,D] coordinates')
    times = torch.as_tensor(control_times, device=points.device, dtype=points.dtype)
    if points.shape[-2] != times.numel():
        raise ValueError('control points and times must align')
    basis = spline_basis(times, query_times).to(points.dtype)
    return torch.einsum('tk,...kd->...td', basis, points)


def fit_control_points(points: Tensor, times: Tensor, valid: Tensor | None = None,
                       num_controls: int = 4) -> tuple[Tensor, Tensor, bool]:
    """Fit a compact spline on the complete requested observation interval.

    Returns ``(controls, control_times, supervised)``. Missing interval endpoints
    or fewer than two valid observations produce masked controls, never an
    extrapolated teacher. With fewer observations than controls, the unique
    natural interpolation of those observations is sampled at the control knots;
    with enough observations a least-squares fit with fixed observed endpoints is used. This sparse
    rule is explicit and deterministic, and does not label missing dense points.
    """
    points = torch.as_tensor(points)
    if not points.is_floating_point():
        points = points.float()
    times = torch.as_tensor(times, dtype=points.dtype, device=points.device)
    if points.ndim != 2 or times.ndim != 1 or len(points) != len(times) or len(times) < 2:
        raise ValueError('fitting requires aligned [T,D] points and at least two times')
    if num_controls < 2:
        raise ValueError('num_controls must be at least two')
    if not bool(torch.isfinite(times).all()) or bool(((times[1:] - times[:-1]) <= 0).any()):
        raise ValueError('fitting times must be finite and strictly increasing')
    valid = torch.ones(len(times), device=points.device, dtype=torch.bool) if valid is None else valid.bool()
    if valid.shape != times.shape:
        raise ValueError('valid mask must have shape [T]')
    valid = valid & torch.isfinite(points).all(-1)
    knots = torch.linspace(times[0], times[-1], num_controls, device=points.device, dtype=points.dtype)
    empty = torch.zeros(num_controls, points.shape[-1], device=points.device, dtype=points.dtype)
    if int(valid.sum()) < 2 or not bool(valid[0] and valid[-1]):
        return empty, knots, False
    x, y = times[valid], points[valid]
    if len(x) < num_controls:
        return decode_spline(y, x, knots), knots, True
    basis = spline_basis(knots, x)
    # Float32/64 solve is intentional: CPU and CUDA linalg do not support bf16.
    dtype = torch.float64 if points.dtype == torch.float64 else torch.float32
    matrix = basis.to(dtype)
    if int(torch.linalg.matrix_rank(matrix)) != num_controls:
        return empty, knots, False
    controls = torch.empty(num_controls, points.shape[-1], device=points.device, dtype=dtype)
    controls[0], controls[-1] = y[0].to(dtype), y[-1].to(dtype)
    if num_controls > 2:
        residual = y.to(dtype) - matrix[:, :1] * controls[0] - matrix[:, -1:] * controls[-1]
        controls[1:-1] = torch.linalg.lstsq(matrix[:, 1:-1], residual).solution
    return controls.to(points.dtype), knots, True
