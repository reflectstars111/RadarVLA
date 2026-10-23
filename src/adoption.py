"""Quintic adoption transitions and continuous polynomial dynamics bounds.

Only the current execution state and a proposed trajectory enter this module.
These bounds concern motion feasibility, not obstacle or road safety.
"""
from __future__ import annotations

import math
import numpy as np
from numpy.polynomial import Polynomial


class AdoptionTransitionError(ValueError):
    def __init__(self, message, diagnostics):
        super().__init__(message)
        self.diagnostics = dict(diagnostics, status='rejected', reason=message)


def quintic_coefficients(velocity, acceleration, endpoint, end_velocity,
                         end_acceleration, duration):
    """Coefficients in normalized time u=t/duration, with the origin at zero."""
    values = np.asarray([velocity, acceleration, endpoint, end_velocity, end_acceleration], float)
    if values.shape != (5, 2) or not np.isfinite(values).all():
        raise ValueError('Transition boundary states must be finite XY vectors')
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError('Transition duration must be finite and positive')
    velocity, acceleration, endpoint, end_velocity, end_acceleration = values
    v, a = velocity * duration, acceleration * duration**2
    distance = endpoint - v - a / 2
    delta_v = end_velocity * duration - v - a
    delta_a = end_acceleration * duration**2 - a
    return np.asarray([np.zeros(2), v, a / 2,
        10 * distance - 4 * delta_v + delta_a / 2,
        -15 * distance + 7 * delta_v - delta_a,
        6 * distance - 3 * delta_v + delta_a / 2])


def sample_quintic(coefficients, duration, times, derivative=0):
    query = np.asarray(times, float)
    if (query.ndim != 1 or not np.isfinite(query).all() or derivative not in (0, 1, 2, 3)
            or (query < 0).any() or (query > duration + 1e-10).any()):
        raise ValueError('Invalid transition sampling request')
    u = np.minimum(query / duration, 1.)
    return np.column_stack([Polynomial(coefficients[:, axis]).deriv(derivative)(u)
                            / duration**derivative for axis in range(2)])


def quintic_dynamics(coefficients, duration):
    """Find vector acceleration/jerk maxima, including interior extrema.

    The derivatives of squared norms are degree five and three respectively.
    Normalized time avoids unnecessary conditioning from short bridge durations.
    """
    critical = {0., 1.}
    peaks = {}
    for derivative, name in ((2, 'max_acceleration_mps2'), (3, 'max_jerk_mps3')):
        axes = [Polynomial(coefficients[:, axis]).deriv(derivative) / duration**derivative
                for axis in range(2)]
        norm_squared = axes[0] * axes[0] + axes[1] * axes[1]
        roots = norm_squared.deriv().roots()
        if not np.isfinite(roots).all():
            raise ValueError('Nonfinite transition dynamics extrema')
        points = [0., 1.]
        points.extend(float(root.real) for root in roots
                      if abs(root.imag) < 1e-8 and 0 < root.real < 1)
        critical.update(points)
        peaks[name] = float(max(np.hypot(axes[0](u), axes[1](u)) for u in points))
    if not all(math.isfinite(value) for value in peaks.values()):
        raise ValueError('Nonfinite transition dynamics')
    return dict(**peaks, dynamics_times_s=[float(u * duration) for u in sorted(critical)])
