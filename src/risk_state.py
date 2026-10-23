"""Prediction-only consistency of the three cumulative conflict probabilities."""
from __future__ import annotations

import torch
from torch import Tensor


def project_cumulative_risk(risk: Tensor) -> Tensor:
    """Project P1/P2/P3 onto P1 <= P2 <= P3; preserve dmin and areq.

    This is the equal-weight Euclidean isotonic projection for three values,
    written with tensor operations so piecewise gradients reach the estimator.
    It uses no targets, calibration fit, thresholds, or new model parameters.
    Consistency is not probability calibration or a trajectory-safety guarantee.
    """
    if risk.ndim < 1 or risk.shape[-1] != 5 or not risk.is_floating_point():
        raise ValueError('risk must be a floating tensor with final dimension 5')
    if not bool(torch.isfinite(risk).all()):
        raise ValueError('risk must be finite')
    p = risk[..., :3]
    if bool(((p < 0) | (p > 1)).any()):
        raise ValueError('conflict probabilities must lie in [0,1]')
    a, b, c = p.unbind(-1)
    ab, bc = (a + b) / 2, (b + c) / 2
    mean = (a + b + c) / 3
    all_pooled = mean[..., None].expand_as(p)
    left = torch.where((ab <= c)[..., None], torch.stack((ab, ab, c), -1), all_pooled)
    right = torch.where((a <= bc)[..., None], torch.stack((a, bc, bc), -1), all_pooled)
    projected = torch.where((a > b)[..., None], left,
                            torch.where((b > c)[..., None], right, p))
    return torch.cat((projected, risk[..., 3:]), -1)
