"""Small explicit physical vocabulary; no pretrained tokenizer or downloads.

Coordinates use signed-log companding. Gaussian labels are distances in physical
units between bin centers, not distances between vocabulary IDs.
"""
from __future__ import annotations

import bisect
import math

import torch
from torch import Tensor
import torch.nn.functional as F


class PhysicalTokenizer:
    SPECIAL = ('<PAD>', '<BOS>', '<EOS>', '<SHORT>', '<LONG>', '<ROAD_UNKNOWN>',
               '<AGENTS>', '<AGENT_FUTURE>', '<EGO_FUTURE>', '<CRITICAL_DYNAMICS>',
               '<IMMEDIATE_ACTION>', '<UNKNOWN>')

    def __init__(self, bins: int = 64, coordinate_limit_m: float = 80.,
                 velocity_limit_mps: float = 40., companding_alpha: float = 1.):
        if bins < 3 or coordinate_limit_m <= 0 or velocity_limit_mps <= 0 or companding_alpha <= 0:
            raise ValueError('bins >= 3 and positive finite limits/alpha are required')
        if not all(math.isfinite(x) for x in (coordinate_limit_m, velocity_limit_mps, companding_alpha)):
            raise ValueError('limits and alpha must be finite')
        self.bins = bins
        self.coordinate_limit_m = coordinate_limit_m
        self.velocity_limit_mps = velocity_limit_mps
        self.alpha = companding_alpha
        self.special_ids = {name: i for i, name in enumerate(self.SPECIAL)}
        self.pad_id, self.bos_id, self.eos_id = range(3)
        self.unknown_id = self.special_ids['<UNKNOWN>']
        self.byte_offset = len(self.SPECIAL)
        self.position_offset = self.byte_offset + 256
        self.velocity_offset = self.position_offset + bins
        self.vocab_size = self.velocity_offset + bins

    def token(self, name: str) -> int:
        return self.special_ids[name]

    def _limits(self, kind):
        if kind == 'position':
            return self.coordinate_limit_m, self.position_offset
        if kind == 'velocity':
            return self.velocity_limit_mps, self.velocity_offset
        raise ValueError(f'unknown physical token kind: {kind}')

    def centers(self, kind='position', *, device=None, dtype=torch.float32) -> Tensor:
        limit, _ = self._limits(kind)
        bound = math.log1p(self.alpha * limit)
        transformed = torch.linspace(-bound, bound, self.bins, device=device, dtype=dtype)
        return transformed.sign() * torch.expm1(transformed.abs()) / self.alpha

    def encode_scalar(self, value: float, kind: str = 'position') -> int:
        if not math.isfinite(float(value)):
            raise ValueError('physical value must be finite')
        limit, offset = self._limits(kind)
        value = max(-limit, min(limit, float(value)))
        transformed = math.copysign(math.log1p(self.alpha * abs(value)), value)
        bound = math.log1p(self.alpha * limit)
        index = round((transformed / bound + 1.) * (self.bins - 1) / 2.)
        return offset + max(0, min(index, self.bins - 1))

    def decode_scalar(self, token_id: int, kind: str = 'position') -> float:
        limit, offset = self._limits(kind)
        index = int(token_id) - offset
        if not 0 <= index < self.bins:
            raise ValueError(f'token {token_id} is not a {kind} token')
        transformed = (2. * index / (self.bins - 1) - 1.) * math.log1p(self.alpha * limit)
        return math.copysign(math.expm1(abs(transformed)) / self.alpha, transformed)

    def physical_ids(self, kind='position') -> list[int]:
        _, offset = self._limits(kind)
        return list(range(offset, offset + self.bins))

    def soft_targets(self, value: float, kind: str = 'position', sigma: float = .75) -> Tensor:
        if sigma <= 0 or not math.isfinite(sigma) or not math.isfinite(value):
            raise ValueError('value and positive sigma must be finite')
        # softmax is stable even if the value is outside the representable range.
        return torch.softmax(-.5 * ((self.centers(kind) - value) / sigma).square(), dim=0)

    def encode_instruction(self, text: str, max_bytes: int = 96) -> list[int]:
        if max_bytes <= 0:
            raise ValueError('max_bytes must be positive')
        return [self.byte_offset + b for b in text.encode('utf-8')[:max_bytes]]


def token_cross_entropy(logits: Tensor, targets: Tensor, tokenizer: PhysicalTokenizer,
                        soft_sigma: float = .75) -> Tensor:
    """Masked SFT loss, Gaussian soft supervision for physical token bins.

    The dataset has already quantized targets, so Gaussian centers here are the
    quantized physical values. No continuous precision is claimed beyond bins.
    Probability mass on structural/byte tokens is penalized at physical slots.
    """
    if soft_sigma <= 0 or not math.isfinite(soft_sigma):
        raise ValueError('soft_sigma must be finite and positive')
    if logits.shape[:-1] != targets.shape or logits.shape[-1] != tokenizer.vocab_size:
        raise ValueError('logits and target shapes/vocabulary do not match')
    valid = targets != tokenizer.pad_id
    if not bool(valid.any()):
        return logits.sum() * 0.
    logp = F.log_softmax(logits[valid], dim=-1)
    target = targets[valid]
    losses = -logp.gather(1, target[:, None]).squeeze(1)
    for kind, offset in [('position', tokenizer.position_offset), ('velocity', tokenizer.velocity_offset)]:
        mask = (target >= offset) & (target < offset + tokenizer.bins)
        if bool(mask.any()):
            centers = tokenizer.centers(kind, device=logits.device, dtype=logits.dtype)
            values = centers[target[mask] - offset]
            q = torch.softmax(-.5 * ((centers[None] - values[:, None]) / soft_sigma).square(), dim=-1)
            losses = losses.clone()
            losses[mask] = -(q * logp[mask, offset:offset + tokenizer.bins]).sum(-1)
    return losses.mean()


def interpolate_trajectory(control_points, control_times, query_times):
    """Piecewise linear interpolation; reject extrapolation or unknown controls."""
    if any(p is None for p in control_points):
        raise ValueError('cannot interpolate missing control points')
    if len(control_points) != len(control_times) or len(control_times) < 2:
        raise ValueError('at least two aligned control points/times are required')
    times = [float(t) for t in control_times]
    points = [[float(v) for v in p] for p in control_points]
    if any(len(p) != 2 for p in points) or not all(math.isfinite(v) for p in points for v in p):
        raise ValueError('control points must be finite 2D coordinates')
    if not all(math.isfinite(t) for t in times) or any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError('control times must be finite and strictly increasing')
    result = []
    for value in query_times:
        t = float(value)
        if not math.isfinite(t) or t < times[0] or t > times[-1]:
            raise ValueError('query time lies outside the finite control-point interval')
        i = max(0, min(bisect.bisect_right(times, t) - 1, len(times) - 2))
        weight = (t - times[i]) / (times[i + 1] - times[i])
        result.append([(1. - weight) * a + weight * b for a, b in zip(points[i], points[i + 1])])
    return result
