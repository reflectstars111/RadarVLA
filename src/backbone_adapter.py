"""Feature adapters for normalized RA inputs; no resampling of raw values."""
import math
import torch
from torch import nn
from torch.nn import functional as F


class TemporalFeatureAdapter(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.projection = nn.Identity() if hidden_dim == 64 else nn.Conv3d(64, hidden_dim, 1)

    def features(self, x, validity=None):
        raise NotImplementedError

    def forward(self, radar):
        if radar.ndim != 5 or radar.shape[2] != 2 or min(radar.shape[1], *radar.shape[-2:]) < 1:
            raise ValueError('Backbone requires [B,T,2,R,A] with positive dimensions')
        b, t, _, r, a = radar.shape
        left = max(4, math.ceil(t / 4) * 4) - t
        pr, pa = (-r) % 8, (-a) % 8
        x = F.pad(radar.permute(0, 2, 1, 3, 4), (0, pa, 0, pr, left, 0))
        valid = torch.ones((b,1,t,r,a),dtype=torch.bool,device=x.device)
        valid = F.pad(valid,(0,pa,0,pr,left,0))
        x = self.features(x,valid)
        expected = (t + left, r + pr, a + pa)
        if x.shape[1] != 64 or x.shape[2:] != expected:
            raise ValueError(f'Backbone feature reconstruction {tuple(x.shape)} differs from {expected}')
        x = x[:, :, left:left+t, :r, :a]
        x = F.adaptive_avg_pool3d(x, (t, math.ceil(r/4), math.ceil(a/4)))
        return self.projection(x).permute(0, 2, 1, 3, 4).contiguous()
