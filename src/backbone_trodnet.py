"""Dynamic-shape T-RODNet feature path, adapted from the pinned MIT upstream.

Copyright (c) 2021 Yizhou Wang. See RODNET_LICENSE.txt and BACKBONE_UPSTREAM.json.
The active stem/DAM/merging/decoder/attention operations are retained; inactive
profiling and unused trainable branches are omitted. Geometry, padding masks
and dynamic relative-position indexing are explicit RadarVLA adaptations.
"""
import itertools
from functools import lru_cache
import math

import torch
from torch import nn
from torch.nn import functional as F

from .backbone_adapter import TemporalFeatureAdapter


class Mlp(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DAMLayerConcat(nn.Module):
    def __init__(self,in_channel):
        super(DAMLayerConcat, self).__init__()

        self.maina =   nn.Conv3d(in_channels=int(in_channel)    , out_channels=int(in_channel / 4), kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0))
        self.mainb=    nn.Conv3d(in_channels=int(in_channel / 4), out_channels=int(in_channel / 4), kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0))
        self.branch2 = nn.Conv3d(in_channels=int(in_channel / 4), out_channels=int(in_channel / 4), kernel_size=(1, 3, 1), stride=(1, 1, 1), padding=(0, 1, 0))
        self.branch3 = nn.Conv3d(in_channels=int(in_channel / 4), out_channels=int(in_channel / 4), kernel_size=(1, 1, 3), stride=(1, 1, 1), padding=(0, 0, 1))
        self.branch4 = nn.Conv3d(in_channels=int(in_channel / 4), out_channels=int(in_channel / 4), kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))
        self.last = nn.Conv3d(in_channels=int(in_channel*2), out_channels=int(in_channel),kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0))

        self.gnmaina =   nn.GroupNorm(num_groups=int(in_channel / 16), num_channels=int(in_channel / 4))
        self.gnmainb =   nn.GroupNorm(num_groups=int(in_channel / 16), num_channels=int(in_channel / 4))
        self.gnbranch2 = nn.GroupNorm(num_groups=int(in_channel / 16), num_channels=int(in_channel / 4))
        self.gnbranch3 = nn.GroupNorm(num_groups=int(in_channel / 16), num_channels=int(in_channel / 4))
        self.gnbranch4 = nn.GroupNorm(num_groups=int(in_channel / 16), num_channels=int(in_channel / 4))
        self.gnlast = nn.GroupNorm(num_groups=int(in_channel / 4), num_channels=int(in_channel))

        self.relu = nn.ReLU()

    def forward(self, x):
        branch1 = x                                      #4 128 8 64 64
        x = self.relu(self.gnmaina(self.maina(branch1))) #4 32 8 64 64
        x = self.relu(self.gnmainb(self.mainb(x)))       #4 32 8 64 64
        branch6 = x
        branch2 = self.relu(self.gnbranch2(self.branch2(x)))#1 16 8 62 64
        branch3 = self.relu(self.gnbranch3(self.branch3(x)))#4 32 8 64 64
        branch4 = self.relu(self.gnbranch4(self.branch4(x)))  # 4 32 8 64 64

        x = torch.cat((branch1, branch2, branch3, branch4, branch6), 1)#4 256 8 64 64

        x = self.relu(self.gnlast(self.last(x)))#4 128 8 64 64
        return x


class DropPath(nn.Module):
    def __init__(self, probability=0.):
        super().__init__()
        self.probability = probability

    def forward(self, x):
        if not self.training or self.probability == 0:
            return x
        keep = 1 - self.probability
        shape = (x.shape[0],) + (1,) * (x.ndim-1)
        return x * x.new_empty(shape).bernoulli_(keep) / keep


def window_partition(x, window):
    b, d, h, w, c = x.shape
    wd, wh, ww = window
    if d % wd or h % wh or w % ww:
        raise ValueError('Window partition requires locally padded dimensions')
    return x.reshape(b,d//wd,wd,h//wh,wh,w//ww,ww,c).permute(
        0,1,3,5,2,4,6,7).contiguous().reshape(-1,wd*wh*ww,c)


def window_reverse(x, window, b, d, h, w):
    wd, wh, ww = window
    return x.reshape(b,d//wd,h//wh,w//ww,wd,wh,ww,-1).permute(
        0,1,4,2,5,3,6,7).contiguous().reshape(b,d,h,w,-1)


@lru_cache(maxsize=32)
def relative_position_index(window, maximum=4):
    """Actual D/H/W coordinates mapped into the fixed max-window bias table."""
    coordinates = torch.stack(torch.meshgrid(*(torch.arange(v,device="cpu") for v in window),indexing='ij')).flatten(1)
    delta = coordinates[:,:,None] - coordinates[:,None,:]
    delta = delta + maximum - 1
    stride = 2*maximum - 1
    return delta[0]*stride*stride + delta[1]*stride + delta[2]


def window_layout(shape, maximum=4, shifted=False):
    window = tuple(min(v,maximum) for v in shape)
    shift = tuple(w//2 if shifted and v>w else 0 for v,w in zip(shape,window))
    padded = tuple(math.ceil(v/w)*w for v,w in zip(shape,window))
    return window,shift,padded


def region_labels(padded, window, shift, device):
    regions = torch.zeros((1,*padded,1),device=device,dtype=torch.int64)
    slices = [((slice(None),) if s==0 else (slice(0,-w),slice(-w,-s),slice(-s,None)))
              for w,s in zip(window,shift)]
    for number,(d,h,w) in enumerate(itertools.product(*slices)):
        regions[:,d,h,w] = number
    return window_partition(regions,window)[...,0]


def masked_softmax(logits, allowed):
    allowed = allowed[:,None]
    safe = logits.masked_fill(~allowed, -torch.inf)
    safe = torch.where(allowed.any(-1,keepdim=True),safe,torch.zeros_like(safe))
    return safe.softmax(-1) * allowed.to(logits.dtype)


def positional_encoding(x):
    """Upstream decoder's sinusoidal 3D positional feature, on the input device."""
    b,d,h,w,c = x.shape
    n = int(math.ceil(c/6)*2)
    inv = 1/(10000**(torch.arange(0,n,2,device=x.device,dtype=torch.float32)/n))
    emb = x.new_zeros(d,h,w,n*3)
    values = []
    for length in (d,h,w):
        phase = torch.arange(length,device=x.device,dtype=torch.float32)[:,None]*inv[None]
        values.append(torch.cat((phase.sin(),phase.cos()),-1).to(x.dtype))
    emb[...,:n] = values[0][:,None,None]
    emb[...,n:2*n] = values[1][None,:,None]
    emb[...,2*n:] = values[2][None,None]
    return emb[None,...,:c].expand(b,-1,-1,-1,-1)


class WindowAttention(nn.Module):
    def __init__(self, channels, heads, decoder=False):
        super().__init__()
        self.heads = heads
        self.scale = (channels//heads)**-.5
        self.relative_position_bias_table = nn.Parameter(torch.zeros(7**3,heads))
        nn.init.trunc_normal_(self.relative_position_bias_table,std=.02)
        self.qkv = nn.Linear(channels,3*channels,bias=True)
        self.proj = nn.Linear(channels,channels)
        self.decoder = decoder
        if decoder:
            self.ratio = nn.Parameter(torch.tensor(.5))

    def forward(self, x, allowed, window, cache=None):
        b,n,c = x.shape
        qkv = self.qkv(x).reshape(b,n,3,self.heads,c//self.heads).permute(2,0,3,1,4)
        q,k,v = qkv.unbind(0)
        q = q*self.scale
        index = relative_position_index(window).to(x.device)
        bias = self.relative_position_bias_table[index].permute(2,0,1)[None]
        weights = masked_softmax(q@k.transpose(-2,-1)+bias,allowed)
        local = self.proj((weights@v).transpose(1,2).reshape(b,n,c))
        if cache is not None:
            # Preserve upstream's additional decoder query scaling explicitly.
            cross = masked_softmax((q*self.scale)@cache['k'].transpose(-2,-1)+bias,allowed)
            cross = self.proj((cross@cache['v']).transpose(1,2).reshape(b,n,c))
            local = (1-self.ratio)*local + self.ratio*cross
        return local,dict(k=k,v=v)


class WindowBlock(nn.Module):
    def __init__(self, channels, heads, shifted, probability, decoder=False):
        super().__init__()
        self.shifted = shifted
        self.decoder = decoder
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        self.attn = WindowAttention(channels,heads,decoder)
        self.mlp = Mlp(channels,4*channels)
        self.drop_path = DropPath(probability)

    def forward(self, x, valid, cache=None):
        b,d,h,w,c = x.shape
        window,shift,padded = window_layout((d,h,w),shifted=self.shifted)
        layout = (b,(d,h,w),window,shift,padded)
        pad = (0,0,0,padded[2]-w,0,padded[1]-h,0,padded[0]-d)
        z = F.pad(self.norm1(x),pad)
        validity = F.pad(valid.permute(0,2,3,4,1),pad)
        if any(shift):
            z = torch.roll(z,tuple(-v for v in shift),(1,2,3))
            validity = torch.roll(validity,tuple(-v for v in shift),(1,2,3))
        tokens = window_partition(z,window)
        key_valid = window_partition(validity,window)[...,0].bool()
        regions = region_labels(padded,window,shift,x.device)
        allowed = (regions[:,:,None]==regions[:,None,:]).repeat(b,1,1) & key_valid[:,None,:]
        if cache is not None:
            if cache['layout'] != layout:
                raise ValueError('Encoder/decoder K/V window layouts do not align')
            if x.device.type != 'meta' and not torch.equal(cache['key_valid'],key_valid):
                raise ValueError('Encoder/decoder K/V valid positions do not align')
        z, kv = self.attn(tokens,allowed,window,cache)
        z = window_reverse(z,window,b,*padded)
        if any(shift):
            z = torch.roll(z,shift,(1,2,3))
        z = z[:,:d,:h,:w]
        out = x + self.drop_path(z)
        out = out + self.drop_path(self.mlp(self.norm2(out)))
        if cache is not None:
            out = out + self.mlp(self.norm2(positional_encoding(x)))
        out = out * valid.permute(0,2,3,4,1)
        kv.update(layout=layout,key_valid=key_valid)
        return out,kv


class ConvStem(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1a = nn.Conv3d(2,64,(9,5,5),padding=(4,2,2))
        self.conv1b = nn.Conv3d(64,64,(9,5,5),stride=(2,2,2),padding=(4,2,2))
        self.gn1a = nn.GroupNorm(16,64)
        self.gn1b = nn.GroupNorm(16,64)
        self.relu = nn.ReLU()

    def forward(self,x):
        return self.relu(self.gn1b(self.conv1b(self.relu(self.gn1a(self.conv1a(x))))))


class Merge(nn.Module):
    def __init__(self, channels, stride):
        super().__init__()
        self.conv_a = nn.Conv3d(channels,channels*2,(9,5,5),padding=(4,2,2))
        self.conv_b = nn.Conv3d(channels*2,channels*2,(9,5,5),stride=stride,padding=(4,2,2))
        self.gn_a = nn.GroupNorm(32,channels*2)
        self.gn_b = nn.GroupNorm(32,channels*2)
        self.relu = nn.ReLU()

    def forward(self,x):
        return self.relu(self.gn_b(self.conv_b(self.relu(self.gn_a(self.conv_a(x))))))


class EncoderStage(nn.Module):
    def __init__(self, channels, heads, probabilities):
        super().__init__()
        self.DAM = DAMLayerConcat(channels)
        self.blocks = nn.ModuleList(WindowBlock(channels,heads,bool(i%2),p) for i,p in enumerate(probabilities))
        self.norm = nn.LayerNorm(channels)

    def forward(self,x,valid):
        x = (self.DAM(x*valid)*valid).permute(0,2,3,4,1)
        caches = []
        for block in self.blocks:
            x,cache = block(x,valid)
            caches.append(cache)
        # Matches upstream: downsampling consumes pre-norm x, skip uses norm(x).
        skip = (self.norm(x)*valid.permute(0,2,3,4,1)).permute(0,4,1,2,3)
        return x.permute(0,4,1,2,3),skip,caches


class DecoderStage(nn.Module):
    def __init__(self, channels, heads, probabilities):
        super().__init__()
        self.blocks = nn.ModuleList(WindowBlock(channels,heads,bool(i%2),p,True) for i,p in enumerate(probabilities))

    def forward(self,x,valid,caches):
        x = x.permute(0,2,3,4,1)
        for block,cache in zip(self.blocks,caches):
            x,_ = block(x,valid,cache)
        return x.permute(0,4,1,2,3)


class T_RODNetFeatures(TemporalFeatureAdapter):
    def __init__(self, hidden_dim=64):
        super().__init__(hidden_dim)
        self.stem = ConvStem()
        probabilities = torch.linspace(0,.1,6,device="cpu").tolist()
        self.encoders = nn.ModuleList(EncoderStage(c,h,probabilities[2*i:2*i+2])
                                      for i,(c,h) in enumerate(zip((64,128,256),(2,4,8))))
        self.merges = nn.ModuleList([Merge(64,(2,2,2)),Merge(128,(1,2,2))])
        self.first_expand = nn.Sequential(nn.Conv3d(256,128,1),nn.GroupNorm(32,128),nn.ReLU())
        self.fuse128 = nn.Conv3d(256,128,1)
        self.decode128 = DecoderStage(128,4,probabilities[2:4])
        self.second_expand = nn.Sequential(nn.Conv3d(128,64,1),nn.GroupNorm(16,64),nn.ReLU())
        self.fuse64 = nn.Conv3d(128,64,1)
        self.decode64 = DecoderStage(64,2,probabilities[:2])

    def features(self,x,validity=None):
        input_shape = x.shape[2:]
        if validity is None:
            validity = torch.ones((x.shape[0],1,*input_shape),device=x.device,dtype=torch.bool)
        x = self.stem(x)
        valid = F.adaptive_max_pool3d(validity.float(),x.shape[2:]).bool()
        skips,caches,valids = [],[],[]
        for i,encoder in enumerate(self.encoders):
            raw,skip,cache = encoder(x,valid)
            skips.append(skip);caches.append(cache);valids.append(valid)
            if i<2:
                x = self.merges[i](raw)
                valid = F.adaptive_max_pool3d(valid.float(),x.shape[2:]).bool()
            else:
                x = skip
        x = F.interpolate(self.first_expand(x),size=skips[1].shape[2:],mode='trilinear',align_corners=True)
        x = self.decode128(self.fuse128(torch.cat((x,skips[1]),1)),valids[1],caches[1])
        x = F.interpolate(self.second_expand(x),size=skips[0].shape[2:],mode='trilinear',align_corners=True)
        x = self.decode64(self.fuse64(torch.cat((x,skips[0]),1)),valids[0],caches[0])
        return F.interpolate(x,size=input_shape,mode='trilinear',align_corners=True)
