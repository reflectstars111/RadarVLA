import ast
import math
from pathlib import Path

import pytest
import torch
from torch import nn

from radar_vla.backbone_trodnet import (WindowAttention, WindowBlock, window_layout,
    window_partition, window_reverse, relative_position_index, region_labels, masked_softmax,
    ConvStem, Merge, DAMLayerConcat)
from radar_vla.model import ModelConfig, RadarVLAGrounder
from test_radar_vla_hf import tiny_local_qwen


@pytest.fixture(autouse=True)
def threads():
    torch.set_num_threads(1)


def test_non_square_partition_reverse_preserves_position_ids():
    x = torch.arange(2*2*8*12*3).reshape(2,2,8,12,3)
    windows = window_partition(x,(2,4,4))
    torch.testing.assert_close(window_reverse(windows,(2,4,4),2,2,8,12),x,rtol=0,atol=0)
    with pytest.raises(ValueError,match='padded'):
        window_partition(x[:,:,:7],(2,4,4))


def test_actual_relative_position_index_matches_coordinates_not_prefix():
    window=(1,2,3)
    index=relative_position_index(window)
    positions=[(d,h,w) for d in range(1) for h in range(2) for w in range(3)]
    for i,a in enumerate(positions):
        for j,b in enumerate(positions):
            delta=[x-y+3 for x,y in zip(a,b)]
            assert index[i,j] == delta[0]*49+delta[1]*7+delta[2]
    assert not torch.equal(index,relative_position_index((4,4,4))[:6,:6])


def test_small_dimensions_disable_shift_and_wrap_regions_are_separate():
    window,shift,padded=window_layout((1,4,7),shifted=True)
    assert window==(1,4,4) and shift==(0,0,2) and padded==(1,4,8)
    regions=region_labels(padded,window,shift,'cpu')
    assert (regions[0][:,None]==regions[0][None,:]).all()
    assert not (regions[-1][:,None]==regions[-1][None,:]).all()


def test_all_padding_rows_have_zero_attention_and_finite_gradients():
    logits=torch.randn(2,2,4,4,requires_grad=True)
    allowed=torch.zeros(2,4,4,dtype=torch.bool)
    allowed[0,:,:2]=True
    weights=masked_softmax(logits,allowed)
    assert torch.isfinite(weights).all() and weights[1].sum()==0
    assert weights[0,:,:,2:].sum()==0
    weights.square().sum().backward()
    assert torch.isfinite(logits.grad).all()
    assert logits.grad[1].abs().sum()==0


def test_invalid_keys_cannot_change_valid_queries():
    attention=WindowAttention(8,2).eval()
    x=torch.randn(1,4,8)
    changed=x.clone();changed[:,2:]+=10000
    allowed=torch.ones(1,4,4,dtype=torch.bool);allowed[:,:,2:]=False
    with torch.no_grad():
        a,_=attention(x,allowed,(1,2,2))
        b,_=attention(changed,allowed,(1,2,2))
    torch.testing.assert_close(a[:,:2],b[:,:2],rtol=0,atol=0)


def test_encoder_decoder_cache_alignment_and_padding_mask():
    enc=WindowBlock(8,2,True,0.).eval()
    dec=WindowBlock(8,2,True,0.,True).eval()
    x=torch.randn(1,2,5,7,8,requires_grad=True)
    valid=torch.ones(1,1,2,5,7,dtype=torch.bool);valid[:,:,:,-1]=False
    y,cache=enc(x,valid)
    z,_=dec(y,valid,cache)
    assert z.shape==x.shape and torch.isfinite(z).all()
    assert z[:,:,-1].abs().sum()==0
    z.square().mean().backward()
    assert torch.isfinite(x.grad).all()
    broken=dict(cache,layout=None)
    with pytest.raises(ValueError,match='layouts'):
        dec(y.detach(),valid,broken)
    broken=dict(cache,key_valid=~cache['key_valid'])
    with pytest.raises(ValueError,match='valid positions'):
        dec(y.detach(),valid,broken)


def test_cross_hg_trodnet_checkpoint_and_options_fail_explicitly():
    options=dict(hidden_dim=8,num_heads=2,num_queries=2)
    hg=RadarVLAGrounder(ModelConfig(backbone='rodnet_hg',**options))
    tr=RadarVLAGrounder(ModelConfig(backbone='trodnet',**options))
    with pytest.raises(ValueError,match='backbone'):
        hg.load_state_dict(tr.state_dict(),strict=False)
    with pytest.raises(ValueError,match='backbone'):
        tr.load_state_dict(hg.state_dict(),strict=False)
    for bad in [{'window_size':7},{'depths':[1,1,1]},{'deformable':True}]:
        with pytest.raises(ValueError,match='option'):
            ModelConfig(backbone='trodnet',backbone_options=bad)
    with pytest.raises(ValueError,match='option'):
        ModelConfig(backbone='cnn2d',backbone_options={'stacks':1})


def test_active_official_stem_merge_and_dam_are_numerically_preserved():
    # Execute only pinned class definitions, not optional imports or demo code.
    source=(Path(__file__).parents[1]/'third_party/trodnet/trodnet.py').read_text()
    tree=ast.parse(source)
    names={'PatchEmbed3D','PatchMerging','DAMLayerConcat'}
    module=ast.Module(body=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name in names],type_ignores=[])
    namespace={'torch':torch,'nn':nn,'np':__import__('numpy')}
    exec(compile(module,'pinned_trodnet','exec'),namespace)
    torch.manual_seed(19)
    reference=namespace['PatchEmbed3D'](img_size=(4,16,24),patch_size=(2,2,2),in_chans=2,embed_dim=64).eval()
    stem=ConvStem().eval()
    stem.load_state_dict({k:v for k,v in reference.state_dict().items() if k in stem.state_dict()})
    x=torch.randn(1,2,4,16,24)
    with torch.no_grad():torch.testing.assert_close(stem(x),reference(x),rtol=0,atol=0)
    for channels,stride,last,prefix in [(64,(2,2,2),False,'2'),(128,(1,2,2),True,'3')]:
        old=namespace['PatchMerging'](channels,last=last).eval()
        new=Merge(channels,stride).eval()
        mapping={'conv_a':'conv'+prefix+'a','conv_b':'conv'+prefix+'b',
                 'gn_a':'gn'+prefix+'a','gn_b':'gn'+prefix+'b'}
        new.load_state_dict({k:old.state_dict()[mapping[k.split('.')[0]]+'.'+k.split('.',1)[1]] for k in new.state_dict()})
        x=torch.randn(1,channels,2,8,12)
        with torch.no_grad():torch.testing.assert_close(new(x),old(x.permute(0,2,3,4,1)).permute(0,4,1,2,3),rtol=0,atol=0)
    old=namespace['DAMLayerConcat'](64).eval();new=DAMLayerConcat(64).eval();new.load_state_dict(old.state_dict())
    x=torch.randn(1,64,2,8,12)
    with torch.no_grad():torch.testing.assert_close(new(x),old(x),rtol=0,atol=0)


def test_trodnet_has_no_unused_trainable_backbone_parameters():
    m=RadarVLAGrounder(ModelConfig(backbone='trodnet',hidden_dim=8,num_heads=2,num_queries=2)).train()
    out=m.encoder(torch.rand(2,4,2,16,24))
    out.square().mean().backward()
    missing=[n for n,p in m.encoder.named_parameters() if p.grad is None]
    assert not missing,missing


@pytest.mark.parametrize('backbone',['rodnet_hg','trodnet'])
def test_native_hf_planner_joint_backward_without_optimizer(backbone,tiny_local_qwen):
    from test_radar_vla_hf import make_planner
    from test_radar_vla_detection import batch
    from radar_vla.tokenizer import token_cross_entropy
    g=RadarVLAGrounder(ModelConfig(backbone=backbone,hidden_dim=8,num_heads=2,num_queries=2,max_grid_size=2))
    p=make_planner(tiny_local_qwen,lora=True,gradient_checkpointing=True)
    out=g(batch())
    targets=torch.tensor([[p.tokenizer.token('<SHORT>'),p.tokenizer.eos_id]])
    logits=p(out['radar_tokens'],out['risk'],['Drive safely'],targets)
    loss=token_cross_entropy(logits,targets,p.tokenizer)
    loss.backward()
    assert torch.isfinite(loss)
    for parameters in [g.encoder.parameters(),g.risk_head.parameters(),p.radar_projector.parameters()]:
        assert any(v.grad is not None and torch.isfinite(v.grad).all() and v.grad.abs().sum()>0 for v in parameters)
    assert any(v.grad is not None and v.grad.abs().sum()>0 for n,v in p.named_parameters() if 'lora_' in n)
