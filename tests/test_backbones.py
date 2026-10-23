import copy
import importlib.util
import math
from pathlib import Path
import sys

import pytest
import torch

from radar_vla.model import ModelConfig, RadarVLAGrounder
from radar_vla.losses import loss_grounding
from test_radar_vla_detection import batch as detection_batch


@pytest.fixture(autouse=True)
def single_thread():
    torch.set_num_threads(1)


@pytest.fixture(params=['rodnet_hg','trodnet'])
def backbone(request):
    return request.param


def model(backbone, **options):
    return RadarVLAGrounder(ModelConfig(backbone=backbone, hidden_dim=8, num_heads=2,
                                       num_queries=2, max_grid_size=2, **options))


def test_legacy_cnn_loads_and_outputs_bitwise_identically():
    path = Path(__file__).parent / 'fixtures/legacy_cnn_model.py'
    name = 'radar_vla._legacy_cnn_reference'
    spec = importlib.util.spec_from_file_location(name, path)
    legacy = importlib.util.module_from_spec(spec)
    sys.modules[name] = legacy
    spec.loader.exec_module(legacy)
    old = legacy.RadarVLAGrounder(legacy.ModelConfig(hidden_dim=8, num_heads=2, num_queries=2, max_grid_size=2)).eval()
    new = model('cnn2d').eval()
    new.load_state_dict(old.state_dict())
    assert list(old.state_dict()) == list(new.state_dict())
    with torch.no_grad():
        a, b = old(detection_batch()), new(detection_batch())
    assert a.keys() == b.keys()
    for key in a:
        torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)


def test_checkpoint_boundary_even_when_strict_false(backbone):
    cnn, target = model('cnn2d'), model(backbone)
    with pytest.raises(ValueError, match='backbone'):
        target.load_state_dict(cnn.state_dict(), strict=False)
    with pytest.raises(ValueError, match='backbone'):
        cnn.load_state_dict(target.state_dict(), strict=False)
    target.load_state_dict(target.state_dict())
    with pytest.raises(ValueError, match='backbone'):
        ModelConfig(backbone='misspelled')


@pytest.mark.parametrize('t,r,a', [(4,256,107), (1,17,19), (3,16,21), (5,25,31)])
def test_native_shape_adapter_on_meta(backbone, t, r, a):
    with torch.device('meta'):
        enc = model(backbone).encoder.eval()
        out = enc(torch.empty(2,t,2,r,a))
    assert out.shape == (2,t,8,math.ceil(r/4),math.ceil(a/4))


def test_temporal_features_have_history_gradient(backbone):
    m = model(backbone).eval()
    x = torch.rand(1,4,2,16,20, requires_grad=True)
    feature = m.encoder(x)
    feature[:,-1].square().mean().backward()
    assert x.grad[:,:-1].abs().sum() > 0
    assert torch.isfinite(x.grad).all()


def test_no_temporal_ignores_all_history(backbone):
    m = model(backbone, use_temporal=False).eval()
    batch = detection_batch()
    changed = copy.deepcopy(batch)
    changed['radar'][:,:-1] *= 100
    with torch.no_grad():
        a, b = m(batch), m(changed)
    for key in a:
        torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)


def test_invalid_doppler_is_masked_but_power_retained(backbone):
    m = model(backbone).eval()
    batch = detection_batch()
    batch['radar_doppler_valid'] = torch.zeros_like(batch['radar'][:,:,0], dtype=torch.bool)
    changed = copy.deepcopy(batch)
    changed['radar'][:,:,1] += 999
    captured = []
    hook = m.encoder.register_forward_pre_hook(lambda _, args: captured.append(args[0].detach().clone()))
    with torch.no_grad():
        a, b = m(batch), m(changed)
    hook.remove()
    assert not captured[0][:,:,1].any()
    torch.testing.assert_close(captured[0][:,:,0], torch.log1p(batch['radar'][:,:,0].clamp_min(0)))
    assert captured[0][:,:,0].abs().sum() > 0
    for key in a:
        torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)


def test_empty_and_more_than_eight_targets_finite_loss(backbone):
    m = model(backbone).train()
    one = detection_batch()
    batch = {k: v.repeat(2, *([1]*(v.ndim-1))) for k,v in one.items()}
    fields = ['agent_state', 'agent_mask', 'agent_state_mask', 'agent_future', 'agent_future_mask',
              'agent_class', 'agent_class_mask', 'agent_size', 'agent_heading', 'agent_box_mask']
    for key in fields:
        v = batch[key]
        batch[key] = v.repeat(1, 5, *([1]*(v.ndim-2)))
    batch['agent_mask'][0] = False
    assert batch['agent_mask'][1].sum() == 10
    out = m(batch)
    assert out['detection_logits'].shape == (2,40)
    assert out['risk'].shape == (2,5)
    loss = loss_grounding(out, batch)['total']
    loss.backward()
    assert torch.isfinite(loss)
    for component in [m.encoder, m.detector, m.risk_head]:
        grads = [p.grad for p in component.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        assert sum(float(g.abs().sum()) for g in grads) > 0


def test_planner_training_gradient_route_without_update(backbone, tmp_path):
    from radar_vla import pipeline
    from radar_vla.synthetic import write_synthetic_dataset
    from radar_vla.data import RadarDataset, collate_batch
    manifest = write_synthetic_dataset(tmp_path/'data', scenes_per_split=1, frames_per_scene=1)
    batch = collate_batch([RadarDataset(manifest, 'train')[0]])
    cfg = pipeline.resolve_config(dict(model=dict(backbone=backbone, hidden_dim=8, num_heads=2,
        num_queries=2, max_grid_size=2), planner=dict(num_layers=1), language_model=dict(backend='tiny')))
    g = RadarVLAGrounder(ModelConfig(**cfg['model']))
    p = pipeline.make_planner(cfg)
    total, physical, pred = pipeline._loss(g, p, batch)
    assert torch.isfinite(total)
    total.backward()
    assert any(v.grad is not None and v.grad.abs().sum()>0 for v in p.parameters())
    assert any(v.grad is not None and v.grad.abs().sum()>0 for v in g.encoder.parameters())
