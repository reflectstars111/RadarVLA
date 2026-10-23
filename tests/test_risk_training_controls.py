"""Exercise real gradient routes and a complete frozen SFT optimizer step."""
import copy

import pytest
import torch

from radar_vla import pipeline
from radar_vla.data import RadarDataset, collate_batch
from radar_vla.model import ModelConfig, RadarVLAGrounder
from radar_vla.synthetic import write_synthetic_dataset


@pytest.fixture
def tiny_data(tmp_path):
    torch.set_num_threads(1)
    manifest=write_synthetic_dataset(tmp_path/'data',scenes_per_split=1,frames_per_scene=1)
    config=dict(model=dict(hidden_dim=8,num_heads=2,num_queries=2,planner_ego_context=True),
                planner=dict(num_layers=1),language_model=dict(backend='tiny'))
    return manifest,config


def test_detach_only_blocks_planner_to_risk_head_not_physical_supervision(tiny_data):
    manifest,config=tiny_data
    config=pipeline.resolve_config(config)
    grounder=RadarVLAGrounder(ModelConfig(**config['model']))
    planner=pipeline.make_planner(config)
    grounder.eval();planner.eval()
    batch=collate_batch([RadarDataset(manifest,'train')[0]])
    head=tuple(grounder.risk_head.parameters())
    for detach in (False,True):
        total,components,_=pipeline._loss(grounder,planner,batch,detach_risk_condition=detach)
        if not detach:
            reference_total=total.detach().clone()
        else:
            torch.testing.assert_close(total.detach(),reference_total,rtol=0,atol=0)
        planning=total-components['total']
        gradients=torch.autograd.grad(planning,head,retain_graph=True,allow_unused=True)
        amount=sum(float(g.abs().sum()) for g in gradients if g is not None)
        assert (amount==0.) if detach else (amount>0.)
        physical=torch.autograd.grad(components['total'],head,retain_graph=True,allow_unused=True)
        assert sum(float(g.abs().sum()) for g in physical if g is not None)>0.
        radar=torch.autograd.grad(planning,tuple(grounder.encoder.parameters()),allow_unused=True)
        assert sum(float(g.abs().sum()) for g in radar if g is not None)>0.


def test_frozen_sft_updates_planner_and_keeps_every_grounder_tensor_exact(tiny_data,tmp_path,monkeypatch):
    manifest,config=tiny_data
    pipeline.run_training(manifest,tmp_path/'grounding',epochs=1,config=config)
    initial=torch.load(tmp_path/'grounding/best.pt',weights_only=True)
    original=pipeline.make_planner
    captured={}
    def capture(options):
        planner=original(options)
        captured['planner_initial']=copy.deepcopy(planner.state_dict())
        captured['planner']=planner
        return planner
    monkeypatch.setattr(pipeline,'make_planner',capture)
    original_forward=RadarVLAGrounder.forward
    def ensure_frozen_eval(model,batch):
        if not any(p.requires_grad for p in model.parameters()):
            assert not model.training
        return original_forward(model,batch)
    monkeypatch.setattr(RadarVLAGrounder,'forward',ensure_frozen_eval)
    result=pipeline.run_training(manifest,tmp_path/'frozen',stage='sft',epochs=1,config=config,
        init_grounding=tmp_path/'grounding/best.pt',freeze_grounder=True,validation_generation_samples=0)
    trained=torch.load(tmp_path/'frozen/last.pt',weights_only=True)
    assert result['state']=='complete' and trained['protocol']['freeze_grounder']
    assert not trained['protocol']['detach_risk_condition']
    for name,tensor in initial['grounder'].items():
        torch.testing.assert_close(trained['grounder'][name],tensor,rtol=0,atol=0)
    assert any(not torch.equal(value,trained['planner'][name])
               for name,value in captured['planner_initial'].items())
    optimizer_parameters=trained['optimizer']['param_groups'][0]['params']
    assert len(optimizer_parameters)==sum(p.requires_grad for p in captured['planner'].parameters())
    assert not (tmp_path/'frozen/test_metrics.json').exists()
    with pytest.raises(ValueError,match='protocol|fingerprint'):
        pipeline.run_training(manifest,tmp_path/'frozen',stage='sft',epochs=1,config=config,
            resume=True,freeze_grounder=False,validation_generation_samples=0)


@pytest.mark.parametrize('options',[
    dict(freeze_grounder=True),dict(detach_risk_condition=True),dict(freeze_grounder=1),
    dict(detach_risk_condition='yes'),dict(stage='sft',freeze_grounder=True,detach_risk_condition=True)])
def test_invalid_controls_fail_before_reading_data(tmp_path,options):
    with pytest.raises(ValueError,match='controls|separate'):
        pipeline.run_training(tmp_path/'missing.jsonl',tmp_path/'never-created',**options)
    assert not (tmp_path/'never-created').exists()
