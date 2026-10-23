"""A censored braking target must not remove an observed collision teacher."""
import importlib.util
from pathlib import Path

import torch
from radar_vla.model import ModelConfig,RadarVLAGrounder
from radar_vla.planner import PlannerConfig,RiskConditionedPlanner
from radar_vla.risk_distillation import configure_risk_head_training

spec=importlib.util.spec_from_file_location('pcol3_control',Path(__file__).resolve().parents[1]/'scripts/refine_risk_pcol3_teacher.py')
control=importlib.util.module_from_spec(spec);spec.loader.exec_module(control)


def test_teacher_replaces_only_known_pcol3_and_never_uses_unknown_braking_label():
    predicted=torch.tensor([[.1,.2,.3,7.,1.],[.1,.2,.3,7.,1.]],requires_grad=True)
    target=torch.tensor([[1.,1.,1.,0.,float('nan')],[0.,0.,float('nan'),0.,0.]])
    mask=torch.tensor([[True,True,True,True,False],[True,True,False,True,True]])
    teacher,known=control.pcol3_teacher_condition(predicted,target,mask)
    assert known.tolist()==[True,False]
    assert teacher[0].tolist()==[predicted[0,0].item(),predicted[0,1].item(),1.,7.,1.]
    assert torch.equal(teacher[1],predicted[1])
    assert not teacher.requires_grad and torch.equal(predicted.detach()[:,2],torch.tensor([.3,.3]))


def test_missing_areq_still_has_mode_gradient_but_missing_pcol3_does_not():
    torch.manual_seed(42)
    grounder=RadarVLAGrounder(ModelConfig(hidden_dim=8,num_heads=2,num_queries=2,use_detection=False,planner_ego_context=True))
    planner=RiskConditionedPlanner(PlannerConfig(hidden_dim=8,radar_dim=8,num_heads=2,num_layers=1))
    configure_risk_head_training(grounder,planner)
    teacher=torch.tensor([[5.,-5.]],requires_grad=True)
    batch=dict(radar_tokens=torch.randn(1,3,8),instruction=['drive'],
        risk_target=torch.tensor([[0.,0.,1.,0.,float('nan')]]),
        risk_mask=torch.tensor([[True,True,True,True,False]]),pcol3_teacher_logits=teacher)
    loss=control.pcol3_refinement_loss(grounder,planner,batch)
    assert loss['mode_kl']>0
    loss['mode_kl'].backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in grounder.parameters())
    assert teacher.grad is None and all(p.grad is None for p in planner.parameters())
    batch['risk_mask'][:,2]=False
    assert control.pcol3_refinement_loss(grounder,planner,batch)['mode_kl'].item()==0
