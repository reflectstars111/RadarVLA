"""Task protocol tests: labels and gradients only, never optimizer steps."""
import copy,json
from pathlib import Path
import numpy as np
import pytest
import torch
from radar_vla.task_supervision import (local_mode,annotate_tasks,validate_tasks,stage_available,TASKWISE_PROTOCOL)
from radar_vla.local_risk import migrate_risk_record
from test_local_risk import record,agent
from test_agent50 import example


def safe_label():return dict(pcol=[0.,0.,0.],pcol_mask=[True]*3,dmin=10.,dmin_mask=True,areq=0.,areq_mask=True)


@pytest.mark.parametrize('condition',['pcol1','pcol2','dmin','areq'])
def test_each_short_condition_sufficient_with_missing_other_labels(condition):
    r=safe_label();r.update(pcol_mask=[False]*3,dmin_mask=False,areq_mask=False)
    if condition.startswith('pcol'):
        i=int(condition[-1])-1;r['pcol'][i]=1.;r['pcol_mask'][i]=True
    elif condition=='dmin':r['dmin']=.5;r['dmin_mask']=True
    else:r['areq']=3.;r['areq_mask']=True
    assert local_mode(r,set_known=True,agent_count=1)[0]=='SHORT'


def test_long_needs_all_conditions_except_confirmed_empty():
    r=safe_label();assert local_mode(r,set_known=True,agent_count=1)[0]=='LONG'
    for key in ('dmin_mask','areq_mask'):
        a=copy.deepcopy(r);a[key]=False
        assert local_mode(a,set_known=True,agent_count=1)[0]=='UNKNOWN'
    r.update(dmin_mask=False,areq_mask=False)
    assert local_mode(r,set_known=True,agent_count=0)[0]=='LONG'
    assert local_mode(r,set_known=False,agent_count=0)[0]=='UNKNOWN'


def test_danger_outside_selected_remains_short_not_unknown():
    r=record([agent(5,20+i,identifier=str(i)) for i in range(8)]+[agent(12,identifier='danger')])
    local=migrate_risk_record(r);assert not local['planner_supervision_available']
    new=annotate_tasks(local);validate_tasks(new)
    assert new['mode_label']=='SHORT' and new['mode_mask']
    assert new['grounding_available'] and not new['sft_available']
    assert len(new['agent_sets']['risk_agent_ids'])==9
    assert new['agent_sets']['sft_described_ids']==[]
    assert not stage_available(new,'sft') and stage_available(new,'grounding')


def test_short_text_subset_separate_from_full_risk_and_selected():
    r=record([agent(8+i*.2,identifier=str(i)) for i in range(10)])
    r['planner_agent_ids']=[str(i) for i in range(8)]
    new=annotate_tasks(migrate_risk_record(r));validate_tasks(new)
    assert len(new['agent_sets']['local_agent_ids'])==10
    assert len(new['agent_sets']['planner_selected_ids'])==8
    assert len(new['agent_sets']['sft_described_ids'])==2
    assert new['sft_available']


def task_record(example):
    from radar_vla.agent50_migration import migrate_record
    from radar_vla.agent50_filter import filter_record
    r,ep,meta,lookup=example;r,_=migrate_record(r,ep,lookup.__getitem__,meta);r,_=filter_record(r)
    return annotate_tasks(migrate_risk_record(r))


def load_one(r,tmp_path):
    from radar_vla.data import RadarDataset,collate_batch
    p=tmp_path/'data.jsonl';p.write_text(json.dumps(r)+'\n')
    dataset=RadarDataset(p,r['split']);return dataset,dataset[0],collate_batch([dataset[0]])


def test_unknown_preserves_grounding_and_has_no_planner_gradient(example,tmp_path):
    from radar_vla.planner import create_supervision,planner_physics_loss,PlannerConfig
    from radar_vla.tokenizer import PhysicalTokenizer,token_cross_entropy
    from radar_vla.model import ModelConfig,RadarVLAGrounder
    from radar_vla.losses import loss_grounding
    r=task_record(example);r.pop('supervision_protocol')
    for a in r['agents']:
        a['position']=[45.,10.];a['future_xy']=[[45.,10.]]*6;a['future_valid']=[True,False,False,False,False,False]
    from radar_vla.local_risk import compute_local_risk
    r['local_risk'],r['local_agent_risk_labels']=compute_local_risk(r);r=annotate_tasks(r)
    assert r['mode_label']=='UNKNOWN' and not r['sft_available']
    ds,sample,batch=load_one(r,tmp_path)
    model=RadarVLAGrounder(ModelConfig(hidden_dim=16,num_heads=2,detection_class_schema='agent_v1'))
    before={k:v.clone() for k,v in model.state_dict().items()}
    output=model(batch);loss=loss_grounding(output,batch);assert torch.isfinite(loss['total']);loss['total'].backward()
    assert all(torch.equal(v,before[k]) for k,v in model.state_dict().items())
    pc=PlannerConfig(hidden_dim=16,radar_dim=16,num_heads=2);tok=PhysicalTokenizer(pc.bins)
    sup=create_supervision(batch,tok,pc);assert sup['rows'][0]['mode'] is None
    logits=torch.randn(*sup['target_ids'].shape,tok.vocab_size,requires_grad=True)
    ce=token_cross_entropy(logits,sup['target_ids'],tok,continuous_targets=sup['continuous_targets'],mode_weight=1.)
    phys=planner_physics_loss(logits,sup,batch,tok,pc)
    assert ce==0 and phys['total']==0;(ce+phys['total']).backward();assert logits.grad.count_nonzero()==0


def test_partial_ego_and_missing_road_keep_legal_losses(example,tmp_path):
    from radar_vla.planner import create_supervision,planner_physics_loss,PlannerConfig
    from radar_vla.tokenizer import PhysicalTokenizer,token_cross_entropy
    r=task_record(example);r.pop('supervision_protocol')
    # Explicit safe Local risk is independent of the actually driven ego future.
    r['local_risk'].update(pcol=[0.]*3,pcol_mask=[True]*3,dmin=10.,dmin_mask=True,areq=0.,areq_mask=True)
    # Keep candidate audit consistent with a valid zero-deceleration target.
    r['local_risk']['braking_candidates'][0]['state']='safe'
    r['ego']['future_valid']=[True,True,True,False,False,False]
    r['road_segments']=[];r=annotate_tasks(r);assert r['mode_label']=='LONG'
    ds,sample,batch=load_one(r,tmp_path)
    pc=PlannerConfig(hidden_dim=16,radar_dim=16,num_heads=2,road_format='segments');tok=PhysicalTokenizer(pc.bins)
    sup=create_supervision(batch,tok,pc);row=sup['rows'][0]
    assert row['mode']=='long' and not row['ego']['fitted'] and row['ego']['mask'].sum()==3
    assert sup['target_ids'][0,0]==tok.token('<LONG>')
    assert tok.token('<ROAD>') not in sup['target_ids'][0].tolist()
    logits=torch.randn(*sup['target_ids'].shape,tok.vocab_size,requires_grad=True)
    loss=planner_physics_loss(logits,sup,batch,tok,pc);assert loss['road']==0 and loss['ego_trajectory']>0
    changed=copy.deepcopy(sup);changed['rows'][0]['ego']['target'][~row['ego']['mask']]=999.
    second=planner_physics_loss(logits,changed,batch,tok,pc)
    assert loss['ego_trajectory']==second['ego_trajectory']
    ce=token_cross_entropy(logits,sup['target_ids'],tok,continuous_targets=sup['continuous_targets'],mode_weight=1.)
    (loss['total']+ce).backward();assert torch.isfinite(logits.grad).all()


def test_protocol_config_and_mixed_collate_rejected(example,tmp_path):
    from radar_vla.data import collate_batch
    from radar_vla.pipeline import validate_input_shapes,resolve_config
    r=task_record(example);ds,sample,batch=load_one(r,tmp_path)
    cfg=json.loads(Path('configs/l40s_qwen25_3b_taskwise_local_v1.json').read_text());resolve_config(cfg)
    cfg['data'].pop('radar_shape');validate_input_shapes(ds,cfg)
    old=copy.deepcopy(cfg);old['data'].pop('supervision_protocol')
    with pytest.raises(ValueError,match='protocol'):validate_input_shapes(ds,old)
    legacy=copy.deepcopy(sample);legacy['taskwise_protocol_id']=torch.tensor(0)
    with pytest.raises(ValueError,match='protocol'):collate_batch([sample,legacy])
    r['mode_label']='LONG' if r['mode_label']!='LONG' else 'SHORT'
    with pytest.raises(ValueError,match='Mode'):validate_tasks(r)


def test_all_tasks_masked_zero_and_safe_mixed_batch(example,tmp_path):
    from radar_vla.data import collate_batch
    from radar_vla.model import ModelConfig,RadarVLAGrounder
    from radar_vla.losses import loss_grounding
    from radar_vla.planner import create_supervision,planner_physics_loss,PlannerConfig
    from radar_vla.tokenizer import PhysicalTokenizer,token_cross_entropy
    good=task_record(example);good.pop('supervision_protocol')
    empty=copy.deepcopy(good);empty['agent_supervision_available']=False;empty['agents']=[];empty['planner_agent_ids']=[]
    empty['agent_risk_labels']={};empty['local_agent_risk_labels']={}
    from radar_vla.local_risk import compute_local_risk
    empty['local_risk'],empty['local_agent_risk_labels']=compute_local_risk(empty)
    empty['ego']['future_valid']=[False]*6;empty['road_segments']=[]
    empty=annotate_tasks(empty);assert not stage_available(empty,'grounding') and not stage_available(empty,'sft')
    ds,sample,batch=load_one(empty,tmp_path)
    model=RadarVLAGrounder(ModelConfig(hidden_dim=16,num_heads=2,detection_class_schema='agent_v1'))
    output=model(batch);loss=loss_grounding(output,batch);assert loss['total']==0
    loss['total'].backward();assert all(p.grad is None or p.grad.count_nonzero()==0 for p in model.parameters())
    good=annotate_tasks(good);_,other,_=load_one(good,tmp_path)
    mixed=collate_batch([sample,other]);assert mixed['agent_mask'].shape[0]==2
    output=model(mixed);loss=loss_grounding(output,mixed);assert torch.isfinite(loss['total'])
    pc=PlannerConfig(hidden_dim=16,radar_dim=16,num_heads=2,road_format='segments');tok=PhysicalTokenizer(pc.bins)
    sup=create_supervision(mixed,tok,pc);assert sup['rows'][0]['mode'] is None
    logits=torch.randn(*sup['target_ids'].shape,tok.vocab_size,requires_grad=True)
    total=token_cross_entropy(logits,sup['target_ids'],tok,continuous_targets=sup['continuous_targets'],mode_weight=1.)+planner_physics_loss(logits,sup,mixed,tok,pc)['total']
    total.backward();assert logits.grad[0].count_nonzero()==0 and torch.isfinite(logits.grad).all()


def test_unknown_width_does_not_clear_road_geometry(example,tmp_path):
    from radar_vla.planner import create_supervision,planner_physics_loss,PlannerConfig
    from radar_vla.tokenizer import PhysicalTokenizer
    r=task_record(example);r.pop('supervision_protocol')
    r['local_risk'].update(pcol=[0.]*3,pcol_mask=[True]*3,dmin=10.,dmin_mask=True,areq=0.,areq_mask=True)
    r['local_risk']['braking_candidates'][0]['state']='safe'
    r['road_segments']=[dict(ordinal=0,centerline_ego_m=[[0.,0.],[5.,0.],[10.,1.]],width_m=None,width_valid=False)]
    r=annotate_tasks(r);ds,sample,batch=load_one(r,tmp_path)
    pc=PlannerConfig(hidden_dim=16,radar_dim=16,num_heads=2,road_format='segments');tok=PhysicalTokenizer(pc.bins)
    sup=create_supervision(batch,tok,pc);road=sup['rows'][0]['road']['segments'][0]
    assert road['fitted'] and not road['width_mask'];assert sup['target_ids'][0,road['width_index']]==tok.pad_id
    logits=torch.randn(*sup['target_ids'].shape,tok.vocab_size,requires_grad=True)
    loss=planner_physics_loss(logits,sup,batch,tok,pc);assert loss['road']>0;loss['total'].backward()
    assert torch.isfinite(logits.grad).all()


def test_partial_agent_future_ignores_invalid_values_and_intervals(example,tmp_path):
    from radar_vla.model import ModelConfig,RadarVLAGrounder
    from radar_vla.losses import loss_grounding
    r=task_record(example);r.pop('supervision_protocol')
    for a in r['agents']:a['future_valid']=[True,False,True,False,False,False]
    from radar_vla.local_risk import compute_local_risk
    r['local_risk'],r['local_agent_risk_labels']=compute_local_risk(r);r=annotate_tasks(r)
    _,sample,batch=load_one(r,tmp_path);model=RadarVLAGrounder(ModelConfig(hidden_dim=16,num_heads=2,detection_class_schema='agent_v1'))
    output=model(batch);first=loss_grounding(output,batch)
    changed={**batch,'agent_future':batch['agent_future'].clone()}
    changed['agent_future'][~batch['agent_future_mask']]=999.
    second=loss_grounding(output,changed)
    for name in ('detection_agent_future','detection_kinematic','detection_smooth'):assert first[name]==second[name]


def test_out_of_vocabulary_future_kept_in_metric_loss(example,tmp_path):
    from radar_vla.planner import create_supervision,planner_physics_loss,PlannerConfig
    from radar_vla.tokenizer import PhysicalTokenizer,token_cross_entropy
    r=task_record(example);r.pop('supervision_protocol')
    r['local_risk'].update(pcol=[0.]*3,pcol_mask=[True]*3,dmin=10.,dmin_mask=True,areq=0.,areq_mask=True)
    r['local_risk']['braking_candidates'][0]['state']='safe'
    r['ego']['future_xy']=[[50.*(i+1),0.] for i in range(6)];r['ego']['future_valid']=[True]*6
    r=annotate_tasks(r);_,sample,batch=load_one(r,tmp_path)
    pc=PlannerConfig(hidden_dim=16,radar_dim=16,num_heads=2,road_format='segments');tok=PhysicalTokenizer(pc.bins)
    sup=create_supervision(batch,tok,pc);assert sup['rows'][0]['masked_out_of_range_fields']>0
    assert sup['rows'][0]['ego']['target'][-1,0]==300.
    logits=torch.randn(*sup['target_ids'].shape,tok.vocab_size,requires_grad=True)
    total=token_cross_entropy(logits,sup['target_ids'],tok,continuous_targets=sup['continuous_targets'])+planner_physics_loss(logits,sup,batch,tok,pc)['total']
    assert torch.isfinite(total);total.backward();assert torch.isfinite(logits.grad).all()


def test_audit_rows_are_json_serializable(example):
    from radar_vla.taskwise_migration import summary_row
    r=task_record(example);json.dumps(summary_row(r,'baseline',r),allow_nan=False)
