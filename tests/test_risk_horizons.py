import copy
import numpy as np
import pytest
import torch

from radar_vla.local_risk import compute_local_risk,check_label
from radar_vla.risk_contract import LOCAL_SHORT_SCHEMA,SHORT_HORIZONS
from test_local_risk import agent,record


@pytest.mark.parametrize('x,expected',[(6.,[1,1,1]),(8.,[0,1,1]),(12.,[0,0,1]),(17.,[0,0,0])])
def test_cumulative_windows_and_2s_geometry(x,expected):
    r=record([agent(x)]);old,_=compute_local_risk(r);new,individual=compute_local_risk(r,schema=LOCAL_SHORT_SCHEMA)
    assert new['pcol']==expected
    assert new['pcol'][1:]==old['pcol'][:2]
    assert new['coverage']['geometry_times_s'][-1]==2.
    assert 'complete_3s' not in new['coverage']
    assert new['geometry_horizon_s']==2. and new['pcol_horizons_s']==[.5,1.,2.]
    for label in [new,*individual.values()]:check_label(label)
    if x==17.:
        assert old['pcol'][-1]==1 and new['dmin']>.5 and new['areq']==0


def test_missing_future_after2s_no_longer_masks_geometry():
    a=agent(30.,10.);a['future_valid']=[True]*4+[False]*2
    r=record([a]);old,_=compute_local_risk(r);new,_=compute_local_risk(r,schema=LOCAL_SHORT_SCHEMA)
    assert not old['dmin_mask'] and not old['areq_mask']
    assert new['dmin_mask'] and new['areq_mask']


def test_models_and_planners_reject_cross_horizon_weights_even_non_strict():
    from radar_vla.model import ModelConfig,RadarVLAGrounder
    from radar_vla.planner import PlannerConfig,RiskConditionedPlanner
    old=RadarVLAGrounder(ModelConfig(hidden_dim=8,num_heads=2));new=RadarVLAGrounder(ModelConfig(hidden_dim=8,num_heads=2,risk_horizon_schema=SHORT_HORIZONS))
    for a,b in [(old,new),(new,old)]:
        with pytest.raises(ValueError,match='horizons'):a.load_state_dict(b.state_dict(),strict=False)
    old=RiskConditionedPlanner(PlannerConfig(hidden_dim=8,radar_dim=8,num_heads=2,num_layers=1))
    new=RiskConditionedPlanner(PlannerConfig(hidden_dim=8,radar_dim=8,num_heads=2,num_layers=1,risk_horizon_schema=SHORT_HORIZONS))
    with pytest.raises(ValueError,match='horizons'):new.load_state_dict(old.state_dict(),strict=False)


def test_new_short_rule_includes2s_and_unknown_mask():
    from radar_vla.task_supervision import local_mode
    from radar_vla.planner import _is_hazard,PlannerConfig
    label=dict(pcol=[0,0,1],pcol_mask=[True]*3,dmin=10.,dmin_mask=True,areq=0.,areq_mask=True,pcol_horizons_s=[.5,1.,2.])
    assert local_mode(label,set_known=True,agent_count=1)[0]=='SHORT'
    cfg=PlannerConfig(risk_horizon_schema=SHORT_HORIZONS)
    assert _is_hazard(torch.tensor([0.,0.,1.,10.,0.]),torch.ones(5,dtype=torch.bool),cfg)
    label['pcol']=[0,0,0];label['pcol_mask'][2]=False
    assert local_mode(label,set_known=True,agent_count=1)[0]=='UNKNOWN'


def test_metrics_keys_and_no_positive_ap_are_explicit():
    from radar_vla.metrics import MetricAccumulator
    m=MetricAccumulator(SHORT_HORIZONS);m.risk_p=[[.1,.2],[.1,.8],[.3,.9]];m.risk_y=[[0,0],[0,1],[0,1]]
    result=m.compute();assert list(result['collision'])==['0.5s','1s','2s']
    assert result['collision']['0.5s']['average_precision'] is None
    assert result['risk_geometry_horizon_s']==2.


def test_configuration_mismatch_is_rejected_before_loading_data():
    from radar_vla.pipeline import resolve_config
    with pytest.raises(ValueError,match='horizons'):
        resolve_config(dict(model=dict(risk_horizon_schema=SHORT_HORIZONS)))


def test_record_migration_keeps_membership_and_future():
    from test_moving_agents import inputs
    from radar_vla.moving_agents import attach_supervision,rebuild_agents
    from radar_vla.risk_horizon_migration import migrate_record
    a,b,c,d=inputs();old=attach_supervision(rebuild_agents(a,b,c,d));original=copy.deepcopy(old)
    new=migrate_record(old)
    assert old==original and new['agents']==old['agents']
    assert new['future_times_s']==old['future_times_s'] and len(new['agents'])==12
    assert new['risk_supervision_schema']==LOCAL_SHORT_SCHEMA


def test_episode_repartition_preserves_test_and_has_rare_positive_each_split():
    from radar_vla.risk_splits import allocate_episode_splits
    rows=[];labels={}
    for i in range(20):
        split='train' if i<14 else 'val' if i<17 else 'test'
        for j in range(10):
            sid=f'{i}_{j}';rows.append(dict(episode_id=str(i),sample_id=sid,split=split,scene_id=f'Town01_{i}',category='normal'))
            hit=i in [0,1,17];labels[sid]=dict(pcol=[int(hit)]*3,pcol_mask=[True]*3)
    allocation,stats=allocate_episode_splits(rows,labels,tolerance=.025)
    assert all(allocation[str(i)]=='test' for i in [17,18,19])
    assert all(r['positive_counts'][0]>0 for r in stats['actual'].values())
    assert stats['actual']['train']['samples']==140
