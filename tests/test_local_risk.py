"""Local Risk contract; no training or optimizer updates."""
import copy
import numpy as np
import pytest
from radar_vla.geometry import box_distance,build_risk_labels
from radar_vla.local_risk import (compute_local_risk,migrate_risk_record,box_distances,check_label,LOCAL_RISK_SCHEMA)


def agent(x=12.,y=0.,velocity=0.,identifier='a'):
    times=np.arange(1,7)*.5
    return dict(id=identifier,category='agent',position=[x,y],velocity=[velocity,0.],heading=0.,size=[4.,2.],
        future_xy=[[x+velocity*t,y] for t in times],future_yaw=[0.]*6,future_valid=[True]*6)


def record(agents=None):
    agents=[agent()] if agents is None else agents
    r=dict(agents=agents,risk_agents=copy.deepcopy(agents),future_times_s=(np.arange(1,7)*.5).tolist(),
        ego=dict(velocity=[5.,0.],yaw_rate=0.,box_size=[4.,2.]),tracking_coverage_valid=[True]*6,
        agent_supervision_available=True,planner_agent_ids=[a['id'] for a in agents][:8],teacher_mode='short',
        grounding_label_schema='agent50_v1',agent_ignore_policy='pedestrian_cyclist_regions_v1',ignored_agents=[])
    r['risk_label']=build_risk_labels(r)
    r['agent_risk_labels']={a['id']:build_risk_labels(dict(r,risk_agents=[a])) for a in agents}
    return r


def test_vector_geometry_matches_reference():
    rng=np.random.default_rng(42)
    a=rng.normal(size=(100,2))*10;b=rng.normal(size=(100,2))*10
    ah=rng.normal(size=100);bh=rng.normal(size=100)
    az=rng.uniform(.1,8,(100,2));bz=rng.uniform(.1,8,(100,2))
    actual=box_distances(a,ah,az,b,bh,bz)
    expected=np.array([box_distance(*args) for args in zip(a,ah,az,b,bh,bz)])
    assert np.allclose(actual,expected,atol=1e-11)


@pytest.mark.parametrize('yaw_rate',[0.,.2,-.4])
def test_legacy_complete_six_knot_equivalence(yaw_rate):
    r=record([agent(),agent(20,5,2,'b')]);r['ego']['yaw_rate']=yaw_rate
    old=build_risk_labels(r);new,individual=compute_local_risk(r)
    for k in ('pcol','pcol_mask','dmin_mask','areq_mask','braking_feasible'):assert new[k]==old[k]
    assert new['dmin']==pytest.approx(old['dmin'],abs=1e-10)
    assert new['areq']==old['areq'];assert len(new['braking_candidates'])==17
    check_label(new)
    for label in individual.values():check_label(label)


def test_fixed_local_set_empty_and_global_outside():
    r=record([]);r['risk_agents']=[agent()];r['risk_label']=build_risk_labels(r)
    r['tracking_coverage_valid']=[False]*6
    new=migrate_risk_record(r)
    assert new['global_risk']==r['risk_label']
    assert new['local_risk']['pcol']==[0,0,0] and new['local_risk']['pcol_mask']==[True]*3
    assert not new['local_risk']['dmin_mask'] and not new['local_risk']['areq_mask']
    assert new['teacher_mode']=='long';assert new['planner_supervision_available']


def test_missing_future_positive_kept_negative_unknown():
    a=agent(6.);a['future_valid']=[True,False,False,False,False,False]
    label,_=compute_local_risk(record([a]));assert label['pcol']==[1,1,1];assert label['pcol_mask']==[True]*3
    a=agent(30.,10.);a['future_valid']=[True,True,False,False,False,False]
    label,_=compute_local_risk(record([a]));assert label['pcol_mask']==[True,False,False]
    assert not label['dmin_mask'] and not label['areq_mask'];assert label['areq']==0.


def test_native_missing_interval_not_bridged():
    a=agent(20.,8.);a['future30_valid']=[True]*30;a['future30_valid'][1]=False
    label,_=compute_local_risk(record([a]));assert not label['pcol_mask'][0]
    assert not label['dmin_mask'];assert label['areq_reason']=='incomplete_future'


def test_contact_and_no_feasible_solution():
    a=agent(3.)
    label,_=compute_local_risk(record([a]));assert label['dmin']==0.;assert not label['areq_mask']
    assert label['areq_reason']=='current_dangerous_proximity';assert len(label['braking_candidates'])==17
    # Stopped ego struck later: no constant longitudinal brake is a solution.
    a=agent(12.,velocity=-5.);r=record([a]);r['ego']['velocity']=[0.,0.]
    label,_=compute_local_risk(r);assert not label['areq_mask'];assert label['areq_reason']=='no_feasible_deceleration'


def test_future_exits_domain_not_trimmed_and_static_moving_preserved():
    r=record([agent(49.,0.,15.),agent(20.,8.,0.,'b')]);before=copy.deepcopy(r)
    new=migrate_risk_record(r)
    assert r==before and new['agents']==r['agents']
    assert new['local_risk']['coverage']['complete_3s']
    assert new['local_agent_risk_labels']['a']['coverage']['agent_observed_intervals']['a'][-1][1]==3.


def test_more_than_eight_keeps_hazard_and_masks_sft():
    agents=[agent(5.,20.+i,0.,str(i)) for i in range(8)]+[agent(12.,0.,0.,'hazard')]
    r=record(agents);new=migrate_risk_record(r)
    assert new['local_risk']['pcol'][2]==1
    assert 'hazard' in new['local_risk']['pcol_source_agent_ids'][2]
    assert len(new['local_agent_risk_labels'])==9
    assert new['planner_agent_ids']==r['planner_agent_ids'];assert not new['planner_supervision_available']
    assert new['planner_supervision_unavailable_reason']=='local_hazard_outside_planner_selected'


def test_all_grid_values_checked_without_monotonic_assumption():
    # Rear approaching actor: some weak braking is safe, stronger braking is dangerous.
    r=record([agent(-16.,0.,8.)]);r['ego']['velocity']=[8.,0.]
    label,_=compute_local_risk(r);states=[c['state'] for c in label['braking_candidates']]
    assert states[0]=='safe' and 'dangerous' in states[1:]
    assert label['areq_mask'] and label['areq']==0.;assert len(states)==17


def test_local_schema_never_mutates_global_or_ignored():
    r=record([agent(20.,8.)]);r['ignored_agents']=[dict(id='ped')]
    r['planner_supervision_available']=False
    new=migrate_risk_record(r)
    assert new['global_risk']==r['risk_label'] and new['risk_label']==r['risk_label']
    assert new['risk_agents']==r['risk_agents'];assert new['teacher_mode']=='long'
    assert new['planner_supervision_available'];assert not new['legacy_planner_supervision_available']


@pytest.mark.parametrize('coverage',[[True]*6,[False]*6,[True,True,False,False,False,False]])
@pytest.mark.parametrize('agents',[[],[agent(20,8)],[agent(12)],[agent(3)]])
def test_fast_legacy_exclusion_reconstruction_matches_original(coverage,agents):
    from radar_vla.local_risk_audit import legacy_fast
    r=record(copy.deepcopy(agents));r['tracking_coverage_valid']=coverage
    actual=legacy_fast(r);expected=build_risk_labels(r)
    for k,v in actual.items():
        if k=='dmin':assert v==pytest.approx(expected[k],abs=1e-10)
        else:assert v==expected[k]


def test_resume_discards_uncommitted_tails_and_checks_signature(tmp_path,monkeypatch):
    import json
    import radar_vla.local_risk_migration as migration
    source=tmp_path/'source.jsonl';config=tmp_path/'cfg.json';output=tmp_path/'out'
    rows=[]
    for i in range(60):
        r=record([]);r.update(sample_id=f's{i}',episode_id='ep',split='train',category='normal');rows.append(r)
    source.write_text(''.join(json.dumps(r)+'\n' for r in rows));config.write_text('{}')
    interrupted=[True]
    class FakeEvidence:
        def __init__(self,source):self.hashes={}
        def verify(self,r):
            if r['sample_id']=='s50' and interrupted[0]:raise ValueError('synthetic interruption')
            return {}
    monkeypatch.setattr(migration,'RawEvidence',FakeEvidence)
    with pytest.raises(ValueError,match='synthetic interruption'):migration.migrate(source,output,config)
    assert json.loads((output/'progress.json').read_text())['rows']==50
    for name in ('manifest.partial.jsonl','samples.jsonl'):
        with (output/name).open('ab') as f:f.write(b'uncommitted broken tail')
    config.write_text('{"changed":true}')
    with pytest.raises(ValueError,match='mismatch'):migration.migrate(source,output,config,resume=True)
    config.write_text('{}');interrupted[0]=False
    # This deliberately tiny fixture must never publish as the real 8835-frame cohort.
    with pytest.raises(ValueError,match='cohort count'):migration.migrate(source,output,config,resume=True)
    final=[json.loads(l)['sample_id'] for l in (output/'manifest.partial.jsonl').open()]
    assert final==[f's{i}' for i in range(60)]
    assert not (output/'manifest.jsonl').exists()


def test_bbox_boundary_membership_is_not_center_clipping():
    from radar_vla.annotation_domain import box_intersects_annotation_domain
    limits=[50.,-np.radians(53),np.radians(53)];pose=np.eye(4)
    a=agent(51.);assert box_intersects_annotation_domain(a['position'],a['size'],a['heading'],limits,pose)
    boundary=agent(20*np.cos(np.radians(54)),20*np.sin(np.radians(54)),identifier='edge')
    assert box_intersects_annotation_domain(boundary['position'],boundary['size'],boundary['heading'],limits,pose)
    r=record([a,boundary]);label,individual=compute_local_risk(r)
    assert set(individual)=={'a','edge'}
    assert label['coverage']['complete_3s']


def test_unknown_set_preserves_observed_positive_and_other_masks():
    r=record([agent(3.)]);r['agent_supervision_available']=False
    label,_=compute_local_risk(r)
    assert label['pcol']==[1.,1.,1.] and label['pcol_mask']==[True]*3
    assert label['dmin']==0. and label['dmin_mask']
    assert not label['areq_mask']


def test_empty_planner_selection_does_not_certify_nonempty_local_set_safe():
    r=record([agent(20.,8.)]);r['planner_agent_ids']=[]
    r['agents'][0]['future_valid']=[True,True,True,True,False,False]
    new=migrate_risk_record(r)
    assert new['local_risk']['pcol_mask']==[True,True,False]
    assert new['teacher_mode'] is None and not new['planner_supervision_available']
