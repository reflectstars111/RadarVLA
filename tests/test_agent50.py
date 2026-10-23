"""Single-class migration invariants and forward/backward checks; no optimizer steps."""
import copy
import json
import math
from pathlib import Path
import numpy as np
import pytest
import torch
from radar_vla import collection_adapter as adapter
from radar_vla.agent50_migration import migrate_record


@pytest.fixture
def example(tmp_path):
    from radar_vla.synthetic import write_synthetic_dataset
    manifest=write_synthetic_dataset(tmp_path/'legacy',scenes_per_split=3,frames_per_scene=1,frames=4,range_bins=16,azimuth_bins=17)
    r=next(json.loads(l) for l in manifest.open() if json.loads(l)['split']=='train' and json.loads(l)['scenario']=='clear')
    r.update(episode_id='ep',carla_frame=103,planner_agent_ids=[a['id'] for a in r['agents']])
    r=adapter._attach_supervision(r)
    radar=r['sensors']['radar'];radar['azimuth_rad']=np.radians(np.linspace(-53,53,17)).tolist()
    for kind in ('power','unfolded_doppler'):
        data=np.load(manifest.parent/radar[kind]);names=[]
        for i,plane in enumerate(data):
            path=tmp_path/f'{i:06d}_{kind}.npy';np.save(path,plane);names.append(str(path))
        radar[kind]=names
    valid=tmp_path/'valid.npy';np.save(valid,np.ones((4,16,17),bool));radar['doppler_valid']=str(valid)
    episode=tmp_path/'episode';episode.mkdir();r['route_archive']=str(episode/'route.json')
    mount=dict(x=1.5,y=.2,z=0.,roll=0.,pitch=0.,yaw=0.)
    meta=dict(episode_id='ep',agent_label_range_m=50,map_label_range_m=50,sensors=dict(semantic_lidar=mount))
    world=[]
    for a in r['agents']:
        world.append(dict(actor_id=a['id'],type_id='vehicle.test',actor_type='vehicle',role_name='parked',
            pose=dict(position_m=[*a['position'],0.],rotation_deg=[0,0,math.degrees(-a['heading'])]),
            bbox=dict(center_actor_m=[0,0,0],extent_m=[a['size'][0]/2,a['size'][1]/2,1.],rotation_actor_deg=[0,0,0])))
    lookup={}
    for i,dt in enumerate(radar['time_offsets_s']):
        lookup[episode/'frames'/f'{i:06d}'/'frame.json']=dict(index=i,episode_id='ep',carla_frame=100+i,timestamp_s=r['timestamp_s']+dt,
            ego=dict(pose=dict(position_m=[r['ego']['velocity'][0]*dt,0.,0.],rotation_deg=[0,0,0]),
                velocity_world_mps=[*r['ego']['velocity'],0.],angular_velocity_world_degps=[0.,0.,5.]),actors=world)
    lookup[episode/'frames'/'000003'/'agents.json']=dict(carla_frame=103,agents=copy.deepcopy(world))
    return r,episode,meta,lookup


def test_migration_preserves_instances_and_all_future_and_risk(example):
    old,ep,meta,lookup=example;before=copy.deepcopy(old)
    new,detail=migrate_record(old,ep,lookup.__getitem__,meta)
    assert old==before
    assert detail['old_ids']==detail['new_ids']
    assert new['agents'][0]['category']=='agent'
    assert new['agents'][0]['type_id']=='vehicle.test'
    assert new['agents'][0]['role_name']=='parked'
    assert detail['park_role']==1
    for k in ('ego','risk_label','agent_risk_labels','teacher_mode','planner_agent_ids','map'):
        assert new[k]==old[k]
    for a,b in zip(old['agents'],new['agents']):
        for k in a:
            if k!='category':assert a[k]==b[k]
    for k in ('power','unfolded_doppler','doppler_valid','range_m','azimuth_rad','time_offsets_s'):
        assert new['sensors']['radar'][k]==old['sensors']['radar'][k]
    assert max(detail['velocity_error_norms_mps'])>0


def test_unknown_and_stationary_agents_are_not_removed_by_relabel(example):
    old,ep,meta,lookup=example
    old['agents'][0]['category']='unknown';old['agents'][0]['velocity']=[0.,0.]
    old['agents'][0]['future_xy'][-1]=[70.,0.]
    new,detail=migrate_record(old,ep,lookup.__getitem__,meta)
    assert detail['stationary']==1 and len(new['agents'])==1
    assert new['agents'][0]['source_category']=='unknown'
    assert new['agents'][0]['future_xy'][-1]==[70.,0.]


def test_sensor_velocity_full_offset_and_world_rotation():
    frame=dict(ego=dict(pose=dict(position_m=[0,0,0],rotation_deg=[0.,0.,90.]),
        velocity_world_mps=[0.,10.,0.],angular_velocity_world_degps=[0.,0.,90.]))
    # World mount [-1,2,0], omega z=pi/2 -> added world [-pi,-pi/2,0].
    assert adapter._radar_sensor_velocity(frame,90.,dict(x=2.,y=1.,z=0.))==pytest.approx([10-math.pi/2,-math.pi])


def test_new_dataset_loss_decoder_and_metrics(example,tmp_path):
    from radar_vla.data import RadarDataset,collate_batch
    from radar_vla.model import ModelConfig,RadarVLAGrounder
    from radar_vla.losses import loss_grounding
    from radar_vla.detection import decode_detections
    from radar_vla.detection_metrics import DetectionMetrics
    from radar_vla.planner import PlannerConfig,create_supervision
    from radar_vla.tokenizer import PhysicalTokenizer
    old,ep,meta,lookup=example;new,_=migrate_record(old,ep,lookup.__getitem__,meta)
    path=tmp_path/'new.jsonl';path.write_text(json.dumps(new)+'\n')
    sample=RadarDataset(path,'train')[0];batch=collate_batch([sample])
    assert batch['agent_class'].tolist()==[[0]] and batch['agent_class_mask'].all()
    config=ModelConfig(detection_class_schema='agent_v1',hidden_dim=16,num_heads=4,max_grid_size=2)
    model=RadarVLAGrounder(config);output=model(batch)
    assert output['detection_class_logits'].shape[-1]==1
    assert not any('class_head' in k for k in model.state_dict())
    losses=loss_grounding(output,batch)
    assert losses['detection_classification'].item()==0 and torch.isfinite(losses['total'])
    losses['total'].backward()
    assert model.detector.object_head.weight.grad is not None
    decoded=decode_detections(output,config,batch['future_times_s'])
    assert decoded[0]['num_vehicles'] is None and decoded[0]['class_counts']=={'agent':0}
    assert decoded[0]['detection_thresholds']['classification'] is None
    meter=DetectionMetrics(2.,'agent_v1');meter.update(decoded,batch);metrics=meter.compute()
    assert set(metrics['per_class'])=={'agent'} and metrics['vehicle_count_mae'] is None
    cfg=PlannerConfig();tok=PhysicalTokenizer(bins=cfg.bins)
    teacher=create_supervision(batch,tok,cfg)
    assert teacher['rows'][0]['mode']==old['teacher_mode']
    # Legacy checkpoints retain their exact classifier shape; mixing is explicit.
    legacy=RadarVLAGrounder(ModelConfig(hidden_dim=16,num_heads=4,max_grid_size=2))
    legacy.load_state_dict(legacy.state_dict())
    with pytest.raises(ValueError,match='class schema'):model.load_state_dict(legacy.state_dict(),strict=False)
    with pytest.raises(ValueError,match='class schema'):legacy.load_state_dict(model.state_dict())
    with pytest.raises(ValueError,match='schema'):loss_grounding(legacy(batch),batch)


def test_single_class_nms_and_boundary_false_positives():
    from radar_vla.detection import decode_detections
    from radar_vla.model import ModelConfig
    from radar_vla.detection_metrics import DetectionMetrics
    config=ModelConfig(detection_class_schema='agent_v1',detection_class_confidence=.99)
    xy=torch.tensor([[[10.,0.],[10.,0.],[70.,0.]]])
    output=dict(detection_logits=torch.tensor([[9.,8.,9.]]),detection_class_logits=torch.zeros(1,3,1),
        detection_state=torch.cat([xy,torch.zeros(1,3,3)],-1),detection_size=torch.ones(1,3,2)*2,
        detection_heading=torch.zeros(1,3),detection_future=xy[:,:,None],detection_future_velocity=torch.zeros(1,3,1,2))
    rows=decode_detections(output,config,torch.tensor([[.5]]))
    assert rows[0]['num_agents']==2 and all(d['category']=='agent' for d in rows[0]['detections'])
    batch=dict(agent_mask=torch.tensor([[True]]),agent_state=torch.tensor([[[10.,0.,0.,0.,0.]]]),
        agent_class=torch.zeros(1,1,dtype=torch.long),agent_class_mask=torch.ones(1,1,dtype=torch.bool),
        agent_future=xy[:,:1,None],agent_future_mask=torch.ones(1,1,1,dtype=torch.bool),
        radar_pose=torch.eye(4)[None,None],detection_annotation_domain=torch.tensor([[50.,-.925,.925]]),
        detection_annotation_domain_mask=torch.tensor([True]),detection_class_schema_id=torch.tensor([1]))
    meter=DetectionMetrics(2.,'agent_v1');meter.update(rows,batch);r=meter.compute()
    assert (r['true_positives'],r['false_positives'],r['ignored_outside_annotation_domain'])==(1,0,1)


def test_migration_does_not_add_raw_world_only_actor(example):
    old,ep,meta,lookup=example
    world=lookup[ep/'frames'/'000003'/'frame.json']['actors']
    extra=copy.deepcopy(world[0]);extra['actor_id']='not_originally_admitted';extra['pose']['position_m']=[15.,0.,0.]
    world.append(extra)
    new,detail=migrate_record(old,ep,lookup.__getitem__,meta)
    assert detail['old_ids']==detail['new_ids']
    assert detail['original_collection_missing_ids']==['not_originally_admitted']


def test_filter_removes_bicycles_even_when_old_category_is_vehicle(example):
    from radar_vla.agent50_filter import filter_record,excluded_agent_kind
    old,ep,meta,lookup=example;new,_=migrate_record(old,ep,lookup.__getitem__,meta)
    a=new['agents'][0];a['type_id']='vehicle.gazelle.omafiets'
    filtered,detail=filter_record(new)
    assert not filtered['agents'] and len(filtered['ignored_agents'])==1
    assert detail['removed']=={a['id']:'cyclist'}
    assert filtered['risk_label']==new['risk_label'] and filtered['teacher_mode']==new['teacher_mode']
    assert filtered['ego']==new['ego']
    assert filtered.get('risk_agents')==new.get('risk_agents')
    assert excluded_agent_kind(dict(source_category='vehicle',type_id='vehicle.yamaha.yzf')) is None
    assert excluded_agent_kind(dict(source_category='unknown')) is None


def test_ignored_background_has_zero_objectness_gradient_and_real_positive_overrides():
    from radar_vla.losses import loss_grounding
    points=torch.tensor([[[10.,0.],[20.,0.],[30.,0.]]])
    logits=torch.zeros(1,3,requires_grad=True)
    output=dict(detection_logits=logits,detection_class_logits=torch.zeros(1,3,1),
        risk=torch.zeros(1,5,requires_grad=True),risk_logits=torch.zeros(1,3,requires_grad=True),
        detection_state=torch.cat((points,torch.zeros(1,3,3)),-1).requires_grad_(),
        detection_size=torch.ones(1,3,2),detection_heading=torch.zeros(1,3),
        detection_future=points[:,:,None].clone(),detection_future_velocity=torch.zeros(1,3,1,2),detection_anchors=points)
    batch=dict(agent_state=torch.tensor([[[20.,0.,0.,0.,0.]]]),agent_mask=torch.tensor([[True]]),
        agent_class=torch.tensor([[0]]),agent_class_mask=torch.tensor([[True]]),detection_class_schema_id=torch.tensor([1]),
        agent_future=torch.tensor([[[[20.,0.]]]]),agent_future_mask=torch.ones(1,1,1,dtype=torch.bool),
        future_times_s=torch.tensor([[.5]]),
        detection_ignore_boxes=torch.tensor([[[10.,0.,1.,1.,0.],[20.,0.,1.,1.,0.]]]),
        detection_ignore_mask=torch.tensor([[True,True]]),detection_ignore_margin_m=torch.tensor([2.]))
    loss=loss_grounding(output,batch)
    loss['detection_objectness'].backward()
    assert logits.grad[0,0]==0 and logits.grad[0,1]<0 and logits.grad[0,2]>0


def test_ignored_prediction_not_fp_and_retained_match_has_priority():
    from radar_vla.detection_metrics import DetectionMetrics
    def detection(x):return dict(position_m=[x,0.],size_m=[1.,1.],heading_rad=0.,category='agent',future_xy=[[x,0.]])
    row=dict(detections=[detection(10.),detection(20.),detection(30.)],detection_truncated=False,num_vehicles=None)
    batch=dict(agent_mask=torch.tensor([[True]]),agent_state=torch.tensor([[[20.,0.,0.,0.,0.]]]),
        agent_class=torch.tensor([[0]]),agent_class_mask=torch.tensor([[True]]),
        agent_future=torch.tensor([[[[20.,0.]]]]),agent_future_mask=torch.ones(1,1,1,dtype=torch.bool),
        detection_ignore_boxes=torch.tensor([[[10.,0.,1.,1.,0.],[20.,0.,1.,1.,0.]]]),
        detection_ignore_mask=torch.tensor([[True,True]]),detection_ignore_margin_m=torch.tensor([2.]))
    meter=DetectionMetrics(2.,'agent_v1');meter.update([row],batch);r=meter.compute()
    assert (r['true_positives'],r['false_positives'],r['ignored_removed_agent_regions'])==(1,1,1)


def test_short_label_is_preserved_but_unusable_teacher_is_explicitly_masked():
    from radar_vla.agent50_filter import filter_record
    from radar_vla.planner import _teacher_mode,PlannerConfig
    risk=dict(pcol=[1,1,1],pcol_mask=[True]*3,dmin=0.,dmin_mask=True,areq=0.,areq_mask=False)
    r=dict(grounding_label_schema='agent50_v1',sample_id='s',split='train',teacher_mode='short',
        agents=[dict(id='p',category='agent',source_category='pedestrian',position=[10.,0.],size=[1.,1.],heading=0.)],
        planner_agent_ids=['p'],agent_risk_labels={'p':risk},risk_label=risk)
    new,_=filter_record(r)
    assert new['teacher_mode']=='short' and new['risk_label']==risk
    assert not new['planner_supervision_available']
    assert _teacher_mode(dict(planner_supervision_available=torch.tensor([False])),0,PlannerConfig()) is None


def test_collate_empty_positive_frame_with_ignore_and_normal_frame(example,tmp_path):
    from radar_vla.agent50_filter import filter_record
    from radar_vla.data import RadarDataset,collate_batch
    from radar_vla.model import ModelConfig,RadarVLAGrounder
    from radar_vla.losses import loss_grounding
    old,ep,meta,lookup=example;new,_=migrate_record(old,ep,lookup.__getitem__,meta)
    bike=copy.deepcopy(new);bike['sample_id']='bike';bike['agents'][0]['type_id']='vehicle.bh.crossbike'
    car=copy.deepcopy(new);car['sample_id']='car'
    records=[filter_record(r)[0] for r in (bike,car)]
    path=tmp_path/'mixed.jsonl';path.write_text(''.join(json.dumps(r)+'\n' for r in records))
    ds=RadarDataset(path,'train');batch=collate_batch([ds[0],ds[1]])
    assert batch['agent_class'].tolist()==[[-1],[0]]
    assert batch['detection_ignore_mask'].tolist()==[[True],[False]]
    assert batch['agent_mask'].tolist()==[[False],[True]]
    model=RadarVLAGrounder(ModelConfig(hidden_dim=16,num_heads=4,max_grid_size=2,detection_class_schema='agent_v1'))
    loss=loss_grounding(model(batch),batch)
    assert torch.isfinite(loss['total'])
    assert loss['detection_classification']==0


def test_local_risk_dataset_loss_teacher_and_schema_compatibility(example, tmp_path):
    from radar_vla.agent50_filter import filter_record
    from radar_vla.local_risk import migrate_risk_record,LOCAL_RISK_SCHEMA
    from radar_vla.data import RadarDataset,collate_batch
    from radar_vla.model import RadarVLAGrounder,ModelConfig
    from radar_vla.losses import loss_grounding
    from radar_vla.pipeline import validate_input_shapes,resolve_config
    from radar_vla.planner import PlannerConfig,create_supervision
    from radar_vla.tokenizer import PhysicalTokenizer
    r,ep,meta,lookup=example
    r,_=migrate_record(r,ep,lookup.__getitem__,meta);r,_=filter_record(r);new=migrate_risk_record(r)
    p=tmp_path/'local.jsonl';p.write_text(json.dumps(new)+'\n')
    ds=RadarDataset(p,'train');sample=ds[0];batch=collate_batch([sample])
    assert ds.risk_schema==LOCAL_RISK_SCHEMA
    assert sample['risk_target'].tolist()==pytest.approx([*new['local_risk']['pcol'],new['local_risk']['dmin'],new['local_risk']['areq']])
    cfg=resolve_config(dict(data=dict(risk_supervision_schema=LOCAL_RISK_SCHEMA),
        model=dict(hidden_dim=16,num_heads=2,detection_class_schema='agent_v1')))
    validate_input_shapes(ds,cfg)
    legacy_cfg=copy.deepcopy(cfg);legacy_cfg['data'].pop('risk_supervision_schema')
    with pytest.raises(ValueError,match='Risk supervision'):validate_input_shapes(ds,legacy_cfg)
    model=RadarVLAGrounder(ModelConfig(**cfg['model']));before={k:v.clone() for k,v in model.state_dict().items()}
    pred=model(batch);loss=loss_grounding(pred,batch);assert torch.isfinite(loss['total'])
    loss['total'].backward()
    for k,v in model.state_dict().items():assert torch.equal(v,before[k])
    pc=PlannerConfig(hidden_dim=16,radar_dim=16,num_heads=2)
    teacher=create_supervision(batch,PhysicalTokenizer(bins=pc.bins),pc)['rows'][0]['mode']
    assert teacher==new['teacher_mode']
    # Invalid Risk components cannot affect loss even if placeholder values differ.
    masked={**batch,'risk_mask':torch.zeros_like(batch['risk_mask'])}
    first=loss_grounding(pred,masked)
    second=loss_grounding(pred,{**masked,'risk_target':torch.full_like(batch['risk_target'],999.)})
    for key in ('collision','distance','brake','monotonicity'):assert first[key]==second[key]==0.
    bad=copy.deepcopy(new);bad.pop('local_risk');p.write_text(json.dumps(bad)+'\n')
    with pytest.raises(ValueError,match='local_risk'):RadarDataset(p,'train')
    legacy=copy.deepcopy(sample);legacy['risk_supervision_schema_id']=torch.tensor(0)
    with pytest.raises(ValueError,match='Risk supervision'):collate_batch([sample,legacy])
