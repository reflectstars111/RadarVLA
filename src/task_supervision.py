"""Versioned task validity and a shared Local SHORT/LONG/UNKNOWN policy."""
from __future__ import annotations
import copy
import math

TASKWISE_PROTOCOL='taskwise_local_v1'
LEGACY_PROTOCOL='legacy_tasks_v1'
MODE_INDEX={'SHORT':0,'LONG':1,'UNKNOWN':-1}


def supervision_protocol(record):
    protocol=record.get('supervision_protocol',LEGACY_PROTOCOL)
    if protocol not in (TASKWISE_PROTOCOL,LEGACY_PROTOCOL):raise ValueError(f'Unknown supervision protocol: {protocol}')
    return protocol


def local_mode(label, *, set_known, agent_count, collision_threshold=.5,clearance_threshold=.5,brake_threshold=3.):
    """A risk fact, independent of the Planner-selected subset."""
    hit=[]
    for i in (0,1):
        if label['pcol_mask'][i] and label['pcol'][i]>=collision_threshold:hit.append(f'pcol_{i+1}s')
    if label['dmin_mask'] and label['dmin']<=clearance_threshold:hit.append('dmin')
    if label['areq_mask'] and label['areq']>=brake_threshold:hit.append('areq')
    if hit:return 'SHORT',hit
    if set_known and agent_count==0 and all(label['pcol_mask'][:2]):return 'LONG',['confirmed_empty_local_set']
    if all([*label['pcol_mask'][:2],label['dmin_mask'],label['areq_mask']]):return 'LONG',['all_short_conditions_excluded']
    return 'UNKNOWN',['insufficient_local_risk_evidence']


def _risk_order(agent,labels):
    positive=[i for i,(p,m) in enumerate(zip(labels['pcol'],labels['pcol_mask'])) if p>.5 and m]
    return (min(positive,default=4),labels['dmin'] if labels['dmin_mask'] else float('inf'),str(agent['id']))


def annotate_tasks(record, *, max_agents=8,short_max_agents=2,max_road_segments=8):
    from .local_risk import LOCAL_RISK_SCHEMA
    if record.get('risk_supervision_schema')!=LOCAL_RISK_SCHEMA:raise ValueError('Taskwise supervision requires versioned Local Risk')
    if supervision_protocol(record)==TASKWISE_PROTOCOL:raise ValueError('Already taskwise; refusing repeat migration')
    new=copy.deepcopy(record);agents=record['agents'];ids=[str(a['id']) for a in agents]
    risk=record['local_risk'];individual=record['local_agent_risk_labels']
    if set(individual)!=set(ids):raise ValueError('Local Risk set differs from legal Agent set')
    selected=list(map(str,record['planner_agent_ids']))
    if len(selected)>max_agents or len(set(selected))!=len(selected) or not set(selected)<=set(ids):raise ValueError('Invalid Planner selected IDs')
    known=bool(record.get('agent_supervision_available',False))
    mode,reasons=local_mode(risk,set_known=known,agent_count=len(agents))
    critical=[a for a in agents if str(a['id']) in selected and local_mode(individual[str(a['id'])],set_known=True,agent_count=1)[0]=='SHORT']
    critical.sort(key=lambda a:_risk_order(a,individual[str(a['id'])]))
    context_consistent=mode!='SHORT' or (known and bool(critical))
    described=([str(a['id']) for a in sorted(agents,key=lambda a:(math.hypot(*a['position']),str(a['id']))) if str(a['id']) in selected]
               if mode=='LONG' else [str(a['id']) for a in critical[:short_max_agents]] if mode=='SHORT' and context_consistent else [])
    masks=dict(detection=known,risk=[*risk['pcol_mask'],risk['dmin_mask'],risk['areq_mask']],mode=mode!='UNKNOWN',
        agent_future={str(a['id']):list(a.get('future_valid',[False]*6)) for a in agents},
        ego_future=list(record['ego'].get('future_valid',[False]*6)),
        road=[dict(geometry=list(s.get('centerline_valid',[True]*len(s['centerline_ego_m']))),width=bool(s['width_valid'])) for s in record.get('road_segments',[])])
    if not record.get('road_supervision_available',True) or len(masks['road'])>max_road_segments:
        for s in masks['road']:s['geometry']=[False]*len(s['geometry']);s['width']=False
    sft_reasons=[]
    if mode=='UNKNOWN':sft_reasons.append('mode_unknown')
    if not context_consistent:sft_reasons.append('local_hazard_outside_planner_selected')
    new.update(supervision_protocol=TASKWISE_PROTOCOL,mode_label=mode,mode_mask=mode!='UNKNOWN',mode_reasons=reasons,
        task_masks=masks,grounding_available=bool(masks['detection'] or any(masks['risk'])),
        sft_available=not sft_reasons,sft_unavailable_reasons=sft_reasons,
        planner_context_consistent=context_consistent,teacher_mode=mode.lower() if mode!='UNKNOWN' else None,
        planner_supervision_available=not sft_reasons,
        planner_supervision_unavailable_reason=';'.join(sft_reasons) or None,
        previous_teacher_mode=record.get('teacher_mode'),previous_planner_supervision_available=record.get('planner_supervision_available',True),
        agent_sets=dict(local_agent_ids=ids,risk_agent_ids=list(individual),planner_selected_ids=selected,sft_described_ids=described),
        mode_policy=dict(collision_threshold=.5,clearance_threshold_m=.5,brake_threshold_mps2=3.,empty_local_exception=True))
    return new


def validate_tasks(record):
    if supervision_protocol(record)!=TASKWISE_PROTOCOL:return
    if record.get('risk_supervision_schema')!='local_agent_collision_v1':raise ValueError('Task protocol requires Local Risk')
    label=record['local_risk'];agents=record['agents'];known=bool(record.get('agent_supervision_available',False))
    mode,_=local_mode(label,set_known=known,agent_count=len(agents))
    if record.get('mode_label')!=mode or record.get('mode_mask')!=(mode!='UNKNOWN'):raise ValueError('Mode label/mask conflicts with Local evidence')
    masks=record['task_masks']
    if masks['risk']!=[*label['pcol_mask'],label['dmin_mask'],label['areq_mask']]:raise ValueError('Risk task mask mismatch')
    if masks['mode']!=(mode!='UNKNOWN') or masks['detection']!=known:raise ValueError('Task validity mismatch')
    if masks['ego_future']!=record['ego'].get('future_valid',[False]*6):raise ValueError('Ego task mask mismatch')
    if masks['agent_future']!={str(a['id']):a.get('future_valid',[False]*6) for a in agents}:raise ValueError('Agent future task mask mismatch')
    if len(masks['road'])!=len(record.get('road_segments',[])):raise ValueError('Road task masks mismatch')
    for segment,mask in zip(record.get('road_segments',[]),masks['road']):
        if len(mask['geometry'])!=len(segment['centerline_ego_m']) or any(type(v) is not bool for v in mask['geometry']) or type(mask['width']) is not bool:raise ValueError('Invalid road mask')
        if mask['width'] and not segment['width_valid']:raise ValueError('Unobserved width marked valid')
    sets=record['agent_sets'];ids={str(a['id']) for a in agents}
    if set(sets['local_agent_ids'])!=ids or set(sets['risk_agent_ids'])!=ids:raise ValueError('Agent/Risk sets mismatch')
    if sets['planner_selected_ids']!=list(map(str,record['planner_agent_ids'])):raise ValueError('Planner selection changed')
    if len(sets['sft_described_ids'])>8 or not set(sets['sft_described_ids'])<=set(sets['planner_selected_ids']):raise ValueError('Invalid SFT description set')
    if record['sft_available']!=(mode!='UNKNOWN' and record['planner_context_consistent']):raise ValueError('SFT eligibility mismatch')
    if record['planner_supervision_available']!=record['sft_available']:raise ValueError('SFT compatibility flag mismatch')
    if record['grounding_available']!=bool(masks['detection'] or any(masks['risk'])):raise ValueError('Grounding eligibility mismatch')


def stage_available(record,stage):
    if supervision_protocol(record)==TASKWISE_PROTOCOL:
        return bool(record['sft_available'] if stage=='sft' else record['grounding_available'])
    return bool(record.get('planner_supervision_available',True)) if stage=='sft' else True


def attach_local_tasks(record):
    """Prepare new raw candidates without whole-frame Risk/Mode exclusions."""
    from .local_risk_audit import legacy_fast
    from .agent50_filter import filter_record
    from .local_risk import migrate_risk_record
    record=copy.deepcopy(record)
    params=dict(distance_cap_m=80.,safety_margin_m=.5,max_deceleration_mps2=8.,deceleration_step_mps2=.5,sample_dt_s=.05,
                label_semantics='sampled_time_counterfactual_cv_ctrv; dmin capped; no swept-volume guarantee')
    record['risk_label']={**legacy_fast(record),**params}
    record['agent_risk_labels']={str(a['id']):{**legacy_fast(dict(record,risk_agents=[a])),**params} for a in record['agents']}
    record['teacher_mode']=None;record['planner_supervision_available']=True
    record,_=filter_record(record)
    record=migrate_risk_record(record)
    return annotate_tasks(record)
