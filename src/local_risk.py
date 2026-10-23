"""Versioned collision supervision for the fixed current Task-1 Agent set.

Geometry uses the existing six GT knots, with a 0.05s sampled check. No actor
membership, physical trajectory or legacy world-risk label is rewritten here.
"""
from __future__ import annotations

import copy
import numpy as np
from .geometry import nominal_rollout

LOCAL_RISK_SCHEMA = 'local_agent_collision_v1'
LEGACY_RISK_SCHEMA = 'legacy_global_v1'
BRAKING_GRID = np.arange(0., 8.0001, .5)


def risk_schema(record):
    schema = record.get('risk_supervision_schema', LEGACY_RISK_SCHEMA)
    if schema not in (LOCAL_RISK_SCHEMA, LEGACY_RISK_SCHEMA):
        raise ValueError(f'Unknown risk_supervision_schema: {schema}')
    return schema


def box_distances(a, ah, az, b, bh, bz):
    """Broadcast exact oriented-rectangle distance, equivalent to box_distance."""
    def corners(p, h, size):
        local = np.array([[-1.,-1.],[1.,-1.],[1.,1.],[-1.,1.]]) * np.asarray(size)[...,None,:] / 2
        c,s = np.cos(h)[...,None],np.sin(h)[...,None]
        return np.stack((local[...,0]*c-local[...,1]*s,local[...,0]*s+local[...,1]*c),-1)+np.asarray(p)[...,None,:]
    aa,bb = np.broadcast_arrays(corners(a,ah,az), corners(b,bh,bz))
    axes = np.concatenate((aa[...,1:3,:]-aa[...,:2,:], bb[...,1:3,:]-bb[...,:2,:]),-2)
    axes = np.stack((-axes[...,1], axes[...,0]),-1)
    pa = np.einsum('...ij,...kj->...ik',aa,axes)
    pb = np.einsum('...ij,...kj->...ik',bb,axes)
    separated = ((pa.max(-2)<pb.min(-2)) | (pb.max(-2)<pa.min(-2))).any(-1)
    def point_edges(p,q):
        edge = np.roll(q,-1,axis=-2)-q
        offset = p[..., :,None,:]-q[...,None,:,:]
        u = np.clip((offset*edge[...,None,:,:]).sum(-1)/(edge*edge).sum(-1)[...,None,:],0,1)
        delta=offset-u[...,None]*edge[...,None,:,:]
        return np.sqrt((delta*delta).sum(-1)).min(axis=(-2,-1))
    return np.where(separated,np.minimum(point_edges(aa,bb),point_edges(bb,aa)),0.)


def _runs(times, valid):
    """Intervals of consecutive valid sampled observations (not continuous guarantees)."""
    positions=np.flatnonzero(valid)
    if not len(positions):return []
    chunks=np.split(positions,np.flatnonzero(np.diff(positions)>1)+1)
    return [[float(times[c[0]]),float(times[c[-1]])] for c in chunks]


def _trajectory(agent, future, times, *, native_validity=True):
    xy=agent.get('future_xy');valid=np.asarray(agent.get('future_valid',[xy is not None]*len(future)),bool)
    if xy is None:xy=np.zeros((len(future),2));valid[:]=False
    xy=np.asarray(xy,float);headings=np.asarray(agent.get('future_yaw',[agent['heading']]*len(future)),float)
    if xy.shape!=(len(future),2) or headings.shape!=future.shape or valid.shape!=future.shape:
        raise ValueError('Local Agent future shape mismatch')
    if not np.isfinite(xy).all() or not np.isfinite(headings).all():raise ValueError('Nonfinite Local Agent future')
    knots=np.r_[0.,future];observed=np.r_[agent.get('present_valid',True),valid]
    positions=np.vstack((agent['position'],xy));angles=np.unwrap(np.r_[agent['heading'],headings])
    out=np.stack([np.interp(times,knots,positions[:,i]) for i in range(2)],-1)
    yaw=np.interp(times,knots,angles);known=[]
    valid30=agent.get('future30_valid') if native_validity else None
    for t in times:
        right=np.searchsorted(knots,t)
        if right<len(knots) and abs(knots[right]-t)<1e-8:
            ok=bool(observed[right])
        else:
            ok=bool(observed[right-1] and observed[right])
            # Missing native observations inside an interpolation interval must
            # not be bridged by two available 0.5s endpoints.
            if valid30 is not None:
                native=np.arange(1,len(valid30)+1)*.1
                selected=(native>knots[right-1]+1e-8)&(native<knots[right]+1e-8)
                ok=ok and bool(np.asarray(valid30,bool)[selected].all())
        known.append(ok)
    return out,yaw,np.asarray(known,bool)


def _label(values,times,ids,set_known):
    """values [17, agents, times]; evaluate every braking candidate."""
    nominal=values[0];finite=np.isfinite(nominal)
    collision=nominal<.5
    pcol=[];masks=[];sources=[]
    coverage=[]
    for horizon in (1.,2.,3.):
        chosen=times<=horizon+1e-9
        hit=bool(collision[:,chosen].any())
        complete=bool(set_known and times[-1]>=horizon-1e-9 and finite[:,chosen].all())
        pcol.append(float(hit));masks.append(hit or complete)
        sources.append([identifier for i,identifier in enumerate(ids) if collision[i,chosen].any()])
        coverage.append(complete)
    minimum=min(float(nominal[finite].min()),80.) if finite.any() else 80.
    complete=bool(set_known and times[-1]>=3.-1e-9 and finite.all())
    empty=not ids
    dmask=bool(not empty and (complete or finite.any() and minimum==0.))
    candidates=[]
    for deceleration,dist in zip(BRAKING_GRID,values):
        hit=bool((dist<.5).any())
        known=bool(set_known and times[-1]>=3.-1e-9 and np.isfinite(dist).all())
        state='dangerous' if hit else 'safe' if known else 'unknown'
        candidates.append(dict(deceleration_mps2=float(deceleration),state=state,
            source_agent_ids=[identifier for i,identifier in enumerate(ids) if (dist[i]<.5).any()]))
    current=bool((nominal[:,0]<.5).any())
    safe=[i for i,c in enumerate(candidates) if c['state']=='safe']
    valid=bool(not empty and complete and not current and safe)
    areq=float(BRAKING_GRID[safe[0]]) if valid else 0.
    reason=('empty_agent_set' if empty else 'current_dangerous_proximity' if current else
            'incomplete_future' if not complete else 'no_feasible_deceleration' if not safe else 'minimum_safe_grid_value')
    feasible=None if empty or not complete and not current else bool(safe) if not current else False
    dmin_sources=[ids[i] for i in range(len(ids)) if finite[i].any() and float(nominal[i,finite[i]].min())<=minimum+1e-9]
    return dict(pcol=pcol,pcol_mask=masks,dmin=minimum,dmin_mask=dmask,areq=areq,areq_mask=valid,
        braking_feasible=feasible,distance_cap_m=80.,safety_margin_m=.5,max_deceleration_mps2=8.,
        deceleration_step_mps2=.5,sample_dt_s=.05,label_semantics=LOCAL_RISK_SCHEMA,
        pcol_source_agent_ids=sources,dmin_source_agent_ids=dmin_sources,
        pcol_reasons=['observed_danger' if p else 'complete_no_danger' if m else 'incomplete_future_or_set' for p,m in zip(pcol,masks)],
        dmin_reason='empty_agent_set' if empty else 'complete_future' if complete else 'observed_zero_lower_bound' if dmask else 'incomplete_future',
        areq_reason=reason,coverage=dict(set_known=bool(set_known),horizon_complete=coverage,
            complete_3s=complete,agent_observed_intervals={identifier:_runs(times,finite[i]) for i,identifier in enumerate(ids)},
            geometry_times_s=times.tolist(),interpolation='existing six knots; do not bridge invalid native intervals'),
        braking_candidates=candidates)


def compute_local_risk(record):
    """Compute aggregate and individual labels without nearest-8 truncation."""
    agents=record['agents'];ids=[str(a['id']) for a in agents]
    if len(set(ids))!=len(ids):raise ValueError('Duplicate local Agent ID')
    future=np.asarray(record['future_times_s'],float)
    if future.shape!=(6,) or not np.allclose(future,np.arange(1,7)*.5):raise ValueError('Expected existing six future knots')
    times=np.unique(np.r_[np.arange(0.,3.,.05),3.,future])
    ego=record['ego'];ego_xy=[];ego_yaw=[]
    for brake in BRAKING_GRID:
        xy,yaw=nominal_rollout(ego['velocity'],ego['yaw_rate'],times,float(brake));ego_xy.append(xy);ego_yaw.append(yaw)
    ego_xy=np.asarray(ego_xy);ego_yaw=np.asarray(ego_yaw)
    if agents:
        trajectories=[_trajectory(a,future,times) for a in agents]
        xy=np.array([x[0] for x in trajectories]);yaw=np.array([x[1] for x in trajectories]);known=np.array([x[2] for x in trajectories])
        # Bound intermediate broadcast memory; actor membership and geometry are unchanged.
        pieces=[]
        for start in range(0,len(agents),8):
            end=start+8
            pieces.append(box_distances(ego_xy[:,None,:,:],ego_yaw[:,None,:],np.asarray(ego['box_size']),
                xy[None,start:end,:,:],yaw[None,start:end,:],np.asarray([a['size'] for a in agents[start:end]])[None,:,None,:]))
        values=np.concatenate(pieces,axis=1)
        values=np.where(known[None,:,:],values,np.nan)
    else:values=np.empty((len(BRAKING_GRID),0,len(times)))
    set_known=bool(record.get('agent_supervision_available',False))
    total=_label(values,times,ids,set_known)
    individual={identifier:_label(values[:,i:i+1,:],times,[identifier],True) for i,identifier in enumerate(ids)}
    return total,individual


def check_label(label):
    known=[p for p,m in zip(label['pcol'],label['pcol_mask']) if m]
    if any(a>b for a,b in zip(known,known[1:])):raise ValueError('Nonmonotone cumulative risk')
    if label['pcol_mask'][2] and label['dmin_mask']:
        if bool(label['pcol'][2]) != (label['dmin']<.5):raise ValueError('pcol/dmin inconsistency')
    if label['areq_mask']:
        candidate=label['braking_candidates'][int(round(label['areq']/.5))]
        if candidate['state']!='safe':raise ValueError('areq not safe')
        if any(c['state']!='dangerous' for c in label['braking_candidates'][:int(round(label['areq']/.5))]):raise ValueError('areq minimality not established')


def migrate_risk_record(old):
    from .collection_adapter import _mode_from_risk
    if old.get('grounding_label_schema')!='agent50_v1' or old.get('agent_ignore_policy')!='pedestrian_cyclist_regions_v1':
        raise ValueError('Task 2 requires final Task-1 filtered manifest')
    if 'risk_supervision_schema' in old:raise ValueError('Risk already versioned; refusing migration')
    local,individual=compute_local_risk(old)
    for label in [local,*individual.values()]:check_label(label)
    selected_ids=[str(i) for i in old['planner_agent_ids']]
    selected={i:individual[i] for i in selected_ids}
    mode=_mode_from_risk(local,selected,bool(old['agents']))
    reason='available'
    if mode is None:
        from .agent50_filter import _hazard
        reason='local_hazard_outside_planner_selected' if _hazard(local) else 'insufficient_local_risk_evidence'
    elif mode=='long' and any(not all(a['future_valid']) for a in old['agents'] if str(a['id']) in selected):
        mode=None;reason='incomplete_selected_agent_future'
    new=copy.deepcopy(old)
    new.update(risk_supervision_schema=LOCAL_RISK_SCHEMA,
        global_risk=copy.deepcopy(old['risk_label']),global_risk_schema=LEGACY_RISK_SCHEMA,
        local_risk=local,local_agent_risk_labels=individual,
        legacy_teacher_mode=old.get('teacher_mode'),legacy_planner_supervision_available=old.get('planner_supervision_available',True),
        teacher_mode=mode,planner_supervision_available=mode is not None,
        planner_supervision_unavailable_reason=None if mode is not None else reason,
        risk_provenance=dict(local_agent_ids=[str(a['id']) for a in old['agents']],
            ignored_agent_ids=[str(a['id']) for a in old.get('ignored_agents',[])],
            planner_selected_ids=selected_ids,global_agent_ids=[str(a['id']) for a in old.get('risk_agents',[])],
            future_knots_s=list(old['future_times_s']),geometry_dt_s=.05,
            target_set='Task1 retained agents; fixed at current time; no nearest8 truncation',
            global_source='exact copy of legacy risk_label',global_coverage_source=old.get('tracking_coverage_source')))
    return new


def legacy_hazard_sources(record):
    """Attribute legacy nominal conflicts using its original six-knot protocol."""
    agents=record.get('risk_agents',record.get('agents',[]))
    if not agents:return [[],[],[]]
    future=np.asarray(record['future_times_s'],float)
    times=np.unique(np.r_[np.arange(0.,3.,.05),3.,future])
    xy,yaw=nominal_rollout(record['ego']['velocity'],record['ego']['yaw_rate'],times)
    traces=[_trajectory(a,future,times,native_validity=False) for a in agents]
    distance=box_distances(xy[None,:,:],yaw[None,:],np.asarray(record['ego']['box_size']),
        np.asarray([x[0] for x in traces]),np.asarray([x[1] for x in traces]),
        np.asarray([a['size'] for a in agents])[:,None,:])
    valid=np.asarray([x[2] for x in traces]);distance=np.where(valid,distance,np.nan)
    return [[str(a['id']) for i,a in enumerate(agents) if (distance[i,times<=h+1e-9]<.5).any()] for h in (1.,2.,3.)]


def supervision_risk(record):
    return record['local_risk'] if risk_schema(record) == LOCAL_RISK_SCHEMA else record['risk_label']
