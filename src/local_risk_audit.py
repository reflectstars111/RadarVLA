"""Read-only legacy exclusion reconstruction and exact spatial source attribution."""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import copy
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from . import collection_adapter as adapter
from .agent50_filter import excluded_agent_kind
from .agent50_migration import write_json,sha256
from .local_risk import compute_local_risk,legacy_hazard_sources
from .geometry import build_risk_labels,nominal_rollout
from .local_risk import _trajectory, _label, box_distances, BRAKING_GRID


def legacy_fast(record):
    """Exact six-knot legacy rules with batched distances for exclusion auditing."""
    future=np.asarray(record['future_times_s'],float)
    times=np.unique(np.r_[np.arange(0.,3.,.05),3.,future])
    agents=record.get('risk_agents',record['agents']);ego=record['ego']
    roll=[nominal_rollout(ego['velocity'],ego['yaw_rate'],times,float(a)) for a in BRAKING_GRID]
    if agents:
        traces=[_trajectory(a,future,times,native_validity=False) for a in agents]
        # Bound temporary broadcast arrays independently of world actor count.
        pieces=[]
        for start in range(0,len(agents),8):
            selected=traces[start:start+8];selected_agents=agents[start:start+8]
            pieces.append(box_distances(np.asarray([x[0] for x in roll])[:,None,:,:],np.asarray([x[1] for x in roll])[:,None,:],np.asarray(ego['box_size']),
                np.asarray([x[0] for x in selected])[None,:,:,:],np.asarray([x[1] for x in selected])[None,:,:],np.asarray([a['size'] for a in selected_agents])[None,:,None,:]))
        values=np.concatenate(pieces,axis=1)
        values=np.where(np.asarray([x[2] for x in traces])[None,:,:],values,np.nan)
    else:values=np.empty((17,0,len(times)))
    result=_label(values,times,[str(a['id']) for a in agents],True)
    knots=np.r_[0.,future];observed=np.r_[record.get('agent_supervision_available',True),record['tracking_coverage_valid']]
    dense=[]
    for t in times:
        right=np.searchsorted(knots,t)
        dense.append(observed[right] if right<len(knots) and abs(knots[right]-t)<1e-8 else observed[right-1] and observed[right])
    dense=np.asarray(dense,bool);nominal=values[0]
    covered=[bool(np.isfinite(nominal[:,times<=h]).all() and dense[times<=h].all()) for h in (1.,2.,3.)]
    result['pcol_mask']=[bool(p or c) for p,c in zip(result['pcol'],covered)]
    complete=covered[2];result['dmin_mask']=bool(agents and (complete or result['dmin']==0.))
    if not agents:
        result['areq_mask']=False;result['areq']=0.;result['braking_feasible']=None
    elif (nominal[:,0]<.5).any() or np.linalg.norm(ego['velocity'])<1e-12 and (nominal<.5).any():
        result['areq_mask']=False;result['areq']=0.;result['braking_feasible']=False
    elif not complete:
        result['areq_mask']=False;result['areq']=0.;result['braking_feasible']=None
    return {k:result[k] for k in ('pcol','pcol_mask','dmin','dmin_mask','areq','areq_mask','braking_feasible')}


def classify_sources(current,metadata,ids,local_ids,ignored_ids,selected_ids):
    geometry=adapter.collector_geometry();position,rotation=geometry.radar_pose(current['ego']['pose'],metadata['sensors']['semantic_lidar'])
    actors={str(a['actor_id']):a for a in current['actors']};rows=[]
    for identifier in sorted(ids):
        actor=actors.get(identifier)
        if actor is None:
            rows.append(dict(id=identifier,status='raw_actor_unrecoverable'));continue
        intersects=bool(geometry.actor_intersects_fov(actor,position,rotation,50.))
        fov=bool(geometry.actor_intersects_fov(actor,position,rotation,1e9))
        center=adapter._bbox_world_xy(actor['pose']['position_m'],actor['pose']['rotation_deg'][2],actor['bbox']['center_actor_m'])
        distance=math.dist(center,position[:2])
        # This field is diagnostic only; never use centre range to select GT.
        rows.append(dict(id=identifier,type_id=actor.get('type_id'),center_range_m=distance,
            center_beyond_50m=distance>50.,bbox_intersects_original_domain=intersects,
            bbox_intersects_original_fov=fov,local=identifier in local_ids,
            ignored=identifier in ignored_ids,planner_selected=identifier in selected_ids,
            exclusion_kind=excluded_agent_kind(actor),
            outside_reason='inside' if intersects else 'outside_fov' if not fov else 'outside_50m',
            membership_reason='retained_local' if identifier in local_ids else 'ignored_class' if identifier in ignored_ids else
                'outside_spatial_domain' if not intersects else 'original_agent_membership_not_retained'))
    return rows


def audit_sources(directory):
    out=Path(directory);counts=Counter();raw_hashes={};metadata={}
    def read(path):
        p=Path(path);data=p.read_bytes();raw_hashes[str(p.resolve())]=hashlib.sha256(data).hexdigest();return json.loads(data)
    with (out/'manifest.jsonl').open() as manifests,(out/'samples.jsonl').open() as samples,(out/'risk_sources.jsonl').open('w') as result:
        for lm,ls in zip(manifests,samples):
            r=json.loads(lm);s=json.loads(ls)
            if r['sample_id']!=s['sample_id']:raise ValueError('Audit ordering mismatch')
            all_ids=set().union(*map(set,s['global_source_agent_ids']),*map(set,s['local_source_agent_ids']))
            counts['samples']+=1
            if not all_ids:continue
            ep=Path(r['route_archive']).resolve().parent
            if str(ep) not in metadata:metadata[str(ep)]=read(ep/'metadata.json')
            raw=read(r['risk_provenance']['raw_evidence']['frame'])
            entries=classify_sources(raw,metadata[str(ep)],all_ids,set(r['risk_provenance']['local_agent_ids']),
                set(r['risk_provenance']['ignored_agent_ids']),set(r['risk_provenance']['planner_selected_ids']))
            for a in entries:
                counts['hazard_source_observations']+=1
                if a.get('status'):counts[a['status']]+=1;continue
                counts['membership/'+a['membership_reason']]+=1
                counts['spatial/'+a['outside_reason']]+=1
                counts['center_beyond_50m']+=a['center_beyond_50m'];counts['not_planner_selected']+=not a['planner_selected']
            result.write(json.dumps(dict(sample_id=r['sample_id'],episode_id=r['episode_id'],split=r['split'],sources=entries),separators=(',',':'))+'\n')
    write_json(out/'source_audit_report.json',dict(status='PASS',counts=counts,raw_evidence=raw_hashes,
        note='Center range is diagnostic; membership uses original collector bbox/FOV. Missing collection actors are not invented.'))
    print('SOURCE AUDIT PASS',dict(counts),flush=True)


def audit_exclusions(full,exclusions,output):
    full=Path(full);exclusions=Path(exclusions);output=Path(output)
    selected=[];reasons=Counter()
    for line in exclusions.open():
        row=json.loads(line);reasons[row['reason']]+=1
        if 'Risk' in row['reason'] or 'risk' in row['reason']:selected.append(row)
    indexed={(r['job_id'],r['carla_frame']):r for r in map(json.loads,(full/'dataset_index.jsonl').open())}
    groups=defaultdict(list)
    for r in selected:groups[r['job_id']].append(r)
    counters=Counter();raw_evidence={};seen=set()
    def read(path):
        p=Path(path);data=p.read_bytes();raw_evidence[str(p.resolve())]=hashlib.sha256(data).hexdigest();return json.loads(data)
    with (output/'excluded_samples.jsonl').open('w') as dest:
        for job,rows in groups.items():
            ep=full/'jobs'/job/'episode';anchors=adapter._selected_anchors(ep/'anchors.jsonl',{r['carla_frame'] for r in rows})
            meta=read(ep/'metadata.json')
            for old in rows:
                key=(job,old['carla_frame']);detail=dict(job_id=job,carla_frame=old['carla_frame'],original_reason=old['reason'])
                try:
                    if key in seen:raise ValueError('Duplicate old exclusion')
                    seen.add(key);row=indexed[key];i=int(row['source_index']);frame=read(ep/'frames'/f'{i:06d}'/'frame.json')
                    admitted=read(ep/'frames'/f'{i:06d}'/'agents.json')['agents'];anchor=anchors[old['carla_frame']]
                    raw_evidence[f'{ep/"anchors.jsonl"}#carla_frame={old["carla_frame"]}']=hashlib.sha256(json.dumps(anchor,sort_keys=True,separators=(',',':')).encode()).hexdigest()
                    origin=frame['ego']['pose']['position_m'][:2];yaw=frame['ego']['pose']['rotation_deg'][2]
                    futures=anchor['futures_by_actor_id'];indices=[4,9,14,19,24,29]
                    all_admitted=[adapter.agent50_label(adapter._agent(a,frame,origin,yaw,futures.get(str(a['actor_id'])),indices),a) for a in admitted]
                    geometry=adapter.collector_geometry();pos,rot=geometry.radar_pose(frame['ego']['pose'],meta['sensors']['semantic_lidar'])
                    raw_by_id={str(a['actor_id']):a for a in admitted}
                    local=[a for a in all_admitted if excluded_agent_kind(a) is None and geometry.actor_intersects_fov(raw_by_id[str(a['id'])],pos,rot,50.)]
                    # Reconstruct original world-risk bounding rule and coverage, not a new Risk definition.
                    speed=math.hypot(*frame['ego']['velocity_world_mps'][:2]);extent=frame['ego']['bbox']['extent_m']
                    world=[a for a in frame['actors'] if adapter._could_affect_risk(a,futures.get(str(a['actor_id'])),origin,speed,extent)]
                    risk_agents=[adapter._agent(a,frame,origin,yaw,futures.get(str(a['actor_id'])),indices) for a in world]
                    record=dict(agents=local,risk_agents=risk_agents,agent_supervision_available=True,
                        future_times_s=(np.arange(1,7)*.5).tolist(),tracking_coverage_valid=adapter._coverage_from_snapshots(ep,i,frame,anchor),
                        ego=dict(velocity=adapter._vector_to_ego(frame['ego']['velocity_world_mps'],yaw),
                            yaw_rate=-math.radians(frame['ego']['angular_velocity_world_degps'][2]),box_size=[2*extent[0],2*extent[1]]))
                    global_label=legacy_fast(record);local_label,individual=compute_local_risk(record)
                    original_selected=adapter._nearest_planner_ids(all_admitted)
                    selected_ids=[identifier for identifier in original_selected if identifier in individual]
                    mode=adapter._mode_from_risk(local_label,{key:individual[key] for key in selected_ids},bool(selected_ids))
                    sources=legacy_hazard_sources(record);hazards=set().union(*map(set,sources))
                    entries=classify_sources(frame,meta,hazards,{str(a['id']) for a in local},
                        {str(a['id']) for a in all_admitted if excluded_agent_kind(a) is not None},set(selected_ids))
                    flags=dict(global_hazard=any(p and m for p,m in zip(global_label['pcol'],global_label['pcol_mask'])),
                        local_hazard=any(p and m for p,m in zip(local_label['pcol'],local_label['pcol_mask'])),
                        local_unknown=not all(local_label['pcol_mask']),
                        domain_excluded_hazard=any(a.get('membership_reason')=='outside_spatial_domain' for a in entries),
                        beyond_50m_hazard=any(a.get('outside_reason')=='outside_50m' for a in entries),
                        ignored_class_hazard=any(a.get('membership_reason')=='ignored_class' for a in entries),
                        unselected_hazard=any(not a.get('planner_selected',False) for a in entries),
                        future_missing=not local_label['coverage']['complete_3s'])
                    detail.update(status='reconstructed',flags=flags,global_risk=global_label,local_risk=local_label,
                        sources=entries,local_planner_teacher_mode=mode,local_agent_ids=[a['id'] for a in local],
                        planner_selected_ids=selected_ids,episode_id=frame['episode_id'],split=row['split'],
                        note='Diagnostic reconstruction only; excluded frame not added to training cohort')
                    counters.update({k:int(v) for k,v in flags.items()});counters['reconstructed']+=1
                except (FileNotFoundError,KeyError,ValueError) as exc:
                    detail.update(status='unrecoverable',error=str(exc));counters['unrecoverable']+=1
                counters['samples']+=1
                dest.write(json.dumps(detail,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n')
                if counters['samples']%50==0:dest.flush();print('EXCLUSIONS',counters['samples'],flush=True)
    write_json(output/'exclusion_audit_report.json',dict(status='PASS',source=str(exclusions.resolve()),source_sha256=sha256(exclusions),
        original_reason_counts=reasons,counts=counters,raw_evidence=raw_evidence,
        limitations='Flags can overlap. Old mixed exclusion reasons do not identify the original cause; attribution is reconstructed using saved raw frames. No samples added.'))
    print('EXCLUSION AUDIT PASS',dict(counters),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--full',type=Path,required=True);p.add_argument('--exclusions',type=Path,required=True);a=p.parse_args()
    audit_sources(a.directory);audit_exclusions(a.full,a.exclusions,a.directory)

if __name__=='__main__':main()
