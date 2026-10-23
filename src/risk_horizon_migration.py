"""Atomic/resumable KRS-v2 migration; source arrays, membership and trajectories untouched."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path

from .local_risk import compute_local_risk,check_label,LOCAL_RISK_SCHEMA
from .risk_contract import LOCAL_SHORT_SCHEMA
from .task_supervision import annotate_tasks


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)


def migrate_record(old):
    from .moving_agents import POLICY
    if old.get('risk_supervision_schema')!=LOCAL_RISK_SCHEMA or old.get('agent_selection_policy')!=POLICY:
        raise ValueError('Migration requires the accepted moving-v1 local-risk manifest')
    new=copy.deepcopy(old);new.pop('supervision_protocol',None)
    risk,individual=compute_local_risk(new,schema=LOCAL_SHORT_SCHEMA)
    for label in [risk,*individual.values()]:check_label(label)
    for k in ['pcol','pcol_mask']:
        if risk[k][1:] != old['local_risk'][k][:2]:raise ValueError('Existing1s/2s targets unexpectedly changed')
    new.update(risk_supervision_schema=LOCAL_SHORT_SCHEMA,local_risk=risk,local_agent_risk_labels=individual,
        risk_label=copy.deepcopy(risk),agent_risk_labels=copy.deepcopy(individual))
    new.setdefault('risk_provenance',{}).update(source_risk_schema=LOCAL_RISK_SCHEMA,
        pcol_horizons_s=[.5,1.,2.],geometry_horizon_s=2.,geometry_dt_s=.05)
    new=annotate_tasks(new)
    for key in ['agents','risk_agents','planner_agent_ids','sensors','ego','future_times_s','episode_id','sample_id']:
        if key in old and new[key]!=old[key]:raise ValueError(f'Unexpected physical data change: {key}')
    return new


def rebase(record,source,destination):
    path_keys={'path','pointcloud','power','unfolded_doppler','folded_doppler','doppler_valid','doppler_prior','raw_points','raw_cube'}
    def relative(p):return os.path.relpath((source/p).resolve(),destination)
    def walk(value):
        if not isinstance(value,dict):return
        for key,v in value.items():
            if key in path_keys and isinstance(v,str):value[key]=relative(v)
            elif key in path_keys and isinstance(v,list) and all(isinstance(p,str) for p in v):value[key]=[relative(p) for p in v]
            elif isinstance(v,dict):walk(v)
    walk(record['sensors'])
    if record.get('route_archive'):record['route_archive']=relative(record['route_archive'])


def migrate(manifest,output,assignment_path):
    from .data import _validate_record
    source=Path(manifest).resolve();out=Path(output).resolve();assignment_path=Path(assignment_path).resolve()
    if out==source.parent:raise ValueError('Use a separate versioned dataset directory')
    assignment=json.loads(assignment_path.read_text())
    if assignment.get('source_manifest_sha256')!=sha(source):raise ValueError('Assignment source manifest differs')
    allocation=assignment['allocation'];out.mkdir(parents=True,exist_ok=True)
    signature=dict(source_manifest_sha256=sha(source),assignment_sha256=sha(assignment_path),
        source_files={p.name:sha(p) for p in sorted(Path(__file__).parent.glob('*.py'))},schema=LOCAL_SHORT_SCHEMA)
    progress=out/'progress.json';partial=out/'manifest.partial.jsonl';final=out/'manifest.jsonl'
    state=json.loads(progress.read_text()) if progress.exists() else dict(rows=0,bytes=0,signature=signature)
    if state['signature']!=signature:raise ValueError('Migration source/code/assignment changed; refusing resume')
    if final.exists():
        if state.get('state')!='complete' or sha(final)!=state['manifest_sha256']:raise ValueError('Published manifest differs')
        return final
    if partial.exists():
        with partial.open('r+b') as f:f.truncate(state['bytes'])
    elif state['rows']:raise ValueError('Committed partial manifest missing')
    shared={};ids=set();counts={};index=0
    # Scan committed metadata to retain leakage and duplicate-ID checks on resume.
    def audit(record):
        sid=record['sample_id'];split=record['split']
        if sid in ids:raise ValueError('Duplicate sample ID')
        ids.add(sid)
        for key in ['power','unfolded_doppler','doppler_valid']:
            paths=record['sensors']['radar'].get(key,[]);paths=paths if isinstance(paths,list) else [paths]
            for p in paths:
                name=str((out/p).resolve())
                if name in shared and shared[name]!=split:raise ValueError('Radar history source shared across splits')
                shared[name]=split
        counts[split]=counts.get(split,0)+1
    if partial.exists():
        for line in partial.open():audit(json.loads(line))
    with source.open() as src,partial.open('ab') as dst:
        for line in src:
            if not line.strip():continue
            index+=1
            if index<=state['rows']:continue
            old=json.loads(line);new=migrate_record(old);new['split']=allocation[new['episode_id']]
            new['split_provenance']=dict(source_split=old['split'],policy='episode_stratified_70_15_15_frozen_test_v2')
            rebase(new,source.parent,out);_validate_record(new,index);audit(new)
            dst.write((json.dumps(new,allow_nan=False)+'\n').encode())
            if index%100==0:
                dst.flush();os.fsync(dst.fileno());state.update(rows=index,bytes=dst.tell());atomic(progress,state)
                print(json.dumps(dict(rows=index,counts=counts)),flush=True)
        dst.flush();os.fsync(dst.fileno());state.update(rows=index,bytes=dst.tell());atomic(progress,state)
    if sha(source)!=signature['source_manifest_sha256']:raise ValueError('Source manifest changed during migration')
    partial.replace(final);state.update(state='complete',manifest_sha256=sha(final),counts=counts)
    atomic(progress,state)
    return final


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--output',required=True);p.add_argument('--assignment',required=True)
    a=p.parse_args();print(migrate(a.manifest,a.output,a.assignment))
