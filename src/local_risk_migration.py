"""Resumable, read-only-source Local Risk migration and exclusion audit."""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import copy
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time
import numpy as np

from . import collection_adapter as adapter
from .agent50_migration import sha256,write_json
from .agent50_filter import excluded_agent_kind
from .local_risk import (LOCAL_RISK_SCHEMA,LEGACY_RISK_SCHEMA,migrate_risk_record,
                         legacy_hazard_sources,compute_local_risk,check_label)

ALLOWED_FIELDS=frozenset(('risk_supervision_schema','global_risk','global_risk_schema','local_risk',
    'local_agent_risk_labels','legacy_teacher_mode','legacy_planner_supervision_available','teacher_mode',
    'planner_supervision_available','planner_supervision_unavailable_reason','risk_provenance'))


def assert_invariants(old,new):
    for key in set(old)|set(new):
        if key not in ALLOWED_FIELDS and old.get(key)!=new.get(key):raise ValueError(f'Forbidden mutation: {key}')
    if new['global_risk']!=old['risk_label']:raise ValueError('Legacy Global Risk changed')
    if set(new['local_agent_risk_labels'])!={str(a['id']) for a in old['agents']}:raise ValueError('Local set changed')
    for label in [new['local_risk'],*new['local_agent_risk_labels'].values()]:check_label(label)


def state(label,h):
    return ('danger' if label['pcol'][h] else 'safe') if label['pcol_mask'][h] else 'unknown'


class RawEvidence:
    def __init__(self,source):
        self.targets=defaultdict(set);self.hashes={}
        with Path(source).open() as f:
            for line in f:
                r=json.loads(line);self.targets[str(Path(r['route_archive']).resolve().parent)].add(r['carla_frame'])
        self.anchors=lru_cache(maxsize=2)(self._anchors)
        self.metadata=lru_cache(maxsize=128)(lambda ep:self.read(Path(ep)/'metadata.json'))
    def read(self,path):
        path=Path(path);data=path.read_bytes();self.hashes[str(path.resolve())]=hashlib.sha256(data).hexdigest()
        return json.loads(data)
    def _anchors(self,ep):
        # Selected-row digests avoid hashing every unrelated future in a large archive.
        path=Path(ep)/'anchors.jsonl';rows=adapter._selected_anchors(path,self.targets[ep])
        for frame,row in rows.items():
            self.hashes[f'{path}#carla_frame={frame}']=hashlib.sha256(json.dumps(row,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        return rows
    def verify(self,r):
        ep=str(Path(r['route_archive']).resolve().parent);meta=self.metadata(ep)
        if meta['episode_id']!=r['episode_id'] or meta['agent_label_range_m']!=50:raise ValueError('Original domain metadata mismatch')
        index=int(Path(r['sensors']['radar']['power'][-1]).name.split('_')[0])
        current=self.read(Path(ep)/'frames'/f'{index:06d}'/'frame.json')
        original=self.read(Path(ep)/'frames'/f'{index:06d}'/'agents.json')
        if current['carla_frame']!=r['carla_frame'] or original['carla_frame']!=r['carla_frame']:raise ValueError('Raw frame identity mismatch')
        admitted={str(a['actor_id']):a for a in original['agents']}
        anchor=self.anchors(ep)[r['carla_frame']]
        geometry=adapter.collector_geometry();position,rotation=geometry.radar_pose(current['ego']['pose'],meta['sensors']['semantic_lidar'])
        origin=current['ego']['pose']['position_m'][:2];yaw=current['ego']['pose']['rotation_deg'][2]
        for a in r['agents']:
            actor=admitted[str(a['id'])]
            if excluded_agent_kind(a) is not None:raise ValueError('Ignored class entered Local set')
            if not geometry.actor_intersects_fov(actor,position,rotation,50.):raise ValueError('Agent outside original bbox/FOV rule')
            actual=adapter._agent(actor,current,origin,yaw,anchor['futures_by_actor_id'].get(str(a['id'])),[4,9,14,19,24,29])
            for key in ('position','velocity','size','heading','future_xy','future_yaw','future_valid','future30_valid'):
                if a.get(key)!=actual.get(key):raise ValueError(f'Raw Agent evidence mismatch {key}, {r["sample_id"]}')
        return dict(frame=str(Path(ep)/'frames'/f'{index:06d}'/'frame.json'),
                    anchor=f'{Path(ep)/"anchors.jsonl"}#carla_frame={r["carla_frame"]}',metadata=str(Path(ep)/'metadata.json'))


def describe(r,new):
    sources=legacy_hazard_sources(r);local_ids={str(a['id']) for a in r['agents']};ignored={str(a['id']) for a in r.get('ignored_agents',[])}
    selected=set(map(str,r['planner_agent_ids']))
    # Non-Local includes ignored classes and true spatially excluded world actors.
    return dict(sample_id=r['sample_id'],episode_id=r['episode_id'],split=r['split'],scenario=r.get('category','unknown'),
        global_risk=r['risk_label'],local_risk=new['local_risk'],global_source_agent_ids=sources,
        local_source_agent_ids=new['local_risk']['pcol_source_agent_ids'],
        global_sources_not_local=[sorted(set(ids)-local_ids) for ids in sources],
        global_sources_ignored=[sorted(set(ids)&ignored) for ids in sources],
        global_sources_not_selected=[sorted(set(ids)-selected) for ids in sources],
        old_teacher_mode=r.get('teacher_mode'),new_teacher_mode=new['teacher_mode'],
        old_planner_available=r.get('planner_supervision_available',True),new_planner_available=new['planner_supervision_available'],
        planner_reason=new['planner_supervision_unavailable_reason'])


def update_stats(counter,row):
    counter['samples']+=1
    for i in range(3):
        g=state(row['global_risk'],i);l=state(row['local_risk'],i)
        counter[f'{i+1}s/global_{g}__local_{l}']+=1
        counter[f'{i+1}s/local_{l}']+=1
        counter[f'{i+1}s/global_{g}']+=1
        for key in ('global_sources_not_local','global_sources_ignored','global_sources_not_selected'):
            counter[f'{i+1}s/{key}']+=bool(row[key][i])
    label=row['local_risk']
    for dim in ('dmin','areq'):
        counter[f'{dim}/valid']+=label[dim+'_mask']
        if label[dim+'_mask']:counter[f'{dim}/valid_zero' if label[dim]==0 else f'{dim}/valid_positive']+=1
        counter[f'{dim}/reason/{label[dim+"_reason"]}']+=1
    counter[f'teacher/{row["old_teacher_mode"]}->{row["new_teacher_mode"]}']+=1
    counter['sft_available']+=row['new_planner_available']
    counter['sft_became_available']+=row['new_planner_available'] and not row['old_planner_available']
    counter['sft_became_unavailable']+=not row['new_planner_available'] and row['old_planner_available']
    if row['planner_reason']:counter['sft_unavailable/'+row['planner_reason']]+=1


def summarize(output):
    groups={name:defaultdict(Counter) for name in ('split','episode','scenario')};all_stats=Counter();episodes={};count=0
    with (output/'samples.jsonl').open() as f:
        for line in f:
            r=json.loads(line);update_stats(all_stats,r)
            for key in groups:update_stats(groups[key][r['episode_id'] if key=='episode' else r[key]],r)
            ep=r['episode_id'];sp=r['split']
            if ep in episodes and episodes[ep]!=sp:raise ValueError('Episode leakage')
            episodes[ep]=sp;count+=1
    return dict(samples=count,episodes=len(episodes),totals=all_stats,groups={k:dict(v) for k,v in groups.items()})


def migrate(source,output,config,resume=False):
    source=Path(source).resolve();output=Path(output).resolve();config=Path(config).resolve()
    output.mkdir(parents=True,exist_ok=True)
    if (output/'manifest.jsonl').exists():raise FileExistsError('Published manifest already exists')
    source_hash=sha256(source);code_hash={p.name:sha256(p) for p in (Path(__file__),Path(__file__).with_name('local_risk.py'))}
    signature=dict(source=str(source),source_sha256=source_hash,config_sha256=sha256(config),code_sha256=code_hash)
    journal=output/'progress.json';partial=output/'manifest.partial.jsonl';audit=output/'samples.jsonl'
    prior=dict(rows=0,manifest_bytes=0,audit_bytes=0,**signature)
    if journal.exists():
        if not resume:raise FileExistsError('Use --resume for an unfinished migration')
        prior=json.loads(journal.read_text())
        if any(prior.get(k)!=v for k,v in signature.items()):raise ValueError('Resume source/config/code mismatch')
    elif any(output.iterdir()):raise FileExistsError('Output directory must be empty')
    for path,key in ((partial,'manifest_bytes'),(audit,'audit_bytes')):
        if path.exists():
            with path.open('r+b') as f:f.truncate(prior[key])
        elif prior['rows']:raise ValueError('Missing resumable output')
    if not journal.exists():write_json(journal,prior)
    evidence=RawEvidence(source)
    if (output/'raw_evidence.json').exists():evidence.hashes.update(json.loads((output/'raw_evidence.json').read_text()))
    seen=set();started=time.monotonic();count=0
    def checkpoint(out,detail):
        out.flush();detail.flush();os.fsync(out.fileno());os.fsync(detail.fileno())
        temp=journal.with_suffix('.tmp');write_json(temp,dict(rows=count,manifest_bytes=out.tell(),audit_bytes=detail.tell(),**signature));temp.replace(journal)
        write_json(output/'raw_evidence.json',evidence.hashes)
    with source.open() as incoming,partial.open('ab') as outgoing,audit.open('ab') as detail:
        try:
            for line in incoming:
                old=json.loads(line);count+=1
                if old['sample_id'] in seen:raise ValueError('Duplicate sample ID')
                seen.add(old['sample_id'])
                if count<=prior['rows']:continue
                raw=evidence.verify(old);new=migrate_risk_record(old);new['risk_provenance']['raw_evidence']=raw
                assert_invariants(old,new);row=describe(old,new)
                outgoing.write((json.dumps(new,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n').encode())
                detail.write((json.dumps(row,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n').encode())
                if count%50==0:
                    checkpoint(outgoing,detail);print(f'MIGRATION {count} elapsed={time.monotonic()-started:.1f}s',flush=True)
            checkpoint(outgoing,detail)
        except BaseException as exc:
            write_json(output/'failure.json',dict(row=count,error=str(exc),resume_from=json.loads(journal.read_text())['rows'] if journal.exists() else 0))
            raise
    summary=summarize(output)
    if summary['samples']!=8835 or summary['episodes']!=64:raise ValueError('Unexpected source cohort count')
    # Independent streamed validation before publication; no radar arrays are rewritten.
    validate_manifest(source,partial)
    report=dict(status='PASS',**summary,training_started=False,optimizer_steps=0,
        risk_supervision_schema=LOCAL_RISK_SCHEMA,source_manifest_sha256=source_hash,
        manifest_sha256=sha256(partial),scope='All splits checked for label integrity; no test parameter selection')
    write_json(output/'migration_report.json',report)
    write_json(output/'provenance.json',dict(**signature,manifest=str(output/'manifest.jsonl'),manifest_sha256=report['manifest_sha256'],
        schema=LOCAL_RISK_SCHEMA,config=str(config),raw_evidence='raw_evidence.json',raw_evidence_sha256=sha256(output/'raw_evidence.json'),
        git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        source_files={str(p):sha256(p) for p in Path(__file__).parent.glob('*.py')},
        legacy_global='exact copy; original six-knot/world-coverage protocol unchanged',
        local='six-knot interpolation; native missing intervals invalidate interpolation; current retained Task1 set'))
    partial.replace(output/'manifest.jsonl');write_json(output/'validation.json',dict(status='PASS',manifest_sha256=report['manifest_sha256'],samples=count,episodes=summary['episodes'],
        mask_and_monotonicity=True,unchanged_fields=True,source_and_output_hashes=report,training_started=False))
    print('MIGRATION PASS',count,flush=True)
    return report


def validate_manifest(source,manifest):
    from itertools import zip_longest
    from .data import _validate_record
    with Path(source).open() as a,Path(manifest).open() as b:
        for i,(la,lb) in enumerate(zip_longest(a,b),1):
            if la is None or lb is None:raise ValueError('Sample count changed')
            old,new=json.loads(la),json.loads(lb);assert_invariants(old,new);_validate_record(new,i)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--resume',action='store_true')
    a=p.parse_args();migrate(a.source,a.output_dir,a.config,a.resume)

if __name__=='__main__':main()
