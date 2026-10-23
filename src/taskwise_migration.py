"""Recover task-valid raw candidates without changing episode splits or radar processing."""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
from functools import lru_cache
import hashlib,json,os,subprocess,time
from pathlib import Path
import numpy as np
from . import collection_adapter as adapter
from .agent50_migration import sha256,write_json
from .task_supervision import annotate_tasks,validate_tasks,TASKWISE_PROTOCOL

TASK_FIELDS=frozenset(('supervision_protocol','mode_label','mode_mask','mode_reasons','task_masks',
    'grounding_available','sft_available','sft_unavailable_reasons','planner_context_consistent','teacher_mode',
    'planner_supervision_available','planner_supervision_unavailable_reason','previous_teacher_mode',
    'previous_planner_supervision_available','agent_sets','mode_policy'))


def assert_baseline(old,new):
    for key in set(old)|set(new):
        if key not in TASK_FIELDS and old.get(key)!=new.get(key):raise ValueError(f'Forbidden baseline change {key}')
    validate_tasks(new)


def summary_row(r,origin,old=None,reason=None):
    masks=r['task_masks'];risk=r['local_risk'];sets=r['agent_sets'];speed=float(np.linalg.norm(r['ego']['velocity']))
    sources=set().union(*map(set,risk['pcol_source_agent_ids']))
    if risk['dmin_mask'] and risk['dmin']<=.5:sources.update(risk['dmin_source_agent_ids'])
    return dict(sample_id=r['sample_id'],episode_id=r['episode_id'],split=r['split'],scenario=r.get('category','unknown'),
        origin=origin,old_exclusion_reason=reason,mode_label=r['mode_label'],mode_reasons=r['mode_reasons'],
        previous_mode=None if old is None else old.get('teacher_mode'),
        grounding_available=r['grounding_available'],sft_available=r['sft_available'],sft_unavailable_reasons=r['sft_unavailable_reasons'],
        detection_valid=masks['detection'],risk_mask=masks['risk'],risk_values=[*risk['pcol'],risk['dmin'],risk['areq']],
        ego_future_valid=masks['ego_future'],agent_future_valid=masks['agent_future'],road=masks['road'],
        agent_sets=sets,risk_source_agent_ids=sorted(sources),unselected_risk_source_ids=sorted(sources-set(sets['planner_selected_ids'])),
        speed_mps=speed,moving=speed>=1.,turning=abs(r['ego']['yaw_rate'])>=.05,
        stationary_agents=int(sum(np.linalg.norm(a['velocity'])<1. for a in r['agents'])),
        moving_agents=int(sum(np.linalg.norm(a['velocity'])>=1. for a in r['agents'])))


def accumulate(c,r):
    c['samples']+=1;c['baseline' if r['origin']=='baseline' else 'recovered']+=1
    for key in ('grounding_available','sft_available','detection_valid','moving','turning'):c[key]+=r[key]
    c['ego_future_any']+=any(r['ego_future_valid']);c['ego_future_points']+=sum(r['ego_future_valid'])
    c['agent_future_any']+=any(any(v) for v in r['agent_future_valid'].values())
    c['agent_future_points']+=sum(sum(v) for v in r['agent_future_valid'].values())
    c['road_geometry_any']+=any(any(s['geometry']) for s in r['road']);c['road_width_any']+=any(s['width'] for s in r['road'])
    c['mode/'+r['mode_label']]+=1;c['transition/'+str(r['previous_mode'])+'->'+r['mode_label']]+=1
    for name,m,v in zip(('pcol_1s','pcol_2s','pcol_3s','dmin','areq'),r['risk_mask'],r['risk_values']):
        c['risk/'+name+'/valid']+=m
        if m:c['risk/'+name+('/positive' if v>0 else '/zero')]+=1
    for reason in r['sft_unavailable_reasons']:c['sft_excluded/'+reason]+=1
    c['unselected_hazard_sources']+=bool(r['unselected_risk_source_ids'])
    c['stationary_agents']+=r['stationary_agents'];c['moving_agents']+=r['moving_agents']
    for name,ids in r['agent_sets'].items():c[name]+=len(ids)


def absolutize(record,output):
    for sensor in record['sensors'].values():
        if isinstance(sensor,dict):
            for key,value in list(sensor.items()):
                if key in ('power','unfolded_doppler','doppler_valid','path','pointcloud'):
                    sensor[key]=[str((output/p).resolve()) for p in value] if isinstance(value,list) else str((output/value).resolve())
                elif isinstance(value,dict) and 'path' in value:value['path']=str((output/value['path']).resolve())
    record['route_archive']=str((output/record['route_archive']).resolve())


class RawInputs:
    def __init__(self,full,targets,output):
        self.full=full;self.targets=targets;self.output=output;self.raw_hashes={};self.cache_hashes={};self.cached={}
        self.products=lru_cache(maxsize=2)(lambda job:adapter._product_frames(full/'jobs'/job))
        self.anchors=lru_cache(maxsize=2)(self._anchors)
    def read(self,path):
        p=Path(path);data=p.read_bytes();self.raw_hashes[str(p.resolve())]=hashlib.sha256(data).hexdigest();return json.loads(data)
    def _anchors(self,job):
        path=self.full/'jobs'/job/'episode/anchors.jsonl';rows=adapter._selected_anchors(path,self.targets[job]) if path.is_file() else {}
        for key,r in rows.items():self.raw_hashes[f'{path}#carla_frame={key}']=hashlib.sha256(json.dumps(r,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        return rows
    def recover(self,row,split):
        job=row['job_id'];ep=self.full/'jobs'/job/'episode';index=int(row['source_index']);products=self.products(job)
        # Label recovery cannot substitute other radar timestamps or create missing products.
        if index<3 or any(i not in products for i in range(index-3,index+1)):raise ValueError('missing_four_causal_radar_products')
        frame=self.read(ep/'frames'/f'{index:06d}'/'frame.json')
        if frame['episode_id']!=row['episode_id'] or frame['carla_frame']!=row['carla_frame']:raise ValueError('candidate_frame_identity_mismatch')
        for name in ('metadata.json','route.json'):self.read(ep/name)
        for i in range(index-3,index+1):self.read(ep/'frames'/f'{i:06d}'/'frame.json')
        for name in ('agents.json','map.json'):
            path=ep/'frames'/f'{index:06d}'/name
            if path.exists():self.read(path)
        old_keys=set(self.cached)
        built=adapter._build_one(dict(row,split=split),self.full/'jobs'/job,products,self.anchors(job),self.output,self.cached,taskwise=True)
        for key in set(self.cached)-old_keys:
            _,folder,product,_=products[key[1]];base=self.cached[key][0]
            with np.load(folder/'folded_ra.npz') as power,np.load(folder/'unfolded_ra.npz') as doppler:
                expected=dict(power=np.asarray(power['power_peak_ra'][:,::-1],np.float32),
                    doppler=np.asarray(doppler['doppler_unfolded_dense_ra_mps'][:,::-1],np.float32),valid=np.asarray(doppler['reliable_mask'][:,::-1],bool))
                for suffix,array in expected.items():
                    path=base.with_name(base.name+'_'+suffix+'.npy')
                    if not np.array_equal(np.load(path,allow_pickle=False),array):raise ValueError('Saved radar cache value mismatch')
                    self.cache_hashes[str(path.resolve())]=sha256(path)
            for path in (folder/'folded_ra.npz',folder/'unfolded_ra.npz',product/'radar_axes.npz',product/'radar_meta.json',product/'complete.json'):
                self.raw_hashes[str(path.resolve())]=sha256(path)
        absolutize(built,self.output)
        packed_mask=Path(built['sensors']['radar']['doppler_valid'])
        expected_mask=np.stack([np.load(self.cached[(job,i)][0].with_name(self.cached[(job,i)][0].name+'_valid.npy'),allow_pickle=False) for i in range(index-3,index+1)])
        if not np.array_equal(np.load(packed_mask,allow_pickle=False),expected_mask):raise ValueError('Saved packed mask mismatch')
        self.cache_hashes[str(packed_mask)]=sha256(packed_mask)
        result=adapter._attach_supervision(built,taskwise=True)
        result['taskwise_recovery_provenance']=dict(job_id=job,source_index=index,carla_frame=row['carla_frame'],
            original_index_split=row['split'],assigned_episode_split=split,raw_episode=str(ep.resolve()),radar='saved products only; current adapter coordinate convention unchanged')
        return result


def migrate(source,full,output,config):
    source=Path(source).resolve();full=Path(full).resolve();output=Path(output).resolve();config=Path(config).resolve()
    if output.exists():raise FileExistsError(f'Refusing to overwrite {output}')
    cfg=json.loads(config.read_text())
    from .pipeline import resolve_config
    cfg=resolve_config(cfg)
    if cfg['data'].get('supervision_protocol')!=TASKWISE_PROTOCOL:raise ValueError('Taskwise config required')
    raw_index=full/'dataset_index.jsonl';rows=[json.loads(l) for l in raw_index.open()]
    if len(rows)!=10000 or len({(r['episode_id'],r['carla_frame']) for r in rows})!=10000:raise ValueError('Unexpected raw cohort')
    source_hash=sha256(source);episode_split={};baseline_ids=set()
    for line in source.open():
        r=json.loads(line);ep=r['episode_id'];split=r['split']
        if ep in episode_split and episode_split[ep]!=split:raise ValueError('Baseline split leakage')
        episode_split[ep]=split;baseline_ids.add((ep,r['carla_frame']))
    if len(baseline_ids)!=8835 or len(episode_split)!=64:raise ValueError('Unexpected baseline cohort')
    raw_ids={(r['episode_id'],r['carla_frame']) for r in rows}
    if not baseline_ids<=raw_ids or any(r['episode_id'] not in episode_split for r in rows):raise ValueError('Raw/baseline episode identity mismatch')
    extra=[r for r in rows if (r['episode_id'],r['carla_frame']) not in baseline_ids];targets=defaultdict(set)
    for r in extra:targets[r['job_id']].add(r['carla_frame'])
    old_exclusions=full.parent/'prepared/full_v3_0a225d38dde3/exclusions.jsonl';old_reasons={}
    for line in old_exclusions.open():
        r=json.loads(line);old_reasons[(r['job_id'],r['carla_frame'])]=r['reason']
    output.mkdir(parents=True);raw=RawInputs(full,targets,output);stats=defaultdict(lambda:defaultdict(Counter));totals=Counter();excluded=Counter();seen=set();count=0;start=time.monotonic()
    core_files=list(Path(__file__).parent.glob('*.py'))
    execution_hashes={str(p.resolve()):sha256(p) for p in core_files}
    def record(r,origin,old=None,reason=None):
        nonlocal count
        from .data import _validate_record
        _validate_record(r,count+1)
        if r['sample_id'] in seen or r['split']!=episode_split[r['episode_id']]:raise ValueError('Duplicate/split mutation')
        seen.add(r['sample_id']);detail=summary_row(r,origin,old,reason)
        manifest.write(json.dumps(r,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n');audit.write(json.dumps(detail,separators=(',',':'))+'\n')
        accumulate(totals,detail)
        for kind,key in (('split',r['split']),('scenario',r.get('category','unknown')),('episode',r['episode_id'])):accumulate(stats[kind][key],detail)
        count+=1
    partial=output/'manifest.partial.jsonl'
    try:
        with partial.open('w') as manifest,(output/'samples.jsonl').open('w') as audit,(output/'hard_exclusions.jsonl').open('w') as hard:
            for line in source.open():
                old=json.loads(line);new=annotate_tasks(old);assert_baseline(old,new);record(new,'baseline',old)
                if count%1000==0:manifest.flush();audit.flush();print('BASELINE',count,flush=True)
            for i,row in enumerate(extra,1):
                key=(row['job_id'],row['carla_frame']);reason=old_reasons.get(key,'not_in_task2_manifest')
                try:
                    new=raw.recover(row,episode_split[row['episode_id']])
                except (ValueError,FileNotFoundError,KeyError) as exc:
                    cause=str(exc)
                    allowed=('missing_four_causal_radar_products','candidate_frame_identity_mismatch','selected anchor/frame mismatch','nonconsecutive CARLA history frames','radar/episode CARLA frame mismatch','final dataset index points to a different radar product/frame','invalid radar','radar RA shape mismatch','unexpected azimuth axis','unexpected radar shape/window','invalid causal timestamps','Missing/nonfinite required ego input','Raw source frame index mismatch','Original Agent50 metadata mismatch')
                    if not isinstance(exc,FileNotFoundError) and not any(cause.startswith(prefix) for prefix in allowed):raise
                    excluded[cause]+=1
                    hard.write(json.dumps(dict(sample_id=f'{row["episode_id"]}_{row["carla_frame"]}',episode_id=row['episode_id'],
                        split=episode_split[row['episode_id']],job_id=row['job_id'],source_index=row['source_index'],
                        old_exclusion_reason=reason,reason=cause),separators=(',',':'))+'\n')
                else:record(new,'recovered',reason=reason)
                if i%25==0:
                    manifest.flush();audit.flush();hard.flush();write_json(output/'progress.json',dict(candidate_checked=i,retained=count,hard_excluded=sum(excluded.values())))
                    print(f'RECOVERY {i}/{len(extra)} retained={count} elapsed={time.monotonic()-start:.1f}s',flush=True)
    except BaseException as exc:
        write_json(output/'failure.json',dict(retained=count,error=str(exc)));raise
    if count+sum(excluded.values())!=10000:raise ValueError('Candidate accounting mismatch')
    if sha256(source)!=source_hash:raise ValueError('Baseline changed during migration')
    # Ensure this run's provenance represents the actual implementation used.
    if any(sha256(p)!=digest for p,digest in execution_hashes.items()):raise ValueError('Source changed during migration; refusing publication')
    report=dict(status='PASS',input_candidates=10000,baseline_samples=8835,additional_candidates=len(extra),samples=count,
        recovered=totals['recovered'],hard_excluded=sum(excluded.values()),hard_exclusion_reasons=excluded,episodes=len(episode_split),
        totals=totals,groups={k:dict(v) for k,v in stats.items()},source_sha256=source_hash,manifest_sha256=sha256(partial),training_started=False,optimizer_steps=0)
    write_json(output/'raw_evidence.json',raw.raw_hashes);write_json(output/'radar_cache_hashes.json',raw.cache_hashes)
    write_json(output/'migration_report.json',report)
    write_json(output/'provenance.json',dict(schema=TASKWISE_PROTOCOL,source=str(source),source_sha256=source_hash,full=str(full),
        raw_index_sha256=sha256(raw_index),old_exclusions=str(old_exclusions),old_exclusions_sha256=sha256(old_exclusions),
        config=str(config),config_sha256=sha256(config),source_files=execution_hashes,manifest_sha256=report['manifest_sha256'],
        raw_evidence_sha256=sha256(output/'raw_evidence.json'),radar_cache_hashes_sha256=sha256(output/'radar_cache_hashes.json'),
        episode_split=episode_split,git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()))
    partial.replace(output/'manifest.jsonl')
    write_json(output/'validation.json',dict(status='PASS',manifest_sha256=report['manifest_sha256'],samples=count,baseline_preserved=8835,episode_split_isolated=True))
    print('MIGRATION PASS',json.dumps(dict(samples=count,recovered=totals['recovered'],excluded=sum(excluded.values()))),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--full',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--config',type=Path,required=True);a=p.parse_args()
    migrate(a.source,a.full,a.output_dir,a.config)

if __name__=='__main__':main()
